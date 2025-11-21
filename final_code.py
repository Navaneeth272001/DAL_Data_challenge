#!/usr/bin/env python3
# ============================================================
# FINAL SCORING MODEL - METRIC LEARNING + CLASSIFICATION (0–10)
# ============================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Starting final training with transformer-based metric learning (classification 0–10)...")

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 24              # tune as per GPU
N_FOLDS = 3
EPOCHS = 15

TEXT_MAX_LENGTH = 256
METRIC_MAX_LENGTH = 64

LR = 1e-5
WEIGHT_DECAY = 0.05

CONTRASTIVE_WEIGHT = 0.2     # λ for metric-learning loss
REG_AUX_WEIGHT = 0.3         # α for regression auxiliary loss
NUM_CLASSES = 11             # scores 0..10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
BASE_PATH = '/mnt/e_disk/nk/DA24S008/DA24S008_DA5401_Data_Challenge/data'
TRAIN_FILE = f'{BASE_PATH}/train_data.json'
TEST_FILE = f'{BASE_PATH}/test_data.json'

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPUs available: {torch.cuda.device_count()}")

# You can switch to a heavier backbone if memory allows:
# MODEL_NAME = 'microsoft/deberta-v3-base'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text, max_len=1000):
    """Safely clean and truncate text."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len].strip()
    return text

def validate_and_clean_data(data, split_name=''):
    """Validate and clean data for train/test."""
    valid_data = []
    removed = 0

    for item in data:
        try:
            if not isinstance(item, dict):
                removed += 1
                continue

            metric_name = clean_text(item.get('metric_name', 'unknown'), 200)
            user_prompt = clean_text(item.get('user_prompt', ''), 2000)
            response = clean_text(item.get('response', ''), 2000)
            system_prompt = clean_text(item.get('system_prompt', ''), 2000)

            score = None
            if 'score' in item:
                try:
                    score = float(item['score'])
                    # clamp to [0, 10] and convert to int class
                    if score < 0 or score > 10:
                        removed += 1
                        continue
                    score = int(round(score))
                except (ValueError, TypeError):
                    removed += 1
                    continue

            # basic sanity checks
            if not metric_name or (not user_prompt and not response):
                removed += 1
                continue

            cleaned_item = {
                'metric_name': metric_name,
                'user_prompt': user_prompt,
                'response': response,
                'system_prompt': system_prompt if system_prompt is not None else ""
            }

            if score is not None:
                cleaned_item['score'] = score

            valid_data.append(cleaned_item)
        except Exception:
            removed += 1
            continue

    print(f"  {split_name}: {len(data)} → {len(valid_data)} valid (removed {removed})")
    return valid_data

# ============================================================
# DATASET
# ============================================================

class FinalScoringDataset(Dataset):
    """
    Dataset that returns:
    - text_input_ids / text_attention_mask: encodes [user_prompt] [SEP] [response]
    - metric_input_ids / metric_attention_mask: encodes [metric_name] [SEP] [system_prompt]
    - label: integer class 0–10 (for train/val only)
    - score_float: float version (for aux regression)
    """
    def __init__(
        self,
        data,
        tokenizer,
        text_max_len=256,
        metric_max_len=64,
        is_test=False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.metric_max_len = metric_max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        metric_name = clean_text(item.get('metric_name', 'unknown'), 200)
        user_prompt = clean_text(item.get('user_prompt', ''), 2000)
        response = clean_text(item.get('response', ''), 2000)
        system_prompt = clean_text(item.get('system_prompt', ''), 2000)

        # Build metric text: metric_name [+ system_prompt if present]
        if system_prompt:
            metric_text = f"{metric_name} [SEP] {system_prompt}"
        else:
            metric_text = metric_name

        # Build main text: user_prompt + response
        if user_prompt and response:
            text = f"{user_prompt} [SEP] {response}"
        elif user_prompt:
            text = user_prompt
        else:
            text = response if response else "placeholder"

        # Tokenize text tower
        text_enc = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize metric tower
        metric_enc = self.tokenizer(
            metric_text,
            max_length=self.metric_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'text_input_ids': text_enc['input_ids'].squeeze(0),
            'text_attention_mask': text_enc['attention_mask'].squeeze(0),
            'metric_input_ids': metric_enc['input_ids'].squeeze(0),
            'metric_attention_mask': metric_enc['attention_mask'].squeeze(0),
        }

        if not self.is_test:
            # score already validated to be integer in [0, 10]
            score_int = int(item.get('score', 5))
            score_int = max(0, min(10, score_int))
            result['label'] = torch.tensor(score_int, dtype=torch.long)
            result['score_float'] = torch.tensor([float(score_int)], dtype=torch.float32)

        return result

# ============================================================
# MODEL
# ============================================================

class ScoringModel(nn.Module):
    """
    Two-tower transformer:
    - Tower 1 encodes (user_prompt + response)
    - Tower 2 encodes (metric_name + system_prompt)
    Heads:
      - classifier: 11 classes (0..10)
      - regressor: scalar aux regression (0..10)
    Metric learning:
      - InfoNCE loss between text_feat and metric_feat.
    """
    def __init__(
        self,
        model_name=MODEL_NAME,
        dropout=0.3,
        num_classes=NUM_CLASSES
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # CLS pooling
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        metric_input_ids,
        metric_attention_mask
    ):
        text_feat = self.encode(text_input_ids, text_attention_mask)
        metric_feat = self.encode(metric_input_ids, metric_attention_mask)
        combined = torch.cat([text_feat, metric_feat], dim=1)

        logits = self.classifier(combined)
        reg_pred = self.regressor(combined)

        return logits, reg_pred, text_feat, metric_feat

# ============================================================
# LOSSES
# ============================================================

def compute_contrastive_loss(text_feat, metric_feat):
    """
    InfoNCE-style loss:
    - text_feat: (B, D)
    - metric_feat: (B, D)
    Each text should match only its corresponding metric (same index).
    """
    text_norm = F.normalize(text_feat, dim=1)
    metric_norm = F.normalize(metric_feat, dim=1)
    # Similarity matrix: (B, B)
    logits = text_norm @ metric_norm.t()
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, targets)
    return loss

# ============================================================
# TRAINING & VALIDATION
# ============================================================

def train_epoch(
    model,
    loader,
    opt,
    device,
    contrastive_weight=CONTRASTIVE_WEIGHT,
    reg_aux_weight=REG_AUX_WEIGHT
):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_ce = 0.0
    total_reg = 0.0
    total_contrast = 0.0
    steps = 0

    for batch in tqdm(loader, desc='Train', leave=False):
        try:
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            metric_ids = batch['metric_input_ids'].to(device)
            metric_mask = batch['metric_attention_mask'].to(device)
            labels = batch['label'].to(device)              # (B,)
            scores_float = batch['score_float'].to(device)  # (B, 1)

            opt.zero_grad()

            logits, reg_pred, text_feat, metric_feat = model(
                text_ids,
                text_mask,
                metric_ids,
                metric_mask
            )

            ce_loss = ce_loss_fn(logits, labels)
            reg_loss = mse_loss_fn(reg_pred, scores_float)
            contrastive_loss = compute_contrastive_loss(text_feat, metric_feat)

            loss = ce_loss + reg_aux_weight * reg_loss + contrastive_weight * contrastive_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_reg += reg_loss.item()
            total_contrast += contrastive_loss.item()
            steps += 1
        except RuntimeError as e:
            # handle OOM or similar
            print(f"  [WARN] Skipping batch due to RuntimeError: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [WARN] Skipping batch due to error: {e}")
            continue

    if steps == 0:
        return float('inf'), float('inf'), float('inf'), float('inf')

    return (
        total_loss / steps,
        total_ce / steps,
        total_reg / steps,
        total_contrast / steps
    )

def validate(model, loader, device):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    total_ce = 0.0
    total_reg = 0.0
    steps = 0

    all_preds_cls = []
    all_targets_cls = []
    all_preds_reg = []
    all_targets_reg = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            try:
                text_ids = batch['text_input_ids'].to(device)
                text_mask = batch['text_attention_mask'].to(device)
                metric_ids = batch['metric_input_ids'].to(device)
                metric_mask = batch['metric_attention_mask'].to(device)
                labels = batch['label'].to(device)              # (B,)
                scores_float = batch['score_float'].to(device)  # (B,1)

                logits, reg_pred, text_feat, metric_feat = model(
                    text_ids,
                    text_mask,
                    metric_ids,
                    metric_mask
                )

                ce_loss = ce_loss_fn(logits, labels)
                reg_loss = mse_loss_fn(reg_pred, scores_float)

                total_ce += ce_loss.item()
                total_reg += reg_loss.item()
                steps += 1

                preds_cls = logits.argmax(dim=1)  # (B,)
                all_preds_cls.append(preds_cls.cpu().numpy())
                all_targets_cls.append(labels.cpu().numpy())

                all_preds_reg.append(reg_pred.cpu().numpy())
                all_targets_reg.append(scores_float.cpu().numpy())
            except Exception as e:
                print(f"  [WARN] Skipping val batch due to error: {e}")
                continue

    if steps == 0:
        return float('inf'), float('inf'), float('inf'), float('inf')

    all_preds_cls = np.concatenate(all_preds_cls, axis=0)
    all_targets_cls = np.concatenate(all_targets_cls, axis=0)

    all_preds_reg = np.concatenate(all_preds_reg, axis=0).reshape(-1)
    all_targets_reg = np.concatenate(all_targets_reg, axis=0).reshape(-1)

    # classification accuracy
    acc = (all_preds_cls == all_targets_cls).mean()

    # RMSE using class predictions vs true scores (ordinal metric)
    rmse_class = np.sqrt(np.mean((all_preds_cls - all_targets_cls) ** 2))

    # RMSE using regression head
    rmse_reg = np.sqrt(np.mean((all_preds_reg - all_targets_reg) ** 2))

    return (
        total_ce / steps,
        total_reg / steps,
        acc,
        rmse_class
    )

# ============================================================
# MAIN
# ============================================================

print("\n" + "="*70)
print("LOADING AND VALIDATING DATA")
print("="*70)

try:
    print(f"\nLoading from paths:")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Test:  {TEST_FILE}")

    with open(TRAIN_FILE, 'r') as f:
        train_data_raw = json.load(f)
    with open(TEST_FILE, 'r') as f:
        test_data_raw = json.load(f)

    print("✓ Train & Test loaded successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    raise SystemExit(1)

print("\nValidating data...")
train_data = validate_and_clean_data(train_data_raw, "Train")
test_data = validate_and_clean_data(test_data_raw, "Test")

if len(train_data) == 0:
    print("❌ No valid training data found.")
    raise SystemExit(1)

print("✓ Data validation complete")

# Tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# K-FOLD TRAINING
# ============================================================

print("\n" + "="*70)
print(f"TRAINING {N_FOLDS} FOLD MODELS WITH METRIC LEARNING + CLASSIFICATION")
print("="*70)

models = []
np.random.seed(42)
np.random.shuffle(train_data)
fold_size = len(train_data) // N_FOLDS

for fold in range(N_FOLDS):
    print(f"\nFold {fold+1}/{N_FOLDS}")
    print("-" * 70)

    fold_start = fold * fold_size
    fold_end = (fold + 1) * fold_size if fold < N_FOLDS - 1 else len(train_data)

    val_data = train_data[fold_start:fold_end]
    train_fold = train_data[:fold_start] + train_data[fold_end:]

    print(f"  Train: {len(train_fold)}, Val: {len(val_data)}")

    train_ds = FinalScoringDataset(
        train_fold,
        tokenizer,
        text_max_len=TEXT_MAX_LENGTH,
        metric_max_len=METRIC_MAX_LENGTH,
        is_test=False
    )
    val_ds = FinalScoringDataset(
        val_data,
        tokenizer,
        text_max_len=TEXT_MAX_LENGTH,
        metric_max_len=METRIC_MAX_LENGTH,
        is_test=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0
    )

    model = ScoringModel(
        model_name=MODEL_NAME,
        dropout=0.3,
        num_classes=NUM_CLASSES
    )
    model.to(DEVICE)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    best_rmse = float('inf')
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_ce, train_reg, train_contrast = train_epoch(
            model,
            train_loader,
            opt,
            DEVICE,
            contrastive_weight=CONTRASTIVE_WEIGHT,
            reg_aux_weight=REG_AUX_WEIGHT
        )
        val_ce, val_reg, val_acc, val_rmse_class = validate(
            model,
            val_loader,
            DEVICE
        )
        # Scheduler on CE loss
        sched.step(val_ce)

        print(
            f"  E{epoch:02d}: "
            f"TrainLoss={train_loss:.4f} | CE={train_ce:.4f} | Reg={train_reg:.4f} | Contr={train_contrast:.4f} || "
            f"ValCE={val_ce:.4f} | ValReg={val_reg:.4f} | Acc={val_acc:.4f} | RMSE_class={val_rmse_class:.4f}"
        )

        # Use RMSE on discrete classes as main early-stopping criterion
        if val_rmse_class < best_rmse:
            best_rmse = val_rmse_class
            best_acc = val_acc
            try:
                torch.save(
                    model.module.state_dict() if isinstance(model, nn.DataParallel)
                    else model.state_dict(),
                    f'model_metric_cls_fold{fold}.pt'
                )
            except Exception as e:
                print(f"  [WARN] Could not save model: {e}")

    # Reload best weights
    try:
        state_dict = torch.load(f'model_metric_cls_fold{fold}.pt', map_location=DEVICE)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"  [WARN] Could not load best model for fold {fold}: {e}")

    models.append(model)
    print(f"  ✓ Fold {fold+1} complete (Best RMSE_class: {best_rmse:.4f}, Best Acc: {best_acc:.4f})")

# ============================================================
# PREDICTION
# ============================================================

print("\n" + "="*70)
print("GENERATING PREDICTIONS (DISCRETE 0–10)")
print("="*70)

test_ds = FinalScoringDataset(
    test_data,
    tokenizer,
    text_max_len=TEXT_MAX_LENGTH,
    metric_max_len=METRIC_MAX_LENGTH,
    is_test=True
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0
)

all_fold_class_preds = []

for fold_idx, model in enumerate(models):
    print(f"\nFold {fold_idx+1} predictions...")
    model.eval()
    preds_fold_cls = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            try:
                text_ids = batch['text_input_ids'].to(DEVICE)
                text_mask = batch['text_attention_mask'].to(DEVICE)
                metric_ids = batch['metric_input_ids'].to(DEVICE)
                metric_mask = batch['metric_attention_mask'].to(DEVICE)

                logits, reg_pred, text_feat, metric_feat = model(
                    text_ids,
                    text_mask,
                    metric_ids,
                    metric_mask
                )

                preds_cls = logits.argmax(dim=1)  # (B,)
                preds_fold_cls.append(preds_cls.cpu().numpy())
            except Exception as e:
                print(f"  [WARN] Skipping test batch due to error: {e}")
                continue

    if len(preds_fold_cls) > 0:
        preds_fold_cls = np.concatenate(preds_fold_cls, axis=0)
        all_fold_class_preds.append(preds_fold_cls)

if len(all_fold_class_preds) == 0:
    print("No predictions generated!")
    raise SystemExit(1)

# Ensemble by majority-vote / mean-then-round
stacked = np.stack(all_fold_class_preds, axis=0)  # (F, N)
ensemble_scores = np.rint(stacked.mean(axis=0)).astype(int)  # average logits→classes, then round
ensemble_scores = np.clip(ensemble_scores, 0, 10)

print("\nPrediction distribution (0–10):")
unique, counts = np.unique(ensemble_scores, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Score {u}: {c} samples")

df = pd.DataFrame({
    'ID': np.arange(1, len(ensemble_scores) + 1),
    'score': ensemble_scores
})
df.to_csv('submission_final_metric_classifier.csv', index=False)

print("\n" + "="*70)
print("✓ TRAINING COMPLETE (CLASSIFICATION 0–10 + METRIC LEARNING)!")
print("="*70)
print(f"  Output file: submission_final_metric_classifier.csv")
print(f"  Rows: {len(df)}")
print("=======================================================")
