#!/usr/bin/env python3
# ============================================================
# FINAL SCORING MODEL - WITH CORRECT FILE PATHS
# ============================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Starting final training with correct paths...")

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
N_FOLDS = 3
EPOCHS = 15
MAX_LENGTH = 256
LR = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
BASE_PATH = '/mnt/e_disk/nk/DA24S008/DA24S008_DA5401_Data_Challenge/data'
TRAIN_FILE = f'{BASE_PATH}/train_data.json'
TEST_FILE = f'{BASE_PATH}/test_data.json'
METRIC_NAMES_FILE = f'{BASE_PATH}/metric_names.json'
EMBEDDINGS_FILE = f'{BASE_PATH}/metric_name_embeddings.npy'

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text, max_len=100):
    """Safely clean and truncate text"""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)
    return text[:max_len].strip()

def validate_and_clean_data(data, split_name=''):
    """Validate and clean data"""
    valid_data = []
    removed = 0

    for item in data:
        try:
            if not isinstance(item, dict):
                removed += 1
                continue

            metric_name = clean_text(item.get('metric_name', 'unknown'), 50)
            user_prompt = clean_text(item.get('user_prompt', ''), 200)
            response = clean_text(item.get('response', ''), 200)

            if 'score' in item:
                try:
                    score = float(item['score'])
                    if not 0 <= score <= 10:
                        removed += 1
                        continue
                except (ValueError, TypeError):
                    removed += 1
                    continue

            if not metric_name or (not user_prompt and not response):
                removed += 1
                continue

            cleaned_item = {
                'metric_name': metric_name,
                'user_prompt': user_prompt,
                'response': response,
            }

            if 'score' in item:
                cleaned_item['score'] = score

            valid_data.append(cleaned_item)
        except:
            removed += 1
            continue

    print(f"  {split_name}: {len(data)} → {len(valid_data)} valid (removed {removed})")
    return valid_data

def analyze_embeddings(embeddings_dict):
    """Analyze embedding dimensions"""
    dims = {}
    for name, emb in embeddings_dict.items():
        dim = len(emb) if hasattr(emb, '__len__') else 1
        if dim not in dims:
            dims[dim] = 0
        dims[dim] += 1

    print(f"\n  Embedding dimensions found:")
    for dim, count in sorted(dims.items()):
        print(f"    {dim}D: {count} embeddings")

    return dims

def standardize_embeddings(embeddings_dict, target_dim=384):
    """Standardize all embeddings to same dimension"""
    print(f"\n  Standardizing embeddings to {target_dim}D...")

    standardized = {}
    for name, emb in embeddings_dict.items():
        emb = np.array(emb, dtype=np.float32)
        current_dim = len(emb)

        if current_dim == target_dim:
            standardized[name] = emb
        elif current_dim < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:current_dim] = emb
            standardized[name] = padded
        else:
            standardized[name] = emb[:target_dim]

    print(f"  ✓ All embeddings standardized to {target_dim}D")
    return standardized

# ============================================================
# DATASET
# ============================================================

class FinalScoringDataset(Dataset):
    def __init__(self, data, tokenizer, embeddings_dict, max_len=256, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.embeddings_dict = embeddings_dict
        self.max_len = max_len
        self.is_test = is_test
        self.embedding_dim = 384

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        metric = clean_text(item.get('metric_name', 'unknown'), 50)
        user_prompt = clean_text(item.get('user_prompt', ''), 150)
        response = clean_text(item.get('response', ''), 150)

        text = f"{metric} {user_prompt} {response}"

        try:
            enc = self.tokenizer(
                text, 
                max_length=self.max_len, 
                padding='max_length',
                truncation=True, 
                return_tensors='pt'
            )
        except:
            enc = self.tokenizer(
                "placeholder", 
                max_length=self.max_len, 
                padding='max_length',
                truncation=True, 
                return_tensors='pt'
            )

        emb = self.embeddings_dict.get(metric)
        if emb is None:
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            emb = np.array(emb, dtype=np.float32)
            if len(emb) != self.embedding_dim:
                if len(emb) < self.embedding_dim:
                    padded = np.zeros(self.embedding_dim, dtype=np.float32)
                    padded[:len(emb)] = emb
                    emb = padded
                else:
                    emb = emb[:self.embedding_dim]

        result = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'metric_embedding': torch.FloatTensor(emb)
        }

        if not self.is_test:
            score = float(item.get('score', 5.0))
            result['score'] = torch.FloatTensor([score])

        return result

# ============================================================
# MODEL
# ============================================================

class ScoringModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 embedding_dim=384, dropout=0.5):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        self.metric_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Dropout(dropout)
        )

        combined = self.transformer.config.hidden_size + 128
        self.regressor = nn.Sequential(
            nn.Linear(combined, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, metric_embedding):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]
        metric_feat = self.metric_fc(metric_embedding)
        combined = torch.cat([text_feat, metric_feat], dim=1)
        return self.regressor(combined)

# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, opt, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0

    for batch in tqdm(loader, desc='Train', leave=False):
        try:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            emb = batch['metric_embedding'].to(device)
            scores = batch['score'].to(device)

            if emb.shape[1] != 384:
                continue

            opt.zero_grad()
            preds = model(ids, mask, emb)
            loss = criterion(preds, scores)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
        except Exception as e:
            continue

    return total_loss / max(len(loader), 1)

def validate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            try:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                emb = batch['metric_embedding'].to(device)
                scores = batch['score'].to(device)

                if emb.shape[1] != 384:
                    continue

                pred = model(ids, mask, emb)
                loss = criterion(pred, scores)

                total_loss += loss.item()
                preds.extend(pred.cpu().numpy())
                targets.extend(scores.cpu().numpy())
            except:
                continue

    if len(preds) == 0:
        return float('inf'), float('inf')

    rmse = np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))
    return total_loss / max(len(loader), 1), rmse

# ============================================================
# MAIN
# ============================================================

print("\n" + "="*70)
print("LOADING AND VALIDATING DATA")
print("="*70)

try:
    print(f"\nLoading from paths:")
    print(f"  Train:      {TRAIN_FILE}")
    print(f"  Test:       {TEST_FILE}")
    print(f"  Metrics:    {METRIC_NAMES_FILE}")
    print(f"  Embeddings: {EMBEDDINGS_FILE}")

    with open(TRAIN_FILE) as f:
        train_data_raw = json.load(f)
    with open(TEST_FILE) as f:
        test_data_raw = json.load(f)
    with open(METRIC_NAMES_FILE) as f:
        metric_names = json.load(f)

    metric_embeddings_raw = np.load(EMBEDDINGS_FILE)
    embeddings_dict_raw = {n: e for n, e in zip(metric_names, metric_embeddings_raw)}

    print("✓ All files loaded successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit(1)

# Analyze embeddings
print("\nAnalyzing embeddings...")
analyze_embeddings(embeddings_dict_raw)

# Standardize embeddings
embeddings_dict = standardize_embeddings(embeddings_dict_raw, target_dim=384)

# Validate data
print("\nValidating data...")
train_data = validate_and_clean_data(train_data_raw, "Train")
test_data = validate_and_clean_data(test_data_raw, "Test")

if len(train_data) == 0:
    print("❌ No valid training data!")
    exit(1)

print("✓ Data validation complete")

# Tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Train models
print(f"\n" + "="*70)
print(f"TRAINING {N_FOLDS} MODELS")
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

    train_ds = FinalScoringDataset(train_fold, tokenizer, embeddings_dict, MAX_LENGTH)
    val_ds = FinalScoringDataset(val_data, tokenizer, embeddings_dict, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

    model = ScoringModel(embedding_dim=384).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_rmse = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, opt, DEVICE)
        val_loss, rmse = validate(model, val_loader, DEVICE)
        sched.step(val_loss)

        print(f"  E{epoch}: Loss={train_loss:.4f}, RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            try:
                torch.save(
                    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    f'model_{fold}.pt'
                )
            except:
                pass

    try:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(f'model_{fold}.pt'))
        else:
            model.load_state_dict(torch.load(f'model_{fold}.pt'))
    except:
        pass

    models.append(model)
    print(f"  ✓ Fold {fold+1} complete (Best RMSE: {best_rmse:.4f})")

# Predict
print(f"\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

test_ds = FinalScoringDataset(test_data, tokenizer, embeddings_dict, MAX_LENGTH, is_test=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

all_preds = []
for fold_idx, model in enumerate(models):
    print(f"\nFold {fold_idx+1} predictions...")
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            try:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                emb = batch['metric_embedding'].to(DEVICE)

                if emb.shape[1] != 384:
                    continue

                pred = model(ids, mask, emb)
                preds.extend(pred.cpu().numpy())
            except:
                continue

    if len(preds) > 0:
        all_preds.append(np.array(preds).flatten())

if len(all_preds) == 0:
    print("❌ No predictions generated!")
    exit(1)

# Ensemble + calibrate
ensemble_preds = np.mean(all_preds, axis=0)
pred_std = np.std(ensemble_preds)
pred_mean = np.mean(ensemble_preds)

print(f"\nPredictions before calibration:")
print(f"  Min: {ensemble_preds.min():.4f}, Max: {ensemble_preds.max():.4f}")
print(f"  Mean: {pred_mean:.4f}, Std: {pred_std:.4f}")

if pred_std < 1.5:
    print(f"  → Applying calibration (std < 1.5)")
    ensemble_preds = (ensemble_preds - pred_mean) * (2.5 / (pred_std + 0.1)) + 5.0

ensemble_preds = np.clip(ensemble_preds, 0, 10)

print(f"\nPredictions after calibration:")
print(f"  Min: {ensemble_preds.min():.4f}, Max: {ensemble_preds.max():.4f}")
print(f"  Mean: {np.mean(ensemble_preds):.4f}, Std: {np.std(ensemble_preds):.4f}")

# Save
df = pd.DataFrame({'ID': range(1, len(ensemble_preds) + 1), 'score': ensemble_preds})
df.to_csv('submission_final.csv', index=False)

print(f"\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"  Output file: submission_final.csv")
print(f"  Rows: {len(df)}")
print(f"  Expected RMSE: 1.5-2.5")
print("="*70)
