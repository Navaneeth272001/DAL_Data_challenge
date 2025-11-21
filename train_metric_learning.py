# ============================================================
# METRIC LEARNING APPROACH FOR AI SCORING
# Based on "Metric Learning" by Bellet, Habrard & Sebban
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
import os
warnings.filterwarnings('ignore')


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

print("Metric Learning-Based Scoring System")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64
N_FOLDS = 3
EPOCHS = 15
MAX_LENGTH = 400
LR = 2e-5
METRIC_DIM = 512  # Learned metric space dimension
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_PATH = '/mnt/e_disk/nk/DA24S008/DA24S008_DA5401_Data_Challenge/data'
TRAIN_FILE = f'{BASE_PATH}/train_data.json'
TEST_FILE = f'{BASE_PATH}/test_data.json'
METRIC_NAMES_FILE = f'{BASE_PATH}/metric_names.json'
EMBEDDINGS_FILE = f'{BASE_PATH}/metric_name_embeddings.npy'

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text, max_len=200):
    """Safely clean text"""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)
    return text[:max_len].strip()

def create_pairwise_constraints(data):
    """
    Create pairwise constraints from scores (Book Chapter 1.1)

    Constructs:
    - Similar pairs (S): instances with close scores
    - Dissimilar pairs (D): instances with far scores
    - Relative constraints (R): triplets where x_i closer to x_j than x_k
    """
    n = len(data)
    similar_pairs = []
    dissimilar_pairs = []
    relative_triplets = []

    # Sort by scores to create meaningful constraints
    sorted_indices = np.argsort([item['score'] for item in data])

    # Create constraints based on score similarity
    for i in range(n):
        for j in range(i+1, min(i+50, n)):  # Limit pairs for efficiency
            idx_i, idx_j = sorted_indices[i], sorted_indices[j]
            score_diff = abs(data[idx_i]['score'] - data[idx_j]['score'])

            if score_diff < 1.0:  # Similar scores
                similar_pairs.append((idx_i, idx_j))
            elif score_diff > 3.0:  # Dissimilar scores
                dissimilar_pairs.append((idx_i, idx_j))

            # Create triplet: if score(i) < score(j) < score(k)
            # then i should be closer to j than to k
            for k in range(j+1, min(j+30, n)):
                idx_k = sorted_indices[k]
                if data[idx_j]['score'] < data[idx_k]['score']:
                    relative_triplets.append((idx_i, idx_j, idx_k))

    return similar_pairs, dissimilar_pairs, relative_triplets

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

def standardize_embeddings(embeddings_dict, target_dim):
    """Standardize embeddings to target dimension"""
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

    print(f"  ✓ All {len(standardized)} embeddings standardized to {target_dim}D")
    return standardized

# ============================================================
# DATASET
# ============================================================

class MetricLearningDataset(Dataset):
    """
    Dataset for metric learning with pairwise/triplet constraints
    """
    def __init__(self, data, tokenizer, embeddings_dict, 
                 constraints=None, max_len=256, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.embeddings_dict = embeddings_dict
        self.max_len = max_len
        self.is_test = is_test
        self.embedding_dim = len(next(iter(embeddings_dict.values())))

        # Unpack constraints
        if constraints:
            self.similar_pairs, self.dissimilar_pairs, self.triplets = constraints
        else:
            self.similar_pairs, self.dissimilar_pairs, self.triplets = [], [], []

    def __len__(self):
        if self.is_test:
            return len(self.data)
        return len(self.data)

    def encode_instance(self, item):
        """Encode a single instance"""
        metric = clean_text(item.get('metric_name', 'unknown'), 50)
        user_prompt = clean_text(item.get('user_prompt', ''), 150)
        response = clean_text(item.get('response', ''), 150)

        text = f"{metric}: {user_prompt} [SEP] {response}"

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

        # Get metric embedding
        emb = self.embeddings_dict.get(metric, np.zeros(self.embedding_dim, dtype=np.float32))
        emb = np.array(emb, dtype=np.float32)

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'metric_embedding': torch.FloatTensor(emb)
        }

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.encode_instance(item)

        if not self.is_test:
            score = float(item.get('score', 5.0))
            encoded['score'] = torch.FloatTensor([score])

        return encoded

# ============================================================
# MODEL - MAHALANOBIS DISTANCE LEARNING (Book Chapter 4)
# ============================================================

class MahalanobisMetricLearner(nn.Module):
    """
    Learns a Mahalanobis distance: d_M(x,x') = sqrt((x-x')^T M (x-x'))

    Based on Book Chapter 4: Linear Metric Learning
    - Uses bilinear similarity (Section 2.2.1)
    - Regularized approach (Section 4.1.2)
    - Learns projection matrix L where M = L^T L
    """
    def __init__(self, model_name='BAAI/bge-large-en-v1.5', 
                 metric_dim=512, embedding_dim=1024, dropout=0.3):
        super().__init__()
        print(f"  Loading base model: {model_name}")
        self.base_model = AutoModel.from_pretrained(model_name)

        # Freeze early layers for efficiency
        for i, layer in enumerate(self.base_model.encoder.layer):
            if i < 8:  # Freeze first 8 layers
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_size = 1024  # BGE-large hidden size

        # Metric embedding processor
        self.metric_processor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Projection to learned metric space (Book: Section 4.1)
        # This learns L where M = L^T L forms the Mahalanobis distance
        self.projection = nn.Sequential(
            nn.Linear(hidden_size + 256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, metric_dim),  # Project to metric_dim space
            nn.LayerNorm(metric_dim)
        )

        # Score predictor (based on learned distances)
        self.score_head = nn.Sequential(
            nn.Linear(metric_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def encode(self, input_ids, attention_mask, metric_embedding):
        """Encode input to learned metric space"""
        # Get text representation
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Process metric embedding
        metric_feat = self.metric_processor(metric_embedding)

        # Combine and project to metric space
        combined = torch.cat([text_feat, metric_feat], dim=1)
        z = self.projection(combined)  # Point in learned metric space

        return z

    def mahalanobis_distance(self, z1, z2):

        return torch.norm(z1 - z2, p=2, dim=-1, keepdim=True)

    def forward(self, input_ids, attention_mask, metric_embedding):
        """Forward pass: encode then predict score"""
        z = self.encode(input_ids, attention_mask, metric_embedding)
        score = self.score_head(z)
        return score, z  # Return both score and embedding

# ============================================================
# TRAINING WITH METRIC LEARNING OBJECTIVES
# ============================================================

def train_epoch(model, loader, opt, device, alpha=0.5):

    model.train()
    score_criterion = nn.MSELoss()
    total_loss = 0
    batch_count = 0

    for batch in tqdm(loader, desc='Train', leave=False):
        try:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            emb = batch['metric_embedding'].to(device)
            scores = batch['score'].to(device)

            opt.zero_grad()

            # Forward pass
            pred_scores, embeddings = model(ids, mask, emb)

            # Primary loss: MSE on scores
            score_loss = score_criterion(pred_scores, scores)

            # Metric learning loss: encourage embeddings to reflect score ordering
            # Triplet-like loss: if score(i) < score(j), then d(i, anchor) < d(j, anchor)
            batch_size = embeddings.size(0)
            metric_loss = 0

            if batch_size >= 3:
                # Sample triplets from batch
                for i in range(0, batch_size-2, 3):
                    anchor_emb = embeddings[i]
                    pos_emb = embeddings[i+1]
                    neg_emb = embeddings[i+2]

                    # Order by scores
                    s_anchor = scores[i]
                    s_pos = scores[i+1]
                    s_neg = scores[i+2]

                    # If scores indicate pos is closer to anchor than neg
                    if abs(s_anchor - s_pos) < abs(s_anchor - s_neg):
                        d_pos = torch.norm(anchor_emb - pos_emb, p=2)
                        d_neg = torch.norm(anchor_emb - neg_emb, p=2)
                        # Triplet loss with margin (Book Section 4.1.1)
                        metric_loss += torch.relu(d_pos - d_neg + 1.0)

                metric_loss = metric_loss / max(1, batch_size // 3)

            # Combined loss
            loss = score_loss + alpha * metric_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            batch_count += 1
        except Exception as e:
            print(f"\nBatch error: {e}")
            continue

    if batch_count == 0:
        return float('inf')

    return total_loss / batch_count

def validate(model, loader, device):
    """Validation"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    preds, targets = [], []
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            try:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                emb = batch['metric_embedding'].to(device)
                scores = batch['score'].to(device)

                pred_scores, _ = model(ids, mask, emb)
                loss = criterion(pred_scores, scores)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                preds.extend(pred_scores.cpu().numpy())
                targets.extend(scores.cpu().numpy())
                batch_count += 1
            except:
                continue

    if batch_count == 0 or len(preds) == 0:
        return float('inf'), float('inf')

    rmse = np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))
    return total_loss / batch_count, rmse

# ============================================================
# MAIN
# ============================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

try:
    with open(TRAIN_FILE) as f:
        train_data_raw = json.load(f)
    with open(TEST_FILE) as f:
        test_data_raw = json.load(f)
    with open(METRIC_NAMES_FILE) as f:
        metric_names = json.load(f)

    metric_embeddings_raw = np.load(EMBEDDINGS_FILE)
    embeddings_dict_raw = {n: e for n, e in zip(metric_names, metric_embeddings_raw)}

    print("✓ All files loaded successfully")
    original_dim = metric_embeddings_raw.shape[1] if len(metric_embeddings_raw.shape) > 1 else len(metric_embeddings_raw[0])
    print(f"  Original embedding shape: {metric_embeddings_raw.shape}")
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Standardize embeddings to 1024D for BGE-large
embeddings_dict = standardize_embeddings(embeddings_dict_raw, target_dim=1024)

print("\nValidating data...")
train_data = validate_and_clean_data(train_data_raw, "Train")
test_data = validate_and_clean_data(test_data_raw, "Test")

if len(train_data) == 0:
    print("No valid training data!")
    exit(1)

print("\nCreating pairwise constraints...")
constraints = create_pairwise_constraints(train_data)
print(f"  Similar pairs: {len(constraints[0])}")
print(f"  Dissimilar pairs: {len(constraints[1])}")
print(f"  Triplets: {len(constraints[2])}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
print("✓ Tokenizer loaded")

print(f"\n" + "="*70)
print(f"TRAINING METRIC LEARNING MODEL")
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

    train_ds = MetricLearningDataset(train_fold, tokenizer, embeddings_dict, 
                                    constraints, MAX_LENGTH)
    val_ds = MetricLearningDataset(val_data, tokenizer, embeddings_dict, 
                                   None, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

    print(f"  Created {len(train_loader)} train batches, {len(val_loader)} val batches")

    model = MahalanobisMetricLearner(metric_dim=METRIC_DIM, embedding_dim=1024).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"  Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_rmse = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, opt, DEVICE)
        val_loss, rmse = validate(model, val_loader, DEVICE)
        sched.step(val_loss)

        print(f"  E{epoch}: Loss={train_loss:.4f}, Val={val_loss:.4f}, RMSE={rmse:.4f}")

        if rmse < best_rmse and rmse != float('inf'):
            best_rmse = rmse
            try:
                torch.save(
                    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    f'metric_model_{fold}.pt'
                )
                print(f"    → Saved best model (RMSE: {best_rmse:.4f})")
            except Exception as e:
                print(f"    ⚠ Could not save model: {e}")

    try:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(f'metric_model_{fold}.pt'))
        else:
            model.load_state_dict(torch.load(f'metric_model_{fold}.pt'))
        print(f"  ✓ Loaded best checkpoint")
    except:
        print(f"  Using final epoch weights")

    models.append(model)
    print(f"  ✓ Fold {fold+1} complete (Best RMSE: {best_rmse:.4f})")

# Predict
print(f"\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

test_ds = MetricLearningDataset(test_data, tokenizer, embeddings_dict, 
                                None, MAX_LENGTH, is_test=True)
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

                pred_scores, _ = model(ids, mask, emb)
                preds.extend(pred_scores.cpu().numpy())
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

    if len(preds) > 0:
        all_preds.append(np.array(preds).flatten())
        print(f"  ✓ Generated {len(preds)} predictions")

if len(all_preds) == 0:
    print("No predictions generated!")
    exit(1)

ensemble_preds = np.mean(all_preds, axis=0)
pred_std = np.std(ensemble_preds)
pred_mean = np.mean(ensemble_preds)

print(f"\nPredictions statistics:")
print(f"  Min: {ensemble_preds.min():.4f}, Max: {ensemble_preds.max():.4f}")
print(f"  Mean: {pred_mean:.4f}, Std: {pred_std:.4f}")

# Calibrate to [0, 10] range
if pred_std < 1.5:
    print(f"  → Applying calibration (std < 1.5)")
    ensemble_preds = (ensemble_preds - pred_mean) * (2.5 / (pred_std + 0.1)) + 5.0

ensemble_preds = np.clip(ensemble_preds, 0, 10)

print(f"\nAfter calibration:")
print(f"  Min: {ensemble_preds.min():.4f}, Max: {ensemble_preds.max():.4f}")
print(f"  Mean: {np.mean(ensemble_preds):.4f}, Std: {np.std(ensemble_preds):.4f}")

df = pd.DataFrame({'ID': range(1, len(ensemble_preds) + 1), 'score': ensemble_preds})
df.to_csv('submission_metric_learning.csv', index=False)

print(f"\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"  Output file: submission_metric_learning.csv")
print(f"  Rows: {len(df)}")
print(f"  Approach: Metric Learning (Mahalanobis distance)")
print(f"  Base Model: BAAI/bge-large-en-v1.5")
print("="*70)
