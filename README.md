
# AI Assistant Response Scoring

## Project Overview

This repository provides several modeling approaches for predicting a score (0–10) that measures how well an assistant’s response satisfies a given metric. It is designed for an AI chatbot evaluation task, where each data point includes a user prompt, an assistant response, (optionally a system prompt), and a metric name. The project includes:

- **Data processing:** Loading and cleaning JSON data (user/system prompts, responses, metric names, and scores).  
- **Transformer models:** Fine-tuning pre-trained language models (e.g. XLM-RoBERTa, DistilBERT, Sentence Transformers like MiniLM, and BGE-Large) for regression or classification on text inputs.  
- **Metric learning:** Dual-tower architectures with contrastive (InfoNCE) loss to align response and metric embeddings, and Mahalanobis metric learning to project inputs into a learned distance space.  
- **Gradient-Boosted Trees (GBDTs):** Training XGBoost, LightGBM, and CatBoost regressors on engineered text features (e.g. TF-IDF of prompts and responses).  
- **Ensemble methods:** Combining multiple model predictions via averaging, majority-vote, weighted sums, and stacking (meta-learning) for improved accuracy.

The final outputs are submission CSV files (`ID`, `score`) for the test set predictions. The code is organized into scripts and a notebook as described below.

## File Descriptions

- **`ensembles.ipynb`**: A Jupyter notebook containing experiments and pipelines for training multiple models and combining them. This includes: (1) transformer-based regressors (e.g. XLM-RoBERTa, DistilBERT) on different text inputs, (2) TF-IDF-based features with Ridge regression, (3) GBDT regressors (XGBoost, LightGBM, CatBoost), and (4) ensemble strategies. It performs data preprocessing, oversampling for balance, model training, and then creates a hybrid ensemble by optimizing weights and training a stacking meta-learner. The notebook ultimately saves predictions to `submission_hybrid_ensemble.csv`.

- **`final_code.py`**: Python script implementing a *two-tower* transformer model with metric learning and classification. One tower encodes the concatenated **[User Prompt + Response]**, and the other encodes **[Metric Name + System Prompt]** (if any). The model has two heads: an 11-class classifier (scores 0–10) and a scalar regressor (float score). It also uses an InfoNCE contrastive loss between the two towers’ embeddings. Key steps:  
  1. Load and clean training/test JSON data (`train_data.json`, `test_data.json`), converting scores to integer classes.  
  2. Perform K-fold cross-validation (default 3 folds). For each fold, train the model (default backbone: `sentence-transformers/all-MiniLM-L6-v2`) with joint losses (cross-entropy + MSE + contrastive). Save the best model per fold.  
  3. Load each fold model to predict on the test set. Ensemble by averaging the class predictions (mean of logits rounded to nearest integer).  
  4. Write output to `submission_final_metric_classifier.csv`.  

- **`final_scoring_model.py`**: Python script implementing a *text+metric-embedding* regression model. The input text (combining metric, user prompt, and response) is encoded by a transformer (`all-MiniLM-L6-v2` by default). Each metric name has a precomputed embedding (from `metric_name_embeddings.npy`). The model concatenates the transformer encoding with a processed metric embedding and passes through a regression head. Key steps:  
  1. Load training/test JSON, metric names (`metric_names.json`), and metric embeddings (`metric_name_embeddings.npy`).  
  2. Standardize embedding dimensions (to e.g. 384D).  
  3. Split data into K folds. For each fold, train the model (MSE regression loss) and save best weights.  
  4. Predict on test set with each fold model. Average the fold predictions. If the ensemble’s standard deviation is low, apply a heuristic calibration scaling (stretching towards mean 5).  
  5. Clip predictions to [0,10] and save `submission_final.csv`.  

- **`train_metric_learning.py`**: Python script for a *Mahalanobis metric learning* approach. It uses a large transformer backbone (e.g. `BAAI/bge-large-en-v1.5`), which outputs a text embedding. Each metric name also has an embedding (standardized to 1024D). The model projects the combined (text + metric) features into a learned metric space \( \mathbb{R}^{D} \) via a linear projection (learning matrix \(L\)). A score head then predicts the final quality. Key steps:  
  1. Load and clean data, metric names, and embeddings. Standardize embeddings to a target dimension.  
  2. Create pairwise constraints (similar/dissimilar pairs and relative triplets) from training scores to guide metric learning.  
  3. Split into K folds. For each fold, train the model using a combined loss: MSE on predicted score plus a triplet-like metric loss that encourages points with closer scores to be nearer in the learned space. Save best model per fold.  
  4. Predict on test set with each fold model. Average the predictions. Apply calibration as needed (similar to above), and save `submission_metric_learning.csv`.  

## Setup Instructions

1. **Python Environment:** Use Python 3.8 or higher. GPU acceleration (CUDA) is recommended for training speed. Ensure drivers and CUDA are set up properly.

2. **Dependencies:** Install required packages. For example:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   pip install transformers tqdm pandas numpy scikit-learn xgboost lightgbm catboost
   ```

3. **Data Files:** Place the dataset files in a directory (e.g. `data/`):
   - `train_data.json`  
   - `test_data.json`  
   - `metric_names.json`  
   - `metric_name_embeddings.npy`

4. **Environment Tips:** Set `CUDA_VISIBLE_DEVICES` if multiple GPUs are available.

## Running the Code

```bash
python final_code.py                 # Transformer + metric learning (classification)
python final_scoring_model.py       # Transformer + metric embedding regression
python train_metric_learning.py     # Mahalanobis metric learning
```

- Run `ensembles.ipynb` in a Jupyter environment to train all models and build the hybrid ensemble.

## Models and Methods

- **Transformers:** XLM-RoBERTa, DistilBERT, MiniLM, BGE-large  
- **Losses:** Cross-entropy, MSE, InfoNCE contrastive loss, triplet-like distance loss  
- **Ensembling:** Averaging, weighting, stacking meta-learner  
- **GBDTs:** XGBoost, LightGBM, CatBoost using TF-IDF features

## Customization

- Change `MODEL_NAME`, `BATCH_SIZE`, `EPOCHS` at the top of each script  
- Modify ensembling logic in the notebook  
- Add new models via the modular structure in the notebook or scripts

---

This README helps ML engineers navigate, run, and modify this model training codebase for AI assistant evaluation scoring tasks.
