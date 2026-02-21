"""
========================================
TCP TRAFFIC CLASSIFICATION - PREDICTIONS ONLY
========================================

Минимальный скрипт ТОЛЬКО для predictions.
Не требует PyTorch, Lifestream, тяжелых зависимостей.

Требуются файлы:
- trained_models.pkl      (обученные LightGBM модели)
- test_features.pkl        (извлеченные test features)
- test_embeddings.pkl      (Lifestream embeddings для test)
- test.csv                 (или будут взяты IDs из test_features.pkl)

Результат:
- submission.csv           (готовый файл для отправки)

Author: AI Assistant
Date: 2025-11-09
"""

import sys
import os
import platform
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TCP TRAFFIC CLASSIFICATION - PREDICTIONS")
print("="*70)

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
print("\n[1/5] Importing libraries...")

try:
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import joblib
    from sklearn.preprocessing import LabelEncoder
    print("✅ All libraries loaded successfully")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("\nInstall required libraries:")
    print("pip install pandas numpy lightgbm scikit-learn joblib")
    sys.exit(1)

# Display versions
print("\n" + "="*70)
print("📦 LIBRARY VERSIONS:")
print("="*70)
print(f"OS:             {platform.system()} {platform.release()}")
print(f"Python:         {sys.version.split()[0]}")
print(f"pandas:         {pd.__version__}")
print(f"numpy:          {np.__version__}")
print(f"lightgbm:       {lgb.__version__}")
import sklearn
print(f"scikit-learn:   {sklearn.__version__}")
print(f"joblib:         {joblib.__version__}")
print("="*70)

# ============================================
# STEP 2: LOAD MODELS AND DATA
# ============================================
print("\n[2/5] Loading trained models and preprocessed data...")

# Check required files
required_files = {
    'trained_models.pkl': 'Trained LightGBM models',
    'test_features.pkl': 'Preprocessed test features',
    'test_embeddings.pkl': 'Test embeddings'
}

missing_files = []
for file, desc in required_files.items():
    if not os.path.exists(file):
        missing_files.append(f"  ❌ {file} ({desc})")
    else:
        file_size = os.path.getsize(file) / (1024**2)
        print(f"  ✅ {file} ({file_size:.1f} MB)")

if missing_files:
    print("\n❌ Missing required files:")
    for msg in missing_files:
        print(msg)
    print("\nPlease ensure all .pkl files are in the current directory.")
    sys.exit(1)

# Load models
print("\nLoading trained models...")
models_data = joblib.load('trained_models.pkl')
models = models_data['models']
label_encoder = models_data['label_encoder']
num_classes = len(label_encoder.classes_)
print(f"✅ Loaded {len(models)} models")
print(f"✅ Number of classes: {num_classes}")

# Load test features
print("\nLoading test features...")
test_data = joblib.load('test_features.pkl')
X_test = test_data['X_test']
test_ids = test_data['test_ids']
print(f"✅ Loaded {len(X_test):,} test samples")
print(f"✅ Features before embeddings: {X_test.shape[1]}")

# Load test embeddings
print("\nLoading test embeddings...")
test_embeddings = joblib.load('test_embeddings.pkl')
print(f"✅ Loaded embeddings with shape: {test_embeddings.shape}")

# Combine features with embeddings
embedding_cols = [f'emb_{i}' for i in range(test_embeddings.shape[1])]
embeddings_df = pd.DataFrame(test_embeddings, columns=embedding_cols)
X_test = pd.concat([X_test.reset_index(drop=True), embeddings_df], axis=1)

print(f"✅ Total features: {X_test.shape[1]}")

# Cleanup
del test_embeddings, embeddings_df, test_data
import gc
gc.collect()

# ============================================
# STEP 3: MAKE PREDICTIONS IN CHUNKS
# ============================================
print("\n[3/5] Making predictions...")
print(f"Total samples: {len(X_test):,}")
print(f"Number of models: {len(models)}")

# Configuration
CHUNK_SIZE = 50000  # Process 50k samples at a time
num_chunks = (len(X_test) + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"Will process {num_chunks} chunks of {CHUNK_SIZE:,} samples each")

# Check for checkpoint
CHECKPOINT_FILE = 'predictions_checkpoint.pkl'
start_chunk = 0
all_predictions = []

if os.path.exists(CHECKPOINT_FILE):
    print("\n✅ Found predictions checkpoint, loading...")
    checkpoint_data = joblib.load(CHECKPOINT_FILE)
    all_predictions = checkpoint_data['predictions']
    start_chunk = checkpoint_data['last_chunk'] + 1
    print(f"✅ Resuming from chunk {start_chunk + 1}/{num_chunks}")
    print(f"   Already processed: {len(all_predictions)} chunks")

# Predictions loop
print("\n🚀 Starting predictions...")
try:
    for i in range(start_chunk * CHUNK_SIZE, len(X_test), CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, len(X_test))
        chunk_num = i // CHUNK_SIZE + 1
        
        print(f"\rChunk {chunk_num}/{num_chunks}: Processing {i:,} to {end_idx:,}...", end='', flush=True)
        
        # Extract chunk
        X_chunk = X_test.iloc[i:end_idx]
        
        # Ensemble predictions
        chunk_preds = np.zeros((len(X_chunk), num_classes))
        for model in models:
            chunk_preds += model.predict(X_chunk)
        chunk_preds /= len(models)
        
        all_predictions.append(chunk_preds)
        
        # Save checkpoint every 10 chunks
        if chunk_num % 10 == 0 or chunk_num == num_chunks:
            joblib.dump({
                'predictions': all_predictions,
                'last_chunk': chunk_num - 1
            }, CHECKPOINT_FILE)
            print(f" ✓ (checkpoint saved)", end='')
        
        # Cleanup
        del X_chunk, chunk_preds
        gc.collect()
    
    print("\n✅ All predictions completed!")

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user!")
    print(f"Progress saved. Run again to resume from chunk {len(all_predictions) + 1}")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error during predictions: {e}")
    print(f"Progress saved. Run again to resume.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# STEP 4: COMBINE AND CONVERT PREDICTIONS
# ============================================
print("\n[4/5] Combining predictions and converting to labels...")

# Combine all chunks
predictions = np.vstack(all_predictions)
print(f"✅ Combined predictions shape: {predictions.shape}")

# Convert to class labels
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)
print(f"✅ Converted to {len(predicted_labels):,} labels")

# Cleanup checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("✅ Checkpoint file cleaned")

# ============================================
# STEP 5: CREATE SUBMISSION FILE
# ============================================
print("\n[5/5] Creating submission file...")

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'app_service': predicted_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print(f"✅ Submission file saved: submission.csv")
print(f"✅ Number of predictions: {len(submission):,}")

# Show sample
print("\n📊 Sample predictions:")
print(submission.head(10))

# Show label distribution
print("\n📊 Predicted labels distribution:")
label_counts = submission['app_service'].value_counts().head(10)
for label, count in label_counts.items():
    print(f"  {label:30s}: {count:,}")

# ============================================
# DONE
# ============================================
print("\n" + "="*70)
print("✅ ALL DONE!")
print("="*70)
print(f"Submission file: submission.csv ({len(submission):,} predictions)")
print("Ready to submit to Kaggle!")
print("="*70)

