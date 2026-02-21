"""
========================================
TCP TRAFFIC CLASSIFICATION - MAIN PIPELINE
========================================
Single file solution with:
- Kaggle data download
- Chunked processing for large datasets
- Memory optimization
- Checkpoint saving/loading
- Google Drive sync between Colab accounts
- Resume training from checkpoints
- SGDClassifier with partial_fit (lightweight, memory-efficient)

BEFORE RUNNING:
1. Execute in Colab cell 1:
    !pip install -q kaggle h5py
    
2. Mount your Google Drive (automatic in Colab)
   The script will automatically sync checkpoints between:
   - Local: ./checkpoints/
   - Google Drive: /content/drive/MyDrive/checkpoints/

3. Then just run this script - no restart needed!

FEATURES:
- Automatic Google Drive sync: Simple two-way sync between local and Drive
- SGDClassifier: Fast, memory-efficient linear model with partial_fit
- Resume training: Automatically continues from where it stopped (checks local checkpoints)
- Time logging: Shows training time for each fold
- Single model per fold (no ensemble): Minimal memory usage
- HDF5 storage: Predictions saved in HDF5 format to avoid RAM overflow
- Chunk-based processing: All predictions processed in chunks without loading all into memory
- No user input required: Fully automatic checkpoint management

For detailed guide, see: GOOGLE_DRIVE_SYNC_GUIDE.md

Author: AI Assistant
Date: 2025-11-08
"""

# ============================================
# STEP 0: LOGGING SETUP (must be first!)
# ============================================
import os
import sys
import atexit
from datetime import datetime

class Logger:
    """Logger class that writes to both console and file"""
    def __init__(self, log_file='training_log.txt'):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        self.start_time = datetime.now()
        self._closed = False
        
        # Write header
        self.log.write("="*80 + "\n")
        self.log.write(f"TRAINING LOG - Started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("="*80 + "\n\n")
        self.log.flush()
        
        # Register cleanup function
        atexit.register(self._cleanup_on_exit)
    
    def write(self, message):
        """Write message to both terminal and log file"""
        if not self._closed:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # Flush immediately to ensure logs are written
    
    def flush(self):
        """Flush both streams"""
        if not self._closed:
            self.terminal.flush()
            self.log.flush()
    
    def close(self):
        """Close log file"""
        if not self._closed and not self.log.closed:
            end_time = datetime.now()
            duration = end_time - self.start_time
            self.log.write("\n" + "="*80 + "\n")
            self.log.write(f"TRAINING LOG - Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log.write(f"Total duration: {duration}\n")
            self.log.write("="*80 + "\n")
            self.log.close()
            self._closed = True
    
    def _cleanup_on_exit(self):
        """Cleanup function called on exit"""
        if not self._closed:
            self.close()
    
    def get_log_path(self):
        """Return path to log file"""
        return self.log_file
    
    def copy_to_drive(self):
        """Copy log file to Google Drive"""
        # Ensure log file is closed before copying
        if not self._closed:
            self.close()
        
        if not os.path.exists(self.log_file):
            return False
            
        try:
            from google.colab import drive
            GDRIVE_LOG_DIR = '/content/drive/MyDrive/checkpoints'
            
            # Ensure Drive is mounted
            if not os.path.exists('/content/drive'):
                try:
                    drive.mount('/content/drive', force_remount=False)
                except Exception:
                    return False
            
            # Create directory if it doesn't exist
            if not os.path.exists(GDRIVE_LOG_DIR):
                try:
                    os.makedirs(GDRIVE_LOG_DIR, exist_ok=True)
                except Exception:
                    return False
            
            # Copy log file
            if os.path.exists(self.log_file) and os.path.exists(GDRIVE_LOG_DIR):
                log_dest = os.path.join(GDRIVE_LOG_DIR, os.path.basename(self.log_file))
                shutil.copy2(self.log_file, log_dest)
                return True
        except ImportError:
            return False
        except Exception:
            return False
        return False

# Initialize logger
logger = Logger('training_log.txt')
sys.stdout = logger

# ============================================
# STEP 1: IMPORTS
# ============================================
print("="*70)
print("STEP 1: Importing libraries...")
print("="*70)

import gc
import json
import platform
import time
import shutil
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import pandas as pd
import numpy as np
import torch
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("⚠️  h5py not available. Install with: pip install h5py")

# Memory monitoring helper
def print_memory_usage():
    """Print current memory usage"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / 1024**3
            print(f"📊 Memory usage: {mem_gb:.2f} GB")
        except:
            pass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
import joblib
import sklearn

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  lightgbm not available. Install with: pip install lightgbm")

# ============================================
# LIBRARY VERSIONS (для совместимости)
# ============================================
print("\n" + "="*70)
print("📦 LIBRARY VERSIONS:")
print("="*70)
print(f"OS:             {platform.system()} {platform.release()}")
print(f"Python:         {sys.version.split()[0]}")
print(f"pandas:         {pd.__version__}")
print(f"numpy:          {np.__version__}")
print(f"torch:          {torch.__version__}")
print(f"scikit-learn:   {sklearn.__version__}")
print(f"joblib:         {joblib.__version__}")
try:
    import importlib.metadata
    tqdm_version = importlib.metadata.version('tqdm')
    print(f"tqdm:           {tqdm_version}")
except:
    print(f"tqdm:           installed (version unknown)")
if PSUTIL_AVAILABLE:
    print(f"psutil:         {psutil.__version__}")
print("="*70)

print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ scikit-learn version: {sklearn.__version__}\n")

# ============================================
# STEP 2: GOOGLE DRIVE SYNC (упрощенная версия)
# ============================================
print("\n" + "="*70)
print("STEP 2: Setting up Google Drive sync...")
print("="*70)

def ensure_checkpoints_sync():
    """
    На старте работы:
      - если папка checkpoints на Google Drive существует,
        копирует всю её структуру в локальную папку
      - если не существует — создает только локальную папку
    """
    GDRIVE_CHECKPOINTS_DIR = '/content/drive/MyDrive/checkpoints'
    LOCAL_CHECKPOINTS_DIR = './checkpoints'
    
    # Монтируем диск (Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive смонтирован")
    except ImportError:
        print("⚠️  Colab не используется, работаем только локально")
    except Exception as e:
        print(f"⚠️  Ошибка монтирования Google Drive: {e}")
        print("   Продолжаем работу только локально")
    
    # Проверяем папки
    if os.path.exists(GDRIVE_CHECKPOINTS_DIR):
        print(f'[SYNC] Найдена папка на Google Drive, копируем содержимое в локальный checkpoints...')
        if not os.path.exists(LOCAL_CHECKPOINTS_DIR):
            os.makedirs(LOCAL_CHECKPOINTS_DIR, exist_ok=True)
        
        # Переносим все файлы и подпапки (поверх если нужно)
        copied_count = 0
        for root, dirs, files in os.walk(GDRIVE_CHECKPOINTS_DIR):
            rel_root = os.path.relpath(root, GDRIVE_CHECKPOINTS_DIR)
            target_root = os.path.join(LOCAL_CHECKPOINTS_DIR, rel_root)
            os.makedirs(target_root, exist_ok=True)
            
            for f in files:
                src_file = os.path.join(root, f)
                dst_file = os.path.join(target_root, f)
                # Копируем только если файл отсутствует или новее
                if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                    if copied_count <= 5:  # Показываем первые 5 файлов
                        print(f'   → скопирован: {f}')
        
        if copied_count > 5:
            print(f'   ... и ещё {copied_count - 5} файлов скопировано')
        print(f'[SYNC] Всего скопировано файлов: {copied_count}')
    else:
        print('[SYNC] Папка checkpoints на Google Drive отсутствует, работаем только локально')
        if not os.path.exists(LOCAL_CHECKPOINTS_DIR):
            os.makedirs(LOCAL_CHECKPOINTS_DIR, exist_ok=True)
    
    print('[SYNC] Инициализация чекпоинтов завершена')
    return LOCAL_CHECKPOINTS_DIR

def save_and_sync_checkpoint(local_filepath):
    """
    После сохранения нового файла в ./checkpoints/ обязательно вызывать эту функцию!
    Она копирует указанный файл на Google Drive (если та папка есть)
    
    Args:
        local_filepath: Путь к файлу относительно корня или абсолютный путь
    """
    GDRIVE_CHECKPOINTS_DIR = '/content/drive/MyDrive/checkpoints'
    LOCAL_CHECKPOINTS_DIR = './checkpoints'
    
    # Нормализуем путь
    local_path = Path(local_filepath)
    if not local_path.is_absolute():
        local_path = Path(LOCAL_CHECKPOINTS_DIR) / local_path.name
    else:
        # Если абсолютный путь, проверяем что он в локальной папке
        try:
            local_path.relative_to(Path(LOCAL_CHECKPOINTS_DIR).resolve())
        except ValueError:
            # Если не в локальной папке, берем только имя файла
            local_path = Path(LOCAL_CHECKPOINTS_DIR) / local_path.name
    
    if not local_path.is_file():
        print(f'[SYNC] ⚠️  Файл {local_path} не найден для синхронизации')
        return
    
    # Куда копируем на диск
    gdrive_path = Path(GDRIVE_CHECKPOINTS_DIR) / local_path.name
    
    # Если папки на диске нет — просто возвращаем
    if not os.path.exists(GDRIVE_CHECKPOINTS_DIR):
        return  # Тихая работа, если диска нет
    
    try:
        # Создаем папку на диске если нужно
        os.makedirs(GDRIVE_CHECKPOINTS_DIR, exist_ok=True)
        shutil.copy2(local_path, gdrive_path)
        print(f'[SYNC] ✓ {local_path.name} → Google Drive')
    except Exception as e:
        print(f'[SYNC] ⚠️  Ошибка синхронизации {local_path.name}: {e}')

# Инициализируем синхронизацию
CHECKPOINT_DIR = ensure_checkpoints_sync()
print(f"📁 Используется папка чекпоинтов: {CHECKPOINT_DIR}")
print("="*70)

# ============================================
# STEP 3: KAGGLE DATA DOWNLOAD
# ============================================
print("\n" + "="*70)
print("STEP 3: Setting up Kaggle and downloading data...")
print("="*70)

def setup_kaggle():
    """Setup Kaggle API and download competition data"""
    # Create .kaggle directory
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    # Kaggle credentials
    kaggle_credentials = {
        "username": "YOUR_KAGGLE_USERNAME",
        "key": "YOUR_KAGGLE_API_KEY"
    }
    
    # Save credentials
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        json.dump(kaggle_credentials, f)
    
    # Set permissions
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
    
    print("✅ Kaggle API configured")
    
    # Download competition data if not exists
    if not os.path.exists('train.csv'):
        print("📥 Downloading competition data...")
        os.system('kaggle competitions download -c whos-talking-classify-the-app-by-its-packets')
        os.system('unzip -q whos-talking-classify-the-app-by-its-packets.zip')
        print("✅ Data downloaded and extracted")
    else:
        print("✅ Data already exists")

setup_kaggle()

# ============================================
# STEP 4: FEATURE EXTRACTION CLASSES
# ============================================
print("\n" + "="*70)
print("STEP 4: Initializing feature extraction classes...")
print("="*70)

class BasicFeatureExtractor:
    """Extract basic statistical features from TCP sequences"""
    
    @staticmethod
    def extract(df, tcp_columns):
        """Extract basic features"""
        features = df[tcp_columns].copy()
        tcp_data = features.values
        tcp_data = np.nan_to_num(tcp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Basic statistics
        features['total_packets'] = np.count_nonzero(tcp_data, axis=1)
        features['mean_size'] = np.mean(np.abs(tcp_data), axis=1)
        features['std_size'] = np.std(np.abs(tcp_data), axis=1)
        features['max_size'] = np.max(np.abs(tcp_data), axis=1)
        features['min_size'] = np.array([
            np.min(np.abs(row[row != 0])) if np.any(row != 0) else 0 
            for row in tcp_data
        ])
        
        # Direction features
        features['out_packets'] = np.sum(tcp_data > 0, axis=1)
        features['in_packets'] = np.sum(tcp_data < 0, axis=1)
        features['out_bytes'] = np.sum(np.maximum(tcp_data, 0), axis=1)
        features['in_bytes'] = np.abs(np.sum(np.minimum(tcp_data, 0), axis=1))
        features['out_in_ratio'] = features['out_packets'] / (features['in_packets'] + 1)
        features['bytes_ratio'] = features['out_bytes'] / (features['in_bytes'] + 1)
        
        # Large packets (MTU)
        features['large_packets'] = np.sum(np.abs(tcp_data) > 1200, axis=1)
        features['large_ratio'] = features['large_packets'] / (features['total_packets'] + 1)
        
        # First packet
        features['first_packet'] = tcp_data[:, 0]
        features['first_abs'] = np.abs(tcp_data[:, 0])
        
        # Variance in windows
        for window in [5, 10, 15]:
            window_data = tcp_data[:, :window]
            features[f'var_{window}'] = np.var(np.abs(window_data), axis=1)
            features[f'mean_{window}'] = np.mean(np.abs(window_data), axis=1)
        
        # Direction changes
        directions = np.sign(tcp_data)
        direction_changes = np.sum(directions[:, 1:] != directions[:, :-1], axis=1)
        features['dir_changes'] = direction_changes
        features['dir_change_rate'] = direction_changes / (features['total_packets'] + 1)
        
        # Burst detection
        large_mask = np.abs(tcp_data) > 1000
        burst_lengths = []
        for row in large_mask:
            current_burst = 0
            max_burst = 0
            for val in row:
                if val:
                    current_burst += 1
                    max_burst = max(max_burst, current_burst)
                else:
                    current_burst = 0
            burst_lengths.append(max_burst)
        features['max_burst'] = burst_lengths
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features


class SimpleTargetEncoder:
    """Simplified target encoding without CV issues"""
    
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = None
    
    def fit_transform(self, df, target, categorical_cols):
        """Fit and transform"""
        self.global_mean = target.mean() if hasattr(target, 'mean') else 0
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            # Calculate means per category
            stats = df.groupby(col)[target.name if hasattr(target, 'name') else 0].agg(['mean', 'count'])
            
            # Smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
            stats['encoded'] = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            
            self.encodings[col] = stats['encoded'].to_dict()
            
            # Map to dataframe
            df_encoded[f'{col}_enc'] = df[col].map(self.encodings[col]).fillna(self.global_mean)
        
        return df_encoded
    
    def transform(self, df, categorical_cols):
        """Transform using fitted encodings"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in self.encodings:
                df_encoded[f'{col}_enc'] = df[col].map(self.encodings[col]).fillna(self.global_mean)
        
        return df_encoded


class RNNEncoder(nn.Module):
    """Lightweight RNN encoder for embeddings"""
    
    def __init__(self, embedding_dim=64, num_layers=2):
        super(RNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_projection = nn.Linear(1, embedding_dim // 2)
        self.lstm = nn.LSTM(
            input_size=embedding_dim // 2,
            hidden_size=embedding_dim // 2,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        output, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        x = self.output_projection(hidden)
        return x


class LifestreamEmbedder:
    """Creates embeddings using RNN"""
    
    def __init__(self, embedding_dim=64, device='cuda'):
        self.embedding_dim = embedding_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.encoder = RNNEncoder(embedding_dim=embedding_dim).to(self.device)
        self.label_encoder = LabelEncoder()
        self.classifier = None  # Will be initialized after knowing num_classes
    
    def train_embeddings(self, sequences, labels, epochs=3, batch_size=512):
        """Train encoder"""
        labels_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Create classifier head for proper cross-entropy
        self.classifier = nn.Linear(self.embedding_dim, num_classes).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(sequences),
            torch.LongTensor(labels_encoded)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), 
            lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()
        
        self.encoder.train()
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for seq_batch, label_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                seq_batch = seq_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                embeddings = self.encoder(seq_batch)
                
                # Classification loss
                logits = self.classifier(embeddings)
                loss = criterion(logits, label_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def extract_embeddings(self, sequences, batch_size=512):
        """Extract embeddings"""
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(sequences))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.encoder.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for (seq_batch,) in tqdm(dataloader, desc="Extracting embeddings"):
                seq_batch = seq_batch.to(self.device)
                embeddings = self.encoder(seq_batch)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


print("✅ Feature extraction classes ready\n")

# ============================================
# STEP 5: MAIN TRAINING PIPELINE
# ============================================
print("\n" + "="*70)
print("STEP 5: Starting main training pipeline...")
print("="*70)

# Configuration
TCP_COLUMNS = [f'tcp_len_{i}' for i in range(1, 31)]
TARGET_COL = 'app_service'
CHUNK_SIZE = 25000  # Process 25k rows at a time (reduced for better memory management)

# CHECKPOINT_DIR уже определен в ensure_checkpoints_sync()


# ============================================
# STEP 4.1: Load and process training data in chunks
# ============================================
print("\n[1/6] Loading and processing training data in chunks...")

def process_train_data_chunked():
    """Process training data in chunks - saves each chunk to disk instead of concatenating"""
    
    # Check if chunk index already exists
    chunk_index_file = f'{CHECKPOINT_DIR}/chunk_index.pkl'
    if os.path.exists(chunk_index_file):
        print("✅ Found preprocessed chunk index, loading...")
        chunk_data = joblib.load(chunk_index_file)
        return chunk_data['feature_files'], chunk_data['target_files'], chunk_data['label_encoder']
    
    label_encoder = LabelEncoder()
    feature_files = []
    target_files = []
    
    # First pass: fit label encoder
    print("Fitting label encoder...")
    all_labels = []
    for chunk in pd.read_csv('train.csv', chunksize=CHUNK_SIZE):
        all_labels.extend(chunk[TARGET_COL].unique())
    label_encoder.fit(list(set(all_labels)))
    print(f"Found {len(label_encoder.classes_)} classes")
    
    # Second pass: process features and save each chunk separately
    print("Processing features in chunks (saving each chunk to disk)...")
    chunk_count = 0
    for i, chunk in enumerate(pd.read_csv('train.csv', chunksize=CHUNK_SIZE)):
        chunk_count += 1
        estimated_chunks = (8200000 // CHUNK_SIZE) + 1
        print(f"\rProcessing chunk {chunk_count} (~{(chunk_count)*CHUNK_SIZE//1000}k rows)...", end='', flush=True)
        
        # Extract basic features
        X_chunk = BasicFeatureExtractor.extract(chunk, TCP_COLUMNS)
        y_chunk = label_encoder.transform(chunk[TARGET_COL].astype(str))
        
        # Save each chunk separately to disk
        feature_path = f'{CHECKPOINT_DIR}/features_chunk_{i}.pkl'
        target_path = f'{CHECKPOINT_DIR}/targets_chunk_{i}.pkl'
        joblib.dump(X_chunk, feature_path)
        joblib.dump(y_chunk, target_path)
        feature_files.append(feature_path)
        target_files.append(target_path)
        
        # Progress update every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"\n✓ Processed {i+1} chunks ({(i+1)*CHUNK_SIZE//1000}k rows)", flush=True)
        
        # Memory cleanup
        del chunk, X_chunk, y_chunk
        gc.collect()
    
    print(f"\n✅ All {chunk_count} chunks processed and saved to disk!")
    
    # Save chunk index (list of file paths)
    print("Saving chunk index...")
    joblib.dump({
        'feature_files': feature_files,
        'target_files': target_files,
        'label_encoder': label_encoder,
        'num_chunks': chunk_count
    }, chunk_index_file)
    
    return feature_files, target_files, label_encoder

feature_files, target_files, label_encoder = process_train_data_chunked()
print(f"\n✅ Processed {len(feature_files)} feature chunks saved to disk")

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================
# STEP 4.2: Add Lifestream embeddings
# ============================================
print("\n[2/6] Creating Lifestream embeddings...")

embedding_index_file = f'{CHECKPOINT_DIR}/lifestream_embeddings_index.pkl'
if os.path.exists(embedding_index_file):
    print("✅ Found preprocessed embeddings index, loading...")
    embeddings_data = joblib.load(embedding_index_file)
    embedding_files = embeddings_data['embedding_files']
    embedder = embeddings_data['embedder']
else:
    # Initialize embedder
    embedder = LifestreamEmbedder(embedding_dim=64, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train embedder on chunks (epochs=3, processing chunk-by-chunk)
    print("Training embedder on chunks (3 epochs)...")
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3:")
        for i, (feat_path, targ_path) in enumerate(zip(feature_files, target_files)):
            if i % 10 == 0:
                print(f"\r  Processing chunk {i+1}/{len(feature_files)}...", end='', flush=True)
            
            # Load feature chunk
            X_chunk = joblib.load(feat_path)
            seq_chunk = X_chunk[TCP_COLUMNS].values.astype(np.float32)
            seq_chunk = np.nan_to_num(seq_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Load target chunk
            y_chunk = joblib.load(targ_path)
            y_chunk_str = label_encoder.inverse_transform(y_chunk).astype(str)
            
            # Train on this chunk (1 epoch per chunk, 3 external iterations)
            embedder.train_embeddings(seq_chunk, y_chunk_str, epochs=1, batch_size=512)
            
            # Memory cleanup
            del X_chunk, seq_chunk, y_chunk, y_chunk_str
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n  ✓ Epoch {epoch+1}/3 completed")
    
    # Extract embeddings chunk-by-chunk and save separately
    print("\nExtracting embeddings chunk-by-chunk...")
    embedding_files = []
    for i, feat_path in enumerate(feature_files):
        if i % 10 == 0:
            print(f"\r  Extracting embeddings chunk {i+1}/{len(feature_files)}...", end='', flush=True)
        
        # Load feature chunk
        X_chunk = joblib.load(feat_path)
        seq_chunk = X_chunk[TCP_COLUMNS].values.astype(np.float32)
        seq_chunk = np.nan_to_num(seq_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract embeddings for this chunk
        emb_chunk = embedder.extract_embeddings(seq_chunk, batch_size=512)
        
        # Save embedding chunk
        emb_path = f'{CHECKPOINT_DIR}/embedding_chunk_{i}.pkl'
        joblib.dump(emb_chunk, emb_path)
        embedding_files.append(emb_path)
        
        # Memory cleanup
        del X_chunk, seq_chunk, emb_chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n✅ All embeddings extracted and saved to {len(embedding_files)} files")
    
    # Save embedding index
    joblib.dump({
        'embedding_files': embedding_files,
        'embedder': embedder
    }, embedding_index_file)
    
    print("✅ Embeddings index saved")

print(f"✅ Embeddings ready: {len(embedding_files)} chunks")

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================
# STEP 4.3: Train LightGBM with CV
# ============================================
print("\n[3/6] Training LightGBM with 5-fold CV...")

# Определяем пути к критическим файлам
MODELS_PATH = f'{CHECKPOINT_DIR}/trained_models.pkl'
OOF_H5_FILE = f'{CHECKPOINT_DIR}/oof_preds_all_folds.h5'

# Проверяем наличие файлов
models_exist = os.path.exists(MODELS_PATH)
oof_exist = os.path.exists(OOF_H5_FILE)

print(f"📋 Статус файлов:")
print(f"   trained_models.pkl: {'✅ найден' if models_exist else '❌ отсутствует'}")
print(f"   oof_preds_all_folds.h5: {'✅ найден' if oof_exist else '❌ отсутствует'}")

# ПРИОРИТЕТ 1: Проверяем наличие обученных моделей (критично для инференса)
if models_exist:
    print(f"\n✅ Файл {MODELS_PATH} найден, загружаем модели для инференса...")
    saved_data = joblib.load(MODELS_PATH)
    
    # Support both old format (just models list) and new format (with scalers)
    if len(saved_data['models']) > 0 and isinstance(saved_data['models'][0], dict) and 'model' in saved_data['models'][0]:
        # New format: models stored as dicts with 'model' and 'scaler'
        models = saved_data['models']  # Keep as dicts for consistency
        scalers = [item.get('scaler', None) for item in saved_data['models']]
        print(f"✅ Загружено {len(models)} обученных моделей с scalers (новый формат)")
    else:
        # Old format: just models list - convert to new format
        old_models = saved_data['models']
        models = [{'model': m, 'scaler': None} for m in old_models]
        scalers = [None] * len(models)
        print(f"✅ Загружено {len(models)} обученных моделей (старый формат без scalers)")
        print(f"⚠️  ВНИМАНИЕ: Модели без scalers! Рекомендуется переобучить модели с масштабированием.")
    
    label_encoder = saved_data['label_encoder']
    num_classes = len(label_encoder.classes_)
    print(f"✅ {num_classes} классов в label_encoder")
    
    # Try to load global scaler if available
    global_scaler_path = f'{CHECKPOINT_DIR}/global_scaler.pkl'
    if os.path.exists(global_scaler_path):
        try:
            global_scaler = joblib.load(global_scaler_path)
            print(f"✅ Global scaler loaded from {global_scaler_path}")
        except Exception as e:
            print(f"⚠️  Could not load global scaler: {e}")
            global_scaler = None
    else:
        global_scaler = None
        print(f"⚠️  Global scaler not found at {global_scaler_path}")
    
    # Диагностика моделей
    print(f"\n📊 Model diagnostics:")
    for i, model_data in enumerate(models):
        # Extract model from dict if needed
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler', None)
        else:
            model = model_data
            scaler = None
        
        if scaler is not None:
            print(f"  Model {i+1}: ✅ Has scaler", flush=True)
        else:
            print(f"  Model {i+1}: ⚠️  No scaler", flush=True)
        
        # Check if LightGBM model
        if LIGHTGBM_AVAILABLE and isinstance(model, lgb.Booster):
            num_trees = model.num_trees()
            num_feature = model.num_feature()
            print(f"  Model {i+1}: LightGBM Booster - {num_trees} trees, {num_feature} features", flush=True)
        elif hasattr(model, 'coef_'):
            # Old SGDClassifier format
            coef = model.coef_
            if coef is not None:
                coef_norm = np.linalg.norm(coef)
                coef_nan = np.isnan(coef).sum()
                coef_inf = np.isinf(coef).sum()
                coef_min = np.nanmin(coef)
                coef_max = np.nanmax(coef)
                
                print(f"  Model {i+1}: coef_norm={coef_norm:.6f}, "
                      f"n_iter={getattr(model, 'n_iter_', 'N/A')}, "
                      f"classes_seen={len(getattr(model, 'classes_', []))}")
                
                if coef_nan > 0 or coef_inf > 0:
                    print(f"    ⚠️  WARNING: Model {i+1} has {coef_nan} NaN and {coef_inf} Inf in coefficients!")
                
                print(f"    coef range: [{coef_min:.6f}, {coef_max:.6f}], "
                      f"shape: {coef.shape}")
                
                # Проверяем intercept
                if hasattr(model, 'intercept_') and model.intercept_ is not None:
                    intercept_nan = np.isnan(model.intercept_).sum()
                    intercept_inf = np.isinf(model.intercept_).sum()
                    if intercept_nan > 0 or intercept_inf > 0:
                        print(f"    ⚠️  WARNING: Model {i+1} intercept has {intercept_nan} NaN and {intercept_inf} Inf!")
            else:
                print(f"  Model {i+1}: ⚠️  coef_ is None!")
        else:
            print(f"  Model {i+1}: type={type(model)}", flush=True)
    
    # Проверяем, что модели обучены
    if len(models) > 0:
        # Extract first model to check format
        first_model_data = models[0]
        if isinstance(first_model_data, dict):
            first_model = first_model_data['model']
        else:
            first_model = first_model_data
        
        all_valid = True
        for i, model_data in enumerate(models):
            # Extract model from dict if needed
            if isinstance(model_data, dict):
                model = model_data['model']
            else:
                model = model_data
            
            # Check LightGBM model
            if LIGHTGBM_AVAILABLE and isinstance(model, lgb.Booster):
                if model.num_trees() == 0:
                    print(f"\n⚠️  WARNING: Model {i+1} has 0 trees! Model might not be trained!")
                    all_valid = False
            # Check SGDClassifier model (old format)
            elif hasattr(model, 'coef_'):
                if model.coef_ is None:
                    print(f"\n⚠️  WARNING: Model {i+1} coefficients are None! Model might not be trained!")
                    all_valid = False
                elif np.linalg.norm(model.coef_) < 1e-10:
                    print(f"\n⚠️  WARNING: Model {i+1} coefficients are zero! Model might not be trained!")
                    all_valid = False
                elif np.isnan(model.coef_).any() or np.isinf(model.coef_).any():
                    print(f"\n⚠️  WARNING: Model {i+1} has NaN/Inf in coefficients!")
                    all_valid = False
        
        if all_valid:
            print(f"\n  ✅ All models appear to be trained")
        else:
            print(f"\n  ⚠️  Some models have issues! This may cause NaN predictions.")
    
    # Если OOF файл тоже есть - пропускаем обучение и объединение OOF
    if oof_exist:
        print(f"✅ Файл {OOF_H5_FILE} также найден, этап обучения по фолдам и объединения OOF будет пропущен!")
        print("   Продолжаем пайплайн с существующими файлами...")
    else:
        print(f"⚠️  Файл {OOF_H5_FILE} отсутствует, но это не критично для инференса на test данных.")
        print("   Если нужен OOF для анализа - запустите обучение заново или удалите trained_models.pkl")
        print("   Продолжаем пайплайн для инференса на test данных...")
    
    # Пропускаем весь блок обучения
    skip_training = True
else:
    # Моделей нет - нужно обучать
    print(f"\n❌ Файл {MODELS_PATH} не найден!")
    
    # Если есть OOF файл, но нет моделей - это проблема
    if oof_exist:
        print(f"⚠️  ВНИМАНИЕ: Файл {OOF_H5_FILE} найден, но {MODELS_PATH} отсутствует!")
        print("   Это означает, что обучение моделей не было завершено.")
        print("   Для инференса на test данных ОБЯЗАТЕЛЬНО нужны обученные модели!")
        print("\n   ВАРИАНТЫ РЕШЕНИЯ:")
        print("   1. Удалите oof_preds_all_folds.h5 и запустите полное обучение заново")
        print("   2. Загрузите trained_models.pkl из предыдущего успешного запуска")
        print("   3. Дождитесь завершения обучения моделей (может занять много времени)")
        print("\n   РЕКОМЕНДАЦИЯ: Удалите oof_preds_all_folds.h5 для чистого обучения:")
        print(f"   import os; os.remove('{OOF_H5_FILE}')")
        raise ValueError(
            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл {OOF_H5_FILE} найден, но {MODELS_PATH} отсутствует!\n"
            f"   Невозможно выполнить инференс на test данных без обученных моделей.\n"
            f"   Удалите {OOF_H5_FILE} и запустите полное обучение, или загрузите {MODELS_PATH}."
        )
    
    # Оба файла отсутствуют - запускаем обучение
    print(f"   Файл {OOF_H5_FILE} также отсутствует.")
    print("   Запускаем цикл обучения по фолдам и объединение OOF...")
    skip_training = False

# Если модели загружены - пропускаем обучение
if skip_training:
    pass  # Модели уже загружены, пропускаем блок обучения
else:
    # Запускаем обучение моделей
    print("Training new models...")
    
    num_classes = len(label_encoder.classes_)
    
    # Load all targets and compute chunk sizes in one pass
    print("Loading all targets for CV split calculation...")
    all_y_parts = []
    chunk_sizes = []
    for t in target_files:
        y_chunk = joblib.load(t)
        chunk_sizes.append(len(y_chunk))
        all_y_parts.append(y_chunk)
    all_y = np.concatenate(all_y_parts)
    sample_count = len(all_y)
    print(f"Total samples: {sample_count:,}")
    
    # Load embedding files list if not already loaded
    if 'embedding_files' not in locals():
        embedding_data = joblib.load(embedding_index_file)
        embedding_files = embedding_data['embedding_files']
    
    # Helper function to map global indices to chunks
    def get_indices_for_files(indices, files, sizes):
        """Map global indices to (file_path, local_indices) pairs"""
        selected = []
        running = 0
        for path, arr_len in zip(files, sizes):
            # Find indices that fall in this chunk
            mask = (indices >= running) & (indices < running + arr_len)
            arr_indices = indices[mask] - running
            if len(arr_indices) > 0:
                selected.append((path, arr_indices))
            running += arr_len
        return selected
    
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")
    
    print(f"✅ Using LightGBM for multiclass classification (252 classes)")
    
    # Function to fit global scaler on sample data
    def fit_global_scaler(sample_feature_paths, embedding_files_list, n_sample_rows=200000):
        """Fit StandardScaler on sample of data for consistent scaling"""
        print(f"  Fitting global scaler on sample data ({n_sample_rows:,} rows)...", flush=True)
        collected = []
        rows = 0
        
        for i, feat_path in enumerate(sample_feature_paths):
            # Load feature chunk
            X_chunk = joblib.load(feat_path)
            
            # Load corresponding embeddings
            emb_chunk = joblib.load(embedding_files_list[i])
            
            # Normalize embeddings (L2 normalization)
            emb_norms = np.linalg.norm(emb_chunk, axis=1, keepdims=True)
            emb_norms = np.where(emb_norms > 0, emb_norms, 1.0)
            emb_chunk = emb_chunk / emb_norms
            
            # Combine features and embeddings
            embedding_cols = [f'emb_{j}' for j in range(emb_chunk.shape[1])]
            emb_df = pd.DataFrame(emb_chunk, columns=embedding_cols, index=X_chunk.index)
            X_combined = pd.concat([X_chunk.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
            
            # Clean data
            X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
            X_combined = X_combined.fillna(0)
            X_combined = X_combined.clip(-1e15, 1e15)
            
            X = X_combined.values.astype(np.float32)
            collected.append(X)
            rows += X.shape[0]
            
            if rows >= n_sample_rows:
                break
            
            del X_chunk, emb_chunk, emb_df, X_combined, X
            gc.collect()
        
        Xs = np.vstack(collected)
        scaler = StandardScaler()
        scaler.fit(Xs)
        
        del collected, Xs
        gc.collect()
        
        print(f"  ✅ Global scaler fitted on {rows:,} rows", flush=True)
        return scaler
    
    # Fit global scaler on sample of chunks (first 40 chunks or ~200k rows)
    sample_size = min(40, len(feature_files))
    sample_feature_paths = feature_files[:sample_size]
    sample_embedding_paths = embedding_files[:sample_size]
    
    global_scaler_path = f'{CHECKPOINT_DIR}/global_scaler.pkl'
    if os.path.exists(global_scaler_path):
        print(f"  Loading global scaler from checkpoint...", flush=True)
        global_scaler = joblib.load(global_scaler_path)
        print(f"  ✅ Global scaler loaded", flush=True)
    else:
        global_scaler = fit_global_scaler(sample_feature_paths, sample_embedding_paths, n_sample_rows=200000)
        joblib.dump(global_scaler, global_scaler_path)
        print(f"  ✅ Global scaler saved to {global_scaler_path}", flush=True)
    
    # Function to load chunks (features + embeddings) for training/validation
    def load_chunk_data(mapped_chunks):
        """Load chunks and combine features with embeddings"""
        chunk_parts = []
        chunk_labels = []
        
        for feat_path, feat_idx in mapped_chunks:
            # Extract chunk index from filename
            chunk_idx = int(feat_path.split('_')[-1].replace('.pkl', ''))
            
            # Load feature chunk
            X_chunk = joblib.load(feat_path).iloc[feat_idx]
            
            # Load target chunk
            y_chunk = joblib.load(target_files[chunk_idx])[feat_idx]
            
            # Load corresponding embeddings
            emb_chunk = joblib.load(embedding_files[chunk_idx])
            emb_chunk = emb_chunk[feat_idx]
            
            # CRITICAL: Normalize embeddings (L2 normalization) to prevent SGD instability
            # This is standard practice for embeddings in ML pipelines
            emb_norms = np.linalg.norm(emb_chunk, axis=1, keepdims=True)
            emb_norms = np.where(emb_norms > 0, emb_norms, 1.0)  # Avoid division by zero
            emb_chunk = emb_chunk / emb_norms
            
            embedding_cols = [f'emb_{i}' for i in range(emb_chunk.shape[1])]
            emb_df = pd.DataFrame(emb_chunk, columns=embedding_cols, index=X_chunk.index)
            
            # Combine features and embeddings
            X_chunk = pd.concat([X_chunk.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
            chunk_parts.append(X_chunk)
            chunk_labels.append(y_chunk)
            
            # Clean up individual chunk data
            del X_chunk, emb_chunk, emb_df
        
        # Concatenate chunks
        if chunk_parts:
            X_combined = pd.concat(chunk_parts, ignore_index=True)
            y_combined = np.concatenate(chunk_labels)
            del chunk_parts, chunk_labels
            
            # Clean data
            X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
            X_combined = X_combined.fillna(0)
            X_combined = X_combined.clip(-1e15, 1e15)
            
            # Convert to numpy array for SGDClassifier
            X_combined = X_combined.values.astype(np.float32)
            
            return X_combined, y_combined
        else:
            return np.array([]), np.array([])
    
    # Training function with SGDClassifier using partial_fit
    def train_lgb_chunkwise(train_mapped, val_mapped, num_classes, fold_num):
        """Train single LightGBM model using chunked incremental training"""
        model_path = f'{CHECKPOINT_DIR}/lgb_fold_{fold_num}.txt'
        
        # LightGBM parameters for multiclass classification
        lgb_params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbosity': -1,
            'seed': 42,
            # 'device': 'gpu'  # Uncomment if GPU LightGBM is available
        }
        
        # Check if model already exists (resume logic)
        bst = None
        if os.path.exists(model_path):
            try:
                bst = lgb.Booster(model_file=model_path)
                print(f"  ✅ Model already trained, loading from checkpoint...", flush=True)
                model_loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}, retraining...", flush=True)
                model_loaded = False
        else:
            model_loaded = False
        
        # Use global scaler (fitted earlier)
        scaler = global_scaler
        
        if not model_loaded:
            print(f"  Training LightGBM on {len(train_mapped)} chunks...", flush=True)
            print(f"  Using global StandardScaler for feature normalization", flush=True)
            print(f"  Parameters: learning_rate={lgb_params['learning_rate']}, num_leaves={lgb_params['num_leaves']}", flush=True)
            
            start_time = time.time()
            rounds_per_chunk = 50  # Number of boosting rounds per chunk
            
            # Train using incremental learning on each chunk
            for chunk_idx, chunk_data in enumerate(train_mapped):
                if (chunk_idx + 1) % 10 == 0:
                    print(f"\r    Processing chunk {chunk_idx+1}/{len(train_mapped)}...", end='', flush=True)
                
                # Load chunk data (embeddings are already normalized in load_chunk_data)
                X_chunk, y_chunk = load_chunk_data([chunk_data])
                
                if len(X_chunk) == 0:
                    continue
                
                # Transform with global scaler
                X_chunk_scaled = scaler.transform(X_chunk)
                
                # Create LightGBM Dataset
                dtrain = lgb.Dataset(X_chunk_scaled, label=y_chunk, free_raw_data=True)
                
                # Train (continue from previous model if exists)
                bst = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=rounds_per_chunk,
                    init_model=bst,
                    keep_training_booster=True
                )
                
                # Periodic saving
                if (chunk_idx + 1) % 20 == 0:
                    bst.save_model(model_path)
                    save_and_sync_checkpoint(model_path)
                
                # Memory cleanup
                del X_chunk, X_chunk_scaled, y_chunk, dtrain
                gc.collect()
            
            elapsed_time = time.time() - start_time
            print(f"\n  ✅ Training completed in {elapsed_time:.1f}s")
            
            # Save final model
            try:
                bst.save_model(model_path)
                save_and_sync_checkpoint(model_path)
                print(f"  ✅ Model saved to {model_path}", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Failed to save model checkpoint: {e}", flush=True)
        # Make predictions on validation set and save each chunk separately
        print(f"  Making predictions on validation ({len(val_mapped)} chunks)...", flush=True)
        oof_chunk_files = []
        
        for chunk_idx, chunk_data in enumerate(val_mapped):
            if (chunk_idx + 1) % 10 == 0:
                print(f"\r    Predicting chunk {chunk_idx+1}/{len(val_mapped)}...", end='', flush=True)
            
            # Load validation chunk
            X_val_chunk, _ = load_chunk_data([chunk_data])
            
            if len(X_val_chunk) == 0:
                continue
            
            # Transform with global scaler
            X_val_chunk_scaled = scaler.transform(X_val_chunk)
            
            # Predict probabilities with LightGBM
            val_preds = bst.predict(X_val_chunk_scaled, num_iteration=bst.best_iteration if hasattr(bst, 'best_iteration') else None)
            val_preds = val_preds.astype(np.float32)
            
            # LightGBM predict returns probabilities directly, but check for validity
            nan_count = np.isnan(val_preds).sum()
            inf_count = np.isinf(val_preds).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"\n    ⚠️  WARNING: Chunk {chunk_idx+1} has {nan_count} NaN and {inf_count} Inf in predictions!", flush=True)
                # Replace NaN/Inf (shouldn't happen with LightGBM, but just in case)
                val_preds = np.nan_to_num(val_preds, nan=1.0/num_classes, posinf=1.0, neginf=0.0)
            
            # Ensure probabilities sum to 1 (LightGBM should already do this, but verify)
            row_sums = val_preds.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            val_preds = val_preds / row_sums
            
            # Save chunk predictions to disk immediately
            chunk_file = f'{CHECKPOINT_DIR}/oof_preds_fold{fold_num}_chunk{chunk_idx}.npy'
            np.save(chunk_file, val_preds)
            oof_chunk_files.append(chunk_file)
            
            # Memory cleanup
            del X_val_chunk, X_val_chunk_scaled, val_preds
            gc.collect()
        
        print(f"\n  ✅ Validation predictions completed and saved to {len(oof_chunk_files)} files")
        
        # Return model and scaler in dict format for compatibility
        return {'model': bst, 'scaler': scaler}, oof_chunk_files, scaler
    
    # Function to combine chunk predictions into HDF5 file
    def combine_chunks_to_hdf5(chunk_files, output_h5_file, val_indices):
        """Combine prediction chunks into HDF5 file without loading all into RAM"""
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required. Install with: pip install h5py")
        
        # First pass: determine dimensions
        print(f"  Determining dimensions from {len(chunk_files)} chunks...")
        n_classes = None
        chunk_sizes = []
        total_rows = 0
        
        for chunk_file in chunk_files:
            if not os.path.exists(chunk_file):
                continue
            arr = np.load(chunk_file, mmap_mode='r')
            rows, n_cls = arr.shape
            chunk_sizes.append(rows)
            total_rows += rows
            if n_classes is None:
                n_classes = n_cls
        
        print(f"  Total rows: {total_rows:,}, Classes: {n_classes}")
        
        # Optimize chunk size: aim for ~1 MB chunks
        optimal_chunk_rows = max(10000, min(100000, int(256000 / n_classes)))
        
        # Create HDF5 file with optimized settings
        # Increase chunk cache for better I/O performance (10 MB cache)
        h5file = h5py.File(output_h5_file, 'w', libver='latest', 
                          rdcc_nbytes=10*1024*1024, rdcc_nslots=1001, rdcc_w0=0.75)
        dset = h5file.create_dataset(
            'preds',
            shape=(total_rows, n_classes),
            dtype='float32',
            chunks=(optimal_chunk_rows, n_classes),
            compression=None,  # Disable compression for speed
            fillvalue=None
        )
        
        # Stream chunks into HDF5 in optimal batch sizes
        print(f"  Writing chunks with batch size: {optimal_chunk_rows:,} rows...")
        pos = 0
        for chunk_idx, (chunk_file, rows) in enumerate(zip(chunk_files, chunk_sizes)):
            if not os.path.exists(chunk_file):
                continue
            
            # Use mmap_mode='r' for efficient reading without full memory load
            arr = np.load(chunk_file, mmap_mode='r')
            
            # Write in batches if chunk is large (better I/O performance)
            if rows > optimal_chunk_rows:
                for batch_start in range(0, rows, optimal_chunk_rows):
                    batch_end = min(batch_start + optimal_chunk_rows, rows)
                    batch_data = arr[batch_start:batch_end]
                    dset[pos:pos+(batch_end-batch_start), :] = batch_data
                    pos += (batch_end - batch_start)
                    del batch_data
            else:
                # Small chunk: write directly
                dset[pos:pos+rows, :] = arr[:]
                pos += rows
            
            # Clean up
            del arr
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()  # Periodic cleanup
        
        h5file.close()
        print(f"  ✅ Combined predictions saved to {output_h5_file}")
        # Sync HDF5 file to Drive
        save_and_sync_checkpoint(output_h5_file)
        return output_h5_file
    
    # Function to compute metrics from HDF5 file (memory-efficient)
    def compute_metrics_from_hdf5(h5_file, y_true, num_classes):
        """Compute metrics reading from HDF5 file in batches without loading all into RAM"""
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required. Install with: pip install h5py")
        
        h5file = h5py.File(h5_file, 'r')
        preds = h5file['preds']
        
        # Check for NaN/Inf in predictions
        print(f"  Checking predictions for NaN/Inf...", flush=True)
        nan_total = 0
        inf_total = 0
        batch_size_check = 100000
        for i in range(0, len(preds), batch_size_check):
            end_idx = min(i + batch_size_check, len(preds))
            batch_check = preds[i:end_idx]
            nan_total += np.isnan(batch_check).sum()
            inf_total += np.isinf(batch_check).sum()
        
        if nan_total > 0 or inf_total > 0:
            print(f"  ⚠️  WARNING: Found {nan_total} NaN and {inf_total} Inf values in predictions!", flush=True)
            print(f"  Will replace with valid values before computing metrics", flush=True)
        
        # For very large datasets, compute metrics incrementally
        # But for reasonable sizes, load in batches and compute
        batch_size = 100000
        total_samples = len(preds)
        
        # If dataset is small enough, load all at once
        if total_samples <= 500000:
            all_preds = preds[:]
            h5file.close()
            
            # Обработка NaN/Inf перед вычислением метрик
            if np.isnan(all_preds).any() or np.isinf(all_preds).any():
                print(f"  Cleaning NaN/Inf in predictions...", flush=True)
                all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=1.0, neginf=0.0)
                # Если все еще есть проблемы, используем равномерное распределение
                if np.isnan(all_preds).any() or np.isinf(all_preds).any():
                    uniform_fallback = np.ones_like(all_preds) / all_preds.shape[1]
                    all_preds = np.where(np.isfinite(all_preds), all_preds, uniform_fallback)
                # Нормализуем
                all_preds = all_preds / all_preds.sum(axis=1, keepdims=True)
            
            try:
                loss = log_loss(y_true, all_preds, labels=np.arange(num_classes))
                acc = accuracy_score(y_true, np.argmax(all_preds, axis=1))
            except Exception as e:
                print(f"  ⚠️  Error computing metrics: {e}", flush=True)
                # Fallback: используем равномерное распределение
                uniform_preds = np.ones_like(all_preds) / all_preds.shape[1]
                loss = log_loss(y_true, uniform_preds, labels=np.arange(num_classes))
                acc = accuracy_score(y_true, np.argmax(uniform_preds, axis=1))
            
            return loss, acc
        
        # For very large datasets, compute incrementally
        # Compute accuracy incrementally
        correct = 0
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_preds = preds[i:end_idx]
            batch_y = y_true[i:end_idx]
            
            # Обработка NaN/Inf перед argmax
            if np.isnan(batch_preds).any() or np.isinf(batch_preds).any():
                batch_preds = np.nan_to_num(batch_preds, nan=0.0, posinf=1.0, neginf=0.0)
                if np.isnan(batch_preds).any() or np.isinf(batch_preds).any():
                    uniform_fallback = np.ones_like(batch_preds) / batch_preds.shape[1]
                    batch_preds = np.where(np.isfinite(batch_preds), batch_preds, uniform_fallback)
                batch_preds = batch_preds / batch_preds.sum(axis=1, keepdims=True)
            
            batch_pred_classes = np.argmax(batch_preds, axis=1)
            correct += np.sum(batch_pred_classes == batch_y)
        
        acc = correct / total_samples
        
        # For log_loss, we need all predictions, but can compute in batches
        # Compute log_loss incrementally using formula
        epsilon = 1e-15
        loss_sum = 0.0
        valid_samples = 0
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_preds = preds[i:end_idx]
            batch_y = y_true[i:end_idx]
            
            # Обработка NaN/Inf перед вычислением loss
            if np.isnan(batch_preds).any() or np.isinf(batch_preds).any():
                batch_preds = np.nan_to_num(batch_preds, nan=0.0, posinf=1.0, neginf=0.0)
                if np.isnan(batch_preds).any() or np.isinf(batch_preds).any():
                    uniform_fallback = np.ones_like(batch_preds) / batch_preds.shape[1]
                    batch_preds = np.where(np.isfinite(batch_preds), batch_preds, uniform_fallback)
                batch_preds = batch_preds / batch_preds.sum(axis=1, keepdims=True)
            
            # Clip predictions to avoid log(0)
            batch_preds = np.clip(batch_preds, epsilon, 1 - epsilon)
            
            # One-hot encode true labels
            y_one_hot = np.zeros((len(batch_y), num_classes))
            y_one_hot[np.arange(len(batch_y)), batch_y] = 1
            
            # Compute log loss for this batch
            batch_loss = -np.sum(y_one_hot * np.log(batch_preds)) / len(batch_y)
            
            # Проверяем, что batch_loss валиден
            if np.isfinite(batch_loss):
                loss_sum += batch_loss * len(batch_y)
                valid_samples += len(batch_y)
            else:
                print(f"  ⚠️  WARNING: Invalid batch_loss at batch {i//batch_size + 1}, skipping...", flush=True)
        
        if valid_samples > 0:
            loss = loss_sum / valid_samples
        else:
            print(f"  ⚠️  WARNING: No valid samples for loss computation! Using fallback.", flush=True)
            # Fallback: используем равномерное распределение
            uniform_preds = np.ones((total_samples, num_classes)) / num_classes
            loss = log_loss(y_true, uniform_preds, labels=np.arange(num_classes))
        
        h5file.close()
        
        return loss, acc
    
    models = []
    oof_chunk_files_by_fold = {}  # Store chunk files for each fold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(sample_count), all_y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*50}")
        
        # Map train/val indices to chunks
        train_mapped = get_indices_for_files(train_idx, feature_files, chunk_sizes)
        val_mapped = get_indices_for_files(val_idx, feature_files, chunk_sizes)
        
        print(f"Train chunks: {len(train_mapped)}, Val chunks: {len(val_mapped)}")
        
        # Train single LightGBM model using chunked incremental training
        fold_model_data, fold_chunk_files, fold_scaler = train_lgb_chunkwise(
            train_mapped, 
            val_mapped, 
            num_classes,
            fold + 1  # Передаем номер fold для resume логики
        )
        
        # Store chunk files for this fold
        oof_chunk_files_by_fold[fold] = {
            'files': fold_chunk_files,
            'val_idx': val_idx
        }
        
        # Combine chunks into HDF5 for this fold
        h5_file = f'{CHECKPOINT_DIR}/oof_preds_fold{fold+1}_full.h5'
        combine_chunks_to_hdf5(fold_chunk_files, h5_file, val_idx)
        
        # Clean up temporary .npy chunk files to save disk space
        if H5PY_AVAILABLE and os.path.exists(h5_file):
            print(f"  Cleaning up temporary .npy chunk files...")
            cleaned_count = 0
            for chunk_file in fold_chunk_files:
                if os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                        cleaned_count += 1
                    except:
                        pass
            print(f"  ✅ Removed {cleaned_count} temporary chunk files")
        
        # Load validation labels for metrics (batch-wise)
        val_labels_list = []
        for feat_path, feat_idx in val_mapped:
            chunk_idx = int(feat_path.split('_')[-1].replace('.pkl', ''))
            y_chunk = joblib.load(target_files[chunk_idx])[feat_idx]
            val_labels_list.append(y_chunk)
        y_fold_val = np.concatenate(val_labels_list)
        del val_labels_list
        
        # Compute metrics from HDF5
        val_loss, val_acc = compute_metrics_from_hdf5(h5_file, y_fold_val, num_classes)
        print(f"✅ Fold {fold + 1} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Store model and scaler (one per fold) - fold_model_data is already a dict
        models.append(fold_model_data)
        
        print(f"   Saved LightGBM model for fold {fold + 1}")
        
        # Memory cleanup after each fold
        del fold_model_data, fold_chunk_files, y_fold_val
        gc.collect()
        print("✅ Memory cleaned after fold")
    
    # Overall metrics: combine all folds into single HDF5
    print(f"\n{'='*50}")
    print("Combining all folds into final OOF predictions...")
    print(f"{'='*50}")
    
    # Create final combined HDF5 file with optimized settings
    final_h5_file = f'{CHECKPOINT_DIR}/oof_preds_all_folds.h5'
    if H5PY_AVAILABLE:
        # Optimize chunk size: aim for ~1 MB chunks
        # Assuming float32 (4 bytes), chunk size ~1MB = 256k elements
        # For shape (N, num_classes), optimal chunk rows = 256000 / num_classes
        optimal_chunk_rows = max(10000, min(100000, int(256000 / num_classes)))
        
        print(f"  Creating HDF5 file with optimized settings...")
        print(f"  Chunk size: ({optimal_chunk_rows}, {num_classes}) ≈ {optimal_chunk_rows * num_classes * 4 / 1024 / 1024:.1f} MB")
        
        # Use libver='latest' for better performance, increase chunk cache
        # Increase chunk cache for better I/O performance (10 MB cache)
        h5file = h5py.File(final_h5_file, 'w', libver='latest',
                          rdcc_nbytes=10*1024*1024, rdcc_nslots=1001, rdcc_w0=0.75)
        
        # Create dataset with optimized chunk size
        dset = h5file.create_dataset(
            'preds',
            shape=(sample_count, num_classes),
            dtype='float32',
            chunks=(optimal_chunk_rows, num_classes),
            compression=None,  # Disable compression for speed (can enable if disk space is critical)
            fillvalue=None  # No fill value needed
        )
        
        # Fill predictions from each fold using batched reads/writes
        batch_size = optimal_chunk_rows  # Read/write in chunks matching dataset chunks
        print(f"  Processing {len(oof_chunk_files_by_fold)} folds with batch size: {batch_size:,} rows...")
        
        for fold, fold_data in oof_chunk_files_by_fold.items():
            val_idx = fold_data['val_idx']
            fold_h5_file = f'{CHECKPOINT_DIR}/oof_preds_fold{fold+1}_full.h5'
            
            print(f"  Processing fold {fold+1}...", end=' ', flush=True)
            fold_start_time = time.time()
            
            # Open fold HDF5 file
            fold_h5 = h5py.File(fold_h5_file, 'r')
            fold_preds = fold_h5['preds']
            fold_size = len(fold_preds)
            
            # Write in batches instead of loading entire array
            # val_idx maps local positions in fold_preds to global positions in final array
            # Process sequentially in batches for optimal I/O
            for batch_start in range(0, fold_size, batch_size):
                batch_end = min(batch_start + batch_size, fold_size)
                batch_local_idx = np.arange(batch_start, batch_end)
                
                # Get corresponding global indices for this batch
                batch_global_idx = val_idx[batch_local_idx]
                
                # Read batch from fold file (sequential read is optimal)
                batch_data = fold_preds[batch_local_idx]
                
                # Write batch to final file
                dset[batch_global_idx] = batch_data
                
                # Memory cleanup
                del batch_data
            
            fold_h5.close()
            fold_time = time.time() - fold_start_time
            print(f"✓ ({fold_time:.1f}s, {fold_size:,} rows)", flush=True)
        
        h5file.close()
        print(f"✅ Final OOF predictions saved to {final_h5_file}")
        # Sync final HDF5 file to Drive
        save_and_sync_checkpoint(final_h5_file)
        
        # Compute overall metrics
        overall_loss, overall_acc = compute_metrics_from_hdf5(final_h5_file, all_y, num_classes)
        print(f"\n{'='*50}")
        print(f"✅ Overall CV - Loss: {overall_loss:.4f}, Accuracy: {overall_acc:.4f}")
        print(f"{'='*50}")
    else:
        print("⚠️  h5py not available, skipping overall metrics computation")
    
    # Save models (КРИТИЧНО для инференса на test данных!)
    models_file = f'{CHECKPOINT_DIR}/trained_models.pkl'
    print(f"\n💾 Сохраняем обученные модели в {models_file}...")
    joblib.dump({'models': models, 'label_encoder': label_encoder}, models_file)
    save_and_sync_checkpoint(models_file)
    print(f"✅ Модели успешно сохранены! Файл {models_file} готов для инференса на test данных.")
    
    # Memory cleanup
    del all_y
    if H5PY_AVAILABLE:
        del oof_chunk_files_by_fold
    gc.collect()

# Aggressive memory cleanup before test processing
print("\n🧹 Cleaning up memory before test processing...")
print_memory_usage()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✅ Memory cleaned")
print_memory_usage()

# ============================================
# STEP 4.4: Process test data
# ============================================
print("\n[4/6] Processing test data in chunks...")

# Check if test chunk index already exists
test_chunk_index_file = f'{CHECKPOINT_DIR}/test_chunk_index.pkl'
if os.path.exists(test_chunk_index_file):
    print("✅ Found preprocessed test chunk index, loading...")
    test_chunk_data = joblib.load(test_chunk_index_file)
    test_feature_files = test_chunk_data['feature_files']
    test_ids = test_chunk_data['test_ids']
    print(f"✅ Loaded {len(test_feature_files)} test feature chunks")
else:
    print("Processing new test features...")
    
    def process_test_data_chunked():
        """Process test data in chunks - saves each chunk to disk"""
        
        feature_files = []
        all_ids = []
        
        print("Processing test features in chunks (saving each chunk to disk)...")
        for i, chunk in enumerate(pd.read_csv('test.csv', chunksize=CHUNK_SIZE)):
            print(f"\rProcessing test chunk {i+1}...", end='', flush=True)
            
            # Save IDs
            if 'id' in chunk.columns:
                all_ids.extend(chunk['id'].values)
                chunk = chunk.drop(columns=['id'])
            else:
                all_ids.extend(chunk.index.values)
            
            # Extract features
            X_chunk = BasicFeatureExtractor.extract(chunk, TCP_COLUMNS)
            
            # Save chunk to disk
            feature_path = f'{CHECKPOINT_DIR}/test_features_chunk_{i}.pkl'
            joblib.dump(X_chunk, feature_path)
            feature_files.append(feature_path)
            
            # Memory cleanup
            del chunk, X_chunk
            gc.collect()
        
        print(f"\n✅ All {len(feature_files)} test chunks processed and saved to disk")
        
        # Save test chunk index
        joblib.dump({
            'feature_files': feature_files,
            'test_ids': all_ids,
            'num_chunks': len(feature_files)
        }, test_chunk_index_file)
        
        return feature_files, all_ids
    
    test_feature_files, test_ids = process_test_data_chunked()
    
    print(f"✅ Processed {len(test_feature_files)} test chunks (checkpoint saved)")
    
    # Memory cleanup
    gc.collect()

# ============================================
# STEP 4.5: Add test embeddings
# ============================================
print("\n[5/6] Creating test embeddings...")
print_memory_usage()

# Check if test embeddings index already exists
test_embedding_index_file = f'{CHECKPOINT_DIR}/test_embeddings_index.pkl'
if os.path.exists(test_embedding_index_file):
    print("✅ Found preprocessed test embeddings index, loading...")
    test_emb_data = joblib.load(test_embedding_index_file)
    test_embedding_files = test_emb_data['embedding_files']
    print(f"✅ Loaded {len(test_embedding_files)} test embedding chunks")
else:
    print("Creating new test embeddings...")
    
    # Process embeddings chunk-by-chunk and save separately
    print("Extracting embeddings chunk-by-chunk...")
    test_embedding_files = []
    
    chunk_num = 0
    for feat_path in test_feature_files:
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"\r  Extracting embeddings chunk {chunk_num}/{len(test_feature_files)}...", end='', flush=True)
        
        # Load feature chunk
        X_chunk = joblib.load(feat_path)
        seq_chunk = X_chunk[TCP_COLUMNS].values.astype(np.float32)
        seq_chunk = np.nan_to_num(seq_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract embeddings for this chunk
        chunk_embeddings = embedder.extract_embeddings(seq_chunk, batch_size=512)
        
        # Save embedding chunk
        emb_path = feat_path.replace('test_features_', 'test_embedding_')
        joblib.dump(chunk_embeddings, emb_path)
        test_embedding_files.append(emb_path)
        
        # Memory cleanup after each chunk
        del X_chunk, seq_chunk, chunk_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n✅ All test embeddings extracted and saved to {len(test_embedding_files)} files")
    
    # Save test embedding index
    joblib.dump({
        'embedding_files': test_embedding_files
    }, test_embedding_index_file)
    
    print("✅ Test embeddings index saved")

print(f"✅ Test embeddings ready: {len(test_embedding_files)} chunks")

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Delete embedder to free memory
print("\n🧹 Freeing embedder memory...")
print_memory_usage()
try:
    del embedder
except:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✅ Memory freed")
print_memory_usage()

# ============================================
# STEP 4.6: Make predictions
# ============================================
print("\n[6/6] Making predictions...")
print(f"Total test chunks: {len(test_feature_files)}")
print(f"Number of models: {len(models)}")
print(f"Number of classes: {num_classes}")

# Проверка наличия моделей
if len(models) == 0:
    raise ValueError("❌ ERROR: No models available for predictions! Models list is empty. "
                     "Make sure models were trained or loaded from checkpoint.")

# Predict chunk-by-chunk (load features + embeddings, predict, save)
print("\nPredicting chunk-by-chunk...")
all_predictions = []

# Check for existing predictions checkpoint
PRED_CHECKPOINT_FILE = f'{CHECKPOINT_DIR}/predictions_checkpoint.pkl'
start_chunk = 0

if os.path.exists(PRED_CHECKPOINT_FILE):
    print("\n✅ Found predictions checkpoint, loading...")
    checkpoint_data = joblib.load(PRED_CHECKPOINT_FILE)
    all_predictions = checkpoint_data['predictions']
    start_chunk = checkpoint_data['last_chunk'] + 1
    print(f"✅ Resuming from chunk {start_chunk}/{len(test_feature_files)}")
    print(f"   Already processed: {len(all_predictions)} chunks")

try:
    for chunk_idx in range(start_chunk, len(test_feature_files)):
        print(f"\nChunk {chunk_idx+1}/{len(test_feature_files)}...", end=' ', flush=True)
        
        # Load feature chunk
        X_chunk = joblib.load(test_feature_files[chunk_idx])
        
        # Load corresponding embedding chunk
        emb_chunk = joblib.load(test_embedding_files[chunk_idx])
        
        # CRITICAL: Normalize embeddings (L2 normalization) - same as in training
        emb_norms = np.linalg.norm(emb_chunk, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms > 0, emb_norms, 1.0)  # Avoid division by zero
        emb_chunk = emb_chunk / emb_norms
        
        embedding_cols = [f'emb_{i}' for i in range(emb_chunk.shape[1])]
        emb_df = pd.DataFrame(emb_chunk, columns=embedding_cols, index=X_chunk.index)
        
        # Combine features and embeddings
        X_chunk = pd.concat([X_chunk.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
        
        # Clean data
        X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)
        X_chunk = X_chunk.fillna(0)
        X_chunk = X_chunk.clip(-1e15, 1e15)
        
        print(f"loaded ({len(X_chunk)} samples)...", end=' ', flush=True)
        
        # Convert to numpy array for SGDClassifier
        X_chunk_array = X_chunk.values.astype(np.float32)
        
        # Диагностика данных для первого чанка
        if chunk_idx == start_chunk:
            print(f"\n  Data diagnostics:", flush=True)
            print(f"    X_chunk_array shape: {X_chunk_array.shape}", flush=True)
            print(f"    X_chunk_array dtype: {X_chunk_array.dtype}", flush=True)
            print(f"    Feature stats: min={X_chunk_array.min():.6f}, max={X_chunk_array.max():.6f}, "
                  f"mean={X_chunk_array.mean():.6f}, std={X_chunk_array.std():.6f}", flush=True)
            
            # Проверяем, не все ли данные одинаковые
            if X_chunk_array.std() < 1e-6:
                print(f"    ⚠️  WARNING: Very low std ({X_chunk_array.std():.6f}), data might be constant!", flush=True)
            
            # Проверяем наличие NaN или Inf
            nan_count = np.isnan(X_chunk_array).sum()
            inf_count = np.isinf(X_chunk_array).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"    ⚠️  WARNING: Found {nan_count} NaN and {inf_count} Inf values!", flush=True)
        
        # Load global scaler for test predictions if not available in models
        test_scaler_path = f'{CHECKPOINT_DIR}/global_scaler.pkl'
        if os.path.exists(test_scaler_path):
            test_global_scaler = joblib.load(test_scaler_path)
        else:
            test_global_scaler = None
            if chunk_idx == start_chunk:
                print(f"\n    ⚠️  WARNING: Global scaler not found! Predictions may be inaccurate.", flush=True)
        
        # Average predictions from all CV folds (simple averaging, no parallelization)
        chunk_preds_list = []
        valid_folds = []
        for fold_idx, model_data in enumerate(models):
            # Extract model and scaler
            if isinstance(model_data, dict):
                model = model_data['model']
                scaler = model_data.get('scaler', None)
            else:
                # Old format: just model
                model = model_data
                scaler = None
            
            # Use scaler from model, or fallback to global scaler, or use unscaled
            if scaler is not None:
                X_chunk_scaled = scaler.transform(X_chunk_array)
            elif test_global_scaler is not None:
                X_chunk_scaled = test_global_scaler.transform(X_chunk_array)
            else:
                # Fallback: use data as-is (not ideal but backward compatible)
                if chunk_idx == start_chunk and fold_idx == 0:
                    print(f"\n    ⚠️  WARNING: Fold {fold_idx+1} has no scaler! Using unscaled data.", flush=True)
                X_chunk_scaled = X_chunk_array
            
            # Predict with LightGBM or SGDClassifier
            if LIGHTGBM_AVAILABLE and isinstance(model, lgb.Booster):
                # LightGBM prediction
                preds = model.predict(X_chunk_scaled, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
                preds = preds.astype(np.float32)
            else:
                # Old SGDClassifier format
                preds = model.predict_proba(X_chunk_scaled)  # должен быть shape (N_samples_in_chunk, N_classes)
            # Проверка размерности предсказания
            assert preds.ndim == 2, f"preds wrong shape: {preds.shape}, expected 2D array (samples, classes)"
            assert preds.shape[0] == len(X_chunk_array), f"preds.shape[0]={preds.shape[0]} != len(X_chunk_array)={len(X_chunk_array)}"
            
            # Проверка на NaN и Inf в предсказаниях
            nan_count = np.isnan(preds).sum()
            inf_count = np.isinf(preds).sum()
            
            # Если есть NaN или Inf, пытаемся исправить
            if nan_count > 0 or inf_count > 0:
                print(f"\n    ⚠️  WARNING: Fold {fold_idx+1} has {nan_count} NaN and {inf_count} Inf values!", flush=True)
                
                # Заменяем Inf на большие числа
                preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Если все еще есть проблемы, заменяем на равномерное распределение
                if np.isnan(preds).any() or np.isinf(preds).any():
                    print(f"    ⚠️  Fold {fold_idx+1}: Using uniform distribution as fallback", flush=True)
                    uniform_preds = np.ones_like(preds) / preds.shape[1]
                    preds = np.where(np.isfinite(preds), preds, uniform_preds)
                
                # Нормализуем вероятности (сумма должна быть 1)
                row_sums = preds.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Избегаем деления на 0
                preds = preds / row_sums
            
            # Проверяем, что вероятности валидны (сумма по строкам ≈ 1)
            row_sums = preds.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=0.01):
                print(f"\n    ⚠️  WARNING: Fold {fold_idx+1} probabilities don't sum to 1! Normalizing...", flush=True)
                preds = preds / preds.sum(axis=1, keepdims=True)
            
            # Диагностика для первого чанка
            if chunk_idx == start_chunk:
                if fold_idx == 0:
                    print(f"\n  preds.shape = {preds.shape}", flush=True)
                    print(f"  First chunk diagnostics:", flush=True)
                
                # Проверяем разнообразие предсказаний от каждой модели
                pred_classes = np.argmax(preds, axis=1)
                unique_classes, counts = np.unique(pred_classes, return_counts=True)
                print(f"    Fold {fold_idx+1}: {len(unique_classes)} unique classes predicted, "
                      f"most common: class {unique_classes[np.argmax(counts)]} ({np.max(counts)}/{len(pred_classes)} samples)", flush=True)
                
                # Проверяем, что вероятности не все одинаковые
                if len(preds) > 0:
                    pred_std = np.std(preds, axis=0)
                    valid_std = pred_std[np.isfinite(pred_std)]
                    if len(valid_std) > 0:
                        print(f"    Fold {fold_idx+1}: std of probabilities per class: min={valid_std.min():.6f}, max={valid_std.max():.6f}", flush=True)
                    else:
                        print(f"    Fold {fold_idx+1}: ⚠️  All std values are NaN/Inf!", flush=True)
            
            chunk_preds_list.append(preds)
            valid_folds.append(fold_idx)
        
        # Проверка, что список не пустой
        if len(chunk_preds_list) == 0:
            raise ValueError(f"❌ ERROR: chunk_preds_list is empty! No predictions were generated. "
                           f"Number of models: {len(models)}, chunk_idx: {chunk_idx}")
        
        # Преобразуем список в numpy массив: shape (N_folds, N_samples, N_classes)
        chunk_preds_array = np.stack(chunk_preds_list, axis=0)
        if chunk_idx == start_chunk:  # Выводим только для первого чанка
            print(f"  chunk_preds_array.shape = {chunk_preds_array.shape}", flush=True)
        
        # Проверяем наличие NaN в массиве перед усреднением
        nan_mask = np.isnan(chunk_preds_array)
        if nan_mask.any():
            nan_folds = nan_mask.any(axis=(1, 2))  # Фолды с NaN
            if nan_folds.any():
                print(f"\n  ⚠️  WARNING: {nan_folds.sum()} fold(s) contain NaN values!", flush=True)
                print(f"  Using nanmean to ignore NaN values in averaging", flush=True)
        
        # Усредняем по фолдам (axis=0), игнорируя NaN: shape (N_samples, N_classes)
        chunk_preds = np.nanmean(chunk_preds_array, axis=0)
        
        # Если после nanmean все еще есть NaN (все фолды дали NaN для некоторых элементов)
        if np.isnan(chunk_preds).any():
            nan_samples = np.isnan(chunk_preds).any(axis=1).sum()
            print(f"\n  ⚠️  WARNING: {nan_samples} samples still have NaN after averaging!", flush=True)
            print(f"  Replacing with uniform distribution", flush=True)
            uniform_fallback = np.ones_like(chunk_preds) / chunk_preds.shape[1]
            chunk_preds = np.where(np.isfinite(chunk_preds), chunk_preds, uniform_fallback)
            # Нормализуем на всякий случай
            chunk_preds = chunk_preds / chunk_preds.sum(axis=1, keepdims=True)
        
        # Проверяем, что усредненные вероятности валидны
        row_sums = chunk_preds.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.01):
            print(f"\n  ⚠️  WARNING: Averaged probabilities don't sum to 1! Normalizing...", flush=True)
            chunk_preds = chunk_preds / chunk_preds.sum(axis=1, keepdims=True)
        
        # Финальная проверка shape
        assert chunk_preds.ndim == 2, f"chunk_preds wrong shape: {chunk_preds.shape}, expected 2D array (samples, classes)"
        assert chunk_preds.shape[0] == len(X_chunk_array), f"chunk_preds.shape[0]={chunk_preds.shape[0]} != len(X_chunk_array)={len(X_chunk_array)}"
        
        print(f"predicting... (shape: {chunk_preds.shape})", end='', flush=True)
        
        all_predictions.append(chunk_preds)
        print(" ✓", flush=True)
        
        # Save checkpoint every 10 chunks
        if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == len(test_feature_files):
            joblib.dump({
                'predictions': all_predictions,
                'last_chunk': chunk_idx
            }, PRED_CHECKPOINT_FILE)
            print(f"   💾 Checkpoint saved (chunk {chunk_idx + 1})")
        
        # Memory cleanup
        del X_chunk, emb_chunk, emb_df, X_chunk_array, chunk_preds, chunk_preds_list
        gc.collect()
        
except Exception as e:
    print(f"\n❌ ERROR during predictions: {e}")
    import traceback
    traceback.print_exc()
    raise

# Combine predictions
print("\nCombining predictions...")
predictions = np.vstack(all_predictions)
print(f"  Combined predictions shape: {predictions.shape}")

# Clean up predictions checkpoint
if os.path.exists(PRED_CHECKPOINT_FILE):
    os.remove(PRED_CHECKPOINT_FILE)
    print("✅ Predictions checkpoint cleaned")

# Convert to labels
predicted_classes = np.argmax(predictions, axis=1)
print(f"\n📊 Prediction statistics:")
print(f"  Total predictions: {len(predicted_classes):,}")
unique_pred_classes, pred_counts = np.unique(predicted_classes, return_counts=True)
print(f"  Unique predicted classes: {len(unique_pred_classes)}")
print(f"  Class distribution:")
for cls, count in zip(unique_pred_classes[:10], pred_counts[:10]):  # Показываем первые 10 классов
    pct = count / len(predicted_classes) * 100
    print(f"    Class {cls}: {count:,} ({pct:.2f}%)")
if len(unique_pred_classes) > 10:
    print(f"    ... and {len(unique_pred_classes) - 10} more classes")

# Проверка на проблему с одинаковыми предсказаниями
if len(unique_pred_classes) == 1:
    print(f"\n⚠️  WARNING: All predictions are the same class ({unique_pred_classes[0]})!")
    print(f"  This indicates a problem with the models or data.")
    print(f"  Checking prediction probabilities...")
    print(f"  Mean probabilities per class: {predictions.mean(axis=0)}")
    print(f"  Std probabilities per class: {predictions.std(axis=0)}")
    print(f"  Max probability range: [{predictions.min():.6f}, {predictions.max():.6f}]")
elif pred_counts[0] / len(predicted_classes) > 0.95:
    print(f"\n⚠️  WARNING: {pred_counts[0]/len(predicted_classes)*100:.1f}% of predictions are the same class!")
    print(f"  This might indicate a problem with model training or data preprocessing.")

predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Проверка label_encoder
print(f"\n📊 Label encoder diagnostics:")
print(f"  Number of classes in encoder: {len(label_encoder.classes_)}")
print(f"  Classes: {label_encoder.classes_[:10]}{'...' if len(label_encoder.classes_) > 10 else ''}")

# Проверка уникальности предсказанных меток
unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
print(f"\n📊 Predicted labels statistics:")
print(f"  Unique predicted labels: {len(unique_labels)}")
print(f"  Label distribution:")
for label, count in zip(unique_labels[:10], label_counts[:10]):  # Показываем первые 10 меток
    pct = count / len(predicted_labels) * 100
    print(f"    '{label}': {count:,} ({pct:.2f}%)")
if len(unique_labels) > 10:
    print(f"    ... and {len(unique_labels) - 10} more labels")

# Проверка на проблему с одинаковыми метками
if len(unique_labels) == 1:
    print(f"\n❌ ERROR: All predicted labels are the same: '{unique_labels[0]}'!")
    print(f"  This is a critical problem. Possible causes:")
    print(f"  1. Models are not trained properly (all predict same class)")
    print(f"  2. Data preprocessing issue (all samples become identical)")
    print(f"  3. Label encoder issue")
    print(f"  Please check the diagnostics above.")
elif label_counts[0] / len(predicted_labels) > 0.95:
    print(f"\n⚠️  WARNING: {label_counts[0]/len(predicted_labels)*100:.1f}% of predictions are the same label '{unique_labels[0]}'!")
    print(f"  This might indicate a problem with model training or data preprocessing.")

# Проверка соответствия test_ids
print(f"\n📊 Test IDs check:")
print(f"  Number of test_ids: {len(test_ids):,}")
print(f"  Number of predictions: {len(predicted_labels):,}")
if len(test_ids) != len(predicted_labels):
    print(f"  ❌ ERROR: Mismatch! test_ids ({len(test_ids)}) != predictions ({len(predicted_labels)})")
    raise ValueError(f"Length mismatch: test_ids={len(test_ids)}, predictions={len(predicted_labels)}")
else:
    print(f"  ✅ Lengths match")

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    TARGET_COL: predicted_labels
})

# Финальная проверка submission
print(f"\n📊 Final submission check:")
print(f"  Submission shape: {submission.shape}")
print(f"  Unique values in '{TARGET_COL}': {submission[TARGET_COL].nunique()}")
print(f"  Value counts:")
print(submission[TARGET_COL].value_counts().head(10))

submission.to_csv('submission.csv', index=False)

# Memory cleanup
del predictions, all_predictions, predicted_classes
gc.collect()

print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"Submission saved to: submission.csv")
print(f"Number of predictions: {len(submission)}")
print(f"Checkpoints saved in: {CHECKPOINT_DIR}/")
print("="*70)

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n✅ All done! Download submission.csv")

# ============================================
# FINAL STEP: CLOSE LOGGER AND COPY TO GOOGLE DRIVE
# ============================================
print("\n" + "="*70)
print("Saving log file to Google Drive...")
print("="*70)

# Register copy function in atexit as backup (will try to copy if not already done)
atexit.register(lambda: logger.copy_to_drive())

# Restore stdout before closing logger
sys.stdout = logger.terminal

# Close logger and get log file path
log_file_path = logger.get_log_path()
logger.close()

# Copy log file to Google Drive
try:
    if logger.copy_to_drive():
        print(f"✅ Log file copied to Google Drive successfully")
    else:
        print(f"⚠️  Could not copy log to Google Drive (may not be in Colab)")
except Exception as e:
    print(f"⚠️  Error copying log to Google Drive: {e}")

print(f"\n📝 Log file location: {os.path.abspath(log_file_path)}")
print("="*70)

