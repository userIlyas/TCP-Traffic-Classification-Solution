"""
========================================
TCP TRAFFIC CLASSIFICATION - MAIN PIPELINE
========================================
Single file solution with:
- Kaggle data download
- Chunked processing for large datasets
- Memory optimization
- Essential checkpoint saving/loading (only critical files)
- Resume training from checkpoints
- LightGBM for multiclass classification

BEFORE RUNNING:
1. Execute:
    pip install kaggle h5py lightgbm

2. Then just run this script!

FEATURES:
- LightGBM: Fast, memory-efficient gradient boosting
- Resume training: Automatically continues from where it stopped (checks local checkpoints)
- Time logging: Shows training time for each fold
- Single model per fold (no ensemble): Minimal memory usage
- Chunk-based processing: All predictions processed in chunks without loading all into memory
- Essential checkpoints only: Only saves critical files needed for resuming

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

# Handle PyTorch import with workaround for duplicate registration error
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except RuntimeError as e:
    if "duplicate registrations" in str(e):
        print("⚠️  PyTorch duplicate registration error detected. Attempting workaround...")
        # Try to clear torch-related modules and retry
        import sys
        torch_modules = [k for k in sys.modules.keys() if 'torch' in k.lower()]
        for mod in torch_modules:
            del sys.modules[mod]
        try:
            import torch
            TORCH_AVAILABLE = True
            print("✅ PyTorch import successful after workaround")
        except Exception as e2:
            print(f"❌ Failed to import PyTorch after workaround: {e2}")
            print("⚠️  Continuing without PyTorch (RNN embeddings will be disabled)")
            TORCH_AVAILABLE = False
    else:
        raise
except ImportError:
    print("⚠️  PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

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

def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage by converting to more efficient dtypes"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f'Memory usage decreased to {end_mem:.2f} MB ({reduction:.1f}% reduction)')
    
    return df

# Conditional torch imports
if TORCH_AVAILABLE:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
else:
    # Create dummy classes to avoid NameError
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class LSTM:
            pass
    class F:
        pass
    class Dataset:
        pass
    class DataLoader:
        pass

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
if TORCH_AVAILABLE:
    print(f"torch:          {torch.__version__}")
else:
    print(f"torch:          not available")
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

if TORCH_AVAILABLE:
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
else:
    print(f"\n⚠️  PyTorch not available - RNN embeddings will be disabled")
print(f"✅ scikit-learn version: {sklearn.__version__}\n")

# ============================================
# STEP 2: CHECKPOINTS DIRECTORY SETUP
# ============================================
print("\n" + "="*70)
print("STEP 2: Setting up checkpoints directory...")
print("="*70)

# Create local checkpoints directory
CHECKPOINT_DIR = './checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"✅ Created checkpoints directory: {CHECKPOINT_DIR}")
else:
    print(f"✅ Checkpoints directory exists: {CHECKPOINT_DIR}")

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

class MTUPreprocessor:
    """
    Preprocess TCP packet sequences by merging consecutive packets.
    According to competition description: Merge into one packet those consecutive 
    packets that have the same direction and have a size of more than 1200.
    This helps reconstruct the original application data that was split by TCP/MTU.
    """
    
    @staticmethod
    def merge_packets(packet_sequence, threshold=1200):
        """
        Merge consecutive packets with same direction and size > threshold.
        
        Args:
            packet_sequence: Array of packet sizes (can contain positive/negative for direction)
            threshold: Minimum packet size to consider for merging (default 1200 bytes)
        
        Returns:
            Merged packet sequence (same length, padded with zeros)
        """
        if len(packet_sequence) == 0:
            return packet_sequence
        
        # Convert to list for easier manipulation
        packets = list(packet_sequence)
        merged = []
        i = 0
        
        while i < len(packets):
            current_packet = packets[i]
            
            # Skip zero packets
            if current_packet == 0:
                merged.append(0)
                i += 1
                continue
            
            # Check if packet is large enough and has same direction as next packets
            if abs(current_packet) > threshold:
                # Try to merge with consecutive packets of same direction
                merged_size = current_packet
                j = i + 1
                
                while j < len(packets):
                    next_packet = packets[j]
                    
                    # Stop if we hit zero or different direction
                    if next_packet == 0:
                        break
                    
                    # Check if same direction (both positive or both negative)
                    if (merged_size > 0 and next_packet > 0) or (merged_size < 0 and next_packet < 0):
                        # Merge if next packet is also large
                        if abs(next_packet) > threshold:
                            merged_size += next_packet
                            j += 1
                        else:
                            break
                    else:
                        # Different direction, stop merging
                        break
                
                merged.append(merged_size)
                i = j  # Skip merged packets
            else:
                # Small packet, don't merge
                merged.append(current_packet)
                i += 1
        
        # Pad to original length with zeros
        while len(merged) < len(packet_sequence):
            merged.append(0)
        
        return np.array(merged[:len(packet_sequence)])
    
    @staticmethod
    def preprocess_df(df, tcp_columns):
        """
        Preprocess DataFrame by merging packets in each row.
        
        Args:
            df: DataFrame with TCP packet columns
            tcp_columns: List of TCP column names
        
        Returns:
            DataFrame with preprocessed TCP columns
        """
        df_preprocessed = df.copy()
        tcp_data = df[tcp_columns].values
        
        # Apply merging to each row
        merged_data = np.array([
            MTUPreprocessor.merge_packets(row) 
            for row in tcp_data
        ])
        
        # Update DataFrame with merged data
        for idx, col in enumerate(tcp_columns):
            df_preprocessed[col] = merged_data[:, idx]
        
        return df_preprocessed


class BasicFeatureExtractor:
    """Extract basic statistical features from TCP sequences"""
    
    @staticmethod
    def extract(df, tcp_columns, apply_mtu_preprocessing=True):
        """
        Extract basic features
        
        Args:
            df: DataFrame with TCP packet columns
            tcp_columns: List of TCP column names
            apply_mtu_preprocessing: Whether to apply MTU packet merging (default True)
        """
        # CRITICAL: Apply MTU preprocessing to merge consecutive large packets
        if apply_mtu_preprocessing:
            df = MTUPreprocessor.preprocess_df(df, tcp_columns)
        
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
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Cannot create LifestreamEmbedder. "
                             "Please install PyTorch or fix the duplicate registration error.")
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


print("✅ Feature extraction classes ready")
print("  ✅ MTUPreprocessor: Merges consecutive packets (>1200 bytes, same direction)")
print("  ✅ BasicFeatureExtractor: Extracts statistical features from TCP sequences\n")

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
    print("  ✅ MTU preprocessing enabled: merging consecutive packets (>1200 bytes, same direction)")
    chunk_count = 0
    for i, chunk in enumerate(pd.read_csv('train.csv', chunksize=CHUNK_SIZE)):
        chunk_count += 1
        estimated_chunks = (8200000 // CHUNK_SIZE) + 1
        print(f"\rProcessing chunk {chunk_count} (~{(chunk_count)*CHUNK_SIZE//1000}k rows)...", end='', flush=True)
        
        # Extract basic features
        X_chunk = BasicFeatureExtractor.extract(chunk, TCP_COLUMNS)
        y_chunk = label_encoder.transform(chunk[TARGET_COL].astype(str))
        
        # Optimize memory usage
        X_chunk = reduce_mem_usage(X_chunk, verbose=False)
        
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
if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================
# STEP 4.2: Add Lifestream embeddings
# ============================================
print("\n[2/6] Creating Lifestream embeddings...")

embedding_index_file = f'{CHECKPOINT_DIR}/lifestream_embeddings_index.pkl'
if os.path.exists(embedding_index_file):
    print("✅ Found preprocessed embeddings index, loading...")
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, but trying to load embedder from checkpoint...")
        print("   This may fail if the embedder contains PyTorch models.")
    try:
        embeddings_data = joblib.load(embedding_index_file)
        embedding_files = embeddings_data['embedding_files']
        embedder = embeddings_data['embedder']
        if not TORCH_AVAILABLE:
            print("⚠️  WARNING: Embedder loaded but PyTorch is not available.")
            print("   Test embeddings extraction may fail.")
    except Exception as e:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                f"❌ Failed to load embedder from checkpoint: {e}\n"
                "   PyTorch is not available, which is required for the embedder.\n"
                "   Please fix the PyTorch duplicate registration error or reinstall PyTorch."
            ) from e
        raise
else:
    # Initialize embedder
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "❌ PyTorch is not available. Cannot create RNN embeddings.\n"
            "   This is required for the LifestreamEmbedder.\n"
            "   Please fix the PyTorch duplicate registration error or install PyTorch properly.\n"
            "   Try: pip uninstall torch torchvision torchaudio && pip install torch"
        )
    embedder = LifestreamEmbedder(embedding_dim=64, device='cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
    
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
            if TORCH_AVAILABLE and torch.cuda.is_available():
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
        if TORCH_AVAILABLE and torch.cuda.is_available():
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
if TORCH_AVAILABLE and torch.cuda.is_available():
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
    
    # CRITICAL: Check class distribution
    print("\n📊 Class distribution analysis:")
    unique_classes, class_counts = np.unique(all_y, return_counts=True)
    print(f"  Total classes: {len(unique_classes)}")
    print(f"  Most common class: {unique_classes[np.argmax(class_counts)]} ({class_counts.max():,} samples, {class_counts.max()/sample_count*100:.2f}%)")
    print(f"  Least common class: {unique_classes[np.argmin(class_counts)]} ({class_counts.min():,} samples, {class_counts.min()/sample_count*100:.2f}%)")
    
    # Calculate class weights for balanced learning
    class_weights = {}
    total_samples = len(all_y)
    n_classes = len(unique_classes)
    for cls, count in zip(unique_classes, class_counts):
        # Balanced weight: n_samples / (n_classes * count)
        class_weights[int(cls)] = total_samples / (n_classes * count)
    
    print(f"  Class weights calculated (min: {min(class_weights.values()):.4f}, max: {max(class_weights.values()):.4f})")
    
    # Check if data is severely imbalanced
    max_class_ratio = class_counts.max() / sample_count
    if max_class_ratio > 0.5:
        print(f"  ⚠️  WARNING: Most common class represents {max_class_ratio*100:.1f}% of data!")
        print(f"  This may cause model to predict only the majority class.")
    
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
    
    # Function to fit global scaler using partial_fit on all data
    def fit_global_scaler(feature_paths, embedding_files_list):
        """Fit StandardScaler using partial_fit on all data for consistent scaling"""
        print(f"  Fitting global scaler using partial_fit on all {len(feature_paths)} chunks...", flush=True)
        scaler = StandardScaler()
        rows = 0
        
        # First pass: accumulate statistics using partial_fit
        for i, feat_path in enumerate(feature_paths):
            if (i + 1) % 20 == 0:
                print(f"\r    Processing chunk {i+1}/{len(feature_paths)} for scaler fitting...", end='', flush=True)
            
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
            
            # Use partial_fit to accumulate statistics
            scaler.partial_fit(X)
            rows += X.shape[0]
            
            del X_chunk, emb_chunk, emb_df, X_combined, X
            gc.collect()
        
        print(f"\n  ✅ Global scaler fitted on {rows:,} rows using partial_fit", flush=True)
        return scaler
    
    # Fit global scaler on ALL chunks using partial_fit (critical for consistent scaling)
    global_scaler_path = f'{CHECKPOINT_DIR}/global_scaler.pkl'
    if os.path.exists(global_scaler_path):
        print(f"  Loading global scaler from checkpoint...", flush=True)
        global_scaler = joblib.load(global_scaler_path)
        print(f"  ✅ Global scaler loaded", flush=True)
    else:
        # Use partial_fit on ALL data for proper global scaling
        global_scaler = fit_global_scaler(feature_files, embedding_files)
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
    
    # Training function with LightGBM using incremental learning
    def train_lgb_chunkwise(train_mapped, val_mapped, num_classes, fold_num, class_weights_dict=None):
        """Train single LightGBM model using chunked incremental training"""
        model_path = f'{CHECKPOINT_DIR}/lgb_fold_{fold_num}.txt'
        
        # LightGBM parameters for multiclass classification
        # CRITICAL: Increased learning rate slightly and more trees per chunk for better learning
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
            # Note: class weights are applied via sample_weight in Dataset, not here
            # 'device': 'gpu'  # Uncomment if GPU LightGBM is available
        }
        
        # Check if model already exists (resume logic)
        bst = None
        if os.path.exists(model_path):
            try:
                bst = lgb.Booster(model_file=model_path)
                num_trees = bst.num_trees()
                print(f"  ✅ Model checkpoint found with {num_trees} trees, loading...", flush=True)
                if num_trees == 0:
                    print(f"  ⚠️  Warning: Model has 0 trees, will retrain...", flush=True)
                    bst = None
                    model_loaded = False
                else:
                    model_loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}, retraining...", flush=True)
                model_loaded = False
                bst = None
        else:
            model_loaded = False
        
        # Use global scaler (fitted earlier)
        scaler = global_scaler
        
        if not model_loaded:
            print(f"  Training LightGBM on {len(train_mapped)} chunks...", flush=True)
            print(f"  Using global StandardScaler (fitted with partial_fit on all data)", flush=True)
            print(f"  Parameters: learning_rate={lgb_params['learning_rate']}, num_leaves={lgb_params['num_leaves']}", flush=True)
            if class_weights_dict:
                print(f"  Using class weights for imbalanced data", flush=True)
            
            start_time = time.time()
            # CRITICAL: Increased rounds per chunk for better learning on imbalanced data
            rounds_per_chunk = 100  # Increased from 50 to 100
            total_rounds = 0
            
            # CRITICAL: Prepare validation set ONCE for early stopping (use larger sample)
            val_data = None
            val_X_scaled = None
            val_y = None
            val_size = 0
            if len(val_mapped) > 0:
                # Load larger validation sample (up to 10 chunks or 250k samples)
                val_chunks_to_load = min(10, len(val_mapped))
                X_val_sample, y_val_sample = load_chunk_data(val_mapped[:val_chunks_to_load])
                if len(X_val_sample) > 0:
                    val_X_scaled = scaler.transform(X_val_sample)
                    val_y = y_val_sample.copy()
                    
                    # CRITICAL: Calculate sample weights for validation set
                    val_sample_weights = None
                    if class_weights_dict:
                        val_sample_weights = np.array([class_weights_dict.get(int(cls), 1.0) for cls in y_val_sample])
                    
                    val_data = lgb.Dataset(val_X_scaled, label=val_y, weight=val_sample_weights, free_raw_data=False)
                    val_size = len(X_val_sample)
                    print(f"  Validation set prepared: {val_size:,} samples", flush=True)
                    del X_val_sample, y_val_sample, val_sample_weights
                    gc.collect()
            
            # CRITICAL: Track best validation score for early stopping across all chunks
            best_val_score = float('inf')
            rounds_without_improvement = 0
            max_rounds_without_improvement = 50  # Stop if no improvement for 50 consecutive chunks
            
            # Train using incremental learning on each chunk
            for chunk_idx, chunk_data in enumerate(train_mapped):
                if (chunk_idx + 1) % 10 == 0:
                    print(f"\r    Processing chunk {chunk_idx+1}/{len(train_mapped)} (trees: {bst.num_trees() if bst is not None else 0})...", end='', flush=True)
                
                # Load chunk data (embeddings are already normalized in load_chunk_data)
                X_chunk, y_chunk = load_chunk_data([chunk_data])
                
                if len(X_chunk) == 0:
                    continue
                
                # Transform with global scaler
                X_chunk_scaled = scaler.transform(X_chunk)
                
                # CRITICAL: Calculate sample weights for this chunk
                sample_weights = None
                if class_weights_dict:
                    sample_weights = np.array([class_weights_dict.get(int(cls), 1.0) for cls in y_chunk])
                
                # Create LightGBM Dataset with sample weights
                dtrain = lgb.Dataset(X_chunk_scaled, label=y_chunk, weight=sample_weights, free_raw_data=True)
                
                # CRITICAL: Use early stopping only if we have validation data
                # But don't use it on every chunk - use it periodically or on first few chunks
                use_early_stopping = (val_data is not None and chunk_idx < 5)  # Only first 5 chunks
                callbacks = []
                if use_early_stopping:
                    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
                
                # Train (continue from previous model if exists)
                # CRITICAL: keep_training_booster=True allows incremental learning
                bst = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=rounds_per_chunk,
                    init_model=bst,
                    keep_training_booster=True,  # CRITICAL: Must be True for incremental learning
                    valid_sets=[val_data] if use_early_stopping else None,
                    callbacks=callbacks if callbacks else None
                )
                
                total_rounds += rounds_per_chunk
                
                # Check validation score if available
                if val_X_scaled is not None and val_y is not None and chunk_idx % 5 == 0:  # Check every 5 chunks
                    best_iter = getattr(bst, 'best_iteration', None)
                    if best_iter is None or best_iter == 0:
                        best_iter = None
                    val_pred = bst.predict(val_X_scaled, num_iteration=best_iter)
                    # CRITICAL: Pass all class labels to log_loss since validation set may not contain all classes
                    all_class_labels = np.arange(num_classes)
                    val_loss = log_loss(val_y, val_pred, labels=all_class_labels)
                    if val_loss < best_val_score:
                        best_val_score = val_loss
                        rounds_without_improvement = 0
                        if chunk_idx % 10 == 0:
                            print(f"\n    Validation loss improved to {val_loss:.6f} (trees: {bst.num_trees()})", flush=True)
                    else:
                        rounds_without_improvement += 1
                    
                    if rounds_without_improvement >= max_rounds_without_improvement:
                        print(f"\n  ⚠️  Early stopping: no improvement for {max_rounds_without_improvement} checks", flush=True)
                        break
                
                # Memory cleanup
                del X_chunk, X_chunk_scaled, y_chunk, dtrain, sample_weights
                gc.collect()
            
            elapsed_time = time.time() - start_time
            final_trees = bst.num_trees() if bst is not None else 0
            print(f"\n  ✅ Training completed in {elapsed_time:.1f}s")
            print(f"  Final model: {final_trees} trees (total rounds: {total_rounds})", flush=True)
            
            # Verify model is trained
            if final_trees == 0:
                raise ValueError(f"❌ ERROR: Model has 0 trees after training! Training failed.")
            
            # CRITICAL: Check model predictions on validation set to ensure it's not predicting only one class
            if val_X_scaled is not None and val_y is not None:
                print(f"  Checking model predictions on validation set...", flush=True)
                val_pred_proba = bst.predict(val_X_scaled, num_iteration=None)
                val_pred_classes = np.argmax(val_pred_proba, axis=1)
                unique_pred_classes, pred_counts = np.unique(val_pred_classes, return_counts=True)
                print(f"  Validation predictions: {len(unique_pred_classes)} unique classes predicted", flush=True)
                print(f"  Most common predicted class: {unique_pred_classes[np.argmax(pred_counts)]} ({pred_counts.max()}/{len(val_pred_classes)} samples, {pred_counts.max()/len(val_pred_classes)*100:.2f}%)", flush=True)
                
                if len(unique_pred_classes) == 1:
                    print(f"  ❌ CRITICAL WARNING: Model predicts only ONE class on validation set!", flush=True)
                    print(f"  This indicates the model did not learn properly. Check class distribution and training parameters.", flush=True)
                elif pred_counts.max() / len(val_pred_classes) > 0.9:
                    print(f"  ⚠️  WARNING: {pred_counts.max()/len(val_pred_classes)*100:.1f}% of predictions are the same class!", flush=True)
                else:
                    print(f"  ✅ Model predicts multiple classes (good diversity)", flush=True)
            
            # Save final model (only critical checkpoint)
            try:
                bst.save_model(model_path)
                print(f"  ✅ Model saved to {model_path}", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Failed to save model checkpoint: {e}", flush=True)
        else:
            # Verify loaded model
            num_trees = bst.num_trees()
            print(f"  ✅ Loaded model has {num_trees} trees", flush=True)
            if num_trees == 0:
                raise ValueError(f"❌ ERROR: Loaded model has 0 trees! Model is not trained.")
        
        # Return model and scaler in dict format for compatibility
        return {'model': bst, 'scaler': scaler}, scaler
    
    models = []
    
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
        # CRITICAL: Pass class weights for balanced learning
        fold_model_data, fold_scaler = train_lgb_chunkwise(
            train_mapped, 
            val_mapped, 
            num_classes,
            fold + 1,  # Передаем номер fold для resume логики
            class_weights_dict=class_weights  # Pass class weights
        )
        
        # Store model and scaler (one per fold) - fold_model_data is already a dict
        models.append(fold_model_data)
        
        print(f"   ✅ Saved LightGBM model for fold {fold + 1}")
        
        # Memory cleanup after each fold
        del fold_model_data
        gc.collect()
        print("✅ Memory cleaned after fold")
    
    # Save models (КРИТИЧНО для инференса на test данных!)
    models_file = f'{CHECKPOINT_DIR}/trained_models.pkl'
    print(f"\n💾 Сохраняем обученные модели в {models_file}...")
    joblib.dump({'models': models, 'label_encoder': label_encoder}, models_file)
    print(f"✅ Модели успешно сохранены! Файл {models_file} готов для инференса на test данных.")
    
    # Memory cleanup
    del all_y
    gc.collect()

# Aggressive memory cleanup before test processing
print("\n🧹 Cleaning up memory before test processing...")
print_memory_usage()
gc.collect()
if TORCH_AVAILABLE and torch.cuda.is_available():
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
        print("  ✅ MTU preprocessing enabled: merging consecutive packets (>1200 bytes, same direction)")
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
            
            # Optimize memory usage
            X_chunk = reduce_mem_usage(X_chunk, verbose=False)
            
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
        if TORCH_AVAILABLE and torch.cuda.is_available():
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
if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.cuda.empty_cache()

# Delete embedder to free memory
print("\n🧹 Freeing embedder memory...")
print_memory_usage()
try:
    del embedder
except:
    pass
gc.collect()
if TORCH_AVAILABLE and torch.cuda.is_available():
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

# Predict chunk-by-chunk and save to HDF5 (avoiding memory issues)
print("\nPredicting chunk-by-chunk and saving to HDF5...")

# Calculate total number of test samples
total_test_samples = 0
for feat_path in test_feature_files:
    X_test = joblib.load(feat_path)
    total_test_samples += len(X_test)
    del X_test
    gc.collect()

print(f"Total test samples: {total_test_samples:,}")

# Use HDF5 for storing predictions (solves OOM problem)
PRED_H5_FILE = f'{CHECKPOINT_DIR}/test_predictions.h5'
PRED_CHECKPOINT_FILE = f'{CHECKPOINT_DIR}/predictions_checkpoint.pkl'

# Check for existing predictions checkpoint
start_chunk = 0
if os.path.exists(PRED_CHECKPOINT_FILE):
    print("\n✅ Found predictions checkpoint, loading...")
    checkpoint_data = joblib.load(PRED_CHECKPOINT_FILE)
    start_chunk = checkpoint_data['last_chunk'] + 1
    print(f"✅ Resuming from chunk {start_chunk}/{len(test_feature_files)}")
    
    # Check if HDF5 file exists and has correct shape
    if os.path.exists(PRED_H5_FILE):
        try:
            with h5py.File(PRED_H5_FILE, 'r') as f:
                existing_samples = f['predictions'].shape[0]
                print(f"   HDF5 file exists with {existing_samples:,} samples")
        except:
            print(f"   HDF5 file exists but may be corrupted, will recreate")
            start_chunk = 0
else:
    # Create new HDF5 file for predictions
    if os.path.exists(PRED_H5_FILE):
        os.remove(PRED_H5_FILE)
        print(f"   Removed old HDF5 file")

# Open/create HDF5 file for predictions
if not H5PY_AVAILABLE:
    raise ImportError("h5py is required for storing predictions. Install with: pip install h5py")

try:
    # Open HDF5 file in append mode
    with h5py.File(PRED_H5_FILE, 'a') as h5f:
        # Create dataset if it doesn't exist
        if 'predictions' not in h5f:
            h5f.create_dataset('predictions', 
                              shape=(total_test_samples, num_classes),
                              maxshape=(None, num_classes),
                              chunks=(min(25000, total_test_samples), num_classes),
                              dtype='float32',
                              compression='gzip')
            print(f"✅ Created HDF5 dataset: shape ({total_test_samples:,}, {num_classes})")
        
        dset = h5f['predictions']
        current_offset = dset.shape[0] if dset.shape[0] < total_test_samples else 0
        
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
            
            # Convert to numpy array
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
                    # CRITICAL: Don't use best_iteration if it's None or 0 - use all trees
                    best_iter = getattr(model, 'best_iteration', None)
                    if best_iter is None or best_iter == 0:
                        best_iter = None  # Use all trees
                    preds = model.predict(X_chunk_scaled, num_iteration=best_iter)
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
            
            # Write predictions directly to HDF5 (no memory accumulation)
            end_offset = current_offset + len(chunk_preds)
            if end_offset > dset.shape[0]:
                dset.resize((end_offset, num_classes))
            dset[current_offset:end_offset] = chunk_preds.astype('float32')
            current_offset = end_offset
            
            print(" ✓", flush=True)
            
            # Save checkpoint every 10 chunks
            if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == len(test_feature_files):
                joblib.dump({
                    'last_chunk': chunk_idx,
                    'current_offset': current_offset
                }, PRED_CHECKPOINT_FILE)
                print(f"   💾 Checkpoint saved (chunk {chunk_idx + 1}, offset: {current_offset:,})")
            
            # Memory cleanup
            del X_chunk, emb_chunk, emb_df, X_chunk_array, chunk_preds, chunk_preds_list, chunk_preds_array
            gc.collect()
        
except Exception as e:
    print(f"\n❌ ERROR during predictions: {e}")
    import traceback
    traceback.print_exc()
    raise

# Load predictions from HDF5 (memory-efficient)
print("\nLoading predictions from HDF5...")
if not os.path.exists(PRED_H5_FILE):
    raise FileNotFoundError(f"❌ ERROR: HDF5 predictions file not found: {PRED_H5_FILE}")

with h5py.File(PRED_H5_FILE, 'r') as h5f:
    dset = h5f['predictions']
    print(f"  HDF5 predictions shape: {dset.shape}")
    
    # Load predictions in chunks to convert to classes (avoid loading all at once)
    print("  Converting predictions to classes chunk-by-chunk...")
    predicted_classes = []
    chunk_size = 50000
    
    for i in range(0, dset.shape[0], chunk_size):
        end_idx = min(i + chunk_size, dset.shape[0])
        pred_chunk = dset[i:end_idx]
        classes_chunk = np.argmax(pred_chunk, axis=1)
        predicted_classes.append(classes_chunk)
        
        if (i // chunk_size + 1) % 10 == 0:
            print(f"\r    Processed {end_idx:,}/{dset.shape[0]:,} samples...", end='', flush=True)
    
    predicted_classes = np.concatenate(predicted_classes)
    print(f"\n  ✅ Converted {len(predicted_classes):,} predictions to classes")

# Clean up predictions checkpoint
if os.path.exists(PRED_CHECKPOINT_FILE):
    os.remove(PRED_CHECKPOINT_FILE)
    print("✅ Predictions checkpoint cleaned")
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
    print(f"  Checking prediction probabilities from HDF5...")
    # Load sample from HDF5 for diagnostics
    with h5py.File(PRED_H5_FILE, 'r') as h5f:
        dset = h5f['predictions']
        sample_size = min(10000, dset.shape[0])
        pred_sample = dset[:sample_size]
        print(f"  Mean probabilities per class (sample): {pred_sample.mean(axis=0)}")
        print(f"  Std probabilities per class (sample): {pred_sample.std(axis=0)}")
        print(f"  Max probability range (sample): [{pred_sample.min():.6f}, {pred_sample.max():.6f}]")
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
del predicted_classes
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
if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n✅ All done! Download submission.csv")

# ============================================
# FINAL STEP: CLOSE LOGGER
# ============================================
print("\n" + "="*70)
print("Closing logger...")
print("="*70)

# Restore stdout before closing logger
sys.stdout = logger.terminal

# Close logger and get log file path
log_file_path = logger.get_log_path()
logger.close()

print(f"\n📝 Log file location: {os.path.abspath(log_file_path)}")
print("="*70)

