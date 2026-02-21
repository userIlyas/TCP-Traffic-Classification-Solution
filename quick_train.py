"""
Quick Training Script
Simplified script for faster experimentation and submission generation
Uses a subset of features for speed while maintaining good performance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
import gc
import warnings
warnings.filterwarnings('ignore')


def extract_fast_features(df, tcp_columns):
    """
    Extract fast, effective features for quick training.
    
    Args:
        df: DataFrame with TCP columns
        tcp_columns: List of TCP column names
        
    Returns:
        DataFrame with features
    """
    print("Extracting features...")
    
    features = df[tcp_columns].copy()
    tcp_data = features.values
    
    # Replace NaN and inf
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
    
    # Large packets (MTU-sized)
    features['large_packets'] = np.sum(np.abs(tcp_data) > 1200, axis=1)
    features['large_ratio'] = features['large_packets'] / (features['total_packets'] + 1)
    
    # First packet features
    features['first_packet'] = tcp_data[:, 0]
    features['first_abs'] = np.abs(tcp_data[:, 0])
    features['first_direction'] = np.sign(tcp_data[:, 0])
    
    # Packet size variance in windows
    for window in [5, 10, 15]:
        window_data = tcp_data[:, :window]
        features[f'var_{window}'] = np.var(np.abs(window_data), axis=1)
        features[f'mean_{window}'] = np.mean(np.abs(window_data), axis=1)
    
    # Direction changes
    directions = np.sign(tcp_data)
    direction_changes = np.sum(directions[:, 1:] != directions[:, :-1], axis=1)
    features['dir_changes'] = direction_changes
    features['dir_change_rate'] = direction_changes / (features['total_packets'] + 1)
    
    # Burst detection (consecutive large packets)
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
    
    print(f"Extracted {features.shape[1]} features")
    
    return features


def train_lgbm_cv(X, y, n_folds=5):
    """
    Train LightGBM with cross-validation.
    
    Args:
        X: Features DataFrame
        y: Target Series
        n_folds: Number of CV folds
        
    Returns:
        List of models and label encoder
    """
    print(f"\nTraining with {n_folds}-fold CV...")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    num_classes = len(le.classes_)
    
    print(f"Number of classes: {num_classes}")
    
    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Cross-validation
    models = []
    oof_preds = np.zeros((len(X), num_classes))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=50)
            ]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        
        val_loss = log_loss(y_val, oof_preds[val_idx])
        val_acc = accuracy_score(y_val, np.argmax(oof_preds[val_idx], axis=1))
        
        print(f"Fold {fold + 1} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        models.append(model)
        
        gc.collect()
    
    # Overall metrics
    overall_loss = log_loss(y_encoded, oof_preds)
    overall_acc = accuracy_score(y_encoded, np.argmax(oof_preds, axis=1))
    
    print(f"\nOverall CV - Loss: {overall_loss:.4f}, Accuracy: {overall_acc:.4f}")
    
    return models, le


def predict_ensemble(models, X, le):
    """
    Make ensemble predictions.
    
    Args:
        models: List of trained models
        X: Features DataFrame
        le: Label encoder
        
    Returns:
        Array of predicted labels
    """
    print("\nMaking predictions...")
    
    # Average predictions from all models
    predictions = np.zeros((len(X), len(le.classes_)))
    
    for model in models:
        predictions += model.predict(X)
    
    predictions /= len(models)
    
    # Convert to labels
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = le.inverse_transform(predicted_classes)
    
    return predicted_labels


def main():
    """Main training and prediction pipeline."""
    
    print("="*70)
    print("QUICK TRAINING PIPELINE FOR TCP TRAFFIC CLASSIFICATION")
    print("="*70)
    
    # Configuration
    TCP_COLUMNS = [f'tcp_len_{i}' for i in range(1, 31)]
    TARGET_COL = 'app_service'
    SAMPLE_SIZE = None  # Set to a number (e.g., 100000) for faster training
    
    # Load training data
    print("\n[1/5] Loading training data...")
    if SAMPLE_SIZE:
        print(f"Using sample of {SAMPLE_SIZE} rows")
        train_df = pd.read_csv('train.csv', nrows=SAMPLE_SIZE)
    else:
        print("Loading full dataset (this may take a while)...")
        train_df = pd.read_csv('train.csv')
    
    print(f"Loaded {len(train_df)} training samples")
    
    # Extract features
    print("\n[2/5] Feature extraction...")
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    
    X_features = extract_fast_features(X, TCP_COLUMNS)
    
    # Free memory
    del train_df, X
    gc.collect()
    
    # Train model
    print("\n[3/5] Training LightGBM...")
    models, le = train_lgbm_cv(X_features, y, n_folds=5)
    
    # Free memory
    del X_features, y
    gc.collect()
    
    # Load test data
    print("\n[4/5] Loading test data...")
    test_df = pd.read_csv('test.csv')
    print(f"Loaded {len(test_df)} test samples")
    
    # Extract test features
    print("\nExtracting test features...")
    test_ids = test_df.index
    X_test = extract_fast_features(test_df, TCP_COLUMNS)
    
    # Free memory
    del test_df
    gc.collect()
    
    # Predict
    print("\n[5/5] Generating predictions...")
    predictions = predict_ensemble(models, X_test, le)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        TARGET_COL: predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "="*70)
    print("COMPLETED SUCCESSFULLY!")
    print(f"Submission saved to: submission.csv")
    print(f"Number of predictions: {len(submission)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

