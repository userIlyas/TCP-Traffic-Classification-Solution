"""
Chunked Training Pipeline with LightGBM
Handles large datasets by processing in chunks and incremental training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from typing import List, Optional, Tuple, Dict
import gc
import joblib
from tqdm import tqdm
import os


class ChunkedDataProcessor:
    """
    Processes large CSV files in chunks to avoid memory issues.
    """
    
    def __init__(self, chunk_size: int = 50000):
        """
        Args:
            chunk_size: Number of rows to process at once
        """
        self.chunk_size = chunk_size
    
    def read_in_chunks(self, filepath: str, usecols: Optional[List[str]] = None):
        """
        Generator that yields chunks of data from CSV file.
        
        Args:
            filepath: Path to CSV file
            usecols: Optional list of columns to read
            
        Yields:
            DataFrame chunks
        """
        print(f"Reading {filepath} in chunks of {self.chunk_size}...")
        
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, usecols=usecols):
            yield chunk
    
    def process_chunks_with_function(self, 
                                    filepath: str,
                                    process_func,
                                    usecols: Optional[List[str]] = None,
                                    save_path: Optional[str] = None):
        """
        Process CSV file in chunks and optionally save results.
        
        Args:
            filepath: Path to input CSV
            process_func: Function to apply to each chunk
            usecols: Optional columns to read
            save_path: Optional path to save processed data
            
        Returns:
            List of processed chunks or None if saved to disk
        """
        processed_chunks = []
        
        for i, chunk in enumerate(self.read_in_chunks(filepath, usecols)):
            print(f"Processing chunk {i+1}...")
            
            # Apply processing function
            processed = process_func(chunk)
            
            if save_path:
                # Save to disk incrementally
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                processed.to_csv(save_path, mode=mode, header=header, index=False)
            else:
                processed_chunks.append(processed)
            
            # Free memory
            del chunk, processed
            gc.collect()
        
        if save_path:
            print(f"Processed data saved to {save_path}")
            return None
        else:
            return pd.concat(processed_chunks, ignore_index=True)


class IncrementalLGBMTrainer:
    """
    Trains LightGBM incrementally on chunks of data.
    Uses LightGBM's native incremental training capability.
    """
    
    def __init__(self,
                 params: Optional[Dict] = None,
                 num_boost_round: int = 1000,
                 early_stopping_rounds: int = 50):
        """
        Args:
            params: LightGBM parameters
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping rounds
        """
        self.params = params or self._get_default_params()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def _get_default_params(self) -> Dict:
        """Get default LightGBM parameters optimized for large datasets."""
        return {
            'objective': 'multiclass',
            'boosting_type': 'gbdt',
            'metric': 'multi_logloss',
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
    
    def train_on_chunks(self,
                       train_file: str,
                       target_col: str,
                       feature_cols: List[str],
                       chunk_size: int = 50000,
                       validation_split: float = 0.2):
        """
        Train LightGBM model on data chunks.
        
        Args:
            train_file: Path to training CSV
            target_col: Name of target column
            feature_cols: List of feature column names
            chunk_size: Size of chunks
            validation_split: Fraction of data for validation
        """
        print("Starting chunked training...")
        
        # First pass: fit label encoder
        print("Fitting label encoder...")
        all_labels = []
        for chunk in pd.read_csv(train_file, chunksize=chunk_size, usecols=[target_col]):
            all_labels.extend(chunk[target_col].astype(str).unique())
        
        unique_labels = list(set(all_labels))
        self.label_encoder.fit(unique_labels)
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Found {num_classes} classes")
        
        # Update params with num_class
        self.params['num_class'] = num_classes
        
        # Second pass: train model
        init_model = None
        best_score = float('inf')
        rounds_without_improvement = 0
        
        for i, chunk in enumerate(pd.read_csv(train_file, chunksize=chunk_size)):
            print(f"\nTraining on chunk {i+1}...")
            
            # Prepare data
            X_chunk = chunk[feature_cols].copy()
            y_chunk = self.label_encoder.transform(chunk[target_col].astype(str))
            
            # Handle missing values
            X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)
            X_chunk = X_chunk.fillna(X_chunk.median())
            
            # Split into train and validation
            split_idx = int(len(X_chunk) * (1 - validation_split))
            X_train = X_chunk.iloc[:split_idx]
            y_train = y_chunk[:split_idx]
            X_val = X_chunk.iloc[split_idx:]
            y_val = y_chunk[split_idx:]
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train
            callbacks = [
                lgb.log_evaluation(period=50),
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)
            ]
            
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_data],
                init_model=init_model,
                callbacks=callbacks
            )
            
            # Update init_model for next chunk
            init_model = self.model
            
            # Check validation score
            y_pred_proba = self.model.predict(X_val)
            val_loss = log_loss(y_val, y_pred_proba)
            
            print(f"Chunk {i+1} validation loss: {val_loss:.4f}")
            
            if val_loss < best_score:
                best_score = val_loss
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            
            # Free memory
            del X_chunk, y_chunk, X_train, y_train, X_val, y_val
            del train_data, val_data
            gc.collect()
            
            # Early stopping across chunks
            if rounds_without_improvement >= 3:
                print("Early stopping: no improvement for 3 chunks")
                break
        
        print(f"\nTraining completed. Best validation loss: {best_score:.4f}")
    
    def train_with_cv(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     n_folds: int = 5) -> Tuple[List, np.ndarray]:
        """
        Train with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_folds: Number of CV folds
            
        Returns:
            List of models and OOF predictions
        """
        print(f"Training with {n_folds}-fold cross-validation...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        num_classes = len(self.label_encoder.classes_)
        
        # Update params
        self.params['num_class'] = num_classes
        
        # Prepare for OOF predictions
        oof_predictions = np.zeros((len(X), num_classes))
        models = []
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"{'='*50}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train
            callbacks = [
                lgb.log_evaluation(period=50),
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)
            ]
            
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_data],
                callbacks=callbacks
            )
            
            # Predict on validation
            oof_predictions[val_idx] = model.predict(X_val)
            
            # Calculate metrics
            val_loss = log_loss(y_val, oof_predictions[val_idx])
            val_acc = accuracy_score(y_val, np.argmax(oof_predictions[val_idx], axis=1))
            
            print(f"Fold {fold + 1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            
            models.append(model)
            
            # Free memory
            del train_data, val_data
            gc.collect()
        
        # Overall OOF metrics
        overall_loss = log_loss(y_encoded, oof_predictions)
        overall_acc = accuracy_score(y_encoded, np.argmax(oof_predictions, axis=1))
        
        print(f"\n{'='*50}")
        print(f"Overall OOF - Loss: {overall_loss:.4f}, Accuracy: {overall_acc:.4f}")
        print(f"{'='*50}")
        
        self.model = models  # Store all models for ensemble
        
        return models, oof_predictions
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            use_ensemble: If True and multiple models exist, average predictions
            
        Returns:
            Array of predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        if isinstance(self.model, list) and use_ensemble:
            # Ensemble prediction
            predictions = np.zeros((len(X), self.params['num_class']))
            
            for model in self.model:
                predictions += model.predict(X)
            
            predictions /= len(self.model)
        else:
            # Single model prediction
            model = self.model if not isinstance(self.model, list) else self.model[0]
            predictions = model.predict(X)
        
        # Convert to class labels
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        return predicted_labels
    
    def save_model(self, path: str):
        """Save the trained model(s) and label encoder."""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'params': self.params
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.params = data['params']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test chunked processing
    print("Testing Chunked Data Processor...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'target': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    sample_data.to_csv('test_data.csv', index=False)
    
    # Test chunked reading
    processor = ChunkedDataProcessor(chunk_size=200)
    
    def process_func(chunk):
        chunk['feature_3'] = chunk['feature_1'] + chunk['feature_2']
        return chunk
    
    processed = processor.process_chunks_with_function(
        'test_data.csv',
        process_func
    )
    
    print(f"\nProcessed data shape: {processed.shape}")
    print(f"Columns: {processed.columns.tolist()}")
    
    # Test incremental training
    print("\nTesting Incremental LGBM Trainer...")
    
    trainer = IncrementalLGBMTrainer()
    
    X = processed[['feature_1', 'feature_2', 'feature_3']]
    y = processed['target']
    
    models, oof_preds = trainer.train_with_cv(X, y, n_folds=3)
    
    print(f"\nTrained {len(models)} models")
    
    # Test prediction
    predictions = trainer.predict(X.head(10))
    print(f"Sample predictions: {predictions}")
    
    # Cleanup
    os.remove('test_data.csv')
    
    print("\nChunked training test completed!")

