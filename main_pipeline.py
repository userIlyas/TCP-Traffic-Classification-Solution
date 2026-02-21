"""
Main Training Pipeline
Integrates all components: MTU preprocessing, TSFEL features, Lifestream embeddings, and LightGBM
"""

import pandas as pd
import numpy as np
import os
import gc
import argparse
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocessing import (
    MTUPreprocessor, 
    TargetEncodingFeatures, 
    extract_basic_features
)
from tsfel_features import TSFELFeatureExtractor, CustomTCPFeatures
from lifestream_embeddings import create_lifestream_features
from chunked_training import ChunkedDataProcessor, IncrementalLGBMTrainer


class TCPTrafficClassifier:
    """
    Complete pipeline for TCP traffic classification.
    """
    
    def __init__(self,
                 use_mtu_preprocessing: bool = True,
                 use_target_encoding: bool = True,
                 use_tsfel: bool = True,
                 use_lifestream: bool = True,
                 tsfel_domain: str = "statistical",
                 lifestream_encoder: str = "rnn",
                 lifestream_dim: int = 128,
                 chunk_size: int = 50000):
        """
        Args:
            use_mtu_preprocessing: Whether to apply MTU reconstruction
            use_target_encoding: Whether to use target encoding
            use_tsfel: Whether to extract TSFEL features
            use_lifestream: Whether to use Lifestream embeddings
            tsfel_domain: TSFEL domain ("statistical", "temporal", "spectral", "all")
            lifestream_encoder: Lifestream encoder type ("rnn" or "transformer")
            lifestream_dim: Dimension of Lifestream embeddings
            chunk_size: Chunk size for processing large files
        """
        self.use_mtu_preprocessing = use_mtu_preprocessing
        self.use_target_encoding = use_target_encoding
        self.use_tsfel = use_tsfel
        self.use_lifestream = use_lifestream
        self.tsfel_domain = tsfel_domain
        self.lifestream_encoder = lifestream_encoder
        self.lifestream_dim = lifestream_dim
        self.chunk_size = chunk_size
        
        # Initialize components
        self.mtu_processor = MTUPreprocessor() if use_mtu_preprocessing else None
        self.target_encoder = TargetEncodingFeatures() if use_target_encoding else None
        self.tsfel_extractor = TSFELFeatureExtractor(domain=tsfel_domain) if use_tsfel else None
        self.custom_tcp_extractor = CustomTCPFeatures()
        self.lifestream_embedder = None
        self.lgbm_trainer = IncrementalLGBMTrainer()
        
        # TCP column names
        self.tcp_columns = [f'tcp_len_{i}' for i in range(1, 31)]
        self.target_col = 'app_service'
    
    def extract_features(self, 
                        df: pd.DataFrame, 
                        target: Optional[pd.Series] = None,
                        is_train: bool = True) -> pd.DataFrame:
        """
        Extract all features from raw TCP data.
        
        Args:
            df: DataFrame with TCP packet columns
            target: Optional target column (for training)
            is_train: Whether this is training data
            
        Returns:
            DataFrame with all extracted features
        """
        print("\n" + "="*70)
        print("FEATURE EXTRACTION PIPELINE")
        print("="*70)
        
        # Start with original data
        features_df = df[self.tcp_columns].copy()
        
        # 1. Basic statistical features
        print("\n[1/5] Extracting basic statistical features...")
        basic_features = extract_basic_features(df, self.tcp_columns)
        basic_feature_cols = [col for col in basic_features.columns if col not in self.tcp_columns]
        features_df = pd.concat([features_df, basic_features[basic_feature_cols]], axis=1)
        print(f"   Added {len(basic_feature_cols)} basic features")
        
        # 2. Target encoding features (if enabled and target provided)
        if self.use_target_encoding and target is not None:
            print("\n[2/5] Creating target encoding features...")
            
            # Create categorical features
            categorical_df = self.target_encoder.create_categorical_features(df, self.tcp_columns)
            categorical_cols = ['direction_pattern', 'first_packet_bucket', 'flow_type']
            
            if is_train:
                # Fit and transform
                encoded_df = self.target_encoder.fit_transform(
                    categorical_df, 
                    target, 
                    categorical_cols
                )
            else:
                # Transform only
                encoded_df = self.target_encoder.transform(categorical_df, categorical_cols)
            
            # Add encoded features
            encoded_cols = [col for col in encoded_df.columns if col.endswith('_target_enc')]
            features_df = pd.concat([features_df, encoded_df[encoded_cols]], axis=1)
            print(f"   Added {len(encoded_cols)} target-encoded features")
        else:
            print("\n[2/5] Skipping target encoding (not enabled or no target)")
        
        # 3. Custom TCP features
        print("\n[3/5] Extracting custom TCP features...")
        custom_features = self.custom_tcp_extractor.extract_all_custom_features(df, self.tcp_columns)
        features_df = pd.concat([features_df, custom_features], axis=1)
        print(f"   Added {custom_features.shape[1]} custom TCP features")
        
        # 4. TSFEL features (if enabled)
        if self.use_tsfel:
            print(f"\n[4/5] Extracting TSFEL features ({self.tsfel_domain} domain)...")
            try:
                tsfel_features = self.tsfel_extractor.extract_features_batch(
                    df, 
                    self.tcp_columns,
                    batch_size=1000
                )
                features_df = pd.concat([features_df, tsfel_features], axis=1)
                print(f"   Added {tsfel_features.shape[1]} TSFEL features")
            except Exception as e:
                print(f"   Warning: TSFEL extraction failed: {e}")
                print("   Continuing without TSFEL features...")
        else:
            print("\n[4/5] Skipping TSFEL features (not enabled)")
        
        # 5. Lifestream embeddings (if enabled)
        if self.use_lifestream:
            print(f"\n[5/5] Creating Lifestream embeddings ({self.lifestream_encoder})...")
            try:
                if is_train and target is not None:
                    # Train embedder
                    lifestream_features, self.lifestream_embedder = create_lifestream_features(
                        df,
                        self.tcp_columns,
                        labels=target,
                        encoder_type=self.lifestream_encoder,
                        embedding_dim=self.lifestream_dim,
                        train_epochs=5
                    )
                elif self.lifestream_embedder is not None:
                    # Use pre-trained embedder
                    sequences = df[self.tcp_columns].values.astype(np.float32)
                    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
                    embeddings = self.lifestream_embedder.extract_embeddings(sequences)
                    
                    embedding_cols = [f'lifestream_{i}' for i in range(self.lifestream_dim)]
                    lifestream_features = pd.DataFrame(
                        embeddings, 
                        columns=embedding_cols, 
                        index=df.index
                    )
                else:
                    print("   Warning: No trained embedder available, skipping...")
                    lifestream_features = None
                
                if lifestream_features is not None:
                    features_df = pd.concat([features_df, lifestream_features], axis=1)
                    print(f"   Added {lifestream_features.shape[1]} Lifestream embedding features")
            except Exception as e:
                print(f"   Warning: Lifestream embedding failed: {e}")
                print("   Continuing without Lifestream features...")
        else:
            print("\n[5/5] Skipping Lifestream embeddings (not enabled)")
        
        # Clean up
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        print("\n" + "="*70)
        print(f"FEATURE EXTRACTION COMPLETE: {features_df.shape[1]} total features")
        print("="*70 + "\n")
        
        return features_df
    
    def train(self, 
             train_file: str,
             use_cv: bool = True,
             n_folds: int = 5,
             sample_size: Optional[int] = None):
        """
        Train the complete pipeline.
        
        Args:
            train_file: Path to training CSV file
            use_cv: Whether to use cross-validation
            n_folds: Number of CV folds
            sample_size: Optional sample size for faster training (None = use all data)
        """
        print("\n" + "="*70)
        print("TRAINING PIPELINE")
        print("="*70)
        
        # Load data
        print(f"\nLoading training data from {train_file}...")
        
        if sample_size:
            print(f"Using sample of {sample_size} rows for faster training...")
            train_df = pd.read_csv(train_file, nrows=sample_size)
        else:
            # For large files, load in chunks
            if os.path.getsize(train_file) > 500 * 1024 * 1024:  # > 500MB
                print("Large file detected, using chunked loading...")
                processor = ChunkedDataProcessor(chunk_size=self.chunk_size)
                chunks = []
                for chunk in processor.read_in_chunks(train_file):
                    chunks.append(chunk)
                    if sample_size and len(pd.concat(chunks)) >= sample_size:
                        break
                train_df = pd.concat(chunks, ignore_index=True)
                if sample_size:
                    train_df = train_df.head(sample_size)
            else:
                train_df = pd.read_csv(train_file)
        
        print(f"Loaded {len(train_df)} samples")
        
        # Separate features and target
        X = train_df.drop(columns=[self.target_col])
        y = train_df[self.target_col]
        
        # Extract features
        X_features = self.extract_features(X, target=y, is_train=True)
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM MODEL")
        print("="*70 + "\n")
        
        if use_cv:
            models, oof_preds = self.lgbm_trainer.train_with_cv(
                X_features, 
                y, 
                n_folds=n_folds
            )
        else:
            # Simple train/val split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train single model
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train.astype(str))
            y_val_encoded = le.transform(y_val.astype(str))
            
            self.lgbm_trainer.label_encoder = le
            
            import lightgbm as lgb
            train_data = lgb.Dataset(X_train, label=y_train_encoded)
            val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
            
            params = self.lgbm_trainer.params
            params['num_class'] = len(le.classes_)
            
            self.lgbm_trainer.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.lgbm_trainer.num_boost_round,
                valid_sets=[val_data],
                callbacks=[
                    lgb.log_evaluation(period=50),
                    lgb.early_stopping(stopping_rounds=self.lgbm_trainer.early_stopping_rounds)
                ]
            )
        
        print("\nTraining completed successfully!")
        
        # Free memory
        del train_df, X, y, X_features
        gc.collect()
    
    def predict(self, test_file: str, output_file: str = 'submission.csv'):
        """
        Make predictions on test data.
        
        Args:
            test_file: Path to test CSV file
            output_file: Path to save predictions
        """
        print("\n" + "="*70)
        print("PREDICTION PIPELINE")
        print("="*70)
        
        # Load test data
        print(f"\nLoading test data from {test_file}...")
        
        # Check file size
        if os.path.getsize(test_file) > 500 * 1024 * 1024:  # > 500MB
            print("Large file detected, processing in chunks...")
            
            processor = ChunkedDataProcessor(chunk_size=self.chunk_size)
            all_predictions = []
            all_ids = []
            
            for i, chunk in enumerate(processor.read_in_chunks(test_file)):
                print(f"\nProcessing chunk {i+1}...")
                
                # Save IDs
                if 'id' in chunk.columns:
                    chunk_ids = chunk['id'].values
                    chunk = chunk.drop(columns=['id'])
                else:
                    chunk_ids = chunk.index.values
                
                # Extract features
                X_features = self.extract_features(chunk, is_train=False)
                
                # Predict
                predictions = self.lgbm_trainer.predict(X_features, use_ensemble=True)
                
                all_predictions.extend(predictions)
                all_ids.extend(chunk_ids)
                
                # Free memory
                del chunk, X_features, predictions
                gc.collect()
            
            # Create submission
            submission = pd.DataFrame({
                'id': all_ids,
                self.target_col: all_predictions
            })
        else:
            # Load all at once
            test_df = pd.read_csv(test_file)
            print(f"Loaded {len(test_df)} test samples")
            
            # Save IDs
            if 'id' in test_df.columns:
                test_ids = test_df['id'].values
                test_df = test_df.drop(columns=['id'])
            else:
                test_ids = test_df.index.values
            
            # Extract features
            X_features = self.extract_features(test_df, is_train=False)
            
            # Predict
            predictions = self.lgbm_trainer.predict(X_features, use_ensemble=True)
            
            # Create submission
            submission = pd.DataFrame({
                'id': test_ids,
                self.target_col: predictions
            })
        
        # Save submission
        submission.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
        print("="*70 + "\n")
    
    def save_pipeline(self, path: str = 'pipeline.pkl'):
        """Save the entire pipeline."""
        import joblib
        joblib.dump({
            'mtu_processor': self.mtu_processor,
            'target_encoder': self.target_encoder,
            'tsfel_extractor': self.tsfel_extractor,
            'lifestream_embedder': self.lifestream_embedder,
            'lgbm_trainer': self.lgbm_trainer,
            'config': {
                'use_mtu_preprocessing': self.use_mtu_preprocessing,
                'use_target_encoding': self.use_target_encoding,
                'use_tsfel': self.use_tsfel,
                'use_lifestream': self.use_lifestream,
                'tsfel_domain': self.tsfel_domain,
                'lifestream_encoder': self.lifestream_encoder,
                'lifestream_dim': self.lifestream_dim
            }
        }, path)
        print(f"Pipeline saved to {path}")
    
    def load_pipeline(self, path: str = 'pipeline.pkl'):
        """Load a saved pipeline."""
        import joblib
        data = joblib.load(path)
        
        self.mtu_processor = data['mtu_processor']
        self.target_encoder = data['target_encoder']
        self.tsfel_extractor = data['tsfel_extractor']
        self.lifestream_embedder = data['lifestream_embedder']
        self.lgbm_trainer = data['lgbm_trainer']
        
        config = data['config']
        self.use_mtu_preprocessing = config['use_mtu_preprocessing']
        self.use_target_encoding = config['use_target_encoding']
        self.use_tsfel = config['use_tsfel']
        self.use_lifestream = config['use_lifestream']
        self.tsfel_domain = config['tsfel_domain']
        self.lifestream_encoder = config['lifestream_encoder']
        self.lifestream_dim = config['lifestream_dim']
        
        print(f"Pipeline loaded from {path}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='TCP Traffic Classification Pipeline')
    parser.add_argument('--train', type=str, default='train.csv', help='Path to training data')
    parser.add_argument('--test', type=str, default='test.csv', help='Path to test data')
    parser.add_argument('--output', type=str, default='submission.csv', help='Path to output file')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size for faster training')
    parser.add_argument('--no-tsfel', action='store_true', help='Disable TSFEL features')
    parser.add_argument('--no-lifestream', action='store_true', help='Disable Lifestream embeddings')
    parser.add_argument('--tsfel-domain', type=str, default='statistical', 
                       choices=['statistical', 'temporal', 'spectral', 'all'],
                       help='TSFEL feature domain')
    parser.add_argument('--lifestream-encoder', type=str, default='rnn',
                       choices=['rnn', 'transformer'],
                       help='Lifestream encoder type')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Chunk size for large files')
    
    args = parser.parse_args()
    
    # Create pipeline
    classifier = TCPTrafficClassifier(
        use_tsfel=not args.no_tsfel,
        use_lifestream=not args.no_lifestream,
        tsfel_domain=args.tsfel_domain,
        lifestream_encoder=args.lifestream_encoder,
        chunk_size=args.chunk_size
    )
    
    # Train
    classifier.train(
        args.train,
        use_cv=True,
        n_folds=args.cv_folds,
        sample_size=args.sample_size
    )
    
    # Save pipeline
    classifier.save_pipeline('trained_pipeline.pkl')
    
    # Predict
    classifier.predict(args.test, args.output)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

