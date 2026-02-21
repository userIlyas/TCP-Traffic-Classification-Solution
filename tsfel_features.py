"""
TSFEL Feature Extraction Module
Extracts time-series features from TCP packet sequences using TSFEL library
"""

import pandas as pd
import numpy as np
import tsfel
from typing import List, Dict, Optional
import json
from tqdm import tqdm


class TSFELFeatureExtractor:
    """
    Extracts comprehensive time-series features from TCP packet sequences
    using TSFEL (Time Series Feature Extraction Library).
    
    TSFEL automatically generates statistical, temporal, and spectral features
    that capture patterns in encrypted traffic.
    """
    
    def __init__(self, 
                 domain: str = "statistical",
                 feature_config: Optional[Dict] = None,
                 n_jobs: int = -1):
        """
        Args:
            domain: Feature domain - "statistical", "temporal", "spectral", or "all"
            feature_config: Custom TSFEL configuration dict (None = use defaults)
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.domain = domain
        self.n_jobs = n_jobs
        
        # Get TSFEL configuration
        if feature_config is None:
            if domain == "all":
                self.cfg = tsfel.get_features_by_domain()
            else:
                self.cfg = tsfel.get_features_by_domain(domain)
        else:
            self.cfg = feature_config
        
        # Store feature names after first extraction
        self.feature_names = None
    
    def _prepare_sequence(self, packet_sequence: np.ndarray) -> pd.DataFrame:
        """
        Prepare packet sequence for TSFEL processing.
        
        Args:
            packet_sequence: Array of TCP packet lengths
            
        Returns:
            DataFrame with time-series data
        """
        # Remove padding (zeros at the end)
        non_zero_mask = packet_sequence != 0
        if np.any(non_zero_mask):
            last_nonzero = np.max(np.where(non_zero_mask)[0])
            sequence = packet_sequence[:last_nonzero + 1]
        else:
            sequence = np.array([0])  # Handle all-zero sequences
        
        # Create DataFrame (TSFEL expects DataFrame input)
        df = pd.DataFrame({'packet_size': sequence})
        
        return df
    
    def extract_features_single(self, packet_sequence: np.ndarray) -> np.ndarray:
        """
        Extract TSFEL features from a single packet sequence.
        
        Args:
            packet_sequence: Array of TCP packet lengths
            
        Returns:
            Array of extracted features
        """
        try:
            df = self._prepare_sequence(packet_sequence)
            
            # Extract features
            features = tsfel.time_series_features_extractor(
                self.cfg, 
                df, 
                verbose=0
            )
            
            # Handle NaN and inf values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)
            
            return features.values.flatten()
        
        except Exception as e:
            # Return zeros if extraction fails
            if self.feature_names is not None:
                return np.zeros(len(self.feature_names))
            else:
                return np.zeros(100)  # Default fallback
    
    def extract_features_batch(self, 
                               df: pd.DataFrame, 
                               tcp_columns: List[str],
                               batch_size: int = 1000) -> pd.DataFrame:
        """
        Extract TSFEL features from multiple sequences in batches.
        
        Args:
            df: DataFrame with TCP packet columns
            tcp_columns: List of TCP column names
            batch_size: Number of sequences to process at once
            
        Returns:
            DataFrame with extracted TSFEL features
        """
        all_features = []
        
        print(f"Extracting TSFEL features ({self.domain} domain)...")
        
        # Process in batches to show progress
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i + batch_size]
            batch_features = []
            
            for idx, row in batch.iterrows():
                packet_seq = row[tcp_columns].values.astype(float)
                features = self.extract_features_single(packet_seq)
                batch_features.append(features)
            
            all_features.extend(batch_features)
        
        # Convert to DataFrame
        features_array = np.array(all_features)
        
        # Get feature names from first extraction
        if self.feature_names is None:
            # Extract feature names
            sample_df = self._prepare_sequence(df[tcp_columns].iloc[0].values.astype(float))
            sample_features = tsfel.time_series_features_extractor(
                self.cfg, 
                sample_df, 
                verbose=0
            )
            self.feature_names = [f'tsfel_{col}' for col in sample_features.columns]
        
        features_df = pd.DataFrame(
            features_array, 
            columns=self.feature_names,
            index=df.index
        )
        
        return features_df
    
    def extract_direction_features(self, 
                                   df: pd.DataFrame, 
                                   tcp_columns: List[str]) -> pd.DataFrame:
        """
        Extract separate features for outgoing and incoming packets.
        
        This captures directional patterns in the traffic flow.
        
        Args:
            df: DataFrame with TCP packet columns
            tcp_columns: List of TCP column names
            
        Returns:
            DataFrame with directional TSFEL features
        """
        print("Extracting directional TSFEL features...")
        
        outgoing_features = []
        incoming_features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            packet_seq = row[tcp_columns].values.astype(float)
            
            # Separate by direction
            outgoing = packet_seq[packet_seq > 0]
            incoming = np.abs(packet_seq[packet_seq < 0])
            
            # Extract features for each direction
            if len(outgoing) > 0:
                out_feat = self.extract_features_single(outgoing)
            else:
                out_feat = np.zeros(len(self.feature_names) if self.feature_names else 100)
            
            if len(incoming) > 0:
                inc_feat = self.extract_features_single(incoming)
            else:
                inc_feat = np.zeros(len(self.feature_names) if self.feature_names else 100)
            
            outgoing_features.append(out_feat)
            incoming_features.append(inc_feat)
        
        # Create DataFrames
        out_df = pd.DataFrame(
            outgoing_features,
            columns=[f'tsfel_out_{i}' for i in range(len(outgoing_features[0]))],
            index=df.index
        )
        
        inc_df = pd.DataFrame(
            incoming_features,
            columns=[f'tsfel_inc_{i}' for i in range(len(incoming_features[0]))],
            index=df.index
        )
        
        return pd.concat([out_df, inc_df], axis=1)


class CustomTCPFeatures:
    """
    Custom feature extraction specifically designed for TCP packet analysis.
    Complements TSFEL with domain-specific features.
    """
    
    @staticmethod
    def extract_burst_features(packet_sequence: np.ndarray) -> Dict[str, float]:
        """
        Extract burst-related features (consecutive large packets).
        
        Args:
            packet_sequence: Array of TCP packet lengths
            
        Returns:
            Dictionary of burst features
        """
        # Remove padding
        packet_sequence = packet_sequence[packet_sequence != 0]
        
        if len(packet_sequence) == 0:
            return {
                'max_burst_length': 0,
                'avg_burst_length': 0,
                'burst_count': 0,
                'burst_ratio': 0
            }
        
        # Define burst as consecutive packets > 1000 bytes
        abs_seq = np.abs(packet_sequence)
        is_large = abs_seq > 1000
        
        # Find bursts
        burst_lengths = []
        current_burst = 0
        
        for large in is_large:
            if large:
                current_burst += 1
            else:
                if current_burst > 0:
                    burst_lengths.append(current_burst)
                current_burst = 0
        
        if current_burst > 0:
            burst_lengths.append(current_burst)
        
        if len(burst_lengths) == 0:
            return {
                'max_burst_length': 0,
                'avg_burst_length': 0,
                'burst_count': 0,
                'burst_ratio': 0
            }
        
        return {
            'max_burst_length': max(burst_lengths),
            'avg_burst_length': np.mean(burst_lengths),
            'burst_count': len(burst_lengths),
            'burst_ratio': sum(burst_lengths) / len(packet_sequence)
        }
    
    @staticmethod
    def extract_pattern_features(packet_sequence: np.ndarray) -> Dict[str, float]:
        """
        Extract pattern-based features (alternating directions, repetitions).
        
        Args:
            packet_sequence: Array of TCP packet lengths
            
        Returns:
            Dictionary of pattern features
        """
        # Remove padding
        packet_sequence = packet_sequence[packet_sequence != 0]
        
        if len(packet_sequence) < 2:
            return {
                'direction_changes': 0,
                'direction_change_rate': 0,
                'alternating_pattern': 0,
                'size_variance': 0
            }
        
        # Direction changes
        directions = np.sign(packet_sequence)
        direction_changes = np.sum(directions[1:] != directions[:-1])
        
        # Alternating pattern (consecutive direction changes)
        is_alternating = direction_changes == len(packet_sequence) - 1
        
        # Size variance
        abs_sizes = np.abs(packet_sequence)
        size_variance = np.var(abs_sizes)
        
        return {
            'direction_changes': direction_changes,
            'direction_change_rate': direction_changes / (len(packet_sequence) - 1),
            'alternating_pattern': 1 if is_alternating else 0,
            'size_variance': size_variance
        }
    
    @staticmethod
    def extract_all_custom_features(df: pd.DataFrame, 
                                    tcp_columns: List[str]) -> pd.DataFrame:
        """
        Extract all custom TCP features for a DataFrame.
        
        Args:
            df: DataFrame with TCP packet columns
            tcp_columns: List of TCP column names
            
        Returns:
            DataFrame with custom features
        """
        print("Extracting custom TCP features...")
        
        all_features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            packet_seq = row[tcp_columns].values.astype(float)
            
            # Extract burst features
            burst_feat = CustomTCPFeatures.extract_burst_features(packet_seq)
            
            # Extract pattern features
            pattern_feat = CustomTCPFeatures.extract_pattern_features(packet_seq)
            
            # Combine
            combined = {**burst_feat, **pattern_feat}
            all_features.append(combined)
        
        return pd.DataFrame(all_features, index=df.index)


if __name__ == "__main__":
    # Test the TSFEL feature extraction
    print("Testing TSFEL Feature Extractor...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'tcp_len_1': [1448, 1448, 100, -500],
        'tcp_len_2': [1448, -1448, 200, -600],
        'tcp_len_3': [1448, -1448, -300, 700],
        'tcp_len_4': [100, -100, 400, -800],
        'tcp_len_5': [0, 0, 0, 0],
    })
    
    tcp_cols = [f'tcp_len_{i}' for i in range(1, 6)]
    
    # Test TSFEL extraction (using only statistical domain for speed)
    extractor = TSFELFeatureExtractor(domain="statistical")
    tsfel_features = extractor.extract_features_batch(sample_data, tcp_cols, batch_size=2)
    
    print(f"\nExtracted {tsfel_features.shape[1]} TSFEL features")
    print(f"Sample features:\n{tsfel_features.head()}")
    
    # Test custom features
    custom_features = CustomTCPFeatures.extract_all_custom_features(sample_data, tcp_cols)
    
    print(f"\nExtracted {custom_features.shape[1]} custom features")
    print(f"Custom features:\n{custom_features.head()}")
    
    print("\nTSFEL feature extraction test completed!")

