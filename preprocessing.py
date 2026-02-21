"""
MTU Preprocessing and Feature Engineering Module
Implements MTU reconstruction and Target Encoding for TCP packet sequences
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from category_encoders import CVTargetEncoder


class MTUPreprocessor:
    """
    Preprocesses TCP packet sequences by reconstructing application-level data
    based on MTU (Maximum Transmission Unit) patterns.
    
    As suggested in the problem description: merge consecutive packets with 
    the same direction and size > 1200 bytes to reconstruct application data.
    """
    
    def __init__(self, mtu_threshold: int = 1200):
        """
        Args:
            mtu_threshold: Threshold for considering packets as MTU-sized (default 1200)
        """
        self.mtu_threshold = mtu_threshold
    
    def reconstruct_flow(self, packet_sequence: np.ndarray) -> np.ndarray:
        """
        Reconstruct application-level data flow from TCP packet sequence.
        
        Merges consecutive packets with:
        - Same direction (both positive or both negative)
        - Size > mtu_threshold
        
        Args:
            packet_sequence: Array of packet lengths (positive=client->server, negative=server->client)
            
        Returns:
            Reconstructed packet sequence with merged MTU packets
        """
        if len(packet_sequence) == 0:
            return packet_sequence
        
        reconstructed = []
        current_sum = 0
        current_direction = None
        
        for packet_len in packet_sequence:
            if packet_len == 0:  # Padding
                if current_sum != 0:
                    reconstructed.append(current_sum)
                    current_sum = 0
                break
            
            packet_direction = 1 if packet_len > 0 else -1
            packet_abs = abs(packet_len)
            
            # Check if this is an MTU-sized packet
            is_mtu = packet_abs > self.mtu_threshold
            
            if is_mtu and current_direction == packet_direction and current_sum != 0:
                # Continue merging
                current_sum += packet_len
            else:
                # Start new sequence or add non-MTU packet
                if current_sum != 0:
                    reconstructed.append(current_sum)
                current_sum = packet_len
                current_direction = packet_direction
        
        # Add last accumulated packet
        if current_sum != 0:
            reconstructed.append(current_sum)
        
        return np.array(reconstructed)
    
    def process_dataframe(self, df: pd.DataFrame, tcp_columns: List[str]) -> pd.DataFrame:
        """
        Apply MTU reconstruction to entire dataframe.
        
        Args:
            df: DataFrame with TCP packet length columns
            tcp_columns: List of column names containing TCP packet lengths
            
        Returns:
            DataFrame with reconstructed packet sequences (variable length)
        """
        sequences = []
        
        for idx, row in df.iterrows():
            packet_seq = row[tcp_columns].values.astype(float)
            reconstructed = self.reconstruct_flow(packet_seq)
            sequences.append(reconstructed)
        
        return sequences


class TargetEncodingFeatures:
    """
    Creates Target Encoding features with regularization and cross-validation
    to prevent overfitting and data leakage.
    """
    
    def __init__(self, smoothing: float = 10.0, cv_folds: int = 5):
        """
        Args:
            smoothing: Regularization parameter (alpha) for smoothing
            cv_folds: Number of cross-validation folds
        """
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.encoders = {}
    
    def create_categorical_features(self, df: pd.DataFrame, tcp_columns: List[str]) -> pd.DataFrame:
        """
        Create categorical features from TCP packet sequences for Target Encoding.
        
        Features include:
        - Packet direction patterns (e.g., "+++---+")
        - Packet size buckets
        - Flow patterns
        
        Args:
            df: DataFrame with TCP packet columns
            tcp_columns: List of TCP column names
            
        Returns:
            DataFrame with additional categorical features
        """
        df = df.copy()
        
        # Direction pattern (first N packets)
        def get_direction_pattern(row, n=10):
            pattern = []
            for col in tcp_columns[:n]:
                val = row[col]
                if pd.isna(val) or val == 0:
                    pattern.append('0')
                elif val > 0:
                    pattern.append('+')
                else:
                    pattern.append('-')
            return ''.join(pattern)
        
        df['direction_pattern'] = df.apply(lambda row: get_direction_pattern(row), axis=1)
        
        # Size bucket for first packet
        def get_size_bucket(size):
            if pd.isna(size) or size == 0:
                return 'zero'
            abs_size = abs(size)
            if abs_size < 100:
                return 'tiny'
            elif abs_size < 500:
                return 'small'
            elif abs_size < 1200:
                return 'medium'
            else:
                return 'large'
        
        df['first_packet_bucket'] = df[tcp_columns[0]].apply(get_size_bucket)
        
        # Flow type based on initial handshake pattern
        def get_flow_type(row):
            first_three = [row[tcp_columns[i]] for i in range(min(3, len(tcp_columns)))]
            if all(pd.notna(x) and x != 0 for x in first_three):
                if first_three[0] > 0 and first_three[1] < 0 and first_three[2] > 0:
                    return 'handshake'
                elif all(x > 0 for x in first_three):
                    return 'upload'
                elif all(x < 0 for x in first_three):
                    return 'download'
            return 'mixed'
        
        df['flow_type'] = df.apply(get_flow_type, axis=1)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target: pd.Series, 
                     categorical_cols: List[str]) -> pd.DataFrame:
        """
        Fit target encoders and transform categorical features.
        
        Args:
            df: DataFrame with categorical features
            target: Target variable
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with target-encoded features
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                encoder = CVTargetEncoder(
                    cols=[col], 
                    cv=self.cv_folds, 
                    smoothing=self.smoothing
                )
                encoded_col = encoder.fit_transform(df[[col]], target)
                df_encoded[f'{col}_target_enc'] = encoded_col[col]
                self.encoders[col] = encoder
        
        return df_encoded
    
    def transform(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Transform categorical features using fitted encoders.
        
        Args:
            df: DataFrame with categorical features
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with target-encoded features
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns and col in self.encoders:
                encoder = self.encoders[col]
                encoded_col = encoder.transform(df[[col]])
                df_encoded[f'{col}_target_enc'] = encoded_col[col]
        
        return df_encoded


def extract_basic_features(df: pd.DataFrame, tcp_columns: List[str]) -> pd.DataFrame:
    """
    Extract basic statistical features from TCP packet sequences.
    
    Args:
        df: DataFrame with TCP packet columns
        tcp_columns: List of TCP column names
        
    Returns:
        DataFrame with additional statistical features
    """
    df = df.copy()
    
    # Convert to numpy array for faster computation
    tcp_data = df[tcp_columns].values
    
    # Replace NaN and inf with 0
    tcp_data = np.nan_to_num(tcp_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Basic statistics
    df['total_packets'] = np.count_nonzero(tcp_data, axis=1)
    df['mean_packet_size'] = np.mean(np.abs(tcp_data), axis=1)
    df['std_packet_size'] = np.std(np.abs(tcp_data), axis=1)
    df['max_packet_size'] = np.max(np.abs(tcp_data), axis=1)
    df['min_packet_size_nonzero'] = np.array([
        np.min(np.abs(row[row != 0])) if np.any(row != 0) else 0 
        for row in tcp_data
    ])
    
    # Direction-based features
    df['outgoing_packets'] = np.sum(tcp_data > 0, axis=1)
    df['incoming_packets'] = np.sum(tcp_data < 0, axis=1)
    df['outgoing_bytes'] = np.sum(np.maximum(tcp_data, 0), axis=1)
    df['incoming_bytes'] = np.abs(np.sum(np.minimum(tcp_data, 0), axis=1))
    
    # Ratio features
    df['out_in_ratio'] = df['outgoing_packets'] / (df['incoming_packets'] + 1)
    df['bytes_ratio'] = df['outgoing_bytes'] / (df['incoming_bytes'] + 1)
    
    # Flow characteristics
    df['flow_duration'] = df['total_packets']  # Proxy for duration
    df['avg_packet_interval'] = df['flow_duration'] / (df['total_packets'] + 1)
    
    # Large packet indicators (MTU-sized)
    df['large_packets_count'] = np.sum(np.abs(tcp_data) > 1200, axis=1)
    df['large_packets_ratio'] = df['large_packets_count'] / (df['total_packets'] + 1)
    
    return df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing MTU Preprocessor...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'tcp_len_1': [1448, 1448, 100],
        'tcp_len_2': [1448, -1448, 200],
        'tcp_len_3': [1448, -1448, -300],
        'tcp_len_4': [100, -100, 0],
        'tcp_len_5': [0, 0, 0],
    })
    
    tcp_cols = [f'tcp_len_{i}' for i in range(1, 6)]
    
    # Test MTU reconstruction
    mtu_processor = MTUPreprocessor()
    reconstructed = mtu_processor.process_dataframe(sample_data, tcp_cols)
    
    print("Original sequences:")
    for idx, row in sample_data.iterrows():
        print(f"  {row[tcp_cols].values}")
    
    print("\nReconstructed sequences:")
    for seq in reconstructed:
        print(f"  {seq}")
    
    # Test basic feature extraction
    print("\nTesting basic feature extraction...")
    features_df = extract_basic_features(sample_data, tcp_cols)
    print(features_df[['total_packets', 'mean_packet_size', 'outgoing_packets', 'incoming_packets']].head())
    
    print("\nPreprocessing module test completed!")

