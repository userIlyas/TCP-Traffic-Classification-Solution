"""
PyTorch-Lifestream Embeddings Module
Creates vector representations of TCP packet sequences using self-supervised learning
Inspired by CoLES (Contrastive Learning for Event Sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TCPSequenceDataset(Dataset):
    """
    Dataset for TCP packet sequences compatible with PyTorch.
    """
    
    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Args:
            sequences: Array of shape (n_samples, seq_length) with TCP packet lengths
            labels: Optional array of labels for supervised learning
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for TCP packet sequences.
    Captures temporal dependencies and patterns in encrypted traffic.
    """
    
    def __init__(self, 
                 input_dim: int = 30,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Length of input sequence (number of packets)
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_projection = nn.Linear(1, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, input_dim, embedding_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length)
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Reshape for projection: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Project to embedding dimension
        x = self.input_projection(x)  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer
        x = self.transformer(x)  # (batch, seq_len, embedding_dim)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch, embedding_dim)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class RNNEncoder(nn.Module):
    """
    RNN-based encoder for TCP packet sequences.
    Alternative to Transformer, often faster for sequential data.
    """
    
    def __init__(self,
                 input_dim: int = 30,
                 embedding_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        """
        Args:
            input_dim: Length of input sequence
            embedding_dim: Dimension of embeddings
            num_layers: Number of RNN layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
        """
        super(RNNEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(1, embedding_dim // 2 if bidirectional else embedding_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim // 2 if bidirectional else embedding_dim,
            hidden_size=embedding_dim // 2 if bidirectional else embedding_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length)
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Reshape: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Project
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # LSTM
        output, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Output projection
        x = self.output_projection(hidden)
        
        return x


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning (CoLES approach).
    Learns to distinguish between similar and dissimilar sequences.
    """
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive pairs mask (same label)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of similarities to all samples
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # Sum of similarities to positive samples
        positive_sim = (exp_sim * positive_mask).sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        positive_sim = torch.clamp(positive_sim, min=1e-8)
        
        # Contrastive loss
        loss = -torch.log(positive_sim / sum_exp_sim)
        
        return loss.mean()


class LifestreamEmbedder:
    """
    Main class for creating embeddings from TCP sequences using self-supervised learning.
    """
    
    def __init__(self,
                 encoder_type: str = "transformer",
                 embedding_dim: int = 128,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            encoder_type: "transformer" or "rnn"
            embedding_dim: Dimension of output embeddings
            device: Device to use for training
        """
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize encoder
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(embedding_dim=embedding_dim)
        elif encoder_type == "rnn":
            self.encoder = RNNEncoder(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.encoder = self.encoder.to(device)
        
        # Loss function
        self.criterion = ContrastiveLoss()
    
    def train_embeddings(self,
                        train_sequences: np.ndarray,
                        train_labels: np.ndarray,
                        epochs: int = 10,
                        batch_size: int = 256,
                        learning_rate: float = 1e-3):
        """
        Train the encoder using contrastive learning.
        
        Args:
            train_sequences: Array of shape (n_samples, seq_length)
            train_labels: Array of labels for creating positive pairs
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Create dataset and dataloader
        dataset = TCPSequenceDataset(train_sequences, train_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # Training loop
        self.encoder.train()
        
        print(f"Training {self.encoder_type} encoder for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for sequences, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.encoder(sequences)
                
                # Compute loss
                loss = self.criterion(embeddings, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def extract_embeddings(self,
                          sequences: np.ndarray,
                          batch_size: int = 512) -> np.ndarray:
        """
        Extract embeddings from sequences.
        
        Args:
            sequences: Array of shape (n_samples, seq_length)
            batch_size: Batch size for inference
            
        Returns:
            Array of embeddings of shape (n_samples, embedding_dim)
        """
        # Create dataset and dataloader
        dataset = TCPSequenceDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Extract embeddings
        self.encoder.eval()
        all_embeddings = []
        
        print("Extracting embeddings...")
        
        with torch.no_grad():
            for sequences_batch in tqdm(dataloader):
                if isinstance(sequences_batch, list):
                    sequences_batch = sequences_batch[0]
                
                sequences_batch = sequences_batch.to(self.device)
                embeddings = self.encoder(sequences_batch)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def save_model(self, path: str):
        """Save the trained encoder."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_type': self.encoder_type,
            'embedding_dim': self.embedding_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained encoder."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder = self.encoder.to(self.device)
        print(f"Model loaded from {path}")


def create_lifestream_features(df: pd.DataFrame,
                               tcp_columns: List[str],
                               labels: Optional[pd.Series] = None,
                               encoder_type: str = "rnn",
                               embedding_dim: int = 128,
                               train_epochs: int = 5) -> pd.DataFrame:
    """
    Convenience function to create lifestream embeddings from a DataFrame.
    
    Args:
        df: DataFrame with TCP packet columns
        tcp_columns: List of TCP column names
        labels: Optional labels for supervised pretraining
        encoder_type: "transformer" or "rnn"
        embedding_dim: Dimension of embeddings
        train_epochs: Number of training epochs
        
    Returns:
        DataFrame with embedding features
    """
    # Prepare sequences
    sequences = df[tcp_columns].values.astype(np.float32)
    
    # Replace NaN and inf
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create embedder
    embedder = LifestreamEmbedder(
        encoder_type=encoder_type,
        embedding_dim=embedding_dim
    )
    
    # Train if labels provided
    if labels is not None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        embedder.train_embeddings(
            sequences,
            encoded_labels,
            epochs=train_epochs,
            batch_size=256
        )
    
    # Extract embeddings
    embeddings = embedder.extract_embeddings(sequences, batch_size=512)
    
    # Create DataFrame
    embedding_cols = [f'lifestream_{i}' for i in range(embedding_dim)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
    
    return embeddings_df, embedder


if __name__ == "__main__":
    # Test the lifestream embeddings
    print("Testing Lifestream Embedder...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    seq_length = 30
    
    # Generate synthetic TCP sequences
    sequences = np.random.randn(n_samples, seq_length) * 500
    labels = np.random.randint(0, 5, n_samples)
    
    # Create embedder
    embedder = LifestreamEmbedder(encoder_type="rnn", embedding_dim=64)
    
    # Train
    embedder.train_embeddings(sequences, labels, epochs=2, batch_size=32)
    
    # Extract embeddings
    embeddings = embedder.extract_embeddings(sequences, batch_size=32)
    
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    print(f"Sample embeddings:\n{embeddings[:3]}")
    
    print("\nLifestream embeddings test completed!")

