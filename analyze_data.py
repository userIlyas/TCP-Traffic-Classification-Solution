"""
Data Analysis and Visualization Script
Provides insights into the TCP traffic dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def analyze_tcp_sequences(df, tcp_columns, sample_size=1000):
    """Analyze TCP packet sequences."""
    print("\n" + "="*70)
    print("TCP SEQUENCE ANALYSIS")
    print("="*70 + "\n")
    
    # Sample for faster analysis
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Analyzing sample of {sample_size} sequences\n")
    else:
        df_sample = df
    
    tcp_data = df_sample[tcp_columns].values
    
    # 1. Sequence lengths (non-zero packets)
    seq_lengths = np.count_nonzero(tcp_data, axis=1)
    
    print("Sequence Length Statistics:")
    print(f"  Mean: {seq_lengths.mean():.2f}")
    print(f"  Median: {np.median(seq_lengths):.2f}")
    print(f"  Min: {seq_lengths.min()}")
    print(f"  Max: {seq_lengths.max()}")
    print(f"  Std: {seq_lengths.std():.2f}")
    
    # 2. Packet size distribution
    all_packets = tcp_data[tcp_data != 0].flatten()
    abs_sizes = np.abs(all_packets)
    
    print("\nPacket Size Statistics:")
    print(f"  Mean: {abs_sizes.mean():.2f}")
    print(f"  Median: {np.median(abs_sizes):.2f}")
    print(f"  Min: {abs_sizes.min():.2f}")
    print(f"  Max: {abs_sizes.max():.2f}")
    print(f"  Std: {abs_sizes.std():.2f}")
    
    # 3. Direction statistics
    outgoing = np.sum(tcp_data > 0)
    incoming = np.sum(tcp_data < 0)
    total = outgoing + incoming
    
    print("\nDirection Statistics:")
    print(f"  Outgoing packets: {outgoing} ({outgoing/total*100:.1f}%)")
    print(f"  Incoming packets: {incoming} ({incoming/total*100:.1f}%)")
    
    # 4. MTU-sized packets
    mtu_packets = np.sum(abs_sizes > 1200)
    print(f"\nMTU-sized packets (>1200): {mtu_packets} ({mtu_packets/len(abs_sizes)*100:.1f}%)")
    
    # 5. Common packet sizes
    print("\nTop 10 most common packet sizes:")
    size_counts = Counter(abs_sizes.astype(int))
    for size, count in size_counts.most_common(10):
        print(f"  {size:5d} bytes: {count:6d} occurrences ({count/len(abs_sizes)*100:.2f}%)")
    
    return {
        'seq_lengths': seq_lengths,
        'abs_sizes': abs_sizes,
        'tcp_data': tcp_data
    }


def analyze_target_distribution(df, target_col='app_service'):
    """Analyze target variable distribution."""
    print("\n" + "="*70)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("="*70 + "\n")
    
    target_counts = df[target_col].value_counts()
    
    print(f"Number of unique classes: {len(target_counts)}")
    print(f"Total samples: {len(df)}\n")
    
    print("Class distribution:")
    for i, (cls, count) in enumerate(target_counts.items(), 1):
        print(f"  {i:2d}. {cls:20s}: {count:7d} ({count/len(df)*100:5.2f}%)")
        if i >= 20:  # Show top 20
            print(f"  ... and {len(target_counts) - 20} more classes")
            break
    
    # Check for imbalance
    max_count = target_counts.max()
    min_count = target_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print("  ⚠ Warning: Significant class imbalance detected!")
    
    return target_counts


def analyze_patterns(df, tcp_columns, target_col='app_service', n_classes=5):
    """Analyze patterns for different classes."""
    print("\n" + "="*70)
    print("PATTERN ANALYSIS BY CLASS")
    print("="*70 + "\n")
    
    # Get top N classes
    top_classes = df[target_col].value_counts().head(n_classes).index
    
    for cls in top_classes:
        print(f"\nClass: {cls}")
        print("-" * 50)
        
        cls_df = df[df[target_col] == cls]
        tcp_data = cls_df[tcp_columns].values
        
        # Average sequence length
        seq_lengths = np.count_nonzero(tcp_data, axis=1)
        print(f"  Avg sequence length: {seq_lengths.mean():.2f}")
        
        # Average packet size
        non_zero = tcp_data[tcp_data != 0]
        avg_size = np.abs(non_zero).mean()
        print(f"  Avg packet size: {avg_size:.2f}")
        
        # Direction ratio
        outgoing = np.sum(tcp_data > 0)
        incoming = np.sum(tcp_data < 0)
        ratio = outgoing / (incoming + 1)
        print(f"  Out/In ratio: {ratio:.2f}")
        
        # Large packets ratio
        large_packets = np.sum(np.abs(tcp_data) > 1200)
        total_packets = np.count_nonzero(tcp_data)
        large_ratio = large_packets / total_packets if total_packets > 0 else 0
        print(f"  Large packets ratio: {large_ratio:.2%}")
        
        # First packet average
        first_packets = tcp_data[:, 0]
        first_avg = np.abs(first_packets[first_packets != 0]).mean()
        print(f"  Avg first packet size: {first_avg:.2f}")


def create_visualizations(analysis_results, target_counts, output_dir='plots'):
    """Create visualization plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Sequence length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(analysis_results['seq_lengths'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Sequence Length (number of non-zero packets)')
    plt.ylabel('Frequency')
    plt.title('Distribution of TCP Sequence Lengths')
    plt.savefig(f'{output_dir}/sequence_lengths.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/sequence_lengths.png")
    plt.close()
    
    # 2. Packet size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(analysis_results['abs_sizes'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of TCP Packet Sizes')
    plt.axvline(x=1200, color='r', linestyle='--', label='MTU threshold (1200)')
    plt.legend()
    plt.savefig(f'{output_dir}/packet_sizes.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/packet_sizes.png")
    plt.close()
    
    # 3. Target distribution (top 20)
    plt.figure(figsize=(12, 8))
    top_20 = target_counts.head(20)
    plt.barh(range(len(top_20)), top_20.values)
    plt.yticks(range(len(top_20)), top_20.index)
    plt.xlabel('Number of Samples')
    plt.ylabel('Class')
    plt.title('Top 20 Classes Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/target_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/target_distribution.png")
    plt.close()
    
    # 4. Direction distribution
    tcp_data = analysis_results['tcp_data']
    outgoing = np.sum(tcp_data > 0)
    incoming = np.sum(tcp_data < 0)
    
    plt.figure(figsize=(8, 6))
    plt.pie([outgoing, incoming], labels=['Outgoing', 'Incoming'], 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Packet Direction Distribution')
    plt.savefig(f'{output_dir}/direction_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/direction_distribution.png")
    plt.close()
    
    print(f"\nAll plots saved to '{output_dir}/' directory")


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("TCP TRAFFIC DATA ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading training data...")
    
    # Load sample for analysis
    sample_size = 10000
    df = pd.read_csv('train.csv', nrows=sample_size)
    
    print(f"Loaded {len(df)} samples for analysis")
    
    # Define columns
    tcp_columns = [f'tcp_len_{i}' for i in range(1, 31)]
    target_col = 'app_service'
    
    # Run analyses
    analysis_results = analyze_tcp_sequences(df, tcp_columns, sample_size=sample_size)
    target_counts = analyze_target_distribution(df, target_col)
    analyze_patterns(df, tcp_columns, target_col, n_classes=5)
    
    # Create visualizations
    try:
        create_visualizations(analysis_results, target_counts)
    except Exception as e:
        print(f"\n⚠ Warning: Could not create visualizations: {e}")
        print("  (matplotlib may not be available in headless environment)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

