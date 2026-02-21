"""
Automated Solution Runner
Automatically runs the best configuration based on available resources and time constraints
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def check_file_size(filepath):
    """Check file size in MB."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return size_mb
    return 0


def estimate_time(train_size_mb, use_tsfel, use_lifestream, sample_size=None):
    """
    Estimate training time based on configuration.
    
    Returns:
        Estimated time in minutes
    """
    if sample_size:
        base_time = 5  # 5 minutes for sample
    else:
        # Base time proportional to data size
        base_time = train_size_mb / 50  # ~1 minute per 50MB
    
    if use_tsfel:
        base_time *= 2
    
    if use_lifestream:
        base_time *= 2.5
    
    return int(base_time)


def run_quick_solution():
    """Run quick baseline solution."""
    print("\n" + "="*70)
    print("RUNNING QUICK BASELINE SOLUTION")
    print("="*70 + "\n")
    
    print("This will use fast feature extraction and LightGBM.")
    print("Expected time: 30-60 minutes on full dataset\n")
    
    start_time = time.time()
    
    try:
        subprocess.run([sys.executable, 'quick_train.py'], check=True)
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n✓ Quick solution completed in {elapsed:.1f} minutes")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running quick solution: {e}")
        return False


def run_full_solution(args):
    """Run full pipeline with all features."""
    print("\n" + "="*70)
    print("RUNNING FULL PIPELINE")
    print("="*70 + "\n")
    
    # Build command
    cmd = [sys.executable, 'main_pipeline.py']
    
    if args.sample_size:
        cmd.extend(['--sample-size', str(args.sample_size)])
        print(f"Using sample size: {args.sample_size}")
    
    if args.no_tsfel:
        cmd.append('--no-tsfel')
        print("TSFEL features: DISABLED")
    else:
        print(f"TSFEL features: ENABLED (domain: {args.tsfel_domain})")
        cmd.extend(['--tsfel-domain', args.tsfel_domain])
    
    if args.no_lifestream:
        cmd.append('--no-lifestream')
        print("Lifestream embeddings: DISABLED")
    else:
        print(f"Lifestream embeddings: ENABLED (encoder: {args.lifestream_encoder})")
        cmd.extend(['--lifestream-encoder', args.lifestream_encoder])
    
    cmd.extend(['--cv-folds', str(args.cv_folds)])
    print(f"Cross-validation folds: {args.cv_folds}")
    
    # Estimate time
    train_size = check_file_size('train.csv')
    estimated_time = estimate_time(
        train_size, 
        not args.no_tsfel, 
        not args.no_lifestream,
        args.sample_size
    )
    print(f"\nEstimated time: ~{estimated_time} minutes")
    print(f"Training data size: {train_size:.1f} MB\n")
    
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True)
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n✓ Full pipeline completed in {elapsed:.1f} minutes")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running full pipeline: {e}")
        return False


def check_submission():
    """Check if submission file was created successfully."""
    if os.path.exists('submission.csv'):
        import pandas as pd
        sub = pd.read_csv('submission.csv')
        
        print("\n" + "="*70)
        print("SUBMISSION FILE CHECK")
        print("="*70)
        print(f"✓ Submission file created: submission.csv")
        print(f"  Number of predictions: {len(sub)}")
        print(f"  Columns: {sub.columns.tolist()}")
        print(f"  Unique classes: {sub['app_service'].nunique()}")
        print(f"\nFirst few predictions:")
        print(sub.head(10))
        print("="*70 + "\n")
        
        return True
    else:
        print("\n✗ Submission file not found!")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated TCP Traffic Classification Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick baseline (recommended for first run)
  python run_solution.py --mode quick
  
  # Full pipeline without Lifestream (faster)
  python run_solution.py --mode full --no-lifestream
  
  # Full pipeline with all features
  python run_solution.py --mode full
  
  # Test on small sample
  python run_solution.py --mode full --sample-size 10000 --no-lifestream
  
  # Auto mode (chooses best based on resources)
  python run_solution.py --mode auto
        """
    )
    
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['quick', 'full', 'auto'],
                       help='Solution mode: quick (baseline), full (all features), auto (choose automatically)')
    
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Use only N samples for training (for testing)')
    
    parser.add_argument('--no-tsfel', action='store_true',
                       help='Disable TSFEL features')
    
    parser.add_argument('--no-lifestream', action='store_true',
                       help='Disable Lifestream embeddings')
    
    parser.add_argument('--tsfel-domain', type=str, default='statistical',
                       choices=['statistical', 'temporal', 'spectral', 'all'],
                       help='TSFEL feature domain')
    
    parser.add_argument('--lifestream-encoder', type=str, default='rnn',
                       choices=['rnn', 'transformer'],
                       help='Lifestream encoder type')
    
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("TCP TRAFFIC CLASSIFICATION - AUTOMATED SOLUTION RUNNER")
    print("="*70 + "\n")
    
    # Check if data files exist
    if not os.path.exists('train.csv'):
        print("✗ Error: train.csv not found!")
        sys.exit(1)
    
    if not os.path.exists('test.csv'):
        print("✗ Error: test.csv not found!")
        sys.exit(1)
    
    print("✓ Data files found")
    
    # Check data sizes
    train_size = check_file_size('train.csv')
    test_size = check_file_size('test.csv')
    
    print(f"  train.csv: {train_size:.1f} MB")
    print(f"  test.csv: {test_size:.1f} MB")
    
    # Auto mode: choose best approach
    if args.mode == 'auto':
        print("\n[AUTO MODE] Analyzing optimal configuration...")
        
        # If sample size specified, use full pipeline
        if args.sample_size:
            print(f"→ Sample size specified ({args.sample_size}), using full pipeline")
            args.mode = 'full'
        # If data is very large (>1GB), suggest quick mode
        elif train_size > 1000:
            print(f"→ Large dataset detected ({train_size:.1f} MB)")
            print("→ Recommending quick baseline for faster results")
            
            response = input("\nUse quick baseline? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                args.mode = 'quick'
            else:
                args.mode = 'full'
                args.no_lifestream = True
                print("→ Using full pipeline without Lifestream")
        else:
            print("→ Using full pipeline with TSFEL, without Lifestream")
            args.mode = 'full'
            args.no_lifestream = True
    
    # Run selected mode
    print(f"\nSelected mode: {args.mode.upper()}\n")
    
    success = False
    
    if args.mode == 'quick':
        success = run_quick_solution()
    else:  # full
        success = run_full_solution(args)
    
    # Check submission
    if success:
        check_submission()
        
        print("\n" + "="*70)
        print("✓ SOLUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review submission.csv")
        print("2. Submit to Kaggle")
        print("3. Check leaderboard score")
        print("\nGood luck! 🚀\n")
    else:
        print("\n" + "="*70)
        print("✗ SOLUTION FAILED")
        print("="*70)
        print("\nPlease check the error messages above.")
        print("Try running with --sample-size 10000 for debugging.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

