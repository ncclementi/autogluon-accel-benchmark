import subprocess
import sys
import pandas as pd
import os

SCRIPT = 'autogluon_sklearn_predictor.py'

# Define sample sizes to test
SAMPLE_SIZES = {
    '1K': 1000,
    '10K': 10000, 
    '100K': 100000,
    '1M': 1000000,
    '10M': 10000000
}

LEADERBOARD_DIR = 'leaderboards-per-sample-size'
os.makedirs(LEADERBOARD_DIR, exist_ok=True)

# Create results directory
RESULTS_DIR = 'comparison-results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmark_for_sample_size(display_name, n_samples):
    """Run benchmark for a specific sample size"""
    print(f'\n{"="*50}')
    print(f'Running benchmark for {display_name} samples ({n_samples:,})')
    print(f'{"="*50}')
    
    # Define filenames for this sample size
    leaderboard_sklearn = os.path.join(LEADERBOARD_DIR, f'leaderboard_sklearn_{display_name}.csv')
    leaderboard_cuml = os.path.join(LEADERBOARD_DIR, f'leaderboard_cuml_{display_name}.csv')
    comparison_file = os.path.join(RESULTS_DIR, f'comparison_results_{display_name}.csv')
    
    # Run sklearn mode
    print('Running sklearn mode...')
    subprocess.run([sys.executable, SCRIPT, '--output', leaderboard_sklearn, '--n_samples', str(n_samples)], check=True)

    # Run cuml mode
    print('Running cuml mode...')
    subprocess.run([sys.executable, '-m', 'cuml.accel', SCRIPT, '--output', leaderboard_cuml, '--n_samples', str(n_samples)], check=True)

    # Compare results
    if os.path.exists(leaderboard_sklearn) and os.path.exists(leaderboard_cuml):
        print(f'\nComparing {leaderboard_sklearn} and {leaderboard_cuml} (score_val, fit_time):')
        df1 = pd.read_csv(leaderboard_sklearn)
        df2 = pd.read_csv(leaderboard_cuml)
        if 'model' in df1.columns and 'model' in df2.columns:
            merged = pd.merge(df1, df2, on='model', suffixes=('_sklearn', '_cuml'))
            compare_cols = ['model', 'score_val_sklearn', 'score_val_cuml', 'fit_time_sklearn', 'fit_time_cuml']
            merged = merged[compare_cols]
            print(merged.to_string(index=False))
            
            # Add sample size column and save comparison
            merged['n_samples'] = n_samples
            merged['sample_size_name'] = display_name
            merged.to_csv(comparison_file, index=False)
            print(f'Saved comparison results to {comparison_file}')
        else:
            print("Could not compare: 'model' column missing in one of the leaderboards.")
    else:
        print('One or both leaderboard files are missing.')

def main():
    """Run benchmarks for all sample sizes"""
    print("Starting AutoGluon sklearn vs cuML benchmark comparison")
    print(f"Testing sample sizes: {list(SAMPLE_SIZES.keys())}")
    
    for display_name, n_samples in SAMPLE_SIZES.items():
        try:
            run_benchmark_for_sample_size(display_name, n_samples)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {display_name} samples: {e}")
        except Exception as e:
            print(f"Unexpected error for {display_name} samples: {e}")
    
    print(f'\n{"="*50}')
    print("All benchmarks completed!")
    print(f'{"="*50}')

if __name__ == "__main__":
    main() 