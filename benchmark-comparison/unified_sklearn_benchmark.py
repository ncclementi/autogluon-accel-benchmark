from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import argparse
import time
import gc

def run_single_benchmark(model, n_samples):
    """Run a single benchmark for the specified model."""
    # Generate binary classification data
    X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, n_classes=2)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # Choose predictor based on model argument
    if model == 'lr':
        predictor = LogisticRegression()

        start_time = time.time()
        predictor.fit(X_train, y_train)
        end_time = time.time()

    elif model == 'rf':
        predictor = RandomForestClassifier(n_estimators=300, 
                max_leaf_nodes= 15000,
                bootstrap=True,
                n_jobs=-1,
            )
        
        start_time = time.time()
        predictor.fit(X_train, y_train)
        end_time = time.time()

    elif model == 'knn':
        predictor = KNeighborsClassifier(weights='uniform', n_jobs=-1)

        start_time = time.time()
        predictor.fit(X_train, y_train)
        end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Clear memory
    del X, y, X_train, X_test, y_train, y_test, predictor
    gc.collect()

    
    return execution_time

def run_benchmark_suite(n_samples, n_runs, to_csv):
    """Run benchmark suite for all models multiple times."""
    models = ['lr', 'rf', 'knn']
    results = {model: [] for model in models}
    
    print(f"Running {n_runs} times for each model with {n_samples} samples")
    print("Ensuring fresh data and memory clearing between runs...")
    
    for model in models:
        print(f"Running {model}...")
        for run in range(n_runs):
            execution_time = run_single_benchmark(model, n_samples)
            results[model].append(execution_time)
            print(f"  Run {run+1}: {execution_time:.4f}s")
        
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df)
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    stats = df.agg(['mean', 'median', 'std', 'min', 'max']).round(4)
    print(stats)
    
    # Save results if to_csv is provided
    if to_csv:
        if isinstance(to_csv, str):
            # If to_csv is a string, add it to the filename
            filename = f"sklearn_stats_{n_samples}_{to_csv}.csv"
        else:
            # If to_csv is True, use default filename
            filename = f"sklearn_stats_{n_samples}.csv"
        stats.to_csv(filename)
        print(f"\nSaved to {filename}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Run sklearn model benchmark')
    parser.add_argument('--model', choices=['lr', 'rf', 'knn'], 
                       help='Single model to run (optional)')
    parser.add_argument('--n_samples', type=int, default=10_000, 
                       help='Number of samples (default: 10_000)')
    parser.add_argument('--suite', action='store_true',
                       help='Run benchmark suite for all models')
    parser.add_argument('--to_csv', nargs='?', const=True, default=False,
                       help='Save results to CSV file. If a string is provided, it will be added to the filename.')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of runs for each model in benchmark suite (default: 5)')
    args = parser.parse_args()
    
    if args.suite:
        # Run benchmark suite
        run_benchmark_suite(args.n_samples, n_runs=args.n_runs, to_csv=args.to_csv)
    elif args.model:
        # Run single model
        execution_time = run_single_benchmark(args.model, args.n_samples)
        print(f"Model: {args.model}, Fit time: {execution_time:.4f} seconds, n_samples: {args.n_samples}")
    else:
        print("Please specify either --model for single run or --suite for benchmark suite")

if __name__ == "__main__":
    main()
