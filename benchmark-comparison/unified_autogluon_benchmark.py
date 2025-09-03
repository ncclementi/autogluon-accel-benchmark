import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import argparse
import gc

def run_single_model(model, n_samples):
    """Run the specified model once with its hyperparameters using TabularPredictor."""
    # Generate binary classification data
    X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, n_classes=2)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Split into train/test
    train_data, test_data = train_test_split(df, test_size=0.1)
    label = "target"
    
    if model == 'lr':
        predictor = TabularPredictor(label=label, verbosity=0).fit(
            train_data=train_data, 
            hyperparameters={
                'LR': {"random_state": None},
            })
        
    elif model == 'rf':
        predictor = TabularPredictor(label=label, verbosity=0).fit(
            train_data=train_data, 
            hyperparameters={
                'RF': [{'criterion': 'gini', "random_state": None, 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}]
            })
        
    elif model == 'knn':
        predictor = TabularPredictor(label=label, verbosity=0).fit(
            train_data=train_data, 
            hyperparameters={
                'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}]
            })
    
    print(f"Model: {model.upper()}, n_samples: {n_samples}")
    fit_summary = predictor.fit_summary()
    model_name ={"lr":"LinearModel", "rf":"RandomForestGini", "knn":"KNeighborsUnif"}

    fit_time  = fit_summary['model_fit_times'][model_name[model]]
    fit_accuracy = fit_summary['model_performance'][model_name[model]]
    
    # Clear memory
    del X, y, train_data, test_data, predictor, df 
    gc.collect()
    
    return fit_time, fit_accuracy

def run_benchmark_suite(n_samples, n_runs, to_csv):
    """Run benchmark suite for all models multiple times."""
    models = ['lr', 'rf', 'knn']
    results = {}
    
    print(f"Running {n_runs} times for each model with {n_samples} samples")
    
    for model in models:
        print(f"\nRunning {model.upper()} model {n_runs} times...")
        times = []
        accuracies = []
        
        for run in range(n_runs):
            fit_time, fit_accuracy = run_single_model(model, n_samples)
            times.append(fit_time)
            accuracies.append(fit_accuracy)
        
        # Store results with model-specific column names
        results[f'{model}_time'] = times
        results[f'{model}_accuracy'] = accuracies
    
    df = pd.DataFrame(results, columns=["lr_time", "lr_accuracy", "rf_time", "rf_accuracy", "knn_time", "knn_accuracy"])
    
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
            filename = f"autogluon_stats_{n_samples}_{to_csv}.csv"
        else:
            # If to_csv is True, use default filename
            filename = f"autogluon_stats_{n_samples}.csv"
        stats.to_csv(filename)
        print(f"\nSaved to {filename}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Run AutoGluon model benchmark')
    parser.add_argument('--model', choices=['lr', 'rf', 'knn'], 
                       help='Single model to run (optional)')
    parser.add_argument('--n_samples', type=int, default=10_000,
                       help='Number of samples (default: 10_000)')
    parser.add_argument('--suite', action='store_true',
                       help='Run benchmark suite (multiple runs for each model)')
    parser.add_argument('--to_csv', nargs='?', const=True, default=False,
                       help='Save results to CSV file. If a string is provided, it will be added to the filename.')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of runs for benchmark suite (default: 5)')
    args = parser.parse_args()
    
    if args.suite:
        # Run benchmark suite
        run_benchmark_suite(args.n_samples, n_runs=args.n_runs, to_csv=args.to_csv)
    elif args.model:
        # Run single model
        fit_time, fit_accuracy = run_single_model(args.model, args.n_samples)
        print(f"Model: {args.model.upper()}, Fit time: {fit_time:.4f} seconds, Accuracy: {fit_accuracy:.4f}, n_samples: {args.n_samples}")
    else:
        print("Please specify either --model for single run or --suite for benchmark suite")

if __name__ == "__main__":
    main()
