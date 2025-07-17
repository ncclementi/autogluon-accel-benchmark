import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='Output CSV filename for leaderboard')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to generate')
    args = parser.parse_args()
    output_file = args.output
    n_samples = args.n_samples

    # Generate binary classification data
    X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Split into train/test
    train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
    label = "target"

    # Train AutoGluon predictor with default metric (log_loss for binary classification)
    predictor = TabularPredictor(label=label).fit(
        train_data=train_data, 
        hyperparameters={
        'LR': {}, 
        'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, 
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
        'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, 
           {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, 
           {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}]
        })

    # Print the leaderboard
    leaderboard = predictor.leaderboard()

    # Save leaderboard to CSV in leaderboard directory
    leaderboard.to_csv(output_file, index=False)



if __name__ == "__main__":
    main() 