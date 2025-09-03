#!/usr/bin/env python3
"""
Sklearn Benchmark Analysis Script

This script processes CSV files from sklearn_results directory
to create comparison dataframes showing CPU vs GPU performance with speedup calculations.
"""

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def load_csv_data(file_path):
    """Load CSV data and return as DataFrame"""
    try:
        df = pd.read_csv(file_path, index_col=0)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_sample_size_from_filename(filename):
    """Extract sample size from filename"""
    # Extract number from filename like "sklearn_stats_100000_cpu.csv"
    parts = filename.split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None

def create_comparison_dataframe_sklearn(sklearn_cpu_file, sklearn_gpu_file, sample_size):
    """
    Create a comparison dataframe for a specific sample size (sklearn only)
    """
    # Load sklearn CSV files
    sklearn_cpu_df = load_csv_data(sklearn_cpu_file)
    sklearn_gpu_df = load_csv_data(sklearn_gpu_file)
    
    # Get median values (assuming we want to compare median performance)
    sklearn_cpu_medians = sklearn_cpu_df.loc['median']
    sklearn_gpu_medians = sklearn_gpu_df.loc['median']
    
    # Create comparison dataframe
    comparison_data = []
    
    # Process sklearn models
    for model in sklearn_cpu_medians.index:
        cpu_time = sklearn_cpu_medians[model]
        gpu_time = sklearn_gpu_medians[model]
        speedup = cpu_time / gpu_time 
        
        comparison_data.append({
            'Model': model,
            'CPU_Time': cpu_time,
            'cuml_accel_Time': gpu_time,
            'Speedup': speedup
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df

def create_comparison_dataframe_autogluon(autogluon_cpu_file, autogluon_cuml_file, autogluon_cuml_cudf_file, sample_size):
    """
    Create a comparison dataframe for AutoGluon results with CPU, cuml_accel, and cuml_cudf timings
    """
    # Load AutoGluon CSV files
    autogluon_cpu_df = load_csv_data(autogluon_cpu_file)
    autogluon_cuml_df = load_csv_data(autogluon_cuml_file)
    autogluon_cuml_cudf_df = load_csv_data(autogluon_cuml_cudf_file)
    
    if any(df is None for df in [autogluon_cpu_df, autogluon_cuml_df, autogluon_cuml_cudf_df]):
        print(f"Failed to load AutoGluon files for sample size {sample_size}")
        return None
    
    # Get median values for time columns
    autogluon_cpu_medians = autogluon_cpu_df.loc['median']
    autogluon_cuml_medians = autogluon_cuml_df.loc['median']
    autogluon_cuml_cudf_medians = autogluon_cuml_cudf_df.loc['median']
    
    # Extract time columns (ending with '_time')
    cpu_time_cols = [col for col in autogluon_cpu_medians.index if col.endswith('_time')]
    
    # Create comparison dataframe
    comparison_data = []
    
    # Process AutoGluon models
    for time_col in cpu_time_cols:
        model_name = time_col.replace('_time', '')
        cpu_time = autogluon_cpu_medians[time_col]
        cuml_time = autogluon_cuml_medians[time_col]
        cuml_cudf_time = autogluon_cuml_cudf_medians[time_col]
        
        speedup_cuml = cpu_time / cuml_time 
        speedup_cuml_cudf = cpu_time / cuml_cudf_time
        
        comparison_data.append({
            'Model': f'autogluon_{model_name}',
            'CPU_Time': cpu_time,
            'cuml_accel_Time': cuml_time,
            'Speedup_CPU_vs_cuml': speedup_cuml,
            'cuml_cudf_Time': cuml_cudf_time,
            'Speedup_CPU_vs_cuml_cudf': speedup_cuml_cudf
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df

def create_accuracy_dataframe_autogluon(autogluon_cpu_file, autogluon_cuml_file, autogluon_cuml_cudf_file, sample_size):
    """
    Create an accuracy comparison dataframe for AutoGluon results
    """
    # Load AutoGluon CSV files
    autogluon_cpu_df = load_csv_data(autogluon_cpu_file)
    autogluon_cuml_df = load_csv_data(autogluon_cuml_file)
    autogluon_cuml_cudf_df = load_csv_data(autogluon_cuml_cudf_file)
    
    
    # Get median values for accuracy columns
    autogluon_cpu_medians = autogluon_cpu_df.loc['median']
    autogluon_cuml_medians = autogluon_cuml_df.loc['median']
    autogluon_cuml_cudf_medians = autogluon_cuml_cudf_df.loc['median']
    
    # Extract accuracy columns (ending with '_accuracy')
    cpu_accuracy_cols = [col for col in autogluon_cpu_medians.index if col.endswith('_accuracy')]
    
    # Create accuracy comparison dataframe
    accuracy_data = []
    
    # Process AutoGluon models
    for accuracy_col in cpu_accuracy_cols:
        model_name = accuracy_col.replace('_accuracy', '')
        cpu_accuracy = autogluon_cpu_medians[accuracy_col]
        cuml_accuracy = autogluon_cuml_medians[accuracy_col]
        cuml_cudf_accuracy = autogluon_cuml_cudf_medians[accuracy_col]
        
        accuracy_data.append({
            'Model': f'autogluon_{model_name}',
            'CPU_Accuracy': cpu_accuracy,
            'cuml_Accuracy': cuml_accuracy,
            'cuml_cudf_Accuracy': cuml_cudf_accuracy
        })
    
    # Create DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df = accuracy_df.set_index('Model')
    
    return accuracy_df

def create_markdown_report(comparison_dataframes_sklearn, comparison_dataframes_autogluon, accuracy_dataframes_autogluon, sample_sizes, autogluon_plot_file=None, sklearn_plot_file=None):
    """
    Create a simple markdown report with dataframe results
    """ 
       
    # Sample size display mapping
    size_display_map = {
        100000: "100K",
        1000000: "1M", 
        5000000: "5M"
    }
    
    # Create markdown content
    markdown_content = []
    
    # Header
    markdown_content.append("# Benchmark Analysis Report")
    markdown_content.append("")
    
    # Methodology note
    markdown_content.append("## Methodology")
    markdown_content.append("- All reported values are **median of 5 runs**")
    markdown_content.append("- Speedup calculations: CPU_time / GPU_time")
    markdown_content.append("- Sample sizes: 100K, 1M, 5M data points")
    markdown_content.append("- **GPU runs:** we simply add `-m cuml.accel` to the script command. For the `cuml_cudf` case, also add `-m cudf.pandas`.")
    markdown_content.append("")
    
    # Performance plots (only if files are provided)
    if autogluon_plot_file or sklearn_plot_file:
        markdown_content.append("## Performance Comparison Plots")
        markdown_content.append("")
        
        if autogluon_plot_file:
            markdown_content.append("### AutoGluon Performance")
            markdown_content.append(f"![AutoGluon Performance]({autogluon_plot_file})")
            markdown_content.append("")
        
        if sklearn_plot_file:
            markdown_content.append("### Sklearn Performance")
            markdown_content.append(f"![Sklearn Performance]({sklearn_plot_file})")
            markdown_content.append("")
    
    markdown_content.append("")
    markdown_content.append("## Tabular Results (Raw Data)")
    markdown_content.append("""Below are the detailed benchmark results for each sample size and method. These tables present the raw median 
                            timing and accuracy values, as well as calculated speedups, for each model and implementation.""")
    markdown_content.append("")

    # Results for each sample size
    for sample_size in sample_sizes:

        markdown_content.append(f"## Sample Size: {size_display_map[sample_size]}")
        markdown_content.append("")
        
        # Sklearn Results
        markdown_content.append("### Sklearn Results:")
        markdown_content.append(comparison_dataframes_sklearn[sample_size].round(4).to_markdown())
        markdown_content.append("")
        
        # AutoGluon Results
        markdown_content.append("### AutoGluon Results:")
        markdown_content.append(comparison_dataframes_autogluon[sample_size].round(4).to_markdown())
        markdown_content.append("")
        
        # AutoGluon Accuracy Results
        markdown_content.append("### AutoGluon Accuracy Results:")
        markdown_content.append(accuracy_dataframes_autogluon[sample_size].round(4).to_markdown())
        markdown_content.append("")
        
        markdown_content.append("---")
        markdown_content.append("")
    
    # Write to file
    report_filename = "benchmark_report_final.md"
    with open(report_filename, 'w') as f:
        f.write('\n'.join(markdown_content))
    
    print(f"Markdown report generated: {report_filename}")

def create_autogluon_performance_plot(comparison_dataframes_autogluon, sample_sizes, filename='autogluon_performance_comparison.png'):
    """
    Create grouped bar charts showing AutoGluon model performance comparison
    with subplots for each model type (RF, KNN, LR) across all sample sizes in one figure
    """
    # Sample size display mapping
    size_display_map = {
        100000: "100K",
        1000000: "1M", 
        5000000: "5M"
    }
    
    # Define models to plot (LR, RF, KNN)
    target_models = ['autogluon_lr', 'autogluon_rf', 'autogluon_knn']
    
    # Colors for the three implementations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    implementations = ['CPU', 'cuml', 'cuml_cudf']
    
    # Create one figure with subplots for each model type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # If only one model, make axes iterable
    if len(target_models) == 1:
        axes = [axes]
    
    # Create a subplot for each model type
    for idx, model in enumerate(target_models):
        model_name = model.replace('autogluon_', '').upper()
        ax = axes[idx]
        
        # Collect data for this model across all sample sizes
        model_data = []
        sample_labels = []
        
        for sample_size in sample_sizes:
            if sample_size in comparison_dataframes_autogluon:
                df = comparison_dataframes_autogluon[sample_size]
                if model in df.index:
                    model_data.append({
                        'CPU': df.loc[model, 'CPU_Time'],
                        'cuml': df.loc[model, 'cuml_accel_Time'],
                        'cuml_cudf': df.loc[model, 'cuml_cudf_Time']
                    })
                    sample_labels.append(size_display_map[sample_size])
        
        if not model_data:
            ax.text(0.5, 0.5, f'No data found for {model_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Set up bar positions
        x = np.arange(len(sample_labels))
        width = 0.25
        
        # Plot bars for each implementation
        for i, impl in enumerate(implementations):
            values = [data[impl] for data in model_data]
            bars = ax.bar(x + i * width, values, width, label=impl, color=colors[i], alpha=0.8)
            
            # Add speedup annotations on top of cuml and cuml_cudf bars
            if impl in ['cuml', 'cuml_cudf']:
                for j, (bar, data) in enumerate(zip(bars, model_data)):
                    cpu_time = data['CPU']
                    gpu_time = data[impl]
                    speedup = cpu_time / gpu_time
                    
                    # Position annotation on top of the bar
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize subplot
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'{model_name} Performance')
        ax.set_xticks(x + width)
        ax.set_xticklabels(sample_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add main title and adjust layout
    fig.suptitle('AutoGluon times: CPU vs cuml vs cuml_cudf', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"AutoGluon performance comparison plot saved as: {filename}")
    return filename

def create_sklearn_performance_plot(comparison_dataframes_sklearn, sample_sizes, filename='sklearn_performance_comparison.png'):
    """
    Create grouped bar charts showing sklearn model performance comparison
    """
    size_display_map = {100000: "100K", 1000000: "1M", 5000000: "5M"}
    colors = ['#1f77b4', '#ff7f0e']
    implementations = ['CPU', 'cuml_accel']
    
    # Get models from first sample size
    first_sample_size = list(comparison_dataframes_sklearn.keys())[0]
    all_models = comparison_dataframes_sklearn[first_sample_size].index.tolist()
    
    # Create figure
    fig, axes = plt.subplots(1, len(all_models), figsize=(6 * len(all_models), 6))
    if len(all_models) == 1:
        axes = [axes]
    
    for idx, model in enumerate(all_models):
        ax = axes[idx]
        
        # Collect data
        model_data = []
        sample_labels = []
        for sample_size in sample_sizes:
            df = comparison_dataframes_sklearn[sample_size]
            model_data.append({
                'CPU': df.loc[model, 'CPU_Time'],
                'cuml_accel': df.loc[model, 'cuml_accel_Time']
            })
            sample_labels.append(size_display_map[sample_size])
        
        # Plot bars
        x = np.arange(len(sample_labels))
        width = 0.35
        
        for i, impl in enumerate(implementations):
            values = [data[impl] for data in model_data]
            bars = ax.bar(x + i * width, values, width, label=impl, color=colors[i], alpha=0.8)
            
            # Add speedup annotations
            if impl == 'cuml_accel':
                for bar, data in zip(bars, model_data):
                    speedup = data['CPU'] / data[impl]
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'{model.upper()} Performance')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(sample_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add main title and adjust layout
    fig.suptitle('Sklearn times: CPU vs cuml_accel', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sklearn performance comparison plot saved as: {filename}")
    return filename

def main():
    """Main function to process CSV files and create comparison dataframes"""
    
    # Define directories
    sklearn_dir = "sklearn_results"
    autogluon_dir = "autogluon_results"
    
    # Get all sample sizes from files
    sklearn_files = glob.glob(os.path.join(sklearn_dir, "sklearn_stats_*_cpu.csv"))
    sample_sizes = []
    
    for file in sklearn_files:
        sample_size = extract_sample_size_from_filename(os.path.basename(file))
        if sample_size:
            sample_sizes.append(sample_size)
    
    sample_sizes.sort()
    
    print(f"Found sample sizes: {sample_sizes}")
    
    # Create separate comparison dataframes for each sample size and library
    comparison_dataframes_sklearn = {}
    comparison_dataframes_autogluon = {}
    accuracy_dataframes_autogluon = {}
    
    for sample_size in sample_sizes:        
        # Define file paths
        sklearn_cpu_file = os.path.join(sklearn_dir, f"sklearn_stats_{sample_size}_cpu.csv")
        sklearn_gpu_file = os.path.join(sklearn_dir, f"sklearn_stats_{sample_size}_cuml_acc.csv")
        autogluon_cpu_file = os.path.join(autogluon_dir, f"autogluon_stats_{sample_size}_cpu.csv")
        autogluon_cuml_file = os.path.join(autogluon_dir, f"autogluon_stats_{sample_size}_cuml_acc.csv")
        autogluon_cuml_cudf_file = os.path.join(autogluon_dir, f"autogluon_stats_{sample_size}_cuml_cudf_acc.csv")
        
        # Create sklearn comparison dataframe
        sklearn_df = create_comparison_dataframe_sklearn(
            sklearn_cpu_file, sklearn_gpu_file, sample_size
        )

                # Create autogluon comparison dataframe
        autogluon_df = create_comparison_dataframe_autogluon(
            autogluon_cpu_file, autogluon_cuml_file, autogluon_cuml_cudf_file, sample_size
        )
        
        # Create autogluon accuracy dataframe
        autogluon_accuracy_df = create_accuracy_dataframe_autogluon(
            autogluon_cpu_file, autogluon_cuml_file, autogluon_cuml_cudf_file, sample_size
        )
        
        if sklearn_df is not None:
            comparison_dataframes_sklearn[sample_size] = sklearn_df  
        
        if autogluon_df is not None:
            comparison_dataframes_autogluon[sample_size] = autogluon_df
            
        if autogluon_accuracy_df is not None:
            accuracy_dataframes_autogluon[sample_size] = autogluon_accuracy_df

    
    # Create performance plots first
    autogluon_plot_file = 'autogluon_performance_comparison.png'
    sklearn_plot_file = 'sklearn_performance_comparison.png'
    
    create_autogluon_performance_plot(comparison_dataframes_autogluon, sample_sizes, autogluon_plot_file)
    create_sklearn_performance_plot(comparison_dataframes_sklearn, sample_sizes, sklearn_plot_file)
    
    # Create markdown report (after plots are created)
    create_markdown_report(comparison_dataframes_sklearn, 
                           comparison_dataframes_autogluon, 
                           accuracy_dataframes_autogluon, 
                           sample_sizes,
                           autogluon_plot_file,
                           sklearn_plot_file) 

if __name__ == "__main__":
    main()
