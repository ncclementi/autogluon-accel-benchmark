import pandas as pd
import os
import glob

def generate_simple_report(results_dir, output_file):
    """Generate a simple markdown report with tables for each sample size"""
    csv_files = glob.glob(os.path.join(results_dir, 'comparison_results_*.csv'))
    
    if not csv_files:
        print("No comparison results found!")
        return
    
    # Define the desired order
    desired_order = ['1K', '10K', '100K', '1M', '10M']
    
    # Create a mapping of sample size to file path
    file_mapping = {}
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        sample_size_name = df['sample_size_name'].iloc[0]
        file_mapping[sample_size_name] = file_path
    
    report_lines = []
    report_lines.append("# AutoGluon sklearn vs cuml.accel Benchmark Results")
    report_lines.append("")
    
    for sample_size in desired_order:
        if sample_size in file_mapping:
            file_path = file_mapping[sample_size]
        df = pd.read_csv(file_path)
        sample_size_name = df['sample_size_name'].iloc[0]
        n_samples = df['n_samples'].iloc[0]
        
        # Add title
        report_lines.append(f"## {sample_size_name} Samples ({n_samples:,})")
        report_lines.append("")
        
        # Convert dataframe to markdown table
        table_md = df.to_markdown(index=False)
        report_lines.append(table_md)
        report_lines.append("")
    
    # Write report to file
    report_content = "\n".join(report_lines)
    
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated successfully: {output_file}")

def main():
    """Main function"""
    results_dir = 'comparison-results'
    output_file = 'benchmark_report.md'
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    generate_simple_report(results_dir, output_file)

if __name__ == "__main__":
    main() 