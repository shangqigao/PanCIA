import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def parse_filename(filename):
    """
    Parse the filename to extract strategy and other information.
    Expected format: radiopathomics_*_Strategy*_*_metrics.json
    """
    basename = os.path.basename(filename)
    # Remove _metrics.json suffix
    basename = basename.replace('_metrics.json', '')
    
    # Extract strategy - looking for "Strategy" followed by number
    parts = basename.split('_')
    omics = parts[0].capitalize()
    strategy = None
    for part in parts:
        if part.startswith('Strategy'):
            strategy = part
            break
    
    return omics, strategy

def load_metrics_from_files(folder_path, metric_name):
    """
    Load metrics from all JSON files in the folder recursively.
    """
    # Find all matching JSON files
    pattern = os.path.join(folder_path, '**', '*_Strategy*_*_metrics.json')
    json_files = glob.glob(pattern, recursive=True)
    
    if not json_files:
        print(f"No matching JSON files found in {folder_path}")
        return None
    
    data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            # Extract strategy from filename
            omics, strategy = parse_filename(file_path)
            
            if strategy is None:
                print(f"Could not extract strategy from: {file_path}")
                continue
            
            # Get cv_results
            cv_results = content.get('cv_results', {})
            
            # Extract metric values for each fold
            for fold_name, fold_metrics in cv_results.items():
                if metric_name in fold_metrics:
                    value = fold_metrics[metric_name]
                    data.append({
                        'Strategy': f"{omics}:{strategy}",
                        'Fold': fold_name,
                        'Metric': metric_name,
                        'Value': value
                    })
                else:
                    print(f"Warning: {metric_name} not found in {file_path}, fold {fold_name}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(data)

import re

def natural_sort_key(strategy):
    """
    Natural sorting key with custom prefix order:
    Radiomics:Strategy* first, then Pathomics:Strategy*, then Radiopathomics:Strategy*
    """
    # Define the order of prefixes
    prefix_order = {
        'Radiomics': 0,
        'Pathomics': 1,
        'Radiopathomics': 2
    }
    
    # Extract prefix and number
    if ':' in strategy:
        prefix, strategy_part = strategy.split(':', 1)
    else:
        prefix = ''
        strategy_part = strategy
    
    # Get prefix priority (default to a high number if not in dict)
    prefix_priority = prefix_order.get(prefix, 999)
    
    # Extract strategy number
    match = re.search(r'Strategy(\d+)', strategy_part)
    if match:
        number = int(match.group(1))
    else:
        # Try to extract any number
        match = re.search(r'(\d+)', strategy_part)
        if match:
            number = int(match.group(1))
        else:
            number = float('inf')
    
    # Return tuple: (prefix_priority, number)
    return (prefix_priority, number)

def create_boxplot(df, metric_name, output_path=None):
    """
    Create box plot showing metric values across strategies with natural ordering.
    """
    if df is None or df.empty:
        print("No data to plot")
        return
    
    # Sort strategies naturally (Strategy1, Strategy2, ..., Strategy10, etc.)
    strategies = sorted(df['Strategy'].unique(), key=natural_sort_key)
    
    # Convert Strategy to categorical with the sorted order
    df_sorted = df.copy()
    df_sorted['Strategy'] = pd.Categorical(df_sorted['Strategy'], 
                                           categories=strategies, 
                                           ordered=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create box plot with points
    ax = sns.boxplot(data=df_sorted, x='Strategy', y='Value', palette='Set3')
    
    # Add individual points to show actual values
    sns.stripplot(data=df_sorted, x='Strategy', y='Value', 
                  size=4, color='darkblue', alpha=0.6, ax=ax)
    
    # Customize the plot
    plt.title(f'Distribution of {metric_name} Across Strategies', fontsize=16, fontweight='bold')
    plt.xlabel('Strategy', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return ax

def print_summary_statistics(df):
    """
    Print summary statistics for each strategy.
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for strategy in df['Strategy'].unique():
        strategy_data = df[df['Strategy'] == strategy]['Value']
        print(f"\nStrategy: {strategy}")
        print(f"  Count: {len(strategy_data)}")
        print(f"  Mean: {strategy_data.mean():.4f}")
        print(f"  Std: {strategy_data.std():.4f}")
        print(f"  Min: {strategy_data.min():.4f}")
        print(f"  Max: {strategy_data.max():.4f}")
        print(f"  Median: {strategy_data.median():.4f}")
        print(f"  Q1: {strategy_data.quantile(0.25):.4f}")
        print(f"  Q3: {strategy_data.quantile(0.75):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Create box plots for metrics across strategies')
    parser.add_argument('--folder_path', type=str, 
                        default='/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_strategies/TCGA_survival_OS',
                        help='Path to the folder containing JSON files')
    parser.add_argument('--metric', type=str, 
                        default="C-index",
                        help='Metric to plot (e.g., "C-index", "C-index-IPCW", "Mean AUC", "IBS")')
    parser.add_argument('--output', type=str, default="/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Survival_ImmuneSubtype/TCGA_survival_OS", 
                       help='Output path for the plot (optional)')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show summary statistics for each strategy')
    
    args = parser.parse_args()
    
    for metric in ["C-index", "C-index-IPCW", "Mean AUC"]:
        args.metric = metric
        # Check if folder exists
        if not os.path.exists(args.folder_path):
            print(f"Error: Folder '{args.folder_path}' does not exist")
            return
        
        # Load data
        print(f"Loading data from {args.folder_path}...")
        df = load_metrics_from_files(args.folder_path, args.metric)
        
        if df is None or df.empty:
            print("No data loaded. Please check the folder path and metric name.")
            return
        
        print(f"Loaded {len(df)} data points from {len(df['Strategy'].unique())} strategies")
        print(f"Strategies found: {', '.join(df['Strategy'].unique())}")
        
        # Show summary statistics if requested
        if args.show_stats:
            print_summary_statistics(df)
        
        # Create box plot
        output_path = f"{args.output}/strategies_comparison_{args.metric}.png"
        print(f"\nCreating box plot for {args.metric}...")
        create_boxplot(df, args.metric, output_path)
        
        # Print basic info
        print("\n" + "="*60)
        print("PLOT INFORMATION")
        print("="*60)
        print(f"Metric: {args.metric}")
        print(f"Number of strategies: {len(df['Strategy'].unique())}")
        print(f"Total data points: {len(df)}")
        print(f"Folds per strategy: {df.groupby('Strategy').size().iloc[0] if len(df['Strategy'].unique()) > 0 else 0}")

if __name__ == "__main__":
    main()