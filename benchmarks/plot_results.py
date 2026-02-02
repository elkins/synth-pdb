#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot multi-hardware synth-pdb Benchmark Results")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of input CSV files (e.g., machine1.csv machine2.csv)")
    parser.add_argument("--output", type=str, default="benchmarks/bench_plot_comparison.png", help="Output plot image")
    args = parser.parse_args()

    all_dfs = []
    for f in args.inputs:
        if not os.path.exists(f):
            print(f"Warning: {f} not found. Skipping.")
            continue
        df = pd.read_csv(f)
        # Create a unique hardware label
        df['hardware_label'] = f"{df['processor'][0]} ({df['cpu_threads'][0]}T, {df['ram_total_gb'][0]}GB)"
        all_dfs.append(df)

    if not all_dfs:
        print("Error: No valid input files found.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Set plot style
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", n_colors=df_all['hardware_label'].nunique())

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    fig.suptitle('Hardware Performance Comparison for synth-pdb', fontsize=16, y=1.02)

    # Plot 1: Throughput Comparison (Grouped Bar Chart)
    sns.barplot(x='N', y='throughput_batched', hue='hardware_label', data=df_all, ax=ax1, palette=palette)
    ax1.set_xlabel('Batch Size (N)')
    ax1.set_ylabel('Structures per Second (Batched)')
    ax1.set_title('Batched Generation Throughput')
    ax1.legend(title='Hardware')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9)

    # Plot 2: Speedup Scaling (Line Plot)
    sns.lineplot(x='N', y='speedup', hue='hardware_label', data=df_all, ax=ax2, marker='o', palette=palette)
    ax2.set_xlabel('Batch Size (N)')
    ax2.set_ylabel('Speedup Factor (Serial vs. Batched)')
    ax2.set_title('Relative Speedup Scaling')
    ax2.legend(title='Hardware')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Customizing the plot for clarity
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(args.output, dpi=150)
    print(f"✅ Comparison plot saved to {args.output}")

if __name__ == "__main__":
    main()
