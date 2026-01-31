#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot synth-pdb Benchmark Results")
    parser.add_argument("--input", type=str, default="benchmarks/bench_results.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="benchmarks/bench_plot.png", help="Output plot image")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run performance_suite.py first.")
        return

    df = pd.read_csv(args.input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Throughput Comparison
    ax1.bar(df['N'].astype(str), df['throughput_serial'], label='Serial', color='#ff9999', alpha=0.8)
    ax1.bar(df['N'].astype(str), df['throughput_batched'], label='Batched (Vectorized)', color='#667eea', alpha=0.8)
    ax1.set_xlabel('Batch Size (N)')
    ax1.set_ylabel('Structures per Second')
    ax1.set_title('Throughput: Serial vs. Batched')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Speedup Scaling
    ax2.plot(df['N'], df['speedup'], marker='o', linestyle='-', linewidth=2, color='#48bb78')
    ax2.set_xlabel('Batch Size (N)')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.set_title('Relative Speedup Scaling')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels to points
    for i, txt in enumerate(df['speedup']):
        ax2.annotate(f"{txt:.1f}x", (df['N'][i], df['speedup'][i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"✅ Plot saved to {args.output}")

if __name__ == "__main__":
    main()
