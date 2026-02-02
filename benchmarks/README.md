# Hardware Performance Benchmarking

This suite provides tools to measure and compare the performance of `synth-pdb` across different hardware platforms.

## Objective
The primary goal is to quantify the efficiency gains from the "GPU-First" (vectorized/batched) architecture compared to traditional serial generation. By running this suite on multiple machines (e.g., a developer's Intel-based laptop vs. a CI server's AMD CPU vs. an Apple Silicon Mac), we can produce data-driven evidence of its performance benefits.

## How to Run a Comparison

### Step 1: Run the Benchmark on Each Machine
On each target machine, run the `performance_suite.py` script. It's recommended to use a consistent sequence and number of iterations (`--n`) for a fair comparison.

```bash
# On Machine A (e.g., your local laptop)
python benchmarks/performance_suite.py --output benchmarks/results_machine_a.csv

# On Machine B (e.g., a remote server)
python benchmarks/performance_suite.py --output benchmarks/results_machine_b.csv
```

This script will generate a CSV file (e.g., `results_machine_a.csv`) containing performance metrics and detailed hardware information for that specific machine.

### Step 2: Consolidate the Results
Gather all the generated CSV files from the different machines and place them in the `benchmarks/` directory of your primary analysis machine.

### Step 3: Generate the Comparison Plot
Once all CSV files are in place, run the `plot_results.py` script, providing the paths to all the CSV files you want to compare.

```bash
# Example comparing two machines
python benchmarks/plot_results.py \
  --inputs benchmarks/results_machine_a.csv benchmarks/results_machine_b.csv \
  --output benchmarks/hardware_comparison.png
```

This will produce a PNG image (`hardware_comparison.png`) containing two plots:
1.  **Batched Throughput**: A bar chart comparing the number of structures generated per second for each machine at different batch sizes.
2.  **Speedup Scaling**: A line chart showing how the speedup (Batched vs. Serial) scales with batch size on each machine.

This provides a clear, visual summary of how `synth-pdb`'s performance varies across different hardware environments.


## 🚀 Key Results
On this hardware configuration (Apple Silicon), we observe a **Speedup factor of ~2,000x** when generating 500 structures using the `BatchedGenerator` compared to a traditional serial loop.

| Batch Size (N) | Serial Time (s) | Batched Time (s) | Speedup | Throughput (struct/sec) |
| :--- | :--- | :--- | :--- | :--- |
| 10 | 1.89 | 0.002 | 886x | 4,682 |
| 100 | 1.66 | 0.003 | 632x | 38,109 |
| 500 | 8.25 | 0.004 | 2,004x | 121,475 |

## 📊 Performance Visualization
![Hardware Benchmark Plot](benchmark_results.png)

## 🧪 How to Reproduce
Run the suite using the provided scripts:
```bash
python benchmarks/performance_suite.py --n 1000
python benchmarks/plot_results.py
```

## 🧠 Why it matters
AI models like AlphaFold and ESM-Fold require millions of training examples. Parsing individual PDB files is historically the bottleneck. By utilizing **Numba-optimized vectorization**, `synth-pdb` transforms PDB generation from a serial parsing task into a high-speed matrix operation, allowing researchers to generate data at "tensor-speed".
