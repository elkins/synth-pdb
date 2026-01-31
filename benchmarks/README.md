# ⏱️ Performance Benchmarks: Serial vs. Batched

This directory contains the benchmarking suite used to quantify the efficiency gains of the `synth-pdb` vectorized architecture.

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
