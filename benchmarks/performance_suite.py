#!/usr/bin/env python3
import time
import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np

import psutil
import platform

# Ensure local synth_pdb is prioritized
current_path = Path(__file__).resolve().parent
repo_root = current_path.parent
if (repo_root / "synth_pdb").exists():
    sys.path.insert(0, str(repo_root))

from synth_pdb.generator import generate_pdb_content
from synth_pdb.batch_generator import BatchedGenerator

def get_hardware_info():
    """Gathers detailed hardware and platform information."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": platform.processor(), # Fallback, specific brand might need other methods
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

def benchmark_serial(sequence, n_samples):
    print(f"  Running Serial Benchmark (N={n_samples})...")
    start_time = time.time()
    for _ in range(n_samples):
        _ = generate_pdb_content(sequence_str=sequence, minimize_energy=False)
    elapsed = time.time() - start_time
    return elapsed

def benchmark_batched(sequence, n_samples):
    print(f"  Running Batched Benchmark (N={n_samples})...")
    # Warm up
    bg_warm = BatchedGenerator(sequence, n_batch=10, full_atom=False)
    _ = bg_warm.generate_batch()
    
    start_time = time.time()
    bg = BatchedGenerator(sequence, n_batch=n_samples, full_atom=False)
    _ = bg.generate_batch()
    elapsed = time.time() - start_time
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="synth-pdb Performance Benchmarking Suite")
    parser.add_argument("--n", type=int, default=1000, help="Number of structures to generate (default: 1000)")
    parser.add_argument("--seq", type=str, default="L-K-E-L-E-K-E-L-E-K-E-L-E-K-E-L", help="Sequence to use")
    parser.add_argument("--output", type=str, default="benchmarks/bench_results.csv", help="Output CSV file")
    args = parser.parse_args()

    # Create directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    hardware_info = get_hardware_info()
    print("🚀 Hardware Information:")
    for key, value in hardware_info.items():
        print(f"  - {key}: {value}")

    test_sizes = [10, 100, 500, args.n]
    results = []

    print(f"\n🚀 Starting Benchmarks (Sequence Length: {len(args.seq.split('-'))})")
    
    for n in test_sizes:
        print(f"\nTesting N={n}")
        serial_time = benchmark_serial(args.seq, n)
        batched_time = benchmark_batched(args.seq, n)
        
        speedup = serial_time / batched_time
        throughput_serial = n / serial_time
        throughput_batched = n / batched_time
        
        print(f"  Serial:  {serial_time:.3f}s ({throughput_serial:.1f} struct/sec)")
        print(f"  Batched: {batched_time:.3f}s ({throughput_batched:.1f} struct/sec)")
        print(f"  💪 Speedup: {speedup:.1f}x")
        
        result_row = {
            "N": n,
            "serial_time": serial_time,
            "batched_time": batched_time,
            "speedup": speedup,
            "throughput_serial": throughput_serial,
            "throughput_batched": throughput_batched
        }
        result_row.update(hardware_info)
        results.append(result_row)

    # Save results
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
