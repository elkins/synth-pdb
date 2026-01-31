
import numpy as np
import time
import sys
from synth_pdb.geometry import position_atom_3d_from_internal_coords
from synth_pdb.relaxation import spectral_density

def benchmark_nerf():
    print("\n--- Benchmarking NeRF Engine ---")
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.5, 0.0, 0.0])
    p3 = np.array([2.5, 1.2, 0.0])
    
    # Warm up
    _ = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0, 180.0)
    
    n_iter = 100000
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = position_atom_3d_from_internal_coords(p1, p2, p3, 1.5, 110.0, 180.0)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_iter * 1e6 # microseconds
    print(f"Average time per call ({n_iter} iterations): {avg_time:.3f} us")
    return avg_time

def benchmark_spectral_density():
    print("\n--- Benchmarking Spectral Density ---")
    # Warm up
    _ = spectral_density(0.0, 1e-8, 0.85)
    
    n_iter = 100000
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = spectral_density(0.0, 1e-8, 0.85)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_iter * 1e6 # microseconds
    print(f"Average time per call ({n_iter} iterations): {avg_time:.3f} us")
    return avg_time

if __name__ == "__main__":
    try:
        import numba
        print(f"Numba version: {numba.__version__}")
    except ImportError:
        print("Numba NOT found. Running baseline performance.")
        
    benchmark_nerf()
    benchmark_spectral_density()
