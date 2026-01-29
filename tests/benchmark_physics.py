
import time
import logging
from synth_pdb.generator import generate_pdb_content

def run_benchmark():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("benchmark")
    
    length = 100
    sequence = "A" * length # Poly-Alanine
    
    logger.info(f" Benchmarking generation of {length}-residue peptide (Minimization ON)...")
    
    start_time = time.time()
    pdb_content = generate_pdb_content(
        sequence_str=sequence,
        minimize_energy=True, # This triggers the physics engine
    )
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f" Generation took {duration:.4f} seconds (Baseline).")
    
    # 2. Optimized Run
    logger.info(f" Benchmarking Optimized Generation (Tol=100.0, MaxIter=500)...")
    start_time = time.time()
    pdb_content_opt = generate_pdb_content(
        sequence_str=sequence,
        minimize_energy=True,
        minimization_k=100.0,
        minimization_max_iter=500
    )
    end_time = time.time()
    duration_opt = end_time - start_time
    logger.info(f" Optimized Generation took {duration_opt:.4f} seconds.")
    logger.info(f" Speedup: {duration/duration_opt:.2f}x")
    
    if len(pdb_content) < 100:
        logger.error("Generated content seems too short!")

if __name__ == "__main__":
    run_benchmark()
