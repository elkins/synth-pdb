#!/usr/bin/env python3

import sys
import os
import numpy as np
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import EnergyMinimizer
import tempfile
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validate_energy")

def measure_energy(pdb_content, name):
    """
    Minimizes structure and returns time taken.
    Uses EnergyMinimizer class which requires file I/O.
    """
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as temp_in:
        temp_in.write(pdb_content)
        temp_in_path = temp_in.name
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_out:
        temp_out_path = temp_out.name
    
    minimizer = EnergyMinimizer(forcefield_name='amber14-all.xml')
    
    start_time = time.time()
    try:
        # Run minimization (ROBUST mode: Strips and re-adds hydrogens to ensure forcefield compatibility)
        success = minimizer.add_hydrogens_and_minimize(temp_in_path, temp_out_path)
        
        duration = time.time() - start_time
        if success:
            logger.info(f"[{name}] Minimization took {duration:.4f} seconds.")
            return duration, True
        else:
            logger.error(f"[{name}] Minimization failed inside OpenMM.")
            return 0, False
            
    except Exception as e:
        logger.error(f"[{name}] Minimization FAILED with exception: {e}")
        return 0, False
    finally:
        # Cleanup
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

def run_experiment():
    # Sequence: ALA-PRO-ALA-PRO-ALA-PRO 
    # High frequency of Pre-Pro constraints
    seq = "APAPAP"
    
    logger.info(f"Testing Sequence: {seq}")
    
    # 1. "Bad" Geometry (Old Behavior)
    # Force Alpha Helix for the Pre-Pro residues (A at 1, 3, 5)
    # Alpha phi/psi causes steric clash with next Proline
    # We use the region string: "1-1:alpha, 2-2:ppii, 3-3:alpha, 4-4:ppii, 5-5:alpha"
    # Proline likes PPII. Ala forced to Alpha.
    bad_structure_def = "1-1:alpha,3-3:alpha,5-5:alpha" 
    # (Residues 2,4,6 are Pro, let them be default/random or specific)
    
    logger.info("--- Generating 'Bad' Geometry (Forced Alpha Pre-Pro) ---")
    bad_pdb = generate_pdb_content(sequence_str=seq, structure=bad_structure_def, seed=42)
    bad_time, bad_success = measure_energy(bad_pdb, "BAD (Alpha)")
    
    # 2. "Good" Geometry (New Behavior)
    # Use random generation, which now defaults to PRE_PRO (Beta-like) for these positions.
    # To be sure, let's explicitly force Beta which is the "Good" state, or just let the generator do its newly implemented job.
    # Let's let the generator do it (Implicit check of feature).
    # Since we set PRE_PRO preference to 75% Beta, it should pick Beta mostly.
    
    logger.info("--- Generating 'Good' Geometry (Default/Pre-Pro Bias) ---")
    good_pdb = generate_pdb_content(sequence_str=seq, conformation='random', seed=42)
    good_time, good_success = measure_energy(good_pdb, "GOOD (Pre-Pro Bias)")
    
    print("\n=== RESULTS ===")
    print(f"Bad Geometry (Alpha) Time: {bad_time:.4f}s")
    print(f"Good Geometry (Bias) Time: {good_time:.4f}s")
    
    if good_time < bad_time:
        print(f"--> Improvement: {(bad_time - good_time)/bad_time * 100:.1f}% faster minimization!")
    else:
        print("--> No significant speedup detected (noise or minimal clash cost).")

if __name__ == "__main__":
    run_experiment()
