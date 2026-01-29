
import logging
import numpy as np
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

def verify_quality():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("quality_check")
    
    # Use a challenging sequence (Helix + Beta + Turn) to stress test
    # Poly-A is too easy. Let's use the Zinc Finger motif sequence from the notebook
    sequence = "VKITVGGTLTVALGGALALALALALALAA"
    structure_def = "1-5:beta,6-8:random,9-13:beta,14-16:random,17-29:alpha"
    
    logger.info("--- Generating BASELINE Structure (Tol=10.0, MaxIter=0) ---")
    content_base = generate_pdb_content(
        sequence_str=sequence, 
        structure=structure_def,
        minimize_energy=True,
        minimization_k=10.0,
        minimization_max_iter=0
    )
    
    logger.info("--- Generating OPTIMIZED Structure (Tol=100.0, MaxIter=500) ---")
    content_opt = generate_pdb_content(
        sequence_str=sequence, 
        structure=structure_def,
        minimize_energy=True,
        minimization_k=100.0,
        minimization_max_iter=1000
    )
    
    # Helper to count violations
    def analyze(name, content):
        val = PDBValidator(content)
        val.validate_all()
        violations = val.get_violations()
        
        clashes = sum(1 for v in violations if "Clash" in v)
        bonds = sum(1 for v in violations if "Bond Length" in v)
        angles = sum(1 for v in violations if "Bond Angle" in v)
        rama = sum(1 for v in violations if "Ramachandran" in v)
        
        logger.info(f"Report for {name}:")
        logger.info(f"  Total Violations: {len(violations)}")
        logger.info(f"  - Clashes: {clashes}")
        logger.info(f"  - Bond Lengths: {bonds}")
        logger.info(f"  - Bond Angles: {angles}")
        logger.info(f"  - Ramachandran: {rama}")
        return len(violations)

    logger.info("\n=== COMPARISON RESULTS ===")
    v_base = analyze("BASELINE", content_base)
    v_opt = analyze("OPTIMIZED", content_opt)
    
    if v_opt > v_base:
        logger.warning(f"Optimized structure has {v_opt - v_base} MORE violations than baseline!")
    else:
        logger.info("Optimized structure quality is EQUAL or BETTER than baseline.")

if __name__ == "__main__":
    verify_quality()
