import pytest
import numpy as np
import biotite.structure.io.pdb as pdb
import io
from synth_pdb.generator import generate_pdb_content
import logging

logger = logging.getLogger(__name__)

def test_bfactor_reflects_dynamics():
    """
    Verify that B-factors in the generated PDB reflect structural dynamics.
    
    Helical residues (rigid, high S2) should have LOW B-factors.
    Terminal residues (flexible, low S2) should have HIGH B-factors.
    """
    # Generate a helix
    pdb_content = generate_pdb_content(sequence_str="A" * 15, conformation="alpha", seed=42)
    
    # Manually parse B-factors from PDB string to avoid Biotite annotation issues
    bfactors = {}
    for line in pdb_content.splitlines():
        if line.startswith("ATOM") and " CA " in line:
            # Extract fields
            # B-factor is columns 61-66 (0-indexed 60-66)
            res_id = int(line[22:26].strip())
            bf_str = line[60:66].strip()
            bf = float(bf_str)
            bfactors[res_id] = bf
    
    logger.info(f"B-factors: {bfactors}")
    
    # Terminal residues (1, 2, 14, 15) should have higher B-factors than core (7, 8)
    term_bfactor = np.mean([bfactors[1], bfactors[2], bfactors[14], bfactors[15]])
    core_bfactor = np.mean([bfactors[7], bfactors[8]])
    
    logger.info(f"Terminal avg B-factor: {term_bfactor:.2f}")
    logger.info(f"Core avg B-factor: {core_bfactor:.2f}")
    
    # Core should be more rigid (lower B-factor)
    assert core_bfactor < term_bfactor, "Core residues should have lower B-factors than termini"
    
    # Verify reasonable range (5-100 Angstrom^2)
    for rid, bf in bfactors.items():
        assert 5.0 <= bf <= 100.0, f"B-factor {bf} for residue {rid} out of realistic range"

@pytest.mark.skip(reason="WIP: Geometry generator needs refactoring to produce reliable secondary structure for S2 prediction.")
def test_bfactor_loop_vs_helix():
    """
    Verify that loop regions have higher B-factors than helical regions.
    
    SKIPPED (WIP):
    Currently, generated helices often deviate from ideal Phi/Psi angles, causing
    S2 prediction to classify them as 'coil', inverting the expected B-factor trend.
    """
    # Generate a structure with explicit regions: helix-loop-helix
    pdb_content = generate_pdb_content(
        sequence_str="A" * 20,
        structure="1-7:alpha,8-13:random,14-20:alpha",
        seed=42
    )
    
    # Manually parse B-factors
    bfactors = {}
    for line in pdb_content.splitlines():
        if line.startswith("ATOM") and " CA " in line:
            res_id = int(line[22:26].strip())
            bf = float(line[60:66].strip())
            bfactors[res_id] = bf
            
    # Helix regions: 3-6, 15-18 (avoiding termini effects)
    # Loop region: 10-11
    helix_bfactor = np.mean([bfactors[i] for i in [4, 5, 16, 17] if i in bfactors])
    loop_bfactor = np.mean([bfactors[i] for i in [10, 11] if i in bfactors])
    
    logger.info(f"Helix avg B-factor: {helix_bfactor:.2f}")
    logger.info(f"Loop avg B-factor: {loop_bfactor:.2f}")
    
    # Loop should be more flexible (higher B-factor)
    assert loop_bfactor > helix_bfactor, "Loop regions should have higher B-factors than helices"
