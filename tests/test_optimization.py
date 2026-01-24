import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.scoring import calculate_clash_score
from synth_pdb.packing import optimize_sidechains, SideChainPacker
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator
import logging
logger = logging.getLogger(__name__)

def test_clash_score_calculation():
    # Create two atoms very close together
    # Use non-adjacent residues to avoid backbone-backbone exclusion logic
    atoms = struc.array([
        struc.Atom([0,0,0], atom_name="CA", element="C", res_id=1),
        struc.Atom([0.5,0,0], atom_name="CA", element="C", res_id=3)
    ])
    score = calculate_clash_score(atoms)
    assert score > 0, "Clash score should be positive for overlapping atoms"
    
    # Create atoms far apart
    atoms_far = struc.array([
        struc.Atom([0,0,0], atom_name="CA", element="C", res_id=1),
        struc.Atom([10,0,0], atom_name="CA", element="C", res_id=3)
    ])
    score_far = calculate_clash_score(atoms_far)
    assert score_far == 0, "Clash score should be zero for distant atoms"

def test_optimization_improves_score():
    # Generate a peptide with likely clashes (e.g. bulky residues close together)
    # Using a sequence that might be crowded
    
    # Generate random structure
    # Use non-optimized structure first
    pdb_content = generate_pdb_content(sequence_str="WFWFW", optimize_sidechains=False)
    
    # Parse back using biotite to get proper AtomArray
    import biotite.structure.io.pdb as pdb
    import io
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    initial_score = calculate_clash_score(peptide)
    logger.info(f"Initial Score: {initial_score}")
    
    # Run optimizer
    optimized_peptide = optimize_sidechains(peptide, steps=200)
    final_score = calculate_clash_score(optimized_peptide)
    logger.info(f"Final Score: {final_score}")
    
    # It should not get worse
    assert final_score <= initial_score + 1e-6, "Optimization should not worsen the score significantly"
    
    # Ideally it improves, but if initial was already good (0.0), it stays 0.0
    if initial_score > 0.1:
        assert final_score < initial_score, "Optimization should improve score if clashes exist"

def test_packer_class():
    packer = SideChainPacker(steps=10)
    assert packer.steps == 10
