import pytest
import io
import numpy as np
import biotite.structure.io.pdb as pdb_io
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

def test_zinc_finger_generation():
    """Test that a Zinc Finger motif generates a structure with a ZN ion."""
    # Classical Zinc-finger motif CCHH: CPYCKKRFHSH (standard sequences usually have H too)
    sequence = "CPYCKKRFHSH"
    
    # Generate with mental ions auto (default) and minimization
    pdb_content = generate_pdb_content(
        sequence_str=sequence, 
        minimize_energy=True, 
        metal_ions="auto"
    )
    
    # 1. Verify ZN is in the PDB
    assert "HETATM" in pdb_content
    assert "ZN" in pdb_content
    
    # 2. Verify PDBValidator preserves HETATM
    validator = PDBValidator(pdb_content)
    atoms = validator.get_atoms()
    
    zn_atoms = [a for a in atoms if a["element"].upper() == "ZN"]
    assert len(zn_atoms) == 1
    assert zn_atoms[0]["record_name"] == "HETATM"
    
    # 3. Verify ZN is not stripped after reconstruction
    reconstructed = validator.atoms_to_pdb_content(atoms)
    assert "HETATM" in reconstructed
    assert "ZN" in reconstructed

def test_zinc_coordination_restraints_detection():
    """Test that ligands are detected and restrained correctly."""
    # We check if the physics.py logic actually runs without crashing
    # using a known motif.
    sequence = "CPYCKKRFHSH"
    
    # Generate but only verify it doesn't crash
    # (The internal logs would show "Applying harmonic coordination constraints")
    try:
        pdb_content = generate_pdb_content(
            sequence_str=sequence, 
            minimize_energy=True, 
            metal_ions="auto"
        )
    except Exception as e:
        pytest.fail(f"Zinc finger generation crashed: {e}")

def test_no_metal_ions_flag():
    """Test that --metal-ions none actually prevents ion injection."""
    sequence = "CPYCKKRFHSH"
    pdb_content = generate_pdb_content(
        sequence_str=sequence, 
        minimize_energy=True, 
        metal_ions="none"
    )
    
    assert "ZN" not in pdb_content
    # We check for ZN specifically. HETATM records might exist for titratable HIS (HID/HIE),
    # so we don't do a broad "HETATM not in pdb_content" check here anymore.

def test_validator_hetatm_type_preservation():
    """Strict test for HETATM vs ATOM record type in Validator."""
    sample_pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "HETATM    2 ZN    ZN A   2       5.000   5.000   5.000  1.00  0.00          Zn  \n"
    )
    
    validator = PDBValidator(sample_pdb)
    atoms = validator.get_atoms()
    
    assert atoms[0]["record_name"] == "ATOM"
    assert atoms[1]["record_name"] == "HETATM"
    
    reconstructed = validator.atoms_to_pdb_content(atoms)
    lines = reconstructed.strip().splitlines()
    assert lines[0].startswith("ATOM")
    assert lines[1].startswith("HETATM")
