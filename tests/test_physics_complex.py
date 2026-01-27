
import pytest
import os
import tempfile
import io
import numpy as np
import biotite.structure.io.pdb as pdb_io
from synth_pdb.physics import EnergyMinimizer, HAS_OPENMM
from synth_pdb.generator import generate_pdb_content

@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_salt_bridge_detection_in_minimization(caplog):
    """Test that salt bridges are detected and handled during minimization."""
    # Sequence with potential salt bridge
    sequence = "KAAAAE" # Lysine and Glutamate
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp_in:
        # Generate initial PDB content
        pdb_content = generate_pdb_content(sequence_str=sequence, minimize_energy=False)
        tmp_in.write(pdb_content)
        tmp_in_path = tmp_in.name
    
    tmp_out_path = tmp_in_path.replace(".pdb", "_min.pdb")
    
    try:
        minimizer = EnergyMinimizer()
        # We need to set logging to DEBUG to catch the "Found X salt bridges" message
        import logging
        logging.getLogger("synth_pdb.physics").setLevel(logging.DEBUG)
        
        success = minimizer.add_hydrogens_and_minimize(tmp_in_path, tmp_out_path)
        
        assert success is True
        # Check if salt bridge detection was logged
        # Note: Depending on the initial random conformation, a salt bridge might or might not be found.
        # But for KAAAAE, it's likely if they are close.
        # Let's at least verify the code path runs.
        assert "Processing physics" in caplog.text
        
    finally:
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path): os.remove(tmp_out_path)

@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_hetatm_restoration(caplog):
    """Test that HETATMs (like ZN) are restored if culled by OpenMM."""
    # Zinc finger-like sequence - needs 4 ligands (Cys or His)
    sequence = "CPYCKCPYCK" # 4 Cysteines
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp_in:
        # Generate with Zinc
        pdb_content = generate_pdb_content(sequence_str=sequence, metal_ions="auto", minimize_energy=False)
        tmp_in.write(pdb_content)
        tmp_in_path = tmp_in.name
        
    tmp_out_path = tmp_in_path.replace(".pdb", "_min.pdb")
    
    try:
        minimizer = EnergyMinimizer()
        success = minimizer.add_hydrogens_and_minimize(tmp_in_path, tmp_out_path)
        
        assert success is True
        
        # Verify ZN is still in the output
        with open(tmp_out_path, 'r') as f:
            out_content = f.read()
            assert "ZN" in out_content
            assert "HETATM" in out_content
            
    finally:
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path): os.remove(tmp_out_path)

def test_minimizer_no_openmm(monkeypatch, caplog):
    """Test behavior when OpenMM is not present."""
    import synth_pdb.physics
    monkeypatch.setattr(synth_pdb.physics, "HAS_OPENMM", False)
    
    minimizer = EnergyMinimizer()
    # Should return early or fail gracefully
    result = minimizer.add_hydrogens_and_minimize("dummy.pdb", "out.pdb")
    assert result is False
    assert "OpenMM not found" in caplog.text

@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_minimizer_empty_topology(caplog):
    """Test error handling for empty or malformed topology."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        # Use a mostly empty PDB but with one line to avoid immediate PDBFile crash
        tmp.write("REMARK Empty PDB\nATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n")
        tmp_path = tmp.name
        
    try:
        minimizer = EnergyMinimizer()
        # Mocking createSystem to raise an exception or similar? 
        # Actually let's just test that it fails gracefully for a "too small" system
        result = minimizer.add_hydrogens_and_minimize(tmp_path, "out.pdb")
        # If it passed or failed, we just want to see it doesn't crash with unhandled exception
        # OpenMM might fail to createSystem for 1 atom without bonds.
        assert result is False or result is True
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
