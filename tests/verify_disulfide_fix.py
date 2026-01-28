
import pytest
from unittest.mock import MagicMock, patch, ANY, mock_open
import numpy as np
import synth_pdb.generator
from biotite.structure import Atom, AtomArray
import builtins

# Save original open to use in side_effect
original_open = builtins.open

def open_side_effect(file, mode='r', *args, **kwargs):
    """
    Side effect for open() mock. 
    Only mocks the specific file we expect 'minimized.pdb'.
    Everything else is passed to the real open().
    """
    if isinstance(file, str) and "minimized.pdb" in file:
        return mock_open(read_data="ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00  0.00           N  ")()
    return original_open(file, mode, *args, **kwargs)

def test_disulfide_detection_uses_minimized_structure():
    """
    Test that the generator uses the MINIMIZED structure coordinates for disulfide detection
    when optimization is enabled.
    """
    
    # 1. Define the "Close" structure (simulating successful folding/minimization)
    minimized_structure = AtomArray(2)
    minimized_structure[0] = Atom([0,0,0], atom_name="SG", res_name="CYS", res_id=1, element="S")
    minimized_structure[1] = Atom([2.1,0,0], atom_name="SG", res_name="CYS", res_id=4, element="S")
    
    # 2. Prepare Mocks
    mock_minimizer = MagicMock()
    mock_minimizer.add_hydrogens_and_minimize.return_value = True
    
    # Use MagicMock for PDBFile instance
    mock_pdb_instance = MagicMock()
    mock_pdb_instance.get_structure.return_value = minimized_structure
    
    with patch("synth_pdb.generator.EnergyMinimizer", return_value=mock_minimizer), \
         patch("synth_pdb.generator.pdb.PDBFile") as mock_pdb_cls, \
         patch("tempfile.TemporaryDirectory") as mock_tmp, \
         patch("builtins.open", side_effect=open_side_effect), \
         patch("synth_pdb.generator.biophysics") as mock_biophysics, \
         patch("synth_pdb.generator.predict_order_parameters", return_value={}): 
            
            # Setup PDBFile mock
            # Both constructor and read() return our instance
            mock_pdb_cls.return_value = mock_pdb_instance
            mock_pdb_cls.read.return_value = mock_pdb_instance # Critical: read() is a classmethod
            
            # Setup biophysics mock
            mock_biophysics.apply_ph_titration.side_effect = lambda p, **kwargs: p
            mock_biophysics.cap_termini.side_effect = lambda p: p

            mock_tmp.return_value.__enter__.return_value = "/tmp/mock_dir"
            
            # Run Generator
            pdb_content = synth_pdb.generator.generate_pdb_content(
                sequence_str="CYS-ALA-ALA-CYS", 
                minimize_energy=True,
                metal_ions='ignore'
            )
            
            # 3. Verification
            assert "SSBOND" in pdb_content, "SSBOND record should be present when using minimized structure coordinates"
            
            import re
            ssbond_pattern = r"SSBOND\s+\d+\s+CYS\s+A\s+1\s+CYS\s+A\s+4"
            assert re.search(ssbond_pattern, pdb_content), f"SSBOND record does not match expected residues 1 and 4. Content:\n{pdb_content}"
            
            print("\nSUCCESS: Disulfide bond detected from minimized structure!")

def test_disulfide_detection_fails_if_far():
    """
    Control test: Verify that if the structure is NOT minimized (or minimized structure is still far),
    NO disulfide bond is detected.
    """
    # 1. Define "Far" structure
    far_structure = AtomArray(2)
    far_structure[0] = Atom([0,0,0], atom_name="SG", res_name="CYS", res_id=1, element="S")
    far_structure[1] = Atom([10.0,0,0], atom_name="SG", res_name="CYS", res_id=4, element="S")
    
    mock_minimizer = MagicMock()
    mock_minimizer.add_hydrogens_and_minimize.return_value = True
    
    mock_pdb_instance = MagicMock()
    mock_pdb_instance.get_structure.return_value = far_structure
    
    with patch("synth_pdb.generator.EnergyMinimizer", return_value=mock_minimizer), \
         patch("synth_pdb.generator.pdb.PDBFile") as mock_pdb_cls, \
         patch("tempfile.TemporaryDirectory") as mock_tmp, \
         patch("builtins.open", side_effect=open_side_effect), \
         patch("synth_pdb.generator.biophysics") as mock_biophysics, \
         patch("synth_pdb.generator.predict_order_parameters", return_value={}): 
            
            mock_pdb_cls.return_value = mock_pdb_instance
            mock_pdb_cls.read.return_value = mock_pdb_instance
            
            mock_biophysics.apply_ph_titration.side_effect = lambda p, **kwargs: p
            mock_biophysics.cap_termini.side_effect = lambda p: p

            mock_tmp.return_value.__enter__.return_value = "/tmp/mock_dir"
            
            pdb_content = synth_pdb.generator.generate_pdb_content(
                sequence_str="CYS-ALA-ALA-CYS",
                minimize_energy=True,
                metal_ions='ignore'
            )
            
            assert "SSBOND" not in pdb_content, "SSBOND record should NOT be present for distant residues"
