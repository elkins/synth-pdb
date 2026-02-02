
import pytest
import os
import tempfile
from synth_pdb.docking import DockingPrep
from synth_pdb.generator import generate_pdb_content

class TestDockingRobustness:
    
    def test_pqr_generation_with_real_structure(self):
        """
        Verify PQR generation using a realistically generated structure from synth-pdb.
        This ensures OpenMM templates match correctly.
        """
        # Generate a small valid peptide
        pdb_content = generate_pdb_content(length=5, sequence_str="ALA-ALA-ALA-ALA-ALA", minimize_energy=False)
        
        with tempfile.TemporaryDirectory() as tmp:
            in_pdb = os.path.join(tmp, "in.pdb")
            out_pqr = os.path.join(tmp, "out.pqr")
            with open(in_pdb, 'w') as f: f.write(pdb_content)
            
            prep = DockingPrep()
            success = prep.write_pqr(in_pdb, out_pqr)
            
            assert success is True
            assert os.path.exists(out_pqr)
            
            with open(out_pqr, 'r') as f:
                lines = f.readlines()
                
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            assert len(atom_lines) > 20
            
            for line in atom_lines:
                charge_val = line[54:62].strip()
                radius_val = line[62:70].strip()
                assert len(charge_val) > 0
                assert len(radius_val) > 0
                float(charge_val)
                float(radius_val)

    def test_ptm_to_pqr_conversion_robust(self):
        """
        Verify that PTMs are translated correctly by preparer before OpenMM sees them.
        """
        # ALA-ALA-SEP-ALA-ALA
        pdb_content = generate_pdb_content(length=5, sequence_str="ALA-ALA-SEP-ALA-ALA", minimize_energy=False)
        
        with tempfile.TemporaryDirectory() as tmp:
            in_pdb = os.path.join(tmp, "in.pdb")
            out_pqr = os.path.join(tmp, "out.pqr")
            with open(in_pdb, 'w') as f: f.write(pdb_content)
            
            prep = DockingPrep()
            success = prep.write_pqr(in_pdb, out_pqr)
            
            assert success is True
            with open(out_pqr, 'r') as f:
                content = f.read()
            
            assert "SER" in content
            assert "SEP" not in content
            
            atom_lines = [l for l in content.split('\n') if l.startswith("ATOM")]
            for l in atom_lines:
                atom_name = l[12:16].strip()
                assert atom_name not in ["P", "O1P", "O2P", "O3P"]
