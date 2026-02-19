
import unittest
import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
import io
import pytest

joblib = pytest.importorskip("joblib", reason="joblib not installed; install synth-pdb[ai]")

from synth_pdb.generator import generate_pdb_content
from synth_pdb.quality.classifier import ProteinQualityClassifier

class TestQualityFilterDeep(unittest.TestCase):
    def setUp(self):
        self.classifier = ProteinQualityClassifier()
        if self.classifier.model is None:
            self.skipTest("Quality classifier model not found. Skipping deep quality tests.")

    def test_high_quality_helix(self):
        """Test that a perfect alpha helix is classified as High Quality."""
        # Generate a standard alpha helix (should be very good) - Use Poly-ALA to avoid sidechain clashes in unminimized structure
        pdb_content = generate_pdb_content(sequence_str="A"*20, conformation="alpha", minimize_energy=False)
        
        is_good, prob, features = self.classifier.predict(pdb_content)
        
        print(f"\n[High Quality Test] Probability: {prob:.4f}")
        print(f"Features: {features}")
        
        # Should be confident
        self.assertTrue(is_good, "Alpha helix should be classified as Good")
        self.assertGreater(prob, 0.6, "Probability for alpha helix should be high (> 0.6)")

    def test_low_quality_clashes(self):
        """Test that a structure with steric clashes is classified as Low Quality."""
        # Generate a helix first
        pdb_content = generate_pdb_content(length=20, conformation="alpha", minimize_energy=False)
        
        # Parse and introduce a severe clash
        pdb_file = PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        # Move residue 5's CA to exactly matches residue 10's CA
        # This creates a massive steric clash
        # structure is an atom array.
        # coords are in structure.coord
        
        # Find index of res 5 CA and res 10 CA
        res5_ca = (structure.res_id == 5) & (structure.atom_name == "CA")
        res10_ca = (structure.res_id == 10) & (structure.atom_name == "CA")
        
        if np.any(res5_ca) and np.any(res10_ca):
            structure.coord[res5_ca] = structure.coord[res10_ca]
            
            # Write back to PDB string
            f = io.StringIO()
            pdb_file.set_structure(structure)
            pdb_file.write(f)
            bad_pdb_content = f.getvalue()
            
            is_good, prob, features = self.classifier.predict(bad_pdb_content)
            
            print(f"\n[Clash Test] Probability: {prob:.4f}")
            print(f"Features: {features}")
            
            # Should be rejected
            self.assertFalse(is_good, "Clashing structure should be classified as Bad")
            self.assertLess(prob, 0.5, "Probability for clashing structure should be low (< 0.5)")
        else:
            self.fail("Could not find residues to clash in generated structure")

    def test_low_quality_geometry(self):
        """Test that a structure with distorted geometry is classified as Low Quality."""
        # Generate helix
        pdb_content = generate_pdb_content(length=10, conformation="alpha")
        
        # Distort bond lengths significantly
        # We'll just take the string and manually mess with coordinates
        # Or simpler: create a random point cloud
        
        import biotite.structure as struc
        atoms = struc.AtomArray(40) # 10 residues
        atoms.coord = np.random.rand(40, 3) * 10 # Random scatter
        atoms.atom_name = np.tile(["N", "CA", "C", "O"], 10)
        atoms.res_id = np.repeat(range(1, 11), 4)
        atoms.res_name = np.tile(["ALA"], 40)
        atoms.element = np.tile(["N", "C", "C", "O"], 10)
        
        f = io.StringIO()
        file = PDBFile()
        file.set_structure(atoms)
        file.write(f)
        bad_pdb = f.getvalue()
        
        is_good, prob, features = self.classifier.predict(bad_pdb)
        
        print(f"\n[Bad Geometry Test] Probability: {prob:.4f}")
        print(f"Features: {features}")
        
        self.assertFalse(is_good, "Random scatter should be classified as Bad")
        self.assertLess(prob, 0.4, "Probability for random scatter should be very low")

if __name__ == '__main__':
    unittest.main()
