
import unittest
import numpy as np
import biotite.structure as struc
from synth_pdb.ai.features import extract_quality_features, _analyze_ramachandran, get_feature_names
from synth_pdb.validator import PDBValidator

class TestAIFeatures(unittest.TestCase):
    def test_ramachandran_analysis_ideal(self):
        phi = np.array([-60, -60, -60])
        psi = np.array([-45, -45, -45])
        
        # Validator instance (dummy)
        validator = PDBValidator(pdb_content="HEADER")
        
        favored, outliers = _analyze_ramachandran(phi, psi, validator)
        
        # Ideally 3 favored (all of them)
        self.assertEqual(favored, 3)
        self.assertEqual(outliers, 0)

    def test_ramachandran_analysis_outliers(self):
        phi = np.array([0, 0, 0])
        psi = np.array([0, 0, 0])
        
        validator = PDBValidator(pdb_content="HEADER")
        
        favored, outliers = _analyze_ramachandran(phi, psi, validator)
        
        self.assertAlmostEqual(favored, 0.0)
        self.assertTrue(outliers > 0.0)

    def test_extract_features_shape(self):
        # Create a dummy atom array
        atoms = struc.AtomArray(12)
        atoms.coord = np.zeros((12, 3)) # Dummy coords
        atoms.atom_name = np.tile(["N", "CA", "C", "O"], 3)
        atoms.res_id = np.repeat([1, 2, 3], 4)
        atoms.res_name = np.tile(["ALA"], 12)
        atoms.element = np.tile(["N", "C", "C", "O"], 3)
        
        import io
        from biotite.structure.io.pdb import PDBFile
        
        pdb_file = PDBFile()
        pdb_file.set_structure(atoms)
        f = io.StringIO()
        pdb_file.write(f)
        pdb_content = f.getvalue()
        
        features = extract_quality_features(pdb_content)
        names = get_feature_names()
        
        self.assertEqual(len(features), len(names))
        
        # Radius of gyration (Index 6) should be 0 for all-zero coords
        pdb_rg_index = 6
        self.assertAlmostEqual(features[pdb_rg_index], 0.0)
        
        # Bond length violations (Index 3) should be high (all 0 coord diffs)
        # 3 res * 4 atoms = ~11 bonds check
        pdb_bond_len_index = 3
        self.assertTrue(features[pdb_bond_len_index] > 0)

if __name__ == '__main__':
    unittest.main()
