
import unittest
import numpy as np
import tempfile
import os
from biotite.structure.io.pdb import PDBFile
import biotite.structure as struc
from synth_pdb.quality.interpolate import interpolate_structures

class TestQualityInterpolate(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.pdb1_path = os.path.join(self.test_dir, "start.pdb")
        self.pdb2_path = os.path.join(self.test_dir, "end.pdb")
        self.bad_pdb_path = os.path.join(self.test_dir, "bad.pdb")

        # Create dummy structure 1 (len 5)
        atoms1 = struc.AtomArray(20) # 5 res * 4 atoms
        atoms1.coord = np.zeros((20, 3))
        atoms1.atom_name = np.tile(["N", "CA", "C", "O"], 5)
        atoms1.res_id = np.repeat(range(1, 6), 4)
        atoms1.res_name = np.tile(["ALA"], 20)
        atoms1.element = np.tile(["N", "C", "C", "O"], 5)
        
        f1 = PDBFile()
        f1.set_structure(atoms1)
        f1.write(self.pdb1_path)
        
        # Create dummy structure 2 (len 5)
        atoms2 = atoms1.copy()
        # Change coords slightly
        atoms2.coord += 1.0
        f2 = PDBFile()
        f2.set_structure(atoms2)
        f2.write(self.pdb2_path)
        
        # Create dummy structure 3 (len 6) - Mismatch
        atoms3 = struc.AtomArray(24) 
        atoms3.coord = np.zeros((24, 3))
        atoms3.atom_name = np.tile(["N", "CA", "C", "O"], 6)
        atoms3.res_name = np.tile(["ALA"], 24)
        atoms3.element = np.tile(["N", "C", "C", "O"], 6)
        f3 = PDBFile()
        f3.set_structure(atoms3)
        f3.write(self.bad_pdb_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_interpolate_length_mismatch(self):
        # Should raise ValueError
        with self.assertRaises(ValueError):
            interpolate_structures(self.pdb1_path, self.bad_pdb_path, steps=5, output_prefix="out")

    def test_interpolate_valid(self):
        out_prefix = os.path.join(self.test_dir, "morph")
        interpolate_structures(self.pdb1_path, self.pdb2_path, steps=2, output_prefix=out_prefix)
        
        # Should create out_0.pdb, out_1.pdb, out_2.pdb
        for i in range(3):
            self.assertTrue(os.path.exists(f"{out_prefix}_{i}.pdb"))

if __name__ == '__main__':
    unittest.main()
