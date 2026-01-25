
import unittest
import os
import tempfile
from synth_pdb.visualization import generate_pymol_script

class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Create dummy PDB
        self.pdb_file = "dummy.pdb"
        with open(self.pdb_file, 'w') as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
            
        # Create dummy NEF
        self.nef_file = "dummy.nef"
        with open(self.nef_file, 'w') as f:
            f.write("save_synthetic_noes\n")
            f.write("_nef_distance_restraint_list.sf_category nef_distance_restraint_list\n")
            f.write("loop_\n")
            f.write("_nef_distance_restraint.index\n")
            f.write("_nef_distance_restraint.chain_code_1\n")
            f.write("_nef_distance_restraint.sequence_code_1\n")
            f.write("_nef_distance_restraint.residue_name_1\n")
            f.write("_nef_distance_restraint.atom_name_1\n")
            f.write("_nef_distance_restraint.chain_code_2\n")
            f.write("_nef_distance_restraint.sequence_code_2\n")
            f.write("_nef_distance_restraint.residue_name_2\n")
            f.write("_nef_distance_restraint.atom_name_2\n")
            f.write("_nef_distance_restraint.target_value\n")
            f.write("1 A 1 ALA N A 2 GLY CA 5.0\n")
            f.write("stop_\n")
            f.write("save_\n")
            
        self.pml_file = "output.pml"

    def tearDown(self):
        if os.path.exists(self.pdb_file): os.remove(self.pdb_file)
        if os.path.exists(self.nef_file): os.remove(self.nef_file)
        if os.path.exists(self.pml_file): os.remove(self.pml_file)

    def test_generate_pymol_script(self):
        # 1. NEF Reader Format (seq_1)
        restraints_nef = [{
            'chain_1': 'A', 'seq_1': 1, 'res_1': 'ALA', 'atom_1': 'N',
            'chain_2': 'A', 'seq_2': 2, 'res_2': 'GLY', 'atom_2': 'CA',
            'dist': 5.0
        }]
        
        generate_pymol_script(self.pdb_file, restraints_nef, self.pml_file)
        self.assertTrue(os.path.exists(self.pml_file))
        
        # 2. NMR Calc Format (residue_index_1)
        # remove file to test again
        os.remove(self.pml_file)
        
        restraints_nmr = [{
            'chain_1': 'A', 'residue_index_1': 1, 'res_name_1': 'ALA', 'atom_name_1': 'N',
            'chain_2': 'A', 'residue_index_2': 2, 'res_name_2': 'GLY', 'atom_name_2': 'CA',
            'actual_distance': 4.5
        }]
        
        generate_pymol_script(self.pdb_file, restraints_nmr, self.pml_file)
        self.assertTrue(os.path.exists(self.pml_file))
        
        with open(self.pml_file, 'r') as f:
            content = f.read()
            
        self.assertIn("load dummy.pdb", content)
        self.assertIn("distance noe_1, (chain A and resi 1 and name N), (chain A and resi 2 and name CA)", content)
        self.assertIn("hide labels, noes", content)
