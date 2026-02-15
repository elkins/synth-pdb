
import unittest
import numpy as np
import biotite.structure as struc
from synth_pdb.cofactors import find_metal_binding_sites, add_metal_ion

class TestCofactors(unittest.TestCase):

    def setUp(self):
        # Create a more robust structure with a C2H2 zinc finger motif.
        # We will have CYS, HIS, CYS, HIS residues.
        # Ligands will be CYS-SG, HIS-NE2, CYS-SG, HIS-NE2.
        # We'll also include HIS-ND1 atoms but place them far away.
        self.structure = struc.AtomArray(32)
        
        # Residue 1: CYS
        self.structure.res_name[:4] = 'CYS'
        self.structure.atom_name[:4] = ['N', 'CA', 'C', 'SG']
        self.structure.res_id[:4] = 1
        
        # Residue 2: HIS
        self.structure.res_name[4:14] = 'HIS'
        self.structure.atom_name[4:14] = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2']
        self.structure.res_id[4:14] = 2

        # Residue 3: CYS
        self.structure.res_name[14:18] = 'CYS'
        self.structure.atom_name[14:18] = ['N', 'CA', 'C', 'SG']
        self.structure.res_id[14:18] = 3

        # Residue 4: HIS
        self.structure.res_name[18:28] = 'HIS'
        self.structure.atom_name[18:28] = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2']
        self.structure.res_id[18:28] = 4
        
        # Add a dummy residue to make it 32 atoms
        self.structure.res_name[28:] = 'DUM'
        self.structure.atom_name[28:] = ['D1', 'D2', 'D3', 'D4']
        self.structure.res_id[28:] = 5


        # Initialize all coordinates to be far apart
        self.structure.coord = np.arange(32 * 3).reshape(32, 3) * 100.0

        # Place the desired ligands in a nice tight cluster
        self.structure.coord[3] = [0, 0, 0]          # CYS 1 SG (index 3)
        self.structure.coord[13] = [2, 0, 0]         # HIS 2 NE2 (index 13)
        self.structure.coord[17] = [1, 1.732, 0]     # CYS 3 SG (index 17)
        self.structure.coord[27] = [1, 0.577, 1.633] # HIS 4 NE2 (index 27)

        # Place other potential ligands far away
        self.structure.coord[10] = [50, 50, 50]       # HIS 2 ND1 (index 10)
        self.structure.coord[24] = [-50, -50, -50]    # HIS 4 ND1 (index 24)

        self.structure.chain_id[:] = 'A'
        self.structure.hetero[:] = False
        # A simplified element list
        self.structure.element = np.array(
            ['N', 'C', 'C', 'S'] + 
            ['N', 'C', 'C', 'O', 'C', 'C', 'N', 'C', 'C', 'N'] +
            ['N', 'C', 'C', 'S'] +
            ['N', 'C', 'C', 'O', 'C', 'C', 'N', 'C', 'C', 'N'] +
            ['X', 'X', 'X', 'X']
        )



    def test_find_metal_binding_sites_success(self):
        """Test finding a C2H2 zinc finger."""
        sites = find_metal_binding_sites(self.structure, distance_threshold=5.0)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]['type'], 'ZN')
        self.assertEqual(len(sites[0]['ligand_indices']), 4)
        # Check if the correct ligand atoms are identified
        found_indices = sorted(sites[0]['ligand_indices'])
        expected_indices = sorted([3, 13, 17, 27]) # SG, NE2, SG, NE2
        self.assertListEqual(found_indices, expected_indices)

    def test_find_metal_binding_sites_no_site(self):
        """Test with a structure that has no valid binding site."""
        # Scatter the ligands far apart
        self.structure.coord[13] = [20, 0, 0]
        sites = find_metal_binding_sites(self.structure, distance_threshold=5.0)
        self.assertEqual(len(sites), 0)

    def test_add_metal_ion(self):
        """Test adding a zinc ion to a found site."""
        sites = find_metal_binding_sites(self.structure, distance_threshold=5.0)
        self.assertEqual(len(sites), 1)
        
        new_structure = add_metal_ion(self.structure, sites[0])
        
        # Check that one atom was added
        self.assertEqual(len(new_structure), len(self.structure) + 1)
        
        # Check the new atom is a Zinc ion
        ion = new_structure[-1]
        self.assertEqual(ion.res_name, 'ZN')
        self.assertEqual(ion.atom_name, 'ZN')
        self.assertEqual(ion.element, 'ZN')
        self.assertTrue(ion.hetero)
        
        # Check the ion is at the centroid of the ligands
        ligand_coords = self.structure.coord[[3, 13, 17, 27]]
        expected_centroid = np.mean(ligand_coords, axis=0)
        np.testing.assert_allclose(ion.coord, expected_centroid, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
