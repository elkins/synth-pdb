import unittest
import logging

from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

# Suppress logging during tests to keep output clean
logging.getLogger().setLevel(logging.CRITICAL)


class TestGeneratorWithRotamers(unittest.TestCase):

    def test_side_chain_generation_creates_reasonable_chi_angle(self):
        """
        Test that side-chain generation creates a reasonable chi-1 angle for LEU.
        """
        # Generate a peptide with a LEU residue
        pdb_content = generate_pdb_content(sequence_str="LEU")
        atoms = PDBValidator._parse_pdb_atoms(pdb_content)

        # Get the atoms for the chi-1 angle of LEU
        leu_atoms = {atom['atom_name']: atom for atom in atoms if atom['residue_name'] == 'LEU'}
        
        self.assertIn('N', leu_atoms, "N atom not found for LEU")
        self.assertIn('CA', leu_atoms, "CA atom not found for LEU")
        self.assertIn('CB', leu_atoms, "CB atom not found for LEU")
        self.assertIn('CG', leu_atoms, "CG atom not found for LEU")

        # Get coordinates
        n_coords = leu_atoms['N']['coords']
        ca_coords = leu_atoms['CA']['coords']
        cb_coords = leu_atoms['CB']['coords']
        cg_coords = leu_atoms['CG']['coords']

        # Calculate the chi-1 angle
        chi1_angle = PDBValidator._calculate_dihedral_angle(n_coords, ca_coords, cb_coords, cg_coords)

        # Check for a reasonable chi-1 angle for LEU. Common rotamer values are around -60 and 180 degrees.
        # Accept angles in either rotamer range
        is_gauche_minus = -100 < chi1_angle < -20  # Around -60 degrees
        is_trans = chi1_angle > 100 or chi1_angle < -100  # Around 180 degrees (or -180)
        self.assertTrue(
            is_gauche_minus or is_trans, 
            f"Chi-1 angle for LEU is {chi1_angle:.1f}°, expected rotamer ranges: "
            f"-100° to -20° (gauche-) or ±100° to ±180° (trans)"
        )


if __name__ == '__main__':
    unittest.main()