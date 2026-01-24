import unittest
import logging
import numpy as np
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

# EDUCATIONAL NOTE - Molecular Chirality (Handedness)
# Introduction to Stereochemistry in Proteins
# ---------------------------------------------
# Most biological molecules are "chiral", meaning they cannot be superimpsed
# on their mirror image (like your left and right hands).
#
# 1. The Alpha Carbon (C-alpha):
#    The central carbon of an amino acid is bonded to four different groups:
#    - Amine group (-NH2)
#    - Carboxyl group (-COOH)
#    - Hydrogen atom (-H)
#    - Side chain (-R)
#
# 2. L-Isomers vs D-Isomers:
#    Because of this asymmetry, amino acids can exist in two forms (enantiomers):
#    - L-amino acids (Levorotatory): The form found in ALL natural proteins produced by ribosomes.
#    - D-amino acids (Dextrorotatory): Rare in nature, found in some bacterial cell walls
#      but almost never in human proteins.
#
# 3. Why it matters:
#    If a synthetic structure generator accidentally creates D-amino acids (due to
#    incorrect cross-product order in vector math), the protein will not fold
#    correctly and will be biologically seemingly "alien".
#
# 4. Geometric Definition (CORN Rule):
#    Looking down the H-Ca bond, the groups CO, R, N read "CO-R-N" clockwise for L-amino acids.
#    Mathematically, we can check the scalar triple product or improper dihedral angle.

logger = logging.getLogger(__name__)

class TestChirality(unittest.TestCase):

    def test_all_residues_are_L_amino_acids(self):
        """
        Generates a random peptide and verifies that all non-Glycine residues
        have L-chirality.
        """
        # Generate a sufficiently long random peptide to cover many amino acid types
        # Using a fixed seed for reproducibility
        pdb_content = generate_pdb_content(length=50, seed=42)
        
        validator = PDBValidator(pdb_content=pdb_content)
        
        # This method doesn't exist yet (TDD), but we define the contract here.
        # It should check improper dihedrals and raise violations for D-amino acids.
        # EDUCATIONAL: The validate_chirality method will likely implement the
        # mathematical check for the "CORN" rule or equivalent improper torsion.
        if hasattr(validator, 'validate_chirality'):
            validator.validate_chirality()
            violations = validator.get_violations()
            
            chirality_violations = [v for v in violations if "Chirality violation" in v]
            
            if chirality_violations:
                for v in chirality_violations:
                    logger.error(v)
            
            self.assertEqual(len(chirality_violations), 0, 
                             f"Found {len(chirality_violations)} D-amino acids! Proteins must be L-chiral.")
        else:
            # TDD Step 1: Fail if method is missing, or warn
            self.fail("PDBValidator.validate_chirality() not implemented yet.")

if __name__ == '__main__':
    unittest.main()
