import unittest
import logging
import numpy as np
from collections import Counter
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

# Suppress logging during tests
# logging.getLogger().setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

class TestRotamerDistribution(unittest.TestCase):

    def test_valine_rotamer_distribution(self):
        """
        Verify that Valine rotamers follow approximately the expected distribution.
        Expected weights from data.py for VAL:
          -60 deg (g-): 0.70
          180 deg (t):  0.20
          60 deg (g+):  0.10
        """
        n_samples = 100
        rotamer_counts = Counter()
        
        # Use a fixed seed for reproducibility across the test suite run
        # but vary it per generation to get distribution
        np.random.seed(42)
        
        # Generate N samples
        for i in range(n_samples):
            # We must use different seeds to trigger the random sampling
            # Using seed=i ensures deterministic "randomness" for the test
            pdb_content = generate_pdb_content(sequence_str="VAL", seed=i)
            atoms = PDBValidator._parse_pdb_atoms(pdb_content)
            
            # Extract atoms
            val_atoms = {atom['atom_name']: atom for atom in atoms if atom['residue_name'] == 'VAL'}
            
            # VAL has Chi1 defined by N-CA-CB-CG1 (or CG2, standard definition uses CG1/CG2)
            # Dunbrack library typically uses CG1 for Chi1 of Valine
            if 'CG1' in val_atoms:
                cg_atom = val_atoms['CG1']
            else:
                continue

            n_coords = val_atoms['N']['coords']
            ca_coords = val_atoms['CA']['coords']
            cb_coords = val_atoms['CB']['coords']
            cg_coords = cg_atom['coords']
            
            angle = PDBValidator._calculate_dihedral_angle(n_coords, ca_coords, cb_coords, cg_coords)
            
            # Bin the angle
            # IMPORTANT: Due to the 180-degree shift to IUPAC convention (Trans=180),
            # the rotamer bins now reflect those IUPAC values.
            # g- : around -60
            # t  : around 180 (or -180)
            # g+ : around 60
            if -90 <= angle <= -30:
                rotamer_counts['g-'] += 1
            elif (angle > 150) or (angle < -150):
                rotamer_counts['t'] += 1
            elif 30 <= angle <= 90:
                rotamer_counts['g+'] += 1
            else:
                rotamer_counts['other'] += 1
                if rotamer_counts['other'] <= 10:  # Print first 10 failures
                    logger.debug(f"Invalid angle: {angle:.2f}")
                
        # Verify distribution
        # With 100 samples, we expect roughly:
        # g- : ~70
        # t  : ~20
        # g+ : ~10
        
        logger.info(f"Observed Rotamer Counts (N={n_samples}): {rotamer_counts}")
        
        # Allow some statistical variance (e.g., +/- 15)
        # We just want to ensure the "most common" one is indeed most common
        # and the "rare" one is rare.
        
        self.assertGreater(rotamer_counts['g-'], 50, "Valine g- should be the dominant rotamer (>50%)")
        self.assertGreater(rotamer_counts['t'], 5, "Valine trans should benefit from weighted sampling")
        self.assertLess(rotamer_counts['g+'], 30, "Valine g+ should be the minor rotamer")
        
        # Ensure we didn't generate weird angles
        self.assertEqual(rotamer_counts['other'], 0, "Generated outliers outside standard rotamer bins")

if __name__ == '__main__':
    unittest.main()
