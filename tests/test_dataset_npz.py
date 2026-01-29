
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from synth_pdb.dataset import DatasetGenerator

class TestDatasetNPZ(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_npz_generation_creates_files(self):
        """Test that generate creates .npz files when format='npz'."""
        # Note: We anticipate adding a 'dataset_format' argument to __init__ or generate
        generator = DatasetGenerator(
            output_dir=self.test_dir,
            num_samples=2,
            min_length=10,
            max_length=15,
            seed=42,
            dataset_format='npz'  # This argument doesn't exist yet -> EXPECTED FAILURE
        )
        generator.generate()

        # Check directory structure
        train_dir = Path(self.test_dir) / "train"
        test_dir = Path(self.test_dir) / "test"
        
        # Check if any .npz files exist recursively
        npz_files = list(Path(self.test_dir).rglob("*.npz"))
        self.assertEqual(len(npz_files), 2, "Should generate exactly 2 npz files")

    def test_npz_content_schema(self):
        """Test that the generated .npz files contain the correct arrays and shapes."""
        generator = DatasetGenerator(
            output_dir=self.test_dir,
            num_samples=1,
            min_length=10,
            max_length=10, # Fixed length for easier checking
            seed=42,
            dataset_format='npz'
        )
        generator.generate()
        
        npz_files = list(Path(self.test_dir).rglob("*.npz"))
        self.assertTrue(len(npz_files) > 0, "No NPZ files generated")
        
        file_path = npz_files[0]
        data = np.load(file_path)
        
        # Check Keys
        expected_keys = {'coords', 'sequence', 'contact_map'}
        self.assertTrue(expected_keys.issubset(data.files), f"Missing keys. Found: {data.files}")
        
        # Check Shapes
        # Length is 10
        L = 10
        coords = data['coords']
        sequence = data['sequence']
        cmap = data['contact_map']
        
        self.assertEqual(coords.shape, (L, 5, 3), "Coords should be (L, 5, 3) -> [N, CA, C, O, CB]")
        self.assertEqual(sequence.shape, (L, 20), "Sequence should be one-hot (L, 20)")
        self.assertEqual(cmap.shape, (L, L), "Contact map should be (L, L)")
        
        # Check Values (Basic sanity)
        # One-hot sequence should sum to 1 per residue
        self.assertTrue(np.allclose(np.sum(sequence, axis=1), 1.0), "Sequence rows must sum to 1 (one-hot)")
        
        # Contact map diagonal should be 0
        self.assertTrue(np.allclose(np.diag(cmap), 0.0), "Contact map diagonal should be 0")

if __name__ == '__main__':
    unittest.main()
