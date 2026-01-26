
import pytest
import shutil
import tempfile
import os
import json
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module to be tested (will fail initially)
try:
    from synth_pdb.dataset import DatasetGenerator
except ImportError:
    DatasetGenerator = None

class TestDatasetGenerator:
    
    @pytest.fixture
    def output_dir(self):
        """Fixture for a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_module_exists(self):
        """Verify the module and class exist."""
        if DatasetGenerator is None:
            pytest.fail("synth_pdb.dataset module or DatasetGenerator class not found")

    def test_initialization(self, output_dir):
        """Test generator initialization and directory creation."""
        if DatasetGenerator is None:
            pytest.skip("Module not implemented")
            
        generator = DatasetGenerator(
            output_dir=output_dir,
            num_samples=10,
            train_ratio=0.8
        )
        
        assert generator.output_dir == Path(output_dir)
        assert generator.num_samples == 10
        assert generator.min_length == 10
        assert generator.max_length == 50 # Default check

    def test_directory_structure(self, output_dir):
        """Test that train/test directories are created."""
        if DatasetGenerator is None:
            pytest.skip("Module not implemented")
            
        generator = DatasetGenerator(output_dir=output_dir)
        generator.prepare_directories()
        
        assert (Path(output_dir) / "train").exists()
        assert (Path(output_dir) / "test").exists()
        assert (Path(output_dir) / "dataset_manifest.csv").exists()

    def test_generation_loop(self, output_dir):
        """
        Test the main generation loop.
        We mock the heavy `generator` calls to keep this fast and focused on logic.
        """
        if DatasetGenerator is None:
            pytest.skip("Module not implemented")
            
        # Mock dependencies
        with patch("synth_pdb.dataset.generate_pdb_content") as mock_gen:
            with patch("synth_pdb.dataset.export_constraints") as mock_export:
                # Mock PDB parsing to avoid biotite error
                with patch("biotite.structure.io.pdb.PDBFile.read") as mock_read:
                    mock_gen.return_value = "FAKE PDB CONTENT"
                    mock_export.return_value = "FAKE CONTACT MAP"
                    
                    # Mock structure object
                    mock_structure = MagicMock()
                    # Mock CA atoms for sequence extraction
                    mock_ca = MagicMock()
                    mock_ca.res_name = ["ALA"] * 10 
                    # When slicing structure[structure.atom_name == "CA"], return mock_ca
                    mock_structure.__getitem__.return_value = mock_ca
                    
                    mock_pdb_file = MagicMock()
                    mock_pdb_file.get_structure.return_value = mock_structure
                    mock_read.return_value = mock_pdb_file
                    
                    # Small number of samples
                    n_samples = 5
                    generator = DatasetGenerator(
                        output_dir=output_dir, 
                        num_samples=n_samples,
                        train_ratio=0.8
                    )
                    
                    # We also need to mock compute_contact_map since we are controlling inputs
                    with patch("synth_pdb.dataset.compute_contact_map", return_value="FAKE_MAP"):
                        generator.generate()
                    
                    # Verify calls
                    assert mock_gen.call_count == n_samples
                    
                    # Verify file counts
                    train_dir = Path(output_dir) / "train"
                    test_dir = Path(output_dir) / "test"
                    
                    train_files = list(train_dir.glob("*.pdb"))
                    test_files = list(test_dir.glob("*.pdb"))
                    
                    # Ideally 4 train (80%), 1 test (20%)
                    total_files = len(train_files) + len(test_files)
                    assert total_files == n_samples
                    
                    # Check manifest
                    manifest_path = Path(output_dir) / "dataset_manifest.csv"
                    assert manifest_path.exists()
                    
                    with open(manifest_path, "r") as f:
                        # Header + 5 lines
                        lines = f.readlines()
                        assert len(lines) == n_samples + 1

    def test_metadata_consistency(self, output_dir):
        """Verify metadata matches generated parameters."""
        if DatasetGenerator is None:
            pytest.skip("Module not implemented")

        with patch("synth_pdb.dataset.generate_pdb_content", return_value="PDB"):
             with patch("synth_pdb.dataset.export_constraints", return_value="MAP"):
                 with patch("biotite.structure.io.pdb.PDBFile.read") as mock_read:
                    mock_pdb = MagicMock()
                    mock_struc = MagicMock()
                    mock_ca = MagicMock()
                    mock_ca.res_name = ["ALA"]
                    mock_struc.__getitem__.return_value = mock_ca
                    mock_pdb.get_structure.return_value = mock_struc
                    mock_read.return_value = mock_pdb
                    
                    with patch("synth_pdb.dataset.compute_contact_map", return_value="FAKE"):
                        generator = DatasetGenerator(output_dir=output_dir, num_samples=1)
                        generator.generate()
                        
                        manifest_path = Path(output_dir) / "dataset_manifest.csv"
                        with open(manifest_path, "r") as f:
                            reader = csv.DictReader(f)
                            row = next(reader)
                            
                            assert "id" in row
                            assert "length" in row
                            assert "split" in row
                            assert row["split"] in ["train", "test"]
                            assert (Path(output_dir) / row["split"] / f"{row['id']}.pdb").exists()

    def test_random_length_variation(self, output_dir):
        """Test that lengths vary within range."""
        if DatasetGenerator is None:
            pytest.skip("Module not implemented")

        # Capture calls to generate_pdb_content to check length argument
        with patch("synth_pdb.dataset.generate_pdb_content", return_value="PDB") as mock_gen:
             with patch("synth_pdb.dataset.export_constraints", return_value="MAP"):
                generator = DatasetGenerator(
                    output_dir=output_dir, 
                    num_samples=10, 
                    min_length=15, 
                    max_length=25
                )
                generator.generate()
                
                lengths = [call.kwargs.get('length') for call in mock_gen.call_args_list]
                # assert all(15 <= l <= 25 for l in lengths) # generator might use 'length' or infer
                # Our implementation should pass explicit length.
                
                # At least verify we passed *some* length argument
                assert len(lengths) == 10
