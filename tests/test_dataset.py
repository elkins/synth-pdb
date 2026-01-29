
import pytest
import shutil
import tempfile
import os
import json
import csv
import numpy as np
import concurrent.futures
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module to be tested
try:
    from synth_pdb.dataset import DatasetGenerator, _generate_single_sample_task
except ImportError:
    DatasetGenerator = None

class SynchronousExecutor(concurrent.futures.Executor):
    """
    A mock executor that runs tasks synchronously in the current thread.
    This allows mocks on generate_pdb_content to work, as they are in the same process.
    """
    def __init__(self, max_workers=None, *args, **kwargs):
        pass
    
    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future
        
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)
    
    def shutdown(self, wait=True):
        pass

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
        generator = DatasetGenerator(
            output_dir=output_dir,
            num_samples=10,
            train_ratio=0.8
        )
        
        assert generator.output_dir == Path(output_dir).absolute()
        assert generator.num_samples == 10
        assert generator.min_length == 10

    def test_directory_structure(self, output_dir):
        """Test that train/test directories are created."""
        generator = DatasetGenerator(output_dir=output_dir)
        generator.prepare_directories()
        
        assert (Path(output_dir) / "train").exists()
        assert (Path(output_dir) / "test").exists()
        assert (Path(output_dir) / "dataset_manifest.csv").exists()

    def test_generate_single_sample_task(self, output_dir):
        """Test the helper function _generate_single_sample_task in isolation."""
        with patch("synth_pdb.dataset.generate_pdb_content", return_value="PDB_CONTENT") as mock_gen:
            with patch("synth_pdb.dataset.export_constraints", return_value="MAP_CONTENT"):
                 with patch("synth_pdb.dataset.compute_contact_map", return_value="MAP"):
                    with patch("biotite.structure.io.pdb.PDBFile.read") as mock_read:
                        # Mock minimal strcuture
                        mock_file = MagicMock()
                        mock_struct = MagicMock()
                        mock_ca = MagicMock()
                        mock_ca.res_name = ["ALA", "GLY"]
                        mock_struct.__getitem__.return_value = mock_ca
                        mock_file.get_structure.return_value = mock_struct
                        mock_read.return_value = mock_file
                        
                        # Create necessary subdir
                        (Path(output_dir) / "train").mkdir(parents=True, exist_ok=True)

                        args = ("id_001", 20, "alpha", "train", str(output_dir))
                        result = _generate_single_sample_task(args)
                        
                        assert result["success"] is True, f"Failed with error: {result.get('error')}"
                        assert result["sample_id"] == "id_001"
                        assert (Path(output_dir) / "train/id_001.pdb").exists()
                        assert (Path(output_dir) / "train/id_001.casp").exists()

    def test_generation_loop(self, output_dir):
        """
        Test the main generation loop using SynchronousExecutor.
        """
        # We replace ProcessPoolExecutor with our SynchronousExecutor
        with patch("concurrent.futures.ProcessPoolExecutor", side_effect=SynchronousExecutor):
            
            # Now we can treat mocks as usual because everything runs in main thread
            with patch("synth_pdb.dataset.generate_pdb_content", return_value="PDB") as mock_gen:
                with patch("synth_pdb.dataset.export_constraints", return_value="MAP"):
                    with patch("synth_pdb.dataset.compute_contact_map", return_value="RAW_MAP"):
                        with patch("biotite.structure.io.pdb.PDBFile.read") as mock_read:
                            # Mock structure setup
                            mock_file = MagicMock()
                            mock_struct = MagicMock()
                            mock_ca = MagicMock()
                            mock_ca.res_name = ["ALA"]
                            mock_struct.__getitem__.return_value = mock_ca
                            mock_file.get_structure.return_value = mock_struct
                            mock_read.return_value = mock_file
                            
                            n_samples = 5
                            generator = DatasetGenerator(
                                output_dir=output_dir, 
                                num_samples=n_samples
                            )
                            generator.generate()
                            
                            # Verify call count
                            assert mock_gen.call_count == n_samples
                            
                            # Verify manifest
                            manifest_path = Path(output_dir) / "dataset_manifest.csv"
                            with open(manifest_path, "r") as f:
                                lines = f.readlines()
                                assert len(lines) == n_samples + 1

    def test_generate_error_handling(self, output_dir):
        """Verify that a single sample failure doesn't stop the loop."""
        with patch("concurrent.futures.ProcessPoolExecutor", side_effect=SynchronousExecutor):
            with patch("synth_pdb.dataset.generate_pdb_content") as mock_gen:
                # First fail, second success
                mock_gen.side_effect = [Exception("BOOM"), "PDB"]
                
                with patch("synth_pdb.dataset.export_constraints", return_value="MAP"):
                    with patch("synth_pdb.dataset.compute_contact_map", return_value="RAW_MAP"):
                         with patch("biotite.structure.io.pdb.PDBFile.read") as mock_read:
                            mock_read.return_value.get_structure.return_value.__getitem__.return_value.res_name = ["ALA"]

                            generator = DatasetGenerator(output_dir=output_dir, num_samples=2)
                            generator.generate()
                            
                            # Verify manifest has 1 success (header + 1)
                            manifest_path = Path(output_dir) / "dataset_manifest.csv"
                            with open(manifest_path, "r") as f:
                                lines = f.readlines()
                                assert len(lines) == 2 

    def test_progress_logging(self, caplog, output_dir):
        """Verify that progress is logged."""
        # Must patch ProcessPoolExecutor to capture logs in same process usually? 
        # Actually logging from threads/processes is tricky to capture with caplog sometimes.
        # But with SynchronousExecutor it is easy.
        with patch("concurrent.futures.ProcessPoolExecutor", side_effect=SynchronousExecutor):
             with patch("synth_pdb.dataset.generate_pdb_content", return_value="PDB"):
                 with patch("synth_pdb.dataset.export_constraints", return_value="MAP"):
                     with patch("synth_pdb.dataset.compute_contact_map", return_value="RAW_MAP"):
                         with patch("biotite.structure.io.pdb.PDBFile.read") as mock_read:
                            mock_read.return_value.get_structure.return_value.__getitem__.return_value.res_name = ["ALA"]
                            
                            import logging
                            with caplog.at_level(logging.INFO):
                                # 10 samples to trigger at least completion log
                                generator = DatasetGenerator(output_dir=output_dir, num_samples=100)
                                generator.generate()
                                
        assert "Generated 100/100 samples" in caplog.text
