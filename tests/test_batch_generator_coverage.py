import pytest
import numpy as np
import os
import tempfile
from synth_pdb.batch_generator import BatchedGenerator, BatchedPeptide

def test_batched_peptide_utility_methods():
    # Setup a small batch
    coords = np.random.rand(5, 10, 3)
    seq = ["ALA"] * 2
    names = ["N", "CA", "C", "O"] * 2
    indices = [1, 1, 1, 1, 2, 2, 2, 2]
    
    # Adjust atom names/indices to match coords shape (10 atoms)
    # We'll just use dummy data for utility testing
    dummy_names = [f"A{i}" for i in range(10)]
    dummy_indices = [i // 4 + 1 for i in range(10)]
    
    batch = BatchedPeptide(coords, seq, dummy_names, dummy_indices)
    
    # 1. Test __len__
    assert len(batch) == 5
    
    # 2. Test __getitem__ (int)
    single = batch[0]
    assert isinstance(single, BatchedPeptide)
    assert single.coords.shape == (1, 10, 3)
    
    # 3. Test __getitem__ (slice)
    sub_batch = batch[1:3]
    assert len(sub_batch) == 2
    assert sub_batch.coords.shape == (2, 10, 3)

def test_batched_peptide_save_pdb():
    bg = BatchedGenerator("ALA-GLY", n_batch=2)
    batch = bg.generate_batch()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pdb")
        batch.save_pdb(path, index=0)
        assert os.path.exists(path)
        with open(path, 'r') as f:
            content = f.read()
            assert "ATOM      1  N   ALA A   1" in content
            assert "END" in content

def test_batched_generator_drift():
    # Test that drift > 0 actually modifies coordinates
    seq = "ALA-ALA-ALA"
    bg_no_drift = BatchedGenerator(seq, n_batch=10)
    bg_drift = BatchedGenerator(seq, n_batch=10)
    
    # Generate with same seed but different drift
    b1 = bg_no_drift.generate_batch(seed=42, drift=0.0)
    b2 = bg_drift.generate_batch(seed=42, drift=10.0)
    
    # Structures should be different
    assert not np.allclose(b1.coords, b2.coords)
    
    # Check that drift actually adds noise (batch members should be different)
    assert not np.allclose(b2.coords[0], b2.coords[1])
    # Without drift, all batch members are same (currenly, unless we add random seed per member)
    # Wait, BatchedGenerator currently applies SAME phi/psi to all? 
    # Let's check batch_generator.py:199
    # phi = np.full((B, L), p_phi) -> Yes.
    # Drift adds batch-wise noise: phi += np.random.normal(0, drift, (B, L))
    # So members ARE different if drift > 0.
    assert not np.allclose(b2.coords[0], b2.coords[1])

def test_batched_generator_template_error():
    # Use full_atom=True to trigger template lookup
    # and provide a sequence that is not in the library
    with pytest.raises((ValueError, KeyError)):
        # We need to bypass the ONE_TO_THREE_LETTER_CODE fallback to "ALA"
        # We can do this by passing a sequence with "-"
        BatchedGenerator("ALA-INVALID", full_atom=True)

def test_batched_generator_resolving_sequence():
    # Test 1-letter vs 3-letter split
    bg1 = BatchedGenerator("AGS")
    assert bg1.sequence == ["ALA", "GLY", "SER"]
    
    bg2 = BatchedGenerator("ALA-GLY-SER")
    assert bg2.sequence == ["ALA", "GLY", "SER"]
