
import pytest
import numpy as np
from synth_pdb.orientogram import compute_6d_orientations

def test_orientations_batch_consistency():
    """Verify that orientations are identical regardless of batch size."""
    B = 4
    L = 10
    N_atoms = L * 4 # N, CA, C, CB
    
    # Mock coordinates
    coords = np.random.randn(B, N_atoms, 3)
    atom_names = ["N", "CA", "C", "CB"] * L
    residue_indices = []
    for i in range(1, L + 1):
        residue_indices.extend([i] * 4)
        
    # Full batch
    res_full = compute_6d_orientations(coords, atom_names, residue_indices, L)
    
    # Individual runs
    for i in range(B):
        res_single = compute_6d_orientations(coords[i:i+1], atom_names, residue_indices, L)
        
        for key in res_full:
            np.testing.assert_allclose(res_full[key][i], res_single[key][0], atol=1e-6)

def test_orientations_with_missing_atoms():
    """Verify orientations handle missing non-core atoms gracefully."""
    B = 1
    L = 5
    # Only N, CA, C for all
    atom_names = ["N", "CA", "C"] * L
    residue_indices = []
    for i in range(1, L + 1):
        residue_indices.extend([i] * 3)
    
    coords = np.random.randn(B, len(atom_names), 3)
    
    # This should trigger virtual CB reconstruction for ALL residues
    res = compute_6d_orientations(coords, atom_names, residue_indices, L)
    
    assert res['dist'].shape == (B, L, L)
    assert not np.any(np.isnan(res['dist']))
    assert not np.any(np.isnan(res['omega']))
