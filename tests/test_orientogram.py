import pytest
import numpy as np
from synth_pdb.orientogram import compute_6d_orientations

def test_orientations_basic_ala_ala():
    """Test 6D orientation calculation for a simple ALA-ALA dipeptide."""
    # B=1, L=2
    # Define coords for N, CA, C, CB for 2 residues
    # Atom names and residue indices
    atom_names = ["N", "CA", "C", "CB", "N", "CA", "C", "CB"]
    res_indices = [1, 1, 1, 1, 2, 2, 2, 2]
    
    # Simple coordinates: Residue 1 at origin area, Residue 2 shifted by 5A
    coords = np.zeros((1, 8, 3))
    # Res 1
    coords[0, 0] = [0, 0, 0] # N
    coords[0, 1] = [1, 0, 0] # CA
    coords[0, 2] = [2, 0, 0] # C
    coords[0, 3] = [1, 1, 0] # CB
    # Res 2
    coords[0, 4] = [5, 0, 0] # N
    coords[0, 5] = [6, 0, 0] # CA
    coords[0, 6] = [7, 0, 0] # C
    coords[0, 7] = [6, 1, 0] # CB
    
    orientations = compute_6d_orientations(coords, atom_names, res_indices, n_residues=2)
    
    # Assertions on output keys
    for key in ['dist', 'omega', 'theta', 'phi']:
        assert key in orientations
        assert orientations[key].shape == (1, 2, 2)
        
    # Check distances
    # CB1 at (1,1,0), CB2 at (6,1,0) -> dist = 5.0
    assert pytest.approx(orientations['dist'][0, 0, 1]) == 5.0
    assert pytest.approx(orientations['dist'][0, 1, 0]) == 5.0
    assert orientations['dist'][0, 0, 0] == 0.0

def test_orientations_batch():
    """Test that batch processing (B > 1) works correctly."""
    B, L = 2, 3
    atom_names = ["N", "CA", "C", "CB"] * L
    res_indices = []
    for i in range(1, L + 1):
        res_indices.extend([i] * 4)
    
    coords = np.random.rand(B, len(atom_names), 3)
    
    orientations = compute_6d_orientations(coords, atom_names, res_indices, n_residues=L)
    
    for key in ['dist', 'omega', 'theta', 'phi']:
        assert orientations[key].shape == (B, L, L)

def test_orientations_gly_reconstruction():
    """Test that Glycine virtual C-beta is correctly reconstructed."""
    # B=1, L=2 (ALA-GLY)
    atom_names = ["N", "CA", "C", "CB", "N", "CA", "C"]
    res_indices = [1, 1, 1, 1, 2, 2, 2]
    
    coords = np.zeros((1, 7, 3))
    # Res 1 (ALA)
    coords[0, 0] = [0, 0, 0] # N
    coords[0, 1] = [1, 0, 0] # CA
    coords[0, 2] = [1, 1, 0] # C
    coords[0, 3] = [2, 0, 0] # CB
    # Res 2 (GLY) - CB missing
    coords[0, 4] = [5, 0, 0] # N
    coords[0, 5] = [6, 0, 0] # CA
    coords[0, 6] = [6, 1, 0] # C
    
    orientations = compute_6d_orientations(coords, atom_names, res_indices, n_residues=2)
    
    # Check that distances between CB1 and reconstructed CB2 are non-zero and reasonable
    assert orientations['dist'][0, 0, 1] > 0
    assert not np.isnan(orientations['dist'][0, 0, 1])
    
    # The output should have (1, 2, 2) for all keys
    for key in ['dist', 'omega', 'theta', 'phi']:
        assert orientations[key].shape == (1, 2, 2)
        assert not np.any(np.isnan(orientations[key]))

def test_orientations_edge_cases():
    """Test edge cases like single residue (L=1)."""
    atom_names = ["N", "CA", "C", "CB"]
    res_indices = [1, 1, 1, 1]
    coords = np.random.rand(1, 4, 3)
    
    orientations = compute_6d_orientations(coords, atom_names, res_indices, n_residues=1)
    
    assert orientations['dist'][0, 0, 0] == 0.0 # Same residue pair
    for key in ['omega', 'theta', 'phi']:
        assert orientations[key].shape == (1, 1, 1)
        assert not np.isnan(orientations[key][0, 0, 0])
