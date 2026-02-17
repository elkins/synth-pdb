import numpy as np
import pytest
from synth_pdb.geometry import position_atoms_batch

def test_normalize_batch():
    """Test the normalize function inside position_atoms_batch."""
    # This is a bit tricky since normalize is a local function.
    # We can't test it directly, but we can test position_atoms_batch
    # with inputs that would cause division by zero if not for the
    # normalization's safety check.
    
    # p1, p2, p3 are collinear, so the cross product will be zero.
    p1 = np.array([[0., 0., 0.]])
    p2 = np.array([[1., 0., 0.]])
    p3 = np.array([[2., 0., 0.]])
    
    bond_lengths = np.array([1.5])
    bond_angles = np.array([90.])
    dihedral_angles = np.array([90.])
    
    # This would fail with a division-by-zero error if not for the
    # normalization's safety check inside position_atoms_batch.
    p4 = position_atoms_batch(p1, p2, p3, bond_lengths, bond_angles, dihedral_angles)
    
    # The exact output is not as important as the fact that it doesn't crash.
    # The result will be mathematically degenerate but should not be NaN.
    assert not np.any(np.isnan(p4))
    
def test_superimpose_batch_reflection():
    """Test superimpose_batch with a reflection."""
    from synth_pdb.geometry import superimpose_batch

    sources = np.array([
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    ])
    
    # Reflection matrix
    reflection = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., -1.]
    ])
    
    targets = np.matmul(sources, reflection.T)
    
    trans, rot = superimpose_batch(sources, targets)
    
    # The determinant of the rotation matrix should be 1, not -1.
    # This checks if the reflection correction logic is working.
    assert np.allclose(np.linalg.det(rot), 1.0)