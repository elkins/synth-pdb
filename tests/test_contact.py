import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.contact import compute_contact_map

@pytest.fixture
def mock_structure():
    """Create a simple 2-residue structure."""
    # Res 1 at Origin, Res 2 at (10, 0, 0)
    # Distance = 10.0
    
    structure = struc.AtomArray(2)
    structure.coord = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    structure.atom_name = np.array(["CA", "CA"])
    structure.res_name = np.array(["ALA", "ALA"])
    structure.chain_id = np.array(["A", "A"])
    structure.res_id = np.array([1, 2])
    structure.element = np.array(["C", "C"])
    return structure

def test_distance_matrix(mock_structure):
    """Test raw distance calculation."""
    matrix = compute_contact_map(mock_structure, method="ca", power=None)
    
    assert matrix.shape == (2, 2)
    # Diagonal should be 0
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0
    # Off-diagonal should be 10.0
    assert matrix[0, 1] == 10.0
    assert matrix[1, 0] == 10.0

def test_binary_map(mock_structure):
    """Test binary thresholding."""
    # Threshold 8.0 < Distance 10.0 -> Should be 0
    binary_low = compute_contact_map(mock_structure, method="ca", threshold=8.0, power=0)
    assert binary_low[0, 1] == 0.0

    # Threshold 12.0 > Distance 10.0 -> Should be 1
    binary_high = compute_contact_map(mock_structure, method="ca", threshold=12.0, power=0)
    assert binary_high[0, 1] == 1.0

def test_method_selection(mock_structure):
    """Test standard CA method."""
    # Ensure it filters for CA
    # Add a junk atom (CB) to confuse it
    complex_structure = struc.AtomArray(3)
    complex_structure.atom_name = np.array(["CA", "CB", "CA"])
    complex_structure.element = np.array(["C", "C", "C"])
    complex_structure.res_name = np.array(["ALA", "ALA", "ALA"]) # Needs CA for filtering logic if any
    
    matrix = compute_contact_map(complex_structure, method="ca")
    assert matrix.shape == (2, 2) # Should ignore CB

def test_noe_intensity(mock_structure):
    """Test 1/r^6 intensity calculation."""
    # Distance = 10.0
    # Intensity = 1e-6
    matrix = compute_contact_map(mock_structure, method="ca", power=6)
    
    expected = 1.0 / (10.0 ** 6)
    assert pytest.approx(matrix[0, 1]) == expected
    
    # Diagonal should be 0 (handled by fill_diagonal)
    assert matrix[0, 0] == 0.0
