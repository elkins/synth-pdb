import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.export import export_constraints

@pytest.fixture
def mock_matrix():
    """Create a 4x4 matrix representing 4 residues."""
    # Res 1-4
    # Contacts: (1, 4) is Long Range. (1, 2) is neighbor.
    mat = np.zeros((4, 4))
    mat[0, 3] = 1.0 # Contact between 1 and 4
    mat[3, 0] = 1.0
    
    mat[0, 1] = 1.0 # Neighbor
    mat[1, 0] = 1.0
    return mat

def test_export_casp_format(mock_matrix):
    """Test exporting to CASP RR format."""
    # Sequence: A A A A
    seq = "AAAA"
    
    output = export_constraints(mock_matrix, sequence=seq, fmt="casp")
    
    lines = output.strip().split("\n")
    # Header should perform sequence
    assert lines[0] == seq
    
    # Body: i j d1 d2 p
    # Should contain 1 4 0 8 1.0
    # Neighbors (1 2) might be filtered or included depending on logic.
    # Standard CASP usually wants separation >= 6.
    # But let's assume we output ALL for now defined by the matrix.
    
    # Finding the line for 1-4 contact (Indices 0, 3 -> ResIDs 1, 4)
    found = False
    for line in lines:
        if line.startswith("1 4"):
            parts = line.split()
            assert parts[0] == "1"
            assert parts[1] == "4"
            assert float(parts[2]) == 0.0 # d_min
            assert float(parts[3]) == 8.0 # d_max (Standard contact threshold)
            assert float(parts[4]) == 1.0 # Probability
            found = True
            
    assert found

def test_export_csv_format(mock_matrix):
    """Test simple CSV export."""
    seq = "AAAA"
    output = export_constraints(mock_matrix, sequence=seq, fmt="csv")
    
    # Expect: ResID1,ResID2,Probability
    assert "1,4,1.0" in output
