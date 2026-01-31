import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.geometry import (
    position_atom_3d_from_internal_coords,
    calculate_angle,
    calculate_dihedral_angle,
    reconstruct_sidechain
)
import synth_pdb.geometry as geometry_mod

def test_calculate_angle_edge_cases():
    """Test calculate_angle with zero vectors and collinear points."""
    # Zero vector case (p2 == p1)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    assert calculate_angle(p1, p2, p3) == 0.0
    
    # Collinear points (180 degrees)
    p1 = np.array([-1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(calculate_angle(p1, p2, p3)) == 180.0
    
    # Exact 0 degrees (p1 and p3 on same side)
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([2.0, 0.0, 0.0])
    assert pytest.approx(calculate_angle(p1, p2, p3)) == 0.0

def test_calculate_dihedral_angle_edge_cases():
    """Test calculate_dihedral_angle with collinear points (singular planes)."""
    # Collinear p1, p2, p3 (plane 1 is undefined)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([2.0, 0.0, 0.0])
    p4 = np.array([2.0, 1.0, 0.0])
    # The current implementation returns 0.0 or handles with safe normalization
    val = calculate_dihedral_angle(p1, p2, p3, p4)
    assert isinstance(val, float)
    
    # Missing bond (p2 == p3)
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.0])
    p4 = np.array([1.0, 0.0, 0.0])
    val = calculate_dihedral_angle(p1, p2, p3, p4)
    assert val == 0.0

def test_position_atom_3d_edge_cases():
    """Test NeRF geometry with extreme angles."""
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    
    # 0 degree bond angle
    pos = position_atom_3d_from_internal_coords(p1, p2, p3, 1.0, 0.0, 180.0)
    # 0 angle means p4 is along p3-p2 vector but length 1.0 from p3
    # b = p3-p2 = [1, 0, 0]. p4 = p3 + 1.0 * (-b) = [1,0,0] - [1,0,0] = [0,0,0]
    assert pytest.approx(pos) == np.array([0.0, 0.0, 0.0])

def test_reconstruct_sidechain_missing_atoms_coverage():
    """Trigger missing atom warning paths in reconstruct_sidechain."""
    # Create simple structure
    atom = struc.Atom(res_id=1, res_name="ALA", atom_name="CA", coord=[0,0,0])
    peptide = struc.array([atom])
    
    # Missing N and C
    reconstruct_sidechain(peptide, 1, {'chi1': [60.0]})
    
    # Missing N only
    peptide += struc.array([struc.Atom(res_id=1, res_name="ALA", atom_name="C", coord=[1,0,0])])
    reconstruct_sidechain(peptide, 1, {'chi1': [60.0]})

def test_reconstruct_sidechain_achiral_gly():
    """Verify GLY (achiral) handles skip logic."""
    atom_ca = struc.Atom(res_id=1, res_name="GLY", atom_name="CA", coord=[0,0,0])
    atom_n = struc.Atom(res_id=1, res_name="GLY", atom_name="N", coord=[-1,0,0])
    atom_c = struc.Atom(res_id=1, res_name="GLY", atom_name="C", coord=[0,1,0])
    peptide = struc.array([atom_ca, atom_n, atom_c])
    
    # GLY has no CB, so should return early/no-op
    reconstruct_sidechain(peptide, 1, {'chi1': [60.0]})
    assert len(peptide[peptide.atom_name == "CB"]) == 0

def test_numba_fallback_logic():
    """Test the no-op JIT decorator definition."""
    def dummy_func(x):
        return x + 1
    
    # If numba is imported, njit will be the real one. 
    # If not, it's our fallback.
    # We can test the fallback directly if we import it from the module
    from synth_pdb.geometry import njit as fallback_njit
    
    decorated = fallback_njit(dummy_func)
    assert decorated(1) == 2
    
    decorated_with_args = fallback_njit(cache=True)(dummy_func)
    assert decorated_with_args(5) == 6
