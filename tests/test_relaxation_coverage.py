import pytest
import numpy as np
import biotite.structure as struc
from unittest.mock import patch, MagicMock
from synth_pdb.relaxation import predict_order_parameters, calculate_relaxation_rates, spectral_density

def test_spectral_density_tau_f():
    """Test spectral density with fast internal motion (tau_f > 0)."""
    # This hits lines 58-60
    j1 = spectral_density(0.1, 1e-8, 0.8, tau_f=0.0)
    j2 = spectral_density(0.1, 1e-8, 0.8, tau_f=1e-10)
    assert j2 > j1

def test_predict_order_parameters_empty():
    """Test order parameter prediction with empty structure."""
    empty = struc.AtomArray(0)
    assert predict_order_parameters(empty) == {}

def test_predict_order_parameters_ptm_and_ions():
    """Test SASA filtering for PTM atoms and ions."""
    # Create an atom array with a PTM atom (P) and an Ion (ZN)
    atoms = struc.AtomArray(5)
    atoms.res_name = np.array(["SER", "SER", "SEP", "SEP", "ZN"])
    atoms.res_id = np.array([1, 1, 2, 2, 3])
    atoms.atom_name = np.array(["N", "CA", "CA", "P", "ZN"])
    atoms.coord = np.zeros((5, 3))
    atoms.element = np.array(["N", "C", "C", "P", "ZN"])
    
    # We mock struc.sasa to return NaNs to hit line 129
    with patch("biotite.structure.sasa", return_value=np.array([np.nan]*5)):
        s2_map = predict_order_parameters(atoms)
        assert len(s2_map) > 0

def test_calculate_relaxation_rates_proline():
    """Verify Proline is excluded from NH relaxation."""
    atoms = struc.AtomArray(3)
    atoms.res_name = np.array(["PRO", "PRO", "PRO"])
    atoms.res_id = np.array([1, 1, 1])
    atoms.atom_name = np.array(["N", "CA", "CD"]) # No H
    atoms.coord = np.zeros((3, 3))
    
    # Add a mock Hydrogen to PRO to try and trigger the loop check
    atoms_with_h = struc.AtomArray(4)
    atoms_with_h.res_name = np.array(["PRO", "PRO", "PRO", "PRO"])
    atoms_with_h.res_id = np.array([1, 1, 1, 1])
    atoms_with_h.atom_name = np.array(["N", "CA", "CD", "H"])
    atoms_with_h.coord = np.zeros((4, 3))
    
    rates = calculate_relaxation_rates(atoms_with_h)
    assert 1 not in rates # Proline should be skipped even if it somehow has an H in our mock

def test_numba_fallback_coverage():
    """Trigger Numba fallback logic by mocking import error."""
    import sys
    with patch.dict("sys.modules", {"numba": None}):
        # Reload the module to trigger the except ImportError
        import importlib
        import synth_pdb.relaxation as relax
        importlib.reload(relax)
        
        @relax.njit
        def test_func(x):
            return x + 1
        
        assert test_func(5) == 6
        
        @relax.njit()
        def test_func_args(x):
            return x * 2
        assert test_func_args(5) == 10
