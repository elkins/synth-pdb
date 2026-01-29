
import pytest
import numpy as np
from synth_pdb.generator import _sample_ramachandran_angles
from synth_pdb.data import RAMACHANDRAN_REGIONS

def test_sample_ramachandran_arguments():
    """
    TDD: Verify _sample_ramachandran_angles accepts next_res_name argument.
    Currently this will fail with TypeError.
    """
    try:
        phi, psi = _sample_ramachandran_angles('ALA', next_res_name='PRO')
    except TypeError:
        pytest.fail("Function does not accept 'next_res_name' argument")

def test_pre_pro_definitions_exist():
    """
    TDD: Verify PRE_PRO definitions exist in data.py.
    Currently fails because key is missing.
    """
    assert 'PRE_PRO' in RAMACHANDRAN_REGIONS, "PRE_PRO distribution missing from RAMACHANDRAN_REGIONS"

def test_pre_pro_distribution_bias():
    """
    Statistical verification of Pre-Proline bias.
    Pre-Proline residues (residue before PRO) should have:
    - Restricted Alpha region (steric clash)
    - Enhanced Beta region preference
    """
    # Sample many points
    phis = []
    psis = []
    
    # We expect the function to handle the logic: if next is PRO, use PRE_PRO distribution
    for _ in range(1000):
        p, s = _sample_ramachandran_angles('ALA', next_res_name='PRO')
        phis.append(p)
        psis.append(s)
        
    phis = np.array(phis)
    psis = np.array(psis)
    
    # Check 1: Alpha region should be significantly depopulated compared to general
    # General Alpha is roughly -60, -45. 
    # Pre-Pro Alpha is restricted/less favorable.
    # Actually, Pre-Pro favors Beta (extended) significantly more.
    
    # Count Fraction in Beta vs Alpha
    # Alpha: phi < 0, -100 < psi < 0
    # Beta: phi < 0, psi > 90
    
    alpha_mask = (phis < -40) & (phis > -100) & (psis > -80) & (psis < -20)
    beta_mask = (phis < -40) & (phis > -180) & (psis > 90) & (psis < 180)
    
    alpha_frac = np.sum(alpha_mask) / 1000.0
    beta_frac = np.sum(beta_mask) / 1000.0
    
    print(f"Alpha Fraction: {alpha_frac}, Beta Fraction: {beta_frac}")
    
    # In standard Lovell/Richardson data, Pre-Pro is >60% Beta/Extended, <30% Alpha
    # Whereas General ALA is >50% Alpha.
    
    assert beta_frac > alpha_frac, "Pre-Pro should favor Beta/Extended over Alpha"
    assert beta_frac > 0.50, "Pre-Pro should be predominately Extended/Beta"

def test_gly_pro_exception():
    """
    TDD: GLY and PRO as current residues have their own strong preferences 
    that might override Pre-Pro effects or blend with them.
    However, for simplicity in this implementation, if current is GLY/PRO, 
    we often keep their specific maps because they are unique.
    
    Documentation/Plan says: "If current is not Gly/Pro..."
    So if we pass GLY, we should get GLY distribution, even if next is PRO.
    """
    # Sample GLY before PRO
    phis = []
    for _ in range(500):
        p, s = _sample_ramachandran_angles('GLY', next_res_name='PRO')
        phis.append(p)
        
    # Glycine has positive Phi values (left side of plot). General/Pre-Pro does not.
    # If we see significant positive Phis, we know it's using GLY map.
    pos_phi_frac = np.sum(np.array(phis) > 0) / 500.0
    
    assert pos_phi_frac > 0.1, "GLY-PRO should still retain GLY characteristics (positive Phi allowed)"
