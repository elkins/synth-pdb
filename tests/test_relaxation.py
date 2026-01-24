import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.relaxation import calculate_relaxation_rates, spectral_density
import logging

logger = logging.getLogger(__name__)

def test_spectral_density_function():
    """Test standard J(w) behavior."""
    # Tests that J(w) decreases with frequency
    tau_m = 10e-9 # 10ns
    
    j_0 = spectral_density(0, tau_m, s2)
    j_high = spectral_density(1e9, tau_m, s2)
    
    assert j_0 > 0
    assert j_high > 0
    assert j_0 > j_high # Spectral density decays at high frequency

def test_relaxation_trends():
    """Test that rigid regions have different rates than flexible ones."""
    # Create dummy structure: 3 residues
    # 1 and 3 are termini (flexible S2=0.5), 2 is center (rigid S2=0.85)
    
    structure = struc.AtomArray(15)
    # 5 Residues: 1, 2, 3, 4, 5
    # 1,5 = Termini (0.5)
    # 2,4 = Penultimate (0.7)
    # 3 = Core (0.85)
    
    ids = []
    for i in range(1, 6):
        ids.extend([i, i, i])
    
    structure.res_id = np.array(ids)
    structure.res_name = np.array(["ALA"]*15)
    structure.atom_name = np.array(["N", "CA", "H"]*5)
    
    # Set seed for reproducibility of random noise
    np.random.seed(42)
    
    rates = calculate_relaxation_rates(structure, field_mhz=600, tau_m_ns=10.0)
    
    s2_term = rates[1]['S2']
    s2_core = rates[3]['S2']
    
    noe_term = rates[1]['NOE']
    noe_core = rates[3]['NOE']
    
    logger.info(f"Term S2: {s2_term}, Core S2: {s2_core}")
    logger.info(f"Term NOE: {noe_term}, Core NOE: {noe_core}")
    
    # Core should be more rigid (Higher S2)
    assert s2_core > s2_term
    
    # PHYSICS NOTE:
    # In simple Model-Free with te=0, J(w) scales linearly with S2.
    # Since NOE ~ 1 + (Sigma/R1), and both Sigma and R1 scale with S2,
    # S2 cancels out! NOE mainly measures Tumbling Time (tau_m), not S2.
    # To see NOE dips, we'd need fast internal motions (tau_e > 0).
    # So we asserting on NOE here is wrong for this simple model.
    # Instead, R1 and R2 scale directly with S2.
    
    # Rigid -> Larger R2 (faster transverse decay)
    # R2 is roughly proportional to S2 * tau_m
    assert rates[3]['R2'] > rates[1]['R2']
    
    # Rigid -> Larger R1 (usually, unless near min)
    assert rates[3]['R1'] > rates[1]['R1']

def test_proline_exclusion():
    """Ensure Prolines are skipped (no amide proton)."""
    structure = struc.AtomArray(3)
    structure.res_id = [1, 1, 1]
    structure.res_name = ["PRO", "PRO", "PRO"]
    structure.atom_name = ["N", "CA", "CD"] # No H
    
    rates = calculate_relaxation_rates(structure)
    assert len(rates) == 0

