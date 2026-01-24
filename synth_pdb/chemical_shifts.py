
import numpy as np
import biotite.structure as struc
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# --- Random Coil Chemical Shifts (Wishart et al.) ---
# EDUCATIONAL NOTE - Random Coil Shifts:
# ======================================
# "Random Coil" refers to a protein state with no fixed secondary structure (a flexible chain).
# The chemical shift of an atom in a random coil depends primarily on its amino acid type.
#
# These values serve as the "baseline" or "zero point" for structure prediction.
# Any deviation from these values (Secondary Shift) indicates structural formation:
# - Alpha Helix formation moves C-alpha downfield (higher ppm) and N upfield (lower ppm).
# - Beta Sheet formation moves C-alpha upfield (lower ppm) and N downfield (higher ppm).
#
# Reference: Wishart, D.S. et al. (1995) J. Biomol. NMR.
# Referenced to DSS at 25C.
# Values for: HA, CA, CB, C, N, HN (Amide H)
# Units: ppm
RANDOM_COIL_SHIFTS: Dict[str, Dict[str, float]] = {
    "ALA": {"HA": 4.32, "CA": 52.5, "CB": 19.1, "C": 177.8, "N": 123.8, "H": 8.24},
    "ARG": {"HA": 4.34, "CA": 56.0, "CB": 30.9, "C": 176.3, "N": 121.3, "H": 8.23},
    "ASN": {"HA": 4.75, "CA": 53.1, "CB": 38.9, "C": 175.2, "N": 118.7, "H": 8.75},
    "ASP": {"HA": 4.66, "CA": 54.2, "CB": 41.1, "C": 176.3, "N": 120.4, "H": 8.34},
    "CYS": {"HA": 4.69, "CA": 58.2, "CB": 28.0, "C": 174.6, "N": 118.8, "H": 8.32}, # Reduced
    "GLN": {"HA": 4.32, "CA": 56.0, "CB": 29.4, "C": 176.0, "N": 120.4, "H": 8.25},
    "GLU": {"HA": 4.29, "CA": 56.6, "CB": 29.9, "C": 176.6, "N": 120.2, "H": 8.35},
    "GLY": {"HA": 3.96, "CA": 45.1, "CB": 0.0,  "C": 174.9, "N": 108.8, "H": 8.33}, # HA2/HA3 split usually, simplified here
    "HIS": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "ILE": {"HA": 4.17, "CA": 61.1, "CB": 38.8, "C": 176.4, "N": 121.4, "H": 8.00},
    "LEU": {"HA": 4.34, "CA": 55.1, "CB": 42.4, "C": 177.6, "N": 121.8, "H": 8.16},
    "LYS": {"HA": 4.32, "CA": 56.2, "CB": 33.1, "C": 176.6, "N": 120.4, "H": 8.29},
    "MET": {"HA": 4.48, "CA": 55.4, "CB": 32.6, "C": 176.3, "N": 119.6, "H": 8.28},
    "PHE": {"HA": 4.62, "CA": 57.7, "CB": 39.6, "C": 175.8, "N": 120.3, "H": 8.12},
    "PRO": {"HA": 4.42, "CA": 63.3, "CB": 32.1, "C": 177.3, "N": 0.0,   "H": 0.0},  # No Amide N/H
    "SER": {"HA": 4.47, "CA": 58.3, "CB": 63.8, "C": 174.6, "N": 115.7, "H": 8.31},
    "THR": {"HA": 4.35, "CA": 61.8, "CB": 69.8, "C": 174.7, "N": 113.6, "H": 8.15},
    "TRP": {"HA": 4.66, "CA": 57.5, "CB": 29.6, "C": 176.1, "N": 121.3, "H": 8.25},
    "TYR": {"HA": 4.55, "CA": 57.9, "CB": 38.8, "C": 175.9, "N": 120.3, "H": 8.12},
    "VAL": {"HA": 4.12, "CA": 62.2, "CB": 32.9, "C": 176.3, "N": 119.9, "H": 8.03},
}

# --- Secondary Structure Offsets (Sparta-Lite) ---
# EDUCATIONAL NOTE - Secondary Chemical Shifts:
# =============================================
# The local magnetic field experienced by a nucleus is heavily influenced by the
# geometry of the protein backbone (Phi/Psi angles).
#
# SPARTA-lite (Simplified prediction):
# "SPARTA" stands for "Shift Prediction from Analogy in Residue-type and Torsion Angle".
# It predicts chemical shifts by finding homologous structures with similar geometry.
#
# Our "Lite" version uses simple statistical offsets instead of database mining,
# but follows the same principle: Geometry determines Shift.
#
# Reference State: DSS (4,4-dimethyl-4-silapentane-1-sulfonic acid)
# This is the "Zero" for proton/carbon NMR, much like sea level for altitude.
# Using a standard reference ensures shifts are comparable across different labs.
#
# Approximate mean offsets for Helical and Sheet conformations relative to random coil
# Based on general statistics (e.g. Spera & Bax 1991)
# Format: {metric: {Helix: val, Sheet: val}}
SECONDARY_SHIFTS: Dict[str, Dict[str, float]] = {
    # C-alpha: Shifted downfield (positive) in Helix, upfield (negative) in Sheet
    "CA": {"alpha": 3.1,  "beta": -1.5},
    # C-beta: Opposite trend to C-alpha
    "CB": {"alpha": -0.5, "beta": 2.2},
    # Carbonyl Carbon: Follows C-alpha trend
    "C":  {"alpha": 2.2,  "beta": -1.6},
    # H-alpha: Shifted upfield (negative) in Helix, downfield (positive) in Sheet
    "HA": {"alpha": -0.4, "beta": 0.5},
    # Amide N: Complex, but generally upfield in Helix
    "N":  {"alpha": -1.5, "beta": 1.2},
    "H":  {"alpha": -0.2, "beta": 0.3},
}

def predict_chemical_shifts(structure: struc.AtomArray) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Predict chemical shifts based on secondary structure (Phi/Psi).
    
    EDUCATIONAL NOTE - Prediction Algorithm:
    ========================================
    1. Calculate Backbone Dihedrals (Phi/Psi) for every residue.
    2. Classify Secondary Structure:
       - Alpha: Phi ~ -60, Psi ~ -45
       - Beta:  Phi ~ -120, Psi ~ 120
       - Coil:  Everything else
    3. Calculate Shift:
       Shift = Random_Coil + Structure_Offset + Noise
       
    LIMITATIONS:
    - Ring Current Effects: Aromatic rings (Phe, Tyr, Trp) create strong magnetic
      fields that shift nearby protons. We omit this for simplicity ($O(N^2)$ geometry check).
    - H-Bonding: Hydrogen bonds affect Amide H shifts significantly. We omit this.
    - Sequence History: Real shifts depend on (i-1) and (i+1) neighbor types. We omit this.
    
    Args:
        structure: AtomArray containing the protein
        
    Returns:
        shifts: Dict[chain_id, Dict[res_id, Dict[atom_name, value]]]
    """
    logger.info("Predicting Chemical Shifts (SPARTA-lite)...")
    
    # Calculate dihedrals
    phi, psi, omega = struc.dihedral_backbone(structure)
    # phi/psi arrays match residue count.
    
    # Helper to clean invalid angles (NaN at termini)
    # We will assume 'coil' (offset 0) for termini where angle is NaN
    
    # We need to iterate over residues
    res_starts = struc.get_residue_starts(structure)
    
    results = {} # Keyed by Chain -> ResID -> Atom -> Value
    
    for i, start_idx in enumerate(res_starts):
        # Identify residue
        res_atoms = structure[start_idx : res_starts[i+1] if i+1 < len(res_starts) else None]
        res_name = res_atoms.res_name[0]
        chain_id = res_atoms.chain_id[0]
        res_id = res_atoms.res_id[0]
        
        if res_name not in RANDOM_COIL_SHIFTS:
            continue
            
        # Get Angles (degrees)
        p = np.rad2deg(phi[i])
        s = np.rad2deg(psi[i])
        
        # Determine Secondary Structure State
        # Simple regions:
        # Alpha: Phi ~ -60 (+/- 30), Psi ~ -45 (+/- 40)
        # Beta:  Phi ~ -120 (+/- 40), Psi ~ 120 (+/- 50)
        ss_state = "coil"
        
        if not np.isnan(p) and not np.isnan(s):
            if (-90 < p < -30) and (-90 < s < -10):
                ss_state = "alpha"
            elif (-160 < p < -80) and (80 < s < 170):
                ss_state = "beta"
            # Else coil
            logger.debug(f"DEBUG: Res {i} {res_name}: Phi={p:.1f}, Psi={s:.1f} -> {ss_state}")
        else:
             logger.debug(f"DEBUG: Res {i} {res_name}: Phi={p:.1f}, Psi={s:.1f} -> NaN/Coil")
        
        # Calculate Shifts
        rc = RANDOM_COIL_SHIFTS[res_name]
        atom_shifts = {}
        
        for atom_type, base_val in rc.items():
            offset = SECONDARY_SHIFTS.get(atom_type, {}).get(ss_state, 0.0)
            
            # Add small random noise for "realism" (0.1 - 0.3 ppm)
            # Experimental assignments always have error/variation
            noise = np.random.normal(0, 0.15) if base_val != 0 else 0
            
            if base_val != 0:
                final_val = base_val + offset + noise
                atom_shifts[atom_type] = round(final_val, 3)
        
        if chain_id not in results:
            results[chain_id] = {}
        results[chain_id][res_id] = atom_shifts
        
    return results
