import numpy as np
from typing import Dict, Tuple
from .geometry import batched_angle, batched_dihedral, position_atoms_batch

def compute_6d_orientations(
    coords: np.ndarray, 
    atom_names: list, 
    residue_indices: list,
    n_residues: int
) -> Dict[str, np.ndarray]:
    """
    Computes 6D inter-residue orientations for all pairs of residues.
    This follows the trRosetta convention: (dist, omega, theta, phi).
    
    Args:
        coords: (B, N_atoms, 3) tensor of atomic coordinates.
        atom_names: List of atom names for the N_atoms.
        residue_indices: List of residue indices for the N_atoms.
        n_residues: Number of residues in the peptide.
        
    Returns:
        orientations: Dictionary containing:
            'dist': (B, L, L) - C-beta distances
            'omega': (B, L, L) - Ca_i-Cb_i-Cb_j-Ca_j dihedral
            'theta': (B, L, L) - Ca_i-Cb_i-Cb_j angle
            'phi': (B, L, L) - N_i-Ca_i-Cb_i-Cb_j dihedral
    """
    B = coords.shape[0]
    L = n_residues
    
    # 1. Extract core frame atoms (N, Ca, C, Cb)
    # Using a dense index map for speed
    n_coords = np.zeros((B, L, 3))
    ca_coords = np.zeros((B, L, 3))
    c_coords = np.zeros((B, L, 3))
    cb_coords = np.zeros((B, L, 3))
    
    has_cb = np.zeros(L, dtype=bool)
    
    for idx, (name, res_idx) in enumerate(zip(atom_names, residue_indices)):
        r = res_idx - 1 # 0-indexed
        if name == "N": n_coords[:, r] = coords[:, idx]
        elif name == "CA": ca_coords[:, r] = coords[:, idx]
        elif name == "C": c_coords[:, r] = coords[:, idx]
        elif name == "CB": 
            cb_coords[:, r] = coords[:, idx]
            has_cb[r] = True

    # 2. Handle GLY (Virtual C-beta)
    # If C-beta is missing, reconstruct it using standard geometry
    # NeRF: N -> C -> Ca -> Cb
    # Parameters for ideal L-Ala C-beta:
    # bl=1.52, ba=110.1, di=-122.6 (relative to N-C-Ca)
    missing_cb = np.where(~has_cb)[0]
    if len(missing_cb) > 0:
        for r in missing_cb:
            p1 = n_coords[:, r]
            p2 = c_coords[:, r]
            p3 = ca_coords[:, r]
            # Ideal L-Alanine C-beta geometry
            bl = np.full(B, 1.522)
            ba = np.full(B, 110.1)
            di = np.full(B, -122.66)
            cb_coords[:, r] = position_atoms_batch(p1, p2, p3, bl, ba, di)

    # 3. Pairwise Geometric Calculations
    # We expand (B, L, 1, 3) and (B, 1, L, 3) to get all pairs (B, L, L, 3)
    cbi = cb_coords[:, :, np.newaxis, :]  # (B, L, 1, 3)
    cbj = cb_coords[:, np.newaxis, :, :]  # (B, 1, L, 3)
    cai = ca_coords[:, :, np.newaxis, :]
    caj = ca_coords[:, np.newaxis, :, :]
    ni = n_coords[:, :, np.newaxis, :]

    # Broadcast indices to (B, L, L, 3)
    cbi_b = np.broadcast_to(cbi, (B, L, L, 3))
    cbj_b = np.broadcast_to(cbj, (B, L, L, 3))
    cai_b = np.broadcast_to(cai, (B, L, L, 3))
    caj_b = np.broadcast_to(caj, (B, L, L, 3))
    ni_b = np.broadcast_to(ni, (B, L, L, 3))

    # A. Distances (B, L, L)
    dist = np.linalg.norm(cbi_b - cbj_b, axis=-1)

    # B. Omega: Dihedral cai-cbi-cbj-caj (B, L, L)
    omega = batched_dihedral(cai_b, cbi_b, cbj_b, caj_b)

    # C. Theta: Angle cai-cbi-cbj (B, L, L)
    theta = batched_angle(cai_b, cbi_b, cbj_b)

    # D. Phi: Dihedral ni-cai-cbi-cbj (B, L, L)
    phi = batched_dihedral(ni_b, cai_b, cbi_b, cbj_b)

    return {
        'dist': dist,
        'omega': omega,
        'theta': theta,
        'phi': phi
    }
