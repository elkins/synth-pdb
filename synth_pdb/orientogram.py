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

    ### EDUCATIONAL NOTE - trRosetta 6D Orientations:
    -------------------------------------------------
    In protein structure prediction (like trRosetta or AlphaFold), 3D geometry 
    is often described by 4 parameters between residue pairs (i, j):
    1. dist:  Distance between C-beta atoms (or virtual CB for Glycine).
    2. omega: Dihedral angle (Ca_i, Cb_i, Cb_j, Ca_j).
    3. theta: Angle (Ca_i, Cb_i, Cb_j).
    4. phi:   Dihedral angle (N_i, Ca_i, Cb_i, Cb_j).

    These parameters completely define the relative position and orientation 
    of two residues in space. Because they are "Internal Coordinates", they 
    are invariant to global rotation and translation, making them ideal 
    features for Machine Learning models.
    
    In this tool, we use NeRF (Natural Extension Reference Frame) to reconstruct 
    the "Virtual C-beta" for Glycine, ensuring the 6D map is dense and consistent.

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
    # Using vectorized boolean masking for speed
    n_coords = np.zeros((B, L, 3))
    ca_coords = np.zeros((B, L, 3))
    c_coords = np.zeros((B, L, 3))
    cb_coords = np.zeros((B, L, 3))
    has_cb = np.zeros(L, dtype=bool)

    atom_names_arr = np.array(atom_names)
    res_indices_arr = np.array(residue_indices) - 1 # 0-indexed

    # Create masks for each atom type
    n_mask = atom_names_arr == "N"
    ca_mask = atom_names_arr == "CA"
    c_mask = atom_names_arr == "C"
    cb_mask = atom_names_arr == "CB"

    # Assign coordinates using fancy indexing
    # We use res_indices_arr[mask] to map the B-batch of specific atoms to their residues
    n_coords[:, res_indices_arr[n_mask]] = coords[:, n_mask]
    ca_coords[:, res_indices_arr[ca_mask]] = coords[:, ca_mask]
    c_coords[:, res_indices_arr[c_mask]] = coords[:, c_mask]
    cb_coords[:, res_indices_arr[cb_mask]] = coords[:, cb_mask]
    has_cb[res_indices_arr[cb_mask]] = True

    # 2. Handle GLY (Virtual C-beta)
    # If C-beta is missing, reconstruct it using standard geometry
    # NeRF: N -> C -> Ca -> Cb
    missing_cb = np.where(~has_cb)[0]
    if len(missing_cb) > 0:
        # Vectorized reconstruction across B batches for all missing residues
        p1 = n_coords[:, missing_cb] # (B, num_missing, 3)
        p2 = c_coords[:, missing_cb]
        p3 = ca_coords[:, missing_cb]
        
        # Flatten to (B * num_missing, 3) for position_atoms_batch compatibility
        num_missing = len(missing_cb)
        p1_f = p1.reshape(-1, 3)
        p2_f = p2.reshape(-1, 3)
        p3_f = p3.reshape(-1, 3)
        
        # Ideal L-Alanine C-beta geometry (Z-Matrix)
        bl = np.full(B * num_missing, 1.522)
        ba = np.full(B * num_missing, 110.1)
        di = np.full(B * num_missing, -122.66)
        
        cb_placed = position_atoms_batch(p1_f, p2_f, p3_f, bl, ba, di)
        cb_coords[:, missing_cb] = cb_placed.reshape(B, num_missing, 3)

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
