import numpy as np
import random
from typing import List, Dict, Optional
from .data import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
    ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_C_N_CA,
    ONE_TO_THREE_LETTER_CODE, RAMACHANDRAN_PRESETS
)
from .geometry import position_atoms_batch

# EDUCATIONAL OVERVIEW - Batched Generation (GPU-First):
# ----------------------------------------------------
# Traditional protein generators (like the serial Generator in generator.py)
# process structures one-by-one. While easy to code, this is a bottleneck for
# training Deep Learning models which require millions of samples.
#
# BatchedGenerator uses "Vectorized Math":
# 1. Parallelism: It processes B structures at once (e.g., B=1000).
# 2. Broadcasting: Using NumPy's broadcasting, a single mathematical expression
#    calculates positions for all structures in the batch simultaneously.
# 3. Hardware Acceleration: On Apple Silicon (M4), this leverages AMX/Accelerate
#    units, often providing 10-100x speedups over Python loops.
#
# This architecture is "ML-Ready" - the output is a single contiguous tensor
# that can be passed directly to frameworks like MLX, PyTorch, or JAX.

class BatchedPeptide:
    """
    A lightweight container for batched protein coordinates.
    Designed for high-performance handover to ML frameworks.
    """
    def __init__(self, coords: np.ndarray, sequence: List[str]):
        self.coords = coords # (B, N_atoms, 3)
        self.sequence = sequence

class BatchedGenerator:
    """
    High-performance vectorized protein structure generator.
    Optimized for generating millions of labeled samples for AI training.
    """
    def __init__(self, sequence_str: str, n_batch: int = 1):
        # Resolve sequence (simplified for now to match TDD)
        if "-" in sequence_str:
            self.sequence = [s.strip() for s in sequence_str.split("-")]
        else:
            # Assume 1-letter codes if no dashes
            self.sequence = [ONE_TO_THREE_LETTER_CODE.get(c, "ALA") for c in sequence_str]
        
        self.n_batch = n_batch
        self.n_res = len(self.sequence)

    def generate_batch(self, seed: Optional[int] = None, conformation: str = 'alpha', drift: float = 0.0) -> BatchedPeptide:
        """
        Generates B structures in parallel.
        
        This method replaces the traditional per-residue loop with a "Batch Walk".
        Instead of placing atoms for structure 1, then structure 2... it places
        atom 'N' for ALL structures, then 'CA' for ALL structures, and so on.

        Args:
            seed: Random seed for reproducible batch generation.
            conformation: The secondary structure preset to use for all members.
            drift: Gaussian noise (std dev) in degrees. Use this to generate "hard decoys"
                   that challenge AI models with near-native but slightly incorrect geometry.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        B = self.n_batch
        L = self.n_res
        
        # We only generate backbone for now (N, CA, C, O) - 4 atoms per residue
        n_atoms = L * 4
        coords = np.zeros((B, n_atoms, 3))
        
        # 1. Place first residue (N, CA, C) at origin frame
        coords[:, 0] = [0, 0, 0] # N
        coords[:, 1] = [BOND_LENGTH_N_CA, 0, 0] # CA
        ang = np.deg2rad(ANGLE_N_CA_C)
        coords[:, 2] = [
            BOND_LENGTH_N_CA - BOND_LENGTH_CA_C * np.cos(ang),
            BOND_LENGTH_CA_C * np.sin(ang),
            0
        ]
        
        # Resolve preset angles
        preset = RAMACHANDRAN_PRESETS.get(conformation, RAMACHANDRAN_PRESETS['alpha'])
        p_phi = preset['phi']
        p_psi = preset['psi']
        
        # Sample torsions for the entire batch (B, L)
        phi = np.full((B, L), p_phi)
        psi = np.full((B, L), p_psi)
        omega = np.full((B, L), 180.0)
        
        if drift > 0:
            phi += np.random.normal(0, drift, (B, L))
            psi += np.random.normal(0, drift, (B, L))
            omega += np.random.normal(0, 2.0, (B, L)) # Fixed small omega drift
        
        # EDUCATIONAL NOTE - Peptidyl Chain Walk:
        # We construct the chain N -> CA -> C iteratively. 
        # For each residue (i), we use the coordinates of (i-1) to place the new atoms.
        
        from .data import BOND_LENGTH_C_O, ANGLE_CA_C_O, ANGLE_CA_C_N, ANGLE_C_N_CA
        
        for i in range(L):
            idx = i * 4
            if i == 0:
                # Place O(0) using N(0), CA(0), C(0)
                p1, p2, p3 = coords[:, 0], coords[:, 1], coords[:, 2]
                bl, ba, di = np.full(B, BOND_LENGTH_C_O), np.full(B, ANGLE_CA_C_O), np.full(B, 180.0)
                coords[:, 3] = position_atoms_batch(p1, p2, p3, bl, ba, di)
            else:
                # Place N(i) using N(i-1), CA(i-1), C(i-1)
                p1, p2, p3 = coords[:, (i-1)*4], coords[:, (i-1)*4+1], coords[:, (i-1)*4+2]
                bl, ba, di = np.full(B, BOND_LENGTH_C_N), np.full(B, ANGLE_CA_C_N), psi[:, i-1]
                coords[:, idx] = position_atoms_batch(p1, p2, p3, bl, ba, di)
                
                # Place CA(i) using CA(i-1), C(i-1), N(i)
                p1, p2, p3 = coords[:, (i-1)*4+1], coords[:, (i-1)*4+2], coords[:, idx]
                bl, ba, di = np.full(B, BOND_LENGTH_N_CA), np.full(B, ANGLE_C_N_CA), omega[:, i-1]
                coords[:, idx+1] = position_atoms_batch(p1, p2, p3, bl, ba, di)
                
                # Place C(i) using C(i-1), N(i), CA(i)
                p1, p2, p3 = coords[:, (i-1)*4+2], coords[:, idx], coords[:, idx+1]
                bl, ba, di = np.full(B, BOND_LENGTH_CA_C), np.full(B, ANGLE_N_CA_C), phi[:, i]
                coords[:, idx+2] = position_atoms_batch(p1, p2, p3, bl, ba, di)
                
                # Place O(i) using N(i), CA(i), C(i)
                p1, p2, p3 = coords[:, idx], coords[:, idx+1], coords[:, idx+2]
                bl, ba, di = np.full(B, BOND_LENGTH_C_O), np.full(B, ANGLE_CA_C_O), np.full(B, 180.0)
                coords[:, idx+3] = position_atoms_batch(p1, p2, p3, bl, ba, di)

        return BatchedPeptide(coords, self.sequence)
