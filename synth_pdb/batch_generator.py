import numpy as np
import random
from typing import List, Dict, Optional
from .data import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
    ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_C_N_CA,
    ONE_TO_THREE_LETTER_CODE, RAMACHANDRAN_PRESETS
)
from .geometry import position_atoms_batch, superimpose_batch
import biotite.structure as struc

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
#
# EDUCATIONAL NOTE - The "Memory Wall" in AI Training:
# --------------------------------------------------
# When generating millions of protein samples, the bottleneck is rarely the 
# CPU math (thanks to vectorization), but rather the "Memory Wall":
# 
# 1. PCIE Latency: Copying large tensors from CPU to GPU memory can be slower 
#    than actually generating the coordinates. 
# 2. Contiguity: Deep Learning models require contiguous memory blocks. 
#    BatchedGenerator ensures the output is one massive C-style array, 
#    avoiding the "gather" overhead of traditional Python lists.
# 3. Unified Memory: On Apple Silicon (M4), CPU and GPU share the same physical 
#    RAM. This means the coordinate tensor can be "zero-copy" - once generated 
#    by NumPy, it is IMMEDIATELY visible to the Metal/MLX GPU without any 
#    data movement.

class BatchedPeptide:
    """
    A lightweight container for batched protein coordinates.
    Designed for high-performance handover to ML frameworks.
    """
    def __init__(self, coords: np.ndarray, sequence: List[str], atom_names: List[str], residue_indices: List[int]):
        self.coords = coords # (B, N_atoms, 3)
        self.sequence = sequence
        self.atom_names = atom_names
        self.residue_indices = residue_indices
        self.n_structures = self.coords.shape[0]
        self.n_atoms = self.coords.shape[1]
        self.n_residues = len(sequence)

    def __len__(self):
        return self.n_structures

    def __getitem__(self, index):
        if isinstance(index, int):
            return BatchedPeptide(self.coords[index:index+1], self.sequence, self.atom_names, self.residue_indices)
        return BatchedPeptide(self.coords[index], self.sequence, self.atom_names, self.residue_indices)

    def save_pdb(self, path: str, index: int = 0):
        """Saves one structure from the batch to a PDB file."""
        with open(path, 'w') as f:
            f.write(self.to_pdb(index))

    def to_pdb(self, index: int = 0) -> str:
        """Converts one structure in the batch to a PDB string."""
        lines = []
        c = self.coords[index]
        # PDB Format String: ATOM, Serial, Name, ResName, Chain, ResSeq, X, Y, Z, Occupancy, B-factor, Element
        fmt = "ATOM  {:>5d} {:<4s} {:>3s} A{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2s}"
        
        for i in range(self.n_atoms):
            name = self.atom_names[i]
            res_idx = self.residue_indices[i]
            res_name = self.sequence[res_idx - 1]
            element = name[0] if not name[0].isdigit() else name[1]
            
            # Atom names with 4 chars start at col 13, others at col 14
            # We simplify here for the generator tool compatibility
            clean_name = name.strip()
            if len(clean_name) < 4:
                atom_field = " " + clean_name.ljust(3)
            else:
                atom_field = clean_name
                
            lines.append(fmt.format(
                i + 1, atom_field, res_name, res_idx, 
                c[i, 0], c[i, 1], c[i, 2], element
            ))
        lines.append("TER")
        lines.append("END")
        return "\n".join(lines)

class BatchedGenerator:
    """
    High-performance vectorized protein structure generator.
    Optimized for generating millions of labeled samples for AI training.
    """
    def __init__(self, sequence_str: str, n_batch: int = 1, full_atom: bool = False):
        # Resolve sequence
        if "-" in sequence_str:
            self.sequence = [s.strip() for s in sequence_str.split("-")]
        else:
            self.sequence = [ONE_TO_THREE_LETTER_CODE.get(c, "ALA") for c in sequence_str]
        
        self.n_batch = n_batch
        self.n_res = len(self.sequence)
        self.full_atom = full_atom
        
        # Build Atom Topology & Load Templates
        self.atom_names = []
        self.residue_indices = []
        self.templates = []
        self.template_backbones = []
        self.offsets = []
        
        current_offset = 0
        for i, res_name in enumerate(self.sequence):
            if full_atom:
                # Get full-atom template from biotite
                template = struc.info.residue(res_name).copy()
                
                # Remove terminal atoms (OXT, H2, H3) to match peptide chain logic
                # (Simple heuristic for now, matches generator.py logic)
                mask = ~np.isin(template.atom_name, ["OXT", "H2", "H3", "HXT"])
                template = template[mask]
                
                names = template.atom_name.tolist()
                n_atoms = len(names)
                
                # Ensure N, CA, C are present for superimposition
                if not all(a in names for a in ["N", "CA", "C"]):
                    raise ValueError(f"Template for {res_name} missing core backbone atoms.")
                
                self.templates.append(template)
                # Store (N, CA, C) coordinates of template for Kabsch (shape 3, 3)
                n_idx = names.index("N")
                ca_idx = names.index("CA")
                c_idx = names.index("C")
                self.template_backbones.append(template.coord[[n_idx, ca_idx, c_idx]])
            else:
                # Backbone only: N, CA, C, O
                names = ["N", "CA", "C", "O"]
                n_atoms = 4
            
            self.atom_names.extend(names)
            self.residue_indices.extend([i + 1] * n_atoms)
            self.offsets.append(current_offset)
            current_offset += n_atoms
            
        self.total_atoms = current_offset

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

        # 3. Full-Atom Superimposition
        if self.full_atom:
            # Allocate full-atom coords
            fa_coords = np.zeros((B, self.total_atoms, 3))
            
            for i in range(L):
                # Target backbone frame (B, 3, 3) from the NeRF backbone
                # NeRF backbone order: N(0), CA(1), C(2), O(3)
                target_n = coords[:, i*4]
                target_ca = coords[:, i*4+1]
                target_c = coords[:, i*4+2]
                target_bb = np.stack([target_n, target_ca, target_c], axis=1) # (B, 3, 3)
                
                # Source backbone frame (3, 3)
                template_bb = self.template_backbones[i]
                # Broadcast template to batch: (B, 3, 3)
                source_bb = np.repeat(template_bb[np.newaxis, :, :], B, axis=0)
                
                # Align template to target
                trans, rot = superimpose_batch(source_bb, target_bb)
                
                # Apply rotation and translation to all atoms in template
                # template.coord: (N_res_atoms, 3)
                template_coords = self.templates[i].coord
                
                # (B, 3, 3) @ (N_res_atoms, 3)^T -> (B, 3, N_res_atoms)
                # Then transpose back to (B, N_res_atoms, 3)
                rotated = np.matmul(rot, template_coords.T).transpose(0, 2, 1)
                aligned = rotated + trans[:, np.newaxis, :]
                
                # Write to global tensor
                offset = self.offsets[i]
                n_res_atoms = template_coords.shape[0]
                fa_coords[:, offset:offset+n_res_atoms] = aligned
                
            return BatchedPeptide(fa_coords, self.sequence, self.atom_names, self.residue_indices)

        return BatchedPeptide(coords, self.sequence, ["N", "CA", "C", "O"] * L, self.residue_indices)
