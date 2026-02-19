import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import io
import logging
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

logger = logging.getLogger(__name__)

def interpolate_structures(start_pdb_path: str, end_pdb_path: str, steps: int, output_prefix: str):
    """
    Interpolates between two structures by morphing their backbone torsion angles.
    
    Args:
        start_pdb_path: Path to start PDB.
        end_pdb_path: Path to end PDB.
        steps: Number of intermediate frames.
        output_prefix: Prefix for output files (e.g. "morph" -> "morph_0.pdb", "morph_1.pdb"...)
    """
    # 1. Load structures
    pdb_file_start = pdb.PDBFile.read(start_pdb_path)
    start_struct = pdb_file_start.get_structure(model=1)
    
    pdb_file_end = pdb.PDBFile.read(end_pdb_path)
    end_struct = pdb_file_end.get_structure(model=1)
    
    # Check compatibility (same length)
    start_ca = start_struct[start_struct.atom_name == "CA"]
    end_ca = end_struct[end_struct.atom_name == "CA"]
    
    if len(start_ca) != len(end_ca):
        raise ValueError(f"Structures have different lengths: {len(start_ca)} vs {len(end_ca)}. Interpolation requires same length.")
    
    # 2. Extract Dihedrals (Phi, Psi, Omega)
    # Biotite returns radians. shape (L,)
    phi_start, psi_start, omega_start = struc.dihedral_backbone(start_struct)
    phi_end, psi_end, omega_end = struc.dihedral_backbone(end_struct)
    
    # Handle NaNs (Termini) by setting to 0 or 180 (for Omega)
    # Mask Nans
    mask = np.isnan(phi_start)
    phi_start[mask] = 0
    mask = np.isnan(phi_end)
    phi_end[mask] = 0
    
    mask = np.isnan(psi_start)
    psi_start[mask] = 0
    mask = np.isnan(psi_end)
    psi_end[mask] = 0

    mask = np.isnan(omega_start)
    omega_start[mask] = np.pi # Trans
    mask = np.isnan(omega_end)
    omega_end[mask] = np.pi

    # get sequence string
    # We assume same sequence
    res_names = start_ca.res_name
    
    # 3. Interpolate
    for step in range(steps + 1): # Include end
        t = step / steps
        
        # Linear interpolation of angles
        # Note: Proper circular interpolation (slerp-like) is better for angles, but simple lerp works for small steps
        
        # Handle periodicity: minimal path
        # diff = (end - start + pi) % 2pi - pi
        phi_diff = np.mod(phi_end - phi_start + np.pi, 2*np.pi) - np.pi
        phi_t = phi_start + t * phi_diff
        
        psi_diff = np.mod(psi_end - psi_start + np.pi, 2*np.pi) - np.pi
        psi_t = psi_start + t * psi_diff
        
        omega_diff = np.mod(omega_end - omega_start + np.pi, 2*np.pi) - np.pi
        omega_t = omega_start + t * omega_diff
        
        # 4. Reconstruct using NeRF (via Generator or ad-hoc)
        # Since generator.py is complex, we use a specialized reconstruction here or try to reuse generator
        # generate_pdb_content doesn't take raw angles array.
        # So we should use synth_pdb.geometry directly or similar.
        
        # Actually, let's use the BatchedGenerator approach logic but for single struct, 
        # OR just use biotite to modify the structure.
        # But Biotite dihedral modification is tricky.
        
        # Simplest: Generate a new structure using BatchedGenerator logic but adapted.
        # We can implement a simple NeRF reconstructor here.
        
        coords = _reconstruct_backbone(phi_t, psi_t, omega_t)
        
        # Write PDB
        out_name = f"{output_prefix}_{step}.pdb"
        _write_simple_pdb(coords, res_names, out_name)
        logger.info(f"Wrote frame {step}: {out_name}")

def _reconstruct_backbone(phi, psi, omega):
    """Reconstruct backbone coordinates from angles."""
    from synth_pdb.geometry import position_atoms_batch
    from synth_pdb.data import (
        BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
        ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_C_N_CA,
        BOND_LENGTH_C_O, ANGLE_CA_C_O
    )
    
    L = len(phi)
    coords = np.zeros((L*3, 3)) # N, CA, C only for now (simplified)
    # Actually we want N, CA, C, O
    coords = np.zeros((L*4, 3))
    
    # 1. First residue
    coords[0] = [0, 0, 0] # N
    coords[1] = [BOND_LENGTH_N_CA, 0, 0] # CA
    ang = np.deg2rad(ANGLE_N_CA_C)
    coords[2] = [
        BOND_LENGTH_N_CA - BOND_LENGTH_CA_C * np.cos(ang),
        BOND_LENGTH_CA_C * np.sin(ang),
        0
    ] # C
    
    # Place O(0)
    # position_atoms_batch expects arrays (B, ...)
    # adapt single to batch
    def pos(p1, p2, p3, bl, ba, di):
        return position_atoms_batch(
            p1.reshape(1,3), p2.reshape(1,3), p3.reshape(1,3), 
            np.array([bl]), np.array([ba]), np.array([np.degrees(di)])
        )[0]

    coords[3] = pos(coords[0], coords[1], coords[2], BOND_LENGTH_C_O, ANGLE_CA_C_O, np.pi)
    
    for i in range(1, L):
        idx = i * 4
        prev_idx = (i-1) * 4
        
        # Place N(i) using psi(i-1)
        # Atoms: N(i-1), CA(i-1), C(i-1) -> N(i)
        coords[idx] = pos(coords[prev_idx], coords[prev_idx+1], coords[prev_idx+2], 
                          BOND_LENGTH_C_N, ANGLE_CA_C_N, psi[i-1])
                          
        # Place CA(i) using omega(i-1)
        # Atoms: CA(i-1), C(i-1), N(i) -> CA(i)
        coords[idx+1] = pos(coords[prev_idx+1], coords[prev_idx+2], coords[idx],
                            BOND_LENGTH_N_CA, ANGLE_C_N_CA, omega[i-1])
                            
        # Place C(i) using phi(i)
        # Atoms: C(i-1), N(i), CA(i) -> C(i)
        coords[idx+2] = pos(coords[prev_idx+2], coords[idx], coords[idx+1],
                            BOND_LENGTH_CA_C, ANGLE_N_CA_C, phi[i])
                            
        # Place O(i)
        coords[idx+3] = pos(coords[idx], coords[idx+1], coords[idx+2],
                            BOND_LENGTH_C_O, ANGLE_CA_C_O, np.pi)
                            
    return coords

def _write_simple_pdb(coords, res_names, path):
    """Write minimal PDB."""
    with open(path, 'w') as f:
        atom_idx = 1
        for i, res_name in enumerate(res_names):
            idx = i * 4
            # N
            f.write(f"ATOM  {atom_idx:>5d}  N   {res_name:>3s} A{i+1:>4d}    {coords[idx][0]:8.3f}{coords[idx][1]:8.3f}{coords[idx][2]:8.3f}  1.00  0.00           N\n")
            atom_idx += 1
            # CA
            f.write(f"ATOM  {atom_idx:>5d}  CA  {res_name:>3s} A{i+1:>4d}    {coords[idx+1][0]:8.3f}{coords[idx+1][1]:8.3f}{coords[idx+1][2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            # C
            f.write(f"ATOM  {atom_idx:>5d}  C   {res_name:>3s} A{i+1:>4d}    {coords[idx+2][0]:8.3f}{coords[idx+2][1]:8.3f}{coords[idx+2][2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            # O
            f.write(f"ATOM  {atom_idx:>5d}  O   {res_name:>3s} A{i+1:>4d}    {coords[idx+3][0]:8.3f}{coords[idx+3][1]:8.3f}{coords[idx+3][2]:8.3f}  1.00  0.00           O\n")
            atom_idx += 1
        f.write("TER\nEND\n")
