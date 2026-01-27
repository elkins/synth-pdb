
"""
Biophysical Realism Module.

Enhances synthetic structures with realistic physical chemistry properties.
Includes:
- pH Titration (Histidine protonation states)
- Terminal Capping (N-acetyl/C-amide)
- Salt Bridge Stabilization (Automatic detection of ionic interactions)
- Charge assignment

Educational Note - pH and Protonation:
--------------------------------------
Biological function depends on pH. The most sensitive residue near physiological pH (7.4) is Histidine (pKa ~ 6.0).
- pH < 6.0: Imidazole ring is protonated (+1 charge). Code: HIP.
- pH > 6.0: Imidazole ring is neutral (0 charge). Tautomers: HIE (epsilon protonated) or HID (delta protonated).
"""

import biotite.structure as struc
import logging
import random
import numpy as np
from .geometry import position_atom_3d_from_internal_coords, calculate_angle
from .data import (
    BOND_LENGTH_N_CA,
    BOND_LENGTH_CA_C,
    BOND_LENGTH_C_N,
    BOND_LENGTH_C_O,
    ANGLE_N_CA_C,
    ANGLE_CA_C_N,
    ANGLE_C_N_CA,
    ANGLE_CA_C_O,
)

logger = logging.getLogger(__name__)

def apply_ph_titration(structure: struc.AtomArray, ph: float = 7.4) -> struc.AtomArray:
    """
    Apply global pH settings to titratable residues (mainly Histidine).
    
    Args:
        structure: The atom array.
        ph: The pH value (default 7.4).
        
    Returns:
        Modified atom array with updated residue names (HIS -> HIE/HID/HIP).
    """
    logger.info(f"Applying pH Titration (pH={ph})...")
    
    # Iterate residues
    # We need to scan usually.
    # But for HIS, we can mask finding "HIS" and potentially replace.
    # However, replacing RES_NAME in Biotite is easy array operation.
    
    # 1. Low pH (Acidic) -> HIP (Positive)
    if ph < 6.0:
        # Simplistic Henderson-Hasselbalch Logic:
        # If pH < pKa (6.0), predominant species is protonated.
        # Rename ALL HIS to HIP.
        mask = structure.res_name == "HIS"
        if mask.any():
            count = len(set(structure.res_id[mask]))
            structure.res_name[mask] = "HIP"
            logger.info(f"Protonated {count} Histidines to HIP (pH {ph} < 6.0)")
            
    # 2. High/Physiological pH -> Neutral Tautomers (HIE/HID)
    else:
        # Determine tautomer ratios.
        # In solution, N-epsilon (HIE) is favored ~80:20 over N-delta (HID).
        # We will assign probabilistically per residue.
        
        # Get all HIS residue IDs
        his_mask = structure.res_name == "HIS"
        his_res_ids = sorted(list(set(structure.res_id[his_mask])))
        
        for res_id in his_res_ids:
            # 80% chance HIE, 20% chances HID
            # Note: Standard PDB often uses just "HIS" implying neural.
            # But explicit modelling requires choosing one.
            # If we want to be explicit:
            new_name = "HIE" if random.random() < 0.8 else "HID"
            
            # Update this residue
            res_mask = (structure.res_id == res_id) & (structure.res_name == "HIS")
            structure.res_name[res_mask] = new_name
            
        if his_res_ids:
             logger.info(f" assigned tautomers (HIE/HID) to {len(his_res_ids)} Histidines (pH {ph} > 6.0)")

    return structure

def cap_termini(structure: struc.AtomArray) -> struc.AtomArray:
    """
    Add terminal capping groups (ACE/NME) to the peptide.
    
    EDUCATIONAL NOTE - Terminal Capping:
    ====================================
    Biological proteins are usually long chains. However, simulation and experiments
    often use shorter peptide fragments.
    
    Uncapped termini (NH3+ and COO-) introduce strong charges that are often unrealistic
    for an internal fragment of a protein.
    - N-terminus cap: Acetyl (ACE) -> replaces H with CH3-CO-
      Eliminates positive charge at N-term.
      Structure: CH3-C(=O)-NH-...
    - C-terminus cap: N-Methylamide (NME) -> replaces O with NH-CH3
      Eliminates negative charge at C-term.
      Structure: ...-CO-NH-CH3
      
    This function geometrically constructs these caps attached to the start and end residues.
    
    Args:
        structure: Input peptide structure
        
    Returns:
        Structure with ACE and NME residues added.
    """
    logger.info("Adding terminal caps (ACE/NME)...")
    
    # 1. Identify Termini
    res_ids = sorted(list(set(structure.res_id)))
    if not res_ids:
        return structure

    n_term_id = res_ids[0]
    c_term_id = res_ids[-1]
    
    # --- ACE (Acetyl) at N-terminus ---
    # Attaches to N of first residue.
    # Geometry:
    # We need to place C (carbonyl), O (carbonyl oxygen), and CH3 (methyl).
    # Since we are prepending, we work "backwards" from N.
    
    try:
        n_1 = structure[(structure.res_id == n_term_id) & (structure.atom_name == "N")][0]
        ca_1 = structure[(structure.res_id == n_term_id) & (structure.atom_name == "CA")][0]
        c_1 = structure[(structure.res_id == n_term_id) & (structure.atom_name == "C")][0]
        
        # 1. Place ACE C (Carbonyl)
        # We position it relative to N-CA-C frame of first residue.
        # Bond: C(ace)-N = 1.33 A
        # Angle: C(ace)-N-CA = 121.7 deg
        # Dihedral: C(ace)-N-CA-C. This corresponds to the Phi angle definition (C_prev-N-CA-C).
        # We'll assume a standard Extended conformation (Phi=-135) or similar to avoid clashes.
        # Or even better, 180 (Trans) relative to CA-C for maximum clearance.
        phi_assume = -135.0 # Beta sheet-like, safe usually
        
        c_ace_coord = position_atom_3d_from_internal_coords(
            c_1.coord, ca_1.coord, n_1.coord,
            BOND_LENGTH_C_N, ANGLE_C_N_CA, phi_assume
        )
        
        # 2. Place ACE O (Carbonyl Oxygen)
        # Bond: O-C = 1.23 A
        # Angle: O-C-N = 123.0 deg (approx)
        # Dihedral: O-C-N-CA.
        # Peptide bond is planar. O is usually trans to H(N), which means cis to CA?
        # Actually in trans peptide bond, O and H are trans. atomic config: O=C-N-H.
        # O and CA are usually 'trans' (180).
        o_ace_coord = position_atom_3d_from_internal_coords(
            ca_1.coord, n_1.coord, c_ace_coord,
            BOND_LENGTH_C_O, 123.0, 180.0
        )
        
        # 3. Place ACE CH3 (Methyl)
        # Bond: CH3-C = 1.50 A
        # Angle: CH3-C-N = 116.0 deg
        # Dihedral: CH3-C-N-CA (Omega). Standard Trans = 180.
        ch3_ace_coord = position_atom_3d_from_internal_coords(
            ca_1.coord, n_1.coord, c_ace_coord,
            1.50, 116.0, 180.0
        )
        
        # Create ACE atoms (Residue ID: n_term_id - 1)
        # If n_term_id is 1, ACE is 0.
        ace_res_id = n_term_id - 1
        ace_atoms = [
            struc.Atom(ch3_ace_coord, atom_name="CH3", res_id=ace_res_id, res_name="ACE", element="C", hetero=False),
            struc.Atom(c_ace_coord, atom_name="C", res_id=ace_res_id, res_name="ACE", element="C", hetero=False),
            struc.Atom(o_ace_coord, atom_name="O", res_id=ace_res_id, res_name="ACE", element="O", hetero=False),
        ]
        ace_structure = struc.array(ace_atoms)
        ace_structure.chain_id[:] = "A"

    except IndexError:
        logger.warning("Could not build ACE cap: Missing N/CA/C on N-terminus.")
        ace_structure = None


    # --- NME (N-Methylamide) at C-terminus ---
    # Attaches to C of last residue.
    # Geometry:
    # We place N, and CH3 (methyl).
    
    try:
        c_last = structure[(structure.res_id == c_term_id) & (structure.atom_name == "C")][0]
        ca_last = structure[(structure.res_id == c_term_id) & (structure.atom_name == "CA")][0]
        n_last = structure[(structure.res_id == c_term_id) & (structure.atom_name == "N")][0]
        
        # 1. Place NME N
        # We assume Trans peptide bond (Omega=180) relative to previous CA-C.
        # Bond: N-C = 1.33 A
        # Angle: N-C-CA = 116.2 deg
        # Dihedral: N-C-CA-N(prev). This is Psi. 
        # We assume Psi = -47 (Alpha) or 135 (Beta). Let's use 135 (Extended) to stick out.
        psi_assume = 135.0
        
        n_nme_coord = position_atom_3d_from_internal_coords(
            n_last.coord, ca_last.coord, c_last.coord,
            BOND_LENGTH_C_N, ANGLE_CA_C_N, psi_assume
        )
        
        # 2. Place NME CH3
        # Bond: CH3-N = 1.45 A
        # Angle: CH3-N-C = 122.0 deg (approx for amide nitrogen)
        # Dihedral: CH3-N-C-CA.
        # Standard Trans peptide bond (Omega = 180).
        ch3_nme_coord = position_atom_3d_from_internal_coords(
            ca_last.coord, c_last.coord, n_nme_coord,
            1.45, 122.0, 180.0
        )
        
        # Create NME atoms
        nme_res_id = c_term_id + 1
        nme_atoms = [
            struc.Atom(n_nme_coord, atom_name="N", res_id=nme_res_id, res_name="NME", element="N", hetero=False),
            struc.Atom(ch3_nme_coord, atom_name="CH3", res_id=nme_res_id, res_name="NME", element="C", hetero=False),
        ]
        nme_structure = struc.array(nme_atoms)
        nme_structure.chain_id[:] = "A"

    except IndexError:
        logger.warning("Could not build NME cap: Missing N/CA/C on C-terminus.")
        nme_structure = None


    # Combine
    final_structure = structure
    
    if ace_structure:
        # Prepend
        final_structure = ace_structure + final_structure
        
    if nme_structure:
        # Critical Fix for OpenMM: Remove OXT from the C-terminal residue.
        # OpenMM sees OXT and thinks it's a terminal residue (e.g., CLYS).
        # But since we attach NME, it should be an internal residue (LYS).
        # The presence of OXT (+ bond to NME) causes "1 C atom too many" error.
        
        # Identify OXT at c_term_id
        oxt_mask = (final_structure.res_id == c_term_id) & (final_structure.atom_name == "OXT")
        if oxt_mask.any():
            final_structure = final_structure[~oxt_mask]
            logger.debug(f"Removed OXT from residue {c_term_id} to accommodate NME cap.")
            
        # Append
        final_structure = final_structure + nme_structure
        
    return final_structure

# EDUCATIONAL NOTE: Salt Bridges
# ------------------------------
# A salt bridge is a combination of two non-covalent interactions: 
# 1. Hydrogen Bonding
# 2. Electrostatic (Ionic) Attraction
#
# It occurs between a positively charged basic residue and a negatively 
# charged acidic residue. In proteins, these are critical for:
# - Stabilizing tertiary and quaternary structure.
# - Specific ligand binding.
# - pH-dependent conformational changes (as pKa shifts can break the bridge).
#
# Reference:
# Bosshard, H. R., et al. (2004). "The salt bridge in proteins." 
# Journal of Molecular Recognition, 17(1), 1-16.

ACIDIC_RESIDUES = ["ASP", "GLU"]
BASIC_RESIDUES = ["LYS", "ARG", "HIS"]

# Specific atoms that carry the formal charges
ACIDIC_ATOMS = ["OD1", "OD2", "OE1", "OE2"]
BASIC_ATOMS = ["NZ", "NH1", "NH2", "ND1", "NE2"]

def find_salt_bridges(structure: struc.AtomArray, cutoff: float = 5.0):
    """
    Automatically detects potential salt bridges in a protein structure.
    
    A salt bridge is defined here as a pair of acidic and basic residues 
    where any of their side-chain charged atoms are within the specified cutoff.
    
    Args:
        structure: Biotite AtomArray (should include side chains).
        cutoff: Distance threshold in Angstroms (default 4.0).
        
    Returns:
        list of dict: Each dict contains:
            - res_ia: Residue ID of the first residue
            - res_ib: Residue ID of the second residue
            - atom_a: Name of the coordinating atom in res_ia
            - atom_b: Name of the coordinating atom in res_ib
            - distance: The measured distance
    """
    # Filter for Acidic and Basic atoms only to speed up search
    acid_mask = np.isin(structure.res_name, ACIDIC_RESIDUES) & np.isin(structure.atom_name, ACIDIC_ATOMS)
    base_mask = np.isin(structure.res_name, BASIC_RESIDUES) & np.isin(structure.atom_name, BASIC_ATOMS)
    
    acids = structure[acid_mask]
    bases = structure[base_mask]
    
    if len(acids) == 0 or len(bases) == 0:
        return []
        
    # Compute Distance Matrix between all Acid atoms and Base atoms
    # acids.coord: (N, 3), bases.coord: (M, 3)
    # diffs: (N, M, 3)
    diffs = acids.coord[:, np.newaxis, :] - bases.coord[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    
    # Find pairs within cutoff
    indices = np.where(dists < cutoff)
    
    found_pairs = {} # (res_ia, res_ib) -> bridge_dict
    
    for acid_idx, base_idx in zip(*indices):
        a_atom = acids[acid_idx]
        b_atom = bases[base_idx]
        
        # Ensure we don't bridge within the same residue
        if a_atom.res_id == b_atom.res_id:
            continue
            
        pair_key = tuple(sorted([a_atom.res_id, b_atom.res_id]))
        dist = dists[acid_idx, base_idx]
        
        # We pick the closest atom pair for each residue-residue interaction
        if pair_key not in found_pairs or dist < found_pairs[pair_key]["distance"]:
            found_pairs[pair_key] = {
                "res_ia": int(a_atom.res_id),
                "res_ib": int(b_atom.res_id),
                "atom_a": a_atom.atom_name,
                "atom_b": b_atom.atom_name,
                "distance": float(dist)
            }
            
    return list(found_pairs.values())
