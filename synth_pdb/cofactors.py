
"""
Cofactor and Metal Ion Coordination Module.

THE "AI TRINITY" PHASE 15: Inorganic Coordination.

Biological proteins aren't just chains of amino acids; they often require 
inorganic "cofactors" or metal ions to function. Zinc (Zn2+) is one of 
the most common, found in "Zinc Finger" motifs where it stabilizes the 
fold by coordinating with Cysteine (C) and Histidine (H) residues.

Educational Note - Coordination Chemistry:
------------------------------------------
1. Ligands: The atoms that donate electrons to the metal (e.g., Cys Sulfur, His Nitrogen).
2. Coordination Number: The number of ligands (Zinc is usually 4 - Tetrahedral).
3. Geometric Centroid: The ideal position for the metal is the center of its ligands.

This module automatically detects these motifs and inserts the appropriate ions.
"""

import numpy as np
import biotite.structure as struc
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def find_metal_binding_sites(structure: struc.AtomArray, distance_threshold: float = 10.0) -> List[Dict]:
    """
    Scans the structure for clusters of residues that could coordinate a metal ion.
    
    Args:
        structure: Biotite AtomArray.
        distance_threshold: Max distance between any two coordinating atoms in a cluster.
        
    Returns:
        List[Dict]: A list of detected sites, each with 'type' and 'ligand_indices'.
    """
    logger.info("Scanning for metal binding motifs (Zinc Fingers)...")
    
    # Standard ligands for Zinc
    # CYS (SG), HIS (NE2 or ND1) - including HID/HIE/HIP variants
    ligand_mask = (
        ((structure.res_name == "CYS") & (structure.atom_name == "SG")) |
        ((np.isin(structure.res_name, ["HIS", "HID", "HIE", "HIP"])) & ((structure.atom_name == "NE2") | (structure.atom_name == "ND1")))
    )
    
    candidate_indices = np.where(ligand_mask)[0]
    if len(candidate_indices) < 3:
        return []
        
    candidate_coords = structure.coord[candidate_indices]
    sites = []
    assigned_indices = set()
    
    for i in range(len(candidate_indices)):
        idx_i = candidate_indices[i]
        if idx_i in assigned_indices:
            continue
            
        # Find all available neighbors within threshold
        diffs = candidate_coords - candidate_coords[i]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        
        neighbor_mask = dists < distance_threshold
        # Only take neighbors that aren't already assigned
        unassigned_neighbors = [
            candidate_indices[j] for j in range(len(candidate_indices)) 
            if neighbor_mask[j] and candidate_indices[j] not in assigned_indices
        ]
        
        # Zinc fingers typically have 4 ligands (C4, C2H2)
        # We greedily take the largest cluster first
        if len(unassigned_neighbors) >= 4:
            # Sort by distance to center if we had more than 4, but 4 is the sweet spot
            cluster = unassigned_neighbors[:4]
            sites.append({
                "type": "ZN",
                "ligand_indices": cluster
            })
            for idx in cluster:
                assigned_indices.add(idx)
                
    if sites:
        logger.info(f"Found {len(sites)} potential metal binding sites.")
    return sites

def add_metal_ion(structure: struc.AtomArray, site: Dict) -> struc.AtomArray:
    """
    Adds a metal ion (HETATM) to the structure at the centroid of its ligands.
    
    Args:
        structure: Original AtomArray.
        site: Identification of the site (from find_metal_binding_sites).
        
    Returns:
        struc.AtomArray: New AtomArray with the ion appended.
    """
    ion_type = site["type"]
    ligand_indices = site["ligand_indices"]
    
    # Calculate Centroid
    ligand_coords = structure.coord[ligand_indices]
    centroid = np.mean(ligand_coords, axis=0)
    
    # Create the Ion Atom
    ion = struc.AtomArray(1)
    ion.res_name = np.array([ion_type])
    ion.atom_name = np.array([ion_type])
    ion.element = np.array([ion_type]) # "ZN" not "Z" for Zinc
    ion.coord = np.array([centroid])
    
    # Metadata
    # Pick a residue ID higher than existing
    max_res_id = np.max(structure.res_id)
    ion.res_id = np.array([max_res_id + 1])
    ion.chain_id = np.array([structure.chain_id[0]])
    ion.hetero = np.array([True])
    
    logger.info(f"Injected {ion_type} ion at coordinated site {centroid}.")
    
    return structure + ion
