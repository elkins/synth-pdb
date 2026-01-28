import numpy as np
import biotite.structure as struc
import logging
from .data import VAN_DER_WAALS_RADII

logger = logging.getLogger(__name__)

def calculate_clash_score(atom_array: struc.AtomArray) -> float:
    """
    Calculate a simple clash score based on Van der Waals overlaps.
    Lower is better.
    
    Args:
        atom_array: Biotite AtomArray containing the structure
        
    Returns:
        float: Systematic score representing the severity of steric clashes
    
    # EDUCATIONAL NOTE - Steric Repulsion and Forces
    # Atoms are not hard billiard balls, but they do have a "Van der Waals radius".
    # When two atoms get too close, their electron clouds repel each other.
    #
    # The standard "Lennard-Jones Potential" models this energy as:
    # E = 4ε [ (σ/r)^12 - (σ/r)^6 ]
    #
    # - (σ/r)^12 term: Repulsion. Rises steeply as atoms overlap (Pauli exclusion).
    # - (σ/r)^6 term: Attraction. Weak "dispersion" forces that hold non-bonded atoms together.
    #
    # In this function, we use a simplified "Soft Sphere" or "Cubic Penalty" approach.
    # Instead of shooting to infinity (which breaks optimization math), we assume
    # the energy rises polynomially once distance < sum_of_radii.
    """
    if atom_array.array_length() < 2:
        return 0.0

    # Get coordinates and arrays ONCE to avoid object creation overhead
    coords = atom_array.coord
    res_ids = atom_array.res_id
    atom_names = atom_array.atom_name
    elements = atom_array.element
    
    # Pre-calculate radii for all atoms
    # Default 1.5 if not found
    radii = np.array([VAN_DER_WAALS_RADII.get(e, 1.5) for e in elements])
    
    # Get Cell List for efficient neighbor search
    cell_list = struc.CellList(atom_array, cell_size=5.0)
    
    clash_score = 0.0
    
    # Pre-define backbone names for check
    backbone_names = frozenset(['C', 'O', 'N', 'CA'])
    
    # Iterate through all atoms
    for i in range(len(atom_array)):
        # Get potential neighbors within 5A
        indices = cell_list.get_atoms(coords[i], radius=5.0)
        
        # Filter indices to only look at j > i
        indices = indices[indices > i]
        
        if len(indices) == 0:
            continue
            
        # Get properties of atom 1
        r1 = radii[i]
        res_id1 = res_ids[i]
        pos1 = coords[i]
        is_bb1 = atom_names[i] in backbone_names
        
        # Iterate over neighbors
        # Using direct array access is much faster than atom_array[j]
        for j in indices:
            res_id2 = res_ids[j]
            
            # Skip atoms in same residue (simplified exclusion)
            # A full forcefield excludes 1-2, 1-3, and scaled 1-4 interactions
            # Here we just blindly skip intra-residue to avoid self-clashes from bond geometry
            if res_id1 == res_id2:
                continue
                
            # Skip peptide bond connections (adjacent residues)
            # This is a heuristic: adjacent residues have bonded atoms that are close
            if abs(res_id1 - res_id2) == 1:
                # If both are backbone, skip
                if is_bb1 and (atom_names[j] in backbone_names):
                    continue
            
            r2 = radii[j]
            pos2 = coords[j]
            
            # Simple Lennard-Jones-like repulsion term
            # Energy ~ (Rmin / r)^12 
            # We want a soft-ish repulsion to guide optimization
            
            # Distance calculation
            dist_sq = np.sum((pos1 - pos2)**2)
            dist = np.sqrt(dist_sq)
            
            optimal_dist = r1 + r2
            
            if dist < optimal_dist * 0.8: # Overlap threshold
                # Severe clash
                overlap = (optimal_dist * 0.8) - dist
                # Cubic penalty for smoothness
                clash_score += (overlap * 10) ** 2
                
    return clash_score

def calculate_energy_score(atom_array: struc.AtomArray) -> float:
    """
    Calculate a total pseudo-energy score.
    Currently just wraps clash_score, but can be extended with electrostatics/solvation.
    """
    return calculate_clash_score(atom_array)
