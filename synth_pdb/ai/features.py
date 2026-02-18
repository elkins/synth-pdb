import numpy as np
import io
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from typing import Dict, List, Any, Tuple
from synth_pdb.orientogram import compute_6d_orientations
from synth_pdb.validator import PDBValidator, RAMACHANDRAN_POLYGONS

# Re-export core function for Latent Space Explorer
__all__ = ['compute_6d_orientations', 'extract_quality_features', 'get_feature_names']

def get_feature_names() -> List[str]:
    """Returns the list of feature names in the order they appear in the feature vector."""
    return [
        "ramachandran_favored_pct",
        "ramachandran_outliers_pct",
        "clash_count",
        "bond_length_violation_count",
        "bond_angle_violation_count",
        "peptide_bond_violation_count",
        "radius_of_gyration",
        "mean_b_factor"
    ]

def extract_quality_features(pdb_content: str) -> np.ndarray:
    """
    Extracts a feature vector for the Protein Quality Classifier.
    
    Features:
    0. Ramachandran Favored %
    1. Ramachandran Outliers %
    2. Steric Clash Count
    3. Bond Length Violation Count
    4. Bond Angle Violation Count
    5. Peptide Bond Planarity Violation Count
    6. Radius of Gyration (Compactness)
    7. Mean B-factor (Flexibility)
    
    Returns:
        np.ndarray: A 1D array of floats (shape: 8,)
    """
    # 1. Use PDBValidator for geometric violations
    # We subclass/intercept to get counts instead of strings
    validator = PDBValidator(pdb_content)
    
    # --- Ramachandran Analysis ---
    # We need to compute this manually to get percentages, as validator only reports outliers
    phi, psi = _get_dihedrals(validator)
    rama_favored, rama_outliers = _analyze_ramachandran(phi, psi, validator)
    total_residues = len(phi)
    
    rama_favored_pct = (rama_favored / total_residues * 100) if total_residues > 0 else 0
    rama_outliers_pct = (rama_outliers / total_residues * 100) if total_residues > 0 else 0
    
    # --- Steric Clashes ---
    # We count the number of clashes detected by the validator
    initial_violations = len(validator.violations)
    # We use backbone_only=True to avoid noise from sidechain generation issues
    validator.validate_steric_clashes(min_atom_distance=0.5, min_ca_distance=3.0, backbone_only=True)
    clash_count = len(validator.violations) - initial_violations
    
    # --- Bond Lengths ---
    initial_violations = len(validator.violations)
    validator.validate_bond_lengths(tolerance=0.1) # 0.1A tolerance
    bond_len_count = len(validator.violations) - initial_violations
    
    # --- Bond Angles ---
    initial_violations = len(validator.violations)
    validator.validate_bond_angles(tolerance=10.0) # 10 deg tolerance
    bond_ang_count = len(validator.violations) - initial_violations
    
    # --- Peptide Bond Planarity ---
    initial_violations = len(validator.violations)
    validator.validate_peptide_plane(tolerance_deg=30.0)
    peptide_plane_count = len(validator.violations) - initial_violations
    
    # --- Global Properties ---
    coords = np.array([atom['coords'] for atom in validator.atoms])
    b_factors = np.array([atom['temp_factor'] for atom in validator.atoms])
    
    # Calculate total residues for normalization
    num_residues = sum(len(res_dict) for res_dict in validator.grouped_atoms.values())
    if num_residues == 0:
        num_residues = 1.0
    
    # Radius of Gyration
    rg = 0.0
    if len(coords) > 0:
        center_of_mass = np.mean(coords, axis=0)
        rg = np.sqrt(np.sum(np.linalg.norm(coords - center_of_mass, axis=1)**2) / len(coords))
        
    mean_b_factor = np.mean(b_factors) if len(b_factors) > 0 else 0.0
    
    return np.array([
        rama_favored_pct,
        rama_outliers_pct,
        float(clash_count) / num_residues,
        float(bond_len_count) / num_residues,
        float(bond_ang_count) / num_residues,
        float(peptide_plane_count) / num_residues,
        rg,
        mean_b_factor
    ])

def _get_dihedrals(validator: PDBValidator) -> Tuple[List[float], List[float]]:
    """Extracts Phi/Psi angles from the validator's parsed atoms."""
    phi_list = []
    psi_list = []
    
    for chain_id, residues_in_chain in validator.grouped_atoms.items():
        sorted_res_numbers = sorted(residues_in_chain.keys())
        for i, res_num in enumerate(sorted_res_numbers):
            current_res_atoms = residues_in_chain[res_num]
            
            # Phi
            phi = None
            if i > 0:
                prev_res_num = sorted_res_numbers[i - 1]
                prev_res_atoms = residues_in_chain.get(prev_res_num)
                if prev_res_atoms and prev_res_atoms.get("C") and current_res_atoms.get("N") and current_res_atoms.get("CA") and current_res_atoms.get("C"):
                    p1 = prev_res_atoms["C"]["coords"]
                    p2 = current_res_atoms["N"]["coords"]
                    p3 = current_res_atoms["CA"]["coords"]
                    p4 = current_res_atoms["C"]["coords"]
                    phi = validator._calculate_dihedral_angle(p1, p2, p3, p4)
            
            # Psi
            psi = None
            if i < len(sorted_res_numbers) - 1:
                next_res_num = sorted_res_numbers[i + 1]
                next_res_atoms = residues_in_chain.get(next_res_num)
                if current_res_atoms.get("N") and current_res_atoms.get("CA") and current_res_atoms.get("C") and next_res_atoms.get("N"):
                    p1 = current_res_atoms["N"]["coords"]
                    p2 = current_res_atoms["CA"]["coords"]
                    p3 = current_res_atoms["C"]["coords"]
                    p4 = next_res_atoms["N"]["coords"]
                    psi = validator._calculate_dihedral_angle(p1, p2, p3, p4)
            
            if phi is not None and psi is not None:
                phi_list.append(phi)
                psi_list.append(psi)
                
    return phi_list, psi_list

def _analyze_ramachandran(phi_list: List[float], psi_list: List[float], validator: PDBValidator) -> Tuple[int, int]:
    """Counts favored and outlier residues."""
    favored = 0
    outliers = 0
    
    # We use "General" polygons for everything to simplify feature extraction
    # This keeps the model robust and noise-tolerant
    polygons = RAMACHANDRAN_POLYGONS["General"]
    
    for phi, psi in zip(phi_list, psi_list):
        is_favored = False
        for poly in polygons["Favored"]:
            if validator._is_point_in_polygon((phi, psi), poly):
                is_favored = True
                break
        
        if is_favored:
            favored += 1
        else:
            is_allowed = False
            for poly in polygons["Allowed"]:
                if validator._is_point_in_polygon((phi, psi), poly):
                    is_allowed = True
                    break
            
            if not is_allowed:
                outliers += 1
                
    return favored, outliers
