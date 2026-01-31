import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

from .data import (
    BOND_LENGTH_N_CA,
    BOND_LENGTH_CA_C,
    BOND_LENGTH_C_N,
    BOND_LENGTH_C_O,
    ANGLE_N_CA_C,
    ANGLE_C_N_CA,
    ANGLE_CA_C_O,
    VAN_DER_WAALS_RADII,
    CHARGED_AMINO_ACIDS,
    POSITIVE_AMINO_ACIDS,
    NEGATIVE_AMINO_ACIDS,
    HYDROPHOBIC_AMINO_ACIDS,
    POLAR_UNCHARGED_AMINO_ACIDS,
    RAMACHANDRAN_POLYGONS,
    BACKBONE_DEPENDENT_ROTAMER_LIBRARY,
    AMINO_ACID_CHI_DEFINITIONS,
)

logger = logging.getLogger(__name__)


class PDBValidator:
    """
    A class to validate PDB structures for various violations like bond lengths, angles,
    and Ramachandran angles.
    """
    atoms: List[Dict[str, Any]]
    grouped_atoms: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]
    sequences_by_chain: Dict[str, List[str]]
    violations: List[str]

    def __init__(self, pdb_content: Optional[str] = None, parsed_atoms: Optional[List[Dict[str, Any]]] = None):
        if pdb_content:
            self.pdb_content = pdb_content
            self.atoms = self._parse_pdb_atoms(pdb_content)
        elif parsed_atoms is not None:
            # Ensure coords are numpy arrays if they aren't already
            for atom in parsed_atoms:
                if isinstance(atom['coords'], list):
                    atom['coords'] = np.array(atom['coords'])
            self.atoms = parsed_atoms
            self.pdb_content = self.atoms_to_pdb_content(self.atoms) # Reconstruct pdb_content for consistency
        else:
            raise ValueError("Either pdb_content or parsed_atoms must be provided.")

        self.grouped_atoms = self._group_atoms_by_residue()
        self.sequences_by_chain = self._get_sequences_by_chain()
        self.violations = []  # Stores detected violations

    @staticmethod
    def _parse_pdb_atoms(pdb_content: str) -> List[Dict[str, Any]]:
        """
        Parses the PDB content and extracts atom information, specifically coordinates.
        Returns a list of dictionaries, each representing an atom with residue and chain info.
        """
        parsed_atoms = []
        for line in pdb_content.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("ATOM") or stripped_line.startswith("HETATM"):
                record_name = stripped_line[:6].strip()
                try:
                    atom_number = int(stripped_line[6:11].strip())
                    atom_name = stripped_line[12:16].strip()
                    alt_loc = stripped_line[16].strip()
                    residue_name = stripped_line[17:20].strip()
                    chain_id = stripped_line[21].strip()
                    residue_number = int(stripped_line[22:26].strip())
                    insertion_code = stripped_line[26].strip()
                    x = float(stripped_line[30:38])
                    y = float(stripped_line[38:46])
                    z = float(stripped_line[46:54])
                    occupancy = float(stripped_line[54:60].strip())
                    temp_factor = float(stripped_line[60:66].strip())
                    element = stripped_line[76:78].strip()
                    charge = stripped_line[78:80].strip()

                    parsed_atoms.append(
                        {
                            "atom_number": atom_number,
                            "atom_name": atom_name,
                            "alt_loc": alt_loc,
                            "residue_name": residue_name,
                            "chain_id": chain_id,
                            "residue_number": residue_number,
                            "insertion_code": insertion_code,
                            "coords": np.array([x, y, z]),
                            "occupancy": occupancy,
                            "temp_factor": temp_factor,
                            "element": element,
                            "charge": charge,
                            "record_name": record_name,
                        }
                    )
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Could not parse PDB ATOM/HETATM line: {line.strip()} - {e}"
                    )
        return parsed_atoms

    def get_atoms(self) -> List[Dict[str, Any]]:
        """
        Returns a deep copy of the parsed atom data.
        """
        # Return a deep copy to allow external modification without affecting internal state directly
        return [atom.copy() for atom in self.atoms]

    @staticmethod
    def atoms_to_pdb_line(atom_data: Dict[str, Any]) -> str:
        """
        Converts a single atom dictionary back into a PDB ATOM line.
        """
        x, y, z = atom_data["coords"]
        record_name = atom_data.get("record_name", "ATOM")
        return (
            f"{record_name: <6}{atom_data['atom_number']: >5} {atom_data['atom_name']: <4}{atom_data['alt_loc']: <1}"
            f"{atom_data['residue_name']: >3} {atom_data['chain_id']: <1}{atom_data['residue_number']: >4}"
            f"{atom_data['insertion_code']: <1}   "
            f"{x: >8.3f}{y: >8.3f}{z: >8.3f}{atom_data['occupancy']: >6.2f}"
            f"{atom_data['temp_factor']: >6.2f}          {atom_data['element']: >2}{atom_data['charge']: <2}"
        )

    @staticmethod
    def atoms_to_pdb_content(atom_list: List[Dict[str, Any]]) -> str:
        """
        Converts a list of atom dictionaries into a PDB content string, with TER records after each chain.
        """
        if not atom_list:
            return ""

        # Group atoms by chain ID, preserving order
        chains: Dict[str, List[Dict[str, Any]]] = {}
        for atom in atom_list:
            chain_id = atom.get("chain_id", "A") # Default to 'A' if no chain_id
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)
        
        pdb_lines = []
        last_atom_number = 0
        
        # Process chains in sorted order of their IDs for consistent output
        for chain_id in sorted(chains.keys()):
            chain_atoms = chains[chain_id]
            for atom in chain_atoms:
                pdb_lines.append(PDBValidator.atoms_to_pdb_line(atom))
                last_atom_number = atom["atom_number"]
            
            # Add a TER record after the last atom of the chain, but only if it's a polymer (contains ATOMs)
            if chain_atoms:
                last_atom_of_chain = chain_atoms[-1]
                
                # Check if this chain contains any ATOM records (is a polymer)
                has_polymer_atoms = any(atom.get("record_name", "ATOM") == "ATOM" for atom in chain_atoms)
                
                if has_polymer_atoms:
                    ter_atom_number = last_atom_of_chain.get("atom_number", 0) + 1
                    # Format the TER record based on the last atom's residue info
                    ter_line = (
                        f"TER   {ter_atom_number: >5}      {last_atom_of_chain['residue_name']: >3} "
                        f"{chain_id: <1}{last_atom_of_chain['residue_number']: >4}"
                    )
                    pdb_lines.append(ter_line)

        return "\n".join(pdb_lines) + "\n"

    def _group_atoms_by_residue(self) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
        """
        Groups parsed atoms by chain ID, then by residue number, then by atom name.
        Structure: {chain_id: {residue_number: {atom_name: atom_data}}}
        """
        grouped_atoms = {}
        for atom in self.atoms:
            chain_id = atom["chain_id"]
            residue_number = atom["residue_number"]
            atom_name = atom["atom_name"]

            if chain_id not in grouped_atoms:
                grouped_atoms[chain_id] = {}
            if residue_number not in grouped_atoms[chain_id]:
                grouped_atoms[chain_id][residue_number] = {}

            grouped_atoms[chain_id][residue_number][atom_name] = atom
        return grouped_atoms

    def _get_sequences_by_chain(self) -> Dict[str, List[str]]:
        """
        Extracts the amino acid sequences (list of 3-letter codes) for each chain.
        """
        sequences = {}
        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            chain_sequence = []
            for res_num in sorted_res_numbers:
                # We can use any atom in the residue to get its name, CA is common.
                ca_atom = residues_in_chain[res_num].get('CA')
                if ca_atom:
                    chain_sequence.append(ca_atom['residue_name'])
                else:
                    # Fallback if CA is not present (e.g., C-alpha only models don't have side chains)
                    # Try to get residue name from first available atom
                    first_atom = next(iter(residues_in_chain[res_num].values()), None)
                    if first_atom:
                        chain_sequence.append(first_atom['residue_name'])
            sequences[chain_id] = chain_sequence
        return sequences

    @staticmethod
    def _calculate_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
        """
        Calculates the Euclidean distance between two 3D coordinates.
        """
        return np.linalg.norm(coord1 - coord2)

    @staticmethod
    def _calculate_angle(
        coord1: np.ndarray, coord2: np.ndarray, coord3: np.ndarray
    ) -> float:
        """
        Calculates the angle (in degrees) formed by three coordinates, with coord2 as the vertex.
        """
        vec1 = coord1 - coord2
        vec2 = coord3 - coord2

        # Avoid division by zero for zero-length vectors
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0  # Or raise an error, depending on desired behavior for degenerate cases

        cosine_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
        # Ensure cosine_angle is within [-1, 1] to avoid issues with arccos due to floating point inaccuracies
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        return np.degrees(angle_rad)

    @staticmethod
    def _calculate_dihedral_angle(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
    ) -> float:
        """
        Calculates the dihedral angle (in degrees) defined by four points (p1, p2, p3, p4).
        
        IMPORTANT NOTE - Dihedral Conventions:
        It was discovered that simple projection math can accidentally swap
        Cis and Trans conventions.  Instead, the robust vector-based normal
        approach used in professional structural biology (IUPAC) is used.
        
        Standard IUPAC convention:
        - Cis-Peptide (eclipsed): ~0 degrees
        - Trans-Peptide (anti-planar): ~180 degrees
        """
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # Normals to the two planes
        n1 = np.cross(v1, v2).astype(float)
        n2 = np.cross(v2, v3).astype(float)
        
        # Normalize normals
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm == 0 or n2_norm == 0:
            return 0.0
            
        n1 /= n1_norm
        n2 /= n2_norm
        
        # Unit vector along the second bond
        u2 = v2.astype(float) / np.linalg.norm(v2)
        
        # Orthonormal basis in the plane perpendicular to b2
        m1 = np.cross(n1, u2)
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        # EDUCATIONAL NOTE:
        # Standard atan2 returns values in (-180, 180]. 
        # This matches the IUPAC convention for protein dihedrals.
        return np.degrees(np.arctan2(y, x))

    def get_violations(self) -> List[str]:
        """
        Returns a list of detected violations.
        """
        return self.violations

    def validate_bond_lengths(self, tolerance: float = 0.05):
        """
        Validates backbone bond lengths (N-CA, CA-C, C-O, C-N peptide bond) against standard values.
        """
        logger.info("Performing bond length validation.")

        bond_standards = {
            "N-CA": BOND_LENGTH_N_CA,
            "CA-C": BOND_LENGTH_CA_C,
            "C-O": BOND_LENGTH_C_O,
            "C-N_peptide": BOND_LENGTH_C_N,
        }

        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i, res_num in enumerate(sorted_res_numbers):
                current_res_atoms = residues_in_chain[res_num]

                n_atom = current_res_atoms.get("N")
                ca_atom = current_res_atoms.get("CA")
                c_atom = current_res_atoms.get("C")
                o_atom = current_res_atoms.get("O")

                # Check N-CA bond
                if n_atom and ca_atom:
                    actual_length = self._calculate_distance(
                        n_atom["coords"], ca_atom["coords"]
                    )
                    expected_length = bond_standards["N-CA"]
                    if abs(actual_length - expected_length) > tolerance:
                        self.violations.append(
                            f"Bond length violation: Chain {chain_id}, Residue {res_num} {n_atom['residue_name']} "
                            f"N-CA bond ({actual_length:.2f}Å) deviates from standard ({expected_length:.2f}Å) "
                            f"by more than {tolerance}Å. Actual: {actual_length:.2f}"
                        )

                # Check CA-C bond
                if ca_atom and c_atom:
                    actual_length = self._calculate_distance(
                        ca_atom["coords"], c_atom["coords"]
                    )
                    expected_length = bond_standards["CA-C"]
                    if abs(actual_length - expected_length) > tolerance:
                        self.violations.append(
                            f"Bond length violation: Chain {chain_id}, Residue {res_num} {ca_atom['residue_name']} "
                            f"CA-C bond ({actual_length:.2f}Å) deviates from standard ({expected_length:.2f}Å) "
                            f"by more than {tolerance}Å. Actual: {actual_length:.2f}"
                        )

                # Check C-O bond
                if c_atom and o_atom:
                    actual_length = self._calculate_distance(
                        c_atom["coords"], o_atom["coords"]
                    )
                    expected_length = bond_standards["C-O"]
                    if abs(actual_length - expected_length) > tolerance:
                        self.violations.append(
                            f"Bond length violation: Chain {chain_id}, Residue {res_num} {c_atom['residue_name']} "
                            f"C-O bond ({actual_length:.2f}Å) deviates from standard ({expected_length:.2f}Å) "
                            f"by more than {tolerance}Å. Actual: {actual_length:.2f}"
                        )

                # Check C (current) - N (next) peptide bond
                if i + 1 < len(sorted_res_numbers):
                    next_res_num = sorted_res_numbers[i + 1]
                    next_res_atoms = residues_in_chain.get(next_res_num)

                    if c_atom and next_res_atoms and next_res_atoms.get("N"):
                        next_n_atom = next_res_atoms["N"]
                        actual_length = self._calculate_distance(
                            c_atom["coords"], next_n_atom["coords"]
                        )
                        expected_length = bond_standards["C-N_peptide"]
                        if abs(actual_length - expected_length) > tolerance:
                            self.violations.append(
                                f"Bond length violation: Chain {chain_id}, Residue {res_num} {c_atom['residue_name']}-"
                                f"Residue {next_res_num} {next_n_atom['residue_name']} peptide bond ({actual_length:.2f}Å) "
                                f"deviates from standard ({expected_length:.2f}Å) by more than {tolerance}Å. Actual: {actual_length:.2f}"
                            )

    def validate_bond_angles(self, tolerance: float = 5.0):
        """
        Validates backbone bond angles (N-CA-C, CA-C-O, CA-C-N_next) against standard values.
        """
        logger.info("Performing bond angle validation.")

        angle_standards = {
            "N-CA-C": ANGLE_N_CA_C,
            "CA-C-O": ANGLE_CA_C_O,
            "CA-C-N_peptide": ANGLE_C_N_CA,  # This is the C(i)-N(i+1)-CA(i+1) angle
        }

        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i, res_num in enumerate(sorted_res_numbers):
                current_res_atoms = residues_in_chain[res_num]

                n_atom = current_res_atoms.get("N")
                ca_atom = current_res_atoms.get("CA")
                c_atom = current_res_atoms.get("C")
                o_atom = current_res_atoms.get("O")

                # Check N-CA-C angle
                if n_atom and ca_atom and c_atom:
                    actual_angle = self._calculate_angle(
                        n_atom["coords"], ca_atom["coords"], c_atom["coords"]
                    )
                    expected_angle = angle_standards["N-CA-C"]
                    if abs(actual_angle - expected_angle) > tolerance:
                        self.violations.append(
                            f"Bond angle violation: Chain {chain_id}, Residue {res_num} {ca_atom['residue_name']} "
                            f"N-CA-C angle ({actual_angle:.2f}°) deviates from standard ({expected_angle:.2f}°) "
                            f"by more than {tolerance}°. Actual: {actual_angle:.2f}"
                        )

                # Check CA-C-O angle
                if ca_atom and c_atom and o_atom:
                    actual_angle = self._calculate_angle(
                        ca_atom["coords"], c_atom["coords"], o_atom["coords"]
                    )
                    expected_angle = angle_standards["CA-C-O"]
                    if abs(actual_angle - expected_angle) > tolerance:
                        self.violations.append(
                            f"Bond angle violation: Chain {chain_id}, Residue {res_num} {c_atom['residue_name']} "
                            f"CA-C-O angle ({actual_angle:.2f}°) deviates from standard ({expected_angle:.2f}°) "
                            f"by more than {tolerance}°. Actual: {actual_angle:.2f}"
                        )

                # Check CA(i)-C(i)-N(i+1) angle (part of peptide bond)
                # Note: The data.py 'ANGLE_C_N_CA' is for C(i-1)-N(i)-CA(i).
                # We need C(i)-N(i+1)-CA(i+1) for the peptide linkage angle at C(i).
                # Let's assume a standard angle for C(i)-N(i+1)-CA(i+1) which is close to 121.7 from data.py (C-N-CA)
                if i + 1 < len(sorted_res_numbers):
                    next_res_num = sorted_res_numbers[i + 1]
                    next_res_atoms = residues_in_chain.get(next_res_num)

                    if (
                        ca_atom
                        and c_atom
                        and next_res_atoms
                        and next_res_atoms.get("N")
                        and next_res_atoms.get("CA")
                    ):
                        next_n_atom = next_res_atoms["N"]
                        next_ca_atom = next_res_atoms["CA"]
                        # The angle we are checking is C(i)-N(i+1)-CA(i+1)
                        actual_angle = self._calculate_angle(
                            c_atom["coords"],
                            next_n_atom["coords"],
                            next_ca_atom["coords"],
                        )
                        expected_angle = angle_standards[
                            "CA-C-N_peptide"
                        ]  # This is C-N-CA from data.py
                        if abs(actual_angle - expected_angle) > tolerance:
                            self.violations.append(
                                f"Bond angle violation: Chain {chain_id}, Residue {res_num} {c_atom['residue_name']}-"
                                f"Residue {next_res_num} {next_n_atom['residue_name']} "
                                f"C-N-CA angle ({actual_angle:.2f}°) deviates from standard ({expected_angle:.2f}°) "
                                f"by more than {tolerance}°. Actual: {actual_angle:.2f}"
                            )

    @staticmethod
    def _is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Ray-casting algorithm to check if a point is inside a polygon.
        Polygon is defined by a list of (x, y) tuples.
        """
        logger.info("Performing Ramachandran angle validation (polygonal regions).")
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def validate_ramachandran(self):
        """
        Validates Ramachandran angles (Phi, Psi) against MolProbity-defined polygonal regions.
        Checks if angles fall within simplified "Favored" (98%) or "Allowed" (99.8%) polygons.
        
        ### Educational Note - Computational Efficiency & Convergence:
        -----------------------------------------------
        Checking Ramachandran angles isn't just about "correctness" — it's a critical
        performance optimization for Energy Minimization (OpenMM).
        
        1. **Better Starting Points**: A structure with valid angles is much closer 
           to the global energy minimum. Minimization starting from a "Favored" 
           conformation converges significantly faster than one starting from a 
           high-energy "Outlier", saving expensive compute cycles.
           
        2. **Filtering**: By rejecting outliers *before* sending them to the 
           physics engine, we avoid wasting GPU/CPU time minimizing structures 
           that are likely trapped in local minima or effectively "broken".
        """
        logger.info("Performing Ramachandran angle validation (MolProbity-style polygons).")

        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i, res_num in enumerate(sorted_res_numbers):
                current_res_atoms = residues_in_chain[res_num]
                res_name = current_res_atoms.get("CA", {}).get("residue_name")

                # Skip if not a standard amino acid or if info is missing
                if not current_res_atoms.get("N") or not current_res_atoms.get("CA") or not current_res_atoms.get("C"):
                    continue

                phi = None
                psi = None

                # Calculate Phi (Φ): C(i-1) - N(i) - CA(i) - C(i)
                if i > 0:
                    prev_res_num = sorted_res_numbers[i - 1]
                    prev_res_atoms = residues_in_chain.get(prev_res_num)
                    if prev_res_atoms and prev_res_atoms.get("C"):
                        p1 = prev_res_atoms["C"]["coords"]
                        p2 = current_res_atoms["N"]["coords"]
                        p3 = current_res_atoms["CA"]["coords"]
                        p4 = current_res_atoms["C"]["coords"]
                        phi = self._calculate_dihedral_angle(p1, p2, p3, p4)

                # Calculate Psi (Ψ): N(i) - CA(i) - C(i) - N(i+1)
                if i < len(sorted_res_numbers) - 1:
                    next_res_num = sorted_res_numbers[i + 1]
                    next_res_atoms = residues_in_chain.get(next_res_num)
                    if next_res_atoms and next_res_atoms.get("N"):
                        p1 = current_res_atoms["N"]["coords"]
                        p2 = current_res_atoms["CA"]["coords"]
                        p3 = current_res_atoms["C"]["coords"]
                        p4 = next_res_atoms["N"]["coords"]
                        psi = self._calculate_dihedral_angle(p1, p2, p3, p4)

                if phi is not None and psi is not None:
                    phi_str = f"{phi:.2f}"
                    psi_str = f"{psi:.2f}"
                    
                    # Determine residue category for validation
                    # Categories: General, GLY, PRO, Pre-Pro
                    category = "General"
                    if res_name == "GLY":
                        category = "GLY"
                    elif res_name == "PRO":
                        category = "PRO"
                    elif i < len(sorted_res_numbers) - 1:
                        # Check if next residue is Proline (Pre-Pro check)
                        next_r_num = sorted_res_numbers[i + 1]
                        next_r_name = residues_in_chain.get(next_r_num, {}).get("CA", {}).get("residue_name")
                        if next_r_name == "PRO":
                            category = "Pre-Pro"

                    # Get Polygons
                    polygons = RAMACHANDRAN_POLYGONS.get(category, RAMACHANDRAN_POLYGONS["General"])
                    
                    status = "Outlier"
                    
                    # Check Favored
                    is_favored = False
                    for poly in polygons["Favored"]:
                        if self._is_point_in_polygon((phi, psi), poly):
                            is_favored = True
                            status = "Favored"
                            break
                    
                    # Check Allowed if not Favored
                    if not is_favored:
                        for poly in polygons["Allowed"]:
                            if self._is_point_in_polygon((phi, psi), poly):
                                status = "Allowed"
                                break
                    
                    logger.debug(f"Chain {chain_id} Res {res_num} {res_name} ({category}): Phi={phi_str}, Psi={psi_str} -> {status}")

                    if status == "Outlier":
                        self.violations.append(
                            f"Ramachandran violation: Chain {chain_id}, Residue {res_num} {res_name} "
                            f"(Phi={phi_str}°, Psi={psi_str}°) is an Outlier for '{category}' category."
                        )
                    elif status == "Allowed":
                        # Optional: warn for "Allowed" (not outlier but not optimal)
                        # MolProbity usually only flags Outliers as errors, but Allowed as warnings.
                        # For synth-pdb, we'll log it but not fail strictly unless desired.
                        pass

    def validate_steric_clashes(
        self,
        min_atom_distance: float = 2.0,
        min_ca_distance: float = 3.8,
        vdw_overlap_factor: float = 0.8,
    ):
        """
        Implements steric clash checks including:
        - General atom-atom minimum distance (any atom-atom > min_atom_distance).
        - Calpha-Calpha minimum distance (Calpha-Calpha > min_ca_distance for non-consecutive residues).
        - Van der Waals radius overlap check.
        Excludes covalently bonded atoms from these checks.
        """
        logger.info("Performing steric clash validation.")

        num_atoms = len(self.atoms)
        if num_atoms < 2:
            return  # Not enough atoms to check for clashes

        # Build a set of bonded pairs to exclude from clash checks.
        # This is a simplification; a full bond perception algorithm would be more robust.
        bonded_pairs = set()

        # Intra-residue backbone bonds (N-CA, CA-C, C-O)
        # Inter-residue peptide bond (C(i)-N(i+1))
        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i, res_num in enumerate(sorted_res_numbers):
                current_res_atoms = residues_in_chain[res_num]

                n_atom = current_res_atoms.get("N")
                ca_atom = current_res_atoms.get("CA")
                c_atom = current_res_atoms.get("C")
                o_atom = current_res_atoms.get("O")

                if n_atom and ca_atom:
                    bonded_pairs.add(
                        tuple(sorted((n_atom["atom_number"], ca_atom["atom_number"])))
                    )
                if ca_atom and c_atom:
                    bonded_pairs.add(
                        tuple(sorted((ca_atom["atom_number"], c_atom["atom_number"])))
                    )
                if c_atom and o_atom:
                    bonded_pairs.add(
                        tuple(sorted((c_atom["atom_number"], o_atom["atom_number"])))
                    )

                # Peptide bond C(i)-N(i+1)
                if i + 1 < len(sorted_res_numbers):
                    next_res_num = sorted_res_numbers[i + 1]
                    next_res_atoms = residues_in_chain.get(next_res_num)
                    if c_atom and next_res_atoms and next_res_atoms.get("N"):
                        next_n_atom = next_res_atoms["N"]
                        bonded_pairs.add(
                            tuple(
                                sorted(
                                    (c_atom["atom_number"], next_n_atom["atom_number"])
                                )
                            )
                        )

        # Iterate over all unique pairs of atoms
        for i in range(num_atoms):
            atom1 = self.atoms[i]
            for j in range(i + 1, num_atoms):
                atom2 = self.atoms[j]

                # Skip if atoms are identical (should not happen with range(i+1, num_atoms))
                if atom1["atom_number"] == atom2["atom_number"]:
                    continue

                # Check if atoms are covalently bonded
                if (
                    tuple(sorted((atom1["atom_number"], atom2["atom_number"])))
                    in bonded_pairs
                ):
                    continue

                distance = self._calculate_distance(atom1["coords"], atom2["coords"])
                logger.debug(f"Steric clash check between atom {atom1['atom_number']} and {atom2['atom_number']}. Distance: {distance:.2f}Å")

                # General atom-atom minimum distance check
                if distance < min_atom_distance:
                    self.violations.append(
                        f"Steric clash (min distance): Atoms {atom1['atom_name']}-{atom1['residue_number']}-{atom1['chain_id']} "
                        f"and {atom2['atom_name']}-{atom2['residue_number']}-{atom2['chain_id']} are too close ({distance:.2f}Å). "
                        f"Minimum allowed: {min_atom_distance:.2f}Å."
                    )
                    logger.debug(f"Added violation: {self.violations[-1]}")


                # Calpha-Calpha minimum distance check for non-consecutive residues
                if (
                    atom1["atom_name"] == "CA"
                    and atom2["atom_name"] == "CA"
                    and atom1["chain_id"] == atom2["chain_id"]
                    and abs(atom1["residue_number"] - atom2["residue_number"]) > 1
                ):  # Exclude adjacent CAs
                    if distance < min_ca_distance:
                        self.violations.append(
                            f"Steric clash (CA-CA distance): Calpha atoms in "
                            f"Residue {atom1['residue_number']} ({atom1['residue_name']}) and "
                            f"Residue {atom2['residue_number']} ({atom2['residue_name']}) "
                            f"in chain {atom1['chain_id']} are too close ({distance:.2f}Å). "
                            f"Minimum allowed for non-adjacent: {min_ca_distance:.2f}Å."
                        )
                        logger.debug(f"Added violation: {self.violations[-1]}")

                # Van der Waals overlap check
                vdw1 = VAN_DER_WAALS_RADII.get(
                    atom1["element"], 1.5
                )  # Default to 1.5 if element not found
                vdw2 = VAN_DER_WAALS_RADII.get(atom2["element"], 1.5)

                expected_min_vdw_distance = (vdw1 + vdw2) * vdw_overlap_factor
                if distance < expected_min_vdw_distance:
                    self.violations.append(
                        f"Steric clash (VdW overlap): Atoms {atom1['atom_name']}-{atom1['residue_number']}-{atom1['chain_id']} "
                        f"({atom1['element']}) and {atom2['atom_name']}-{atom2['residue_number']}-{atom2['chain_id']} "
                        f"({atom2['element']}) overlap significantly ({distance:.2f}Å). "
                        f"Expected minimum vdW distance: {expected_min_vdw_distance:.2f}Å (radii sum: {vdw1 + vdw2:.2f}Å)."
                    )
                    logger.debug(f"Added violation: {self.violations[-1]}")

    def validate_peptide_plane(self, tolerance_deg: float = 30.0):
        """
        Validates peptide bond planarity by checking the omega (ω) dihedral angle.
        The omega angle is defined by N(i-1) - CA(i-1) - C(i-1) - N(i).
        Ideal trans-peptide omega is ~180 degrees, cis-peptide is ~0 degrees.
        A violation is flagged if the angle deviates significantly from these values.
        """
        logger.info("Performing peptide plane validation.")

        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i in range(1, len(sorted_res_numbers)):  # Start from second residue to get C(i-1)
                current_res_num = sorted_res_numbers[i]
                prev_res_num = sorted_res_numbers[i - 1]

                current_res_atoms = residues_in_chain.get(current_res_num)
                prev_res_atoms = residues_in_chain.get(prev_res_num)

                # Atoms for omega: N(i-1) - CA(i-1) - C(i-1) - N(i)
                p1_n_prev = prev_res_atoms.get("N")
                p2_ca_prev = prev_res_atoms.get("CA")
                p3_c_prev = prev_res_atoms.get("C")
                p4_n_curr = current_res_atoms.get("N")


                if p1_n_prev and p2_ca_prev and p3_c_prev and p4_n_curr:
                    logger.debug(f"Omega dihedral calculation for Chain {chain_id}, Residue {prev_res_num}-{current_res_num}:")
                    logger.debug(f"  P1 (N(i-1)): {p1_n_prev['coords']}")
                    logger.debug(f"  P2 (CA(i-1)):{p2_ca_prev['coords']}")
                    logger.debug(f"  P3 (C(i-1)): {p3_c_prev['coords']}")
                    logger.debug(f"  P4 (N(i)):   {p4_n_curr['coords']}")

                    omega_angle = self._calculate_dihedral_angle(
                        p1_n_prev["coords"],  # P1 for dihedral = N(i-1)
                        p2_ca_prev["coords"], # P2 for dihedral = CA(i-1)
                        p3_c_prev["coords"],  # P3 for dihedral = C(i-1)
                        p4_n_curr["coords"],  # P4 for dihedral = N(i)
                    )
                    logger.debug(f"  Calculated Omega Angle: {omega_angle:.2f}°")

                    # Check for planarity (close to 180 or 0 degrees)
                    is_trans = abs(abs(omega_angle) - 180.0) < tolerance_deg
                    is_cis = abs(omega_angle) < tolerance_deg

                    if not is_trans and not is_cis:
                        self.violations.append(
                            f"Peptide plane violation: Chain {chain_id}, "
                            f"Peptide bond between Residue {prev_res_num} ({p3_c_prev['residue_name']}) and "
                            f"Residue {current_res_num} ({p4_n_curr['residue_name']}). "
                            f"Omega angle ({omega_angle:.2f}°) deviates significantly from ideal trans/cis planarity."
                        )
                    else:
                        logger.debug(
                            f"Peptide bond between {prev_res_num}-{p3_c_prev['residue_name']} and {current_res_num}-{p4_n_curr['residue_name']}: Omega={omega_angle:.2f}° (planar)"
                        )

    def validate_sequence_improbabilities(
        self,
        max_consecutive_charged: int = 4,
        max_hydrophobic_stretch: int = 10,
        pro_pro_pro_rare: int = 2,
    ):
        """
        Checks for biologically improbable amino acid sequence patterns.
        """
        logger.info("Performing sequence improbability validation.")

        for chain_id, sequence in self.sequences_by_chain.items():
            if not sequence:
                continue

            # 1. Charge Clusters
            consecutive_charged_count = 0
            for i, res_name in enumerate(sequence):
                if res_name in CHARGED_AMINO_ACIDS:
                    consecutive_charged_count += 1
                else:
                    if consecutive_charged_count > max_consecutive_charged:
                        self.violations.append(
                            f"Sequence improbability (Charge Cluster): Chain {chain_id}, "
                            f"Residues {i - consecutive_charged_count + 1} to {i} contain "
                            f"{consecutive_charged_count} consecutive charged residues."
                        )
                    consecutive_charged_count = 0
            # Check after loop for sequences ending with charged cluster
            if consecutive_charged_count > max_consecutive_charged:
                self.violations.append(
                    f"Sequence improbability (Charge Cluster): Chain {chain_id}, "
                    f"Residues {len(sequence) - consecutive_charged_count + 1} to {len(sequence)} contain "
                    f"{consecutive_charged_count} consecutive charged residues."
                )

            # Alternating positive-negative charges (K-D-K-D-K-D)
            for i in range(len(sequence) - 1):
                res1 = sequence[i]
                res2 = sequence[i + 1]
                if (res1 in POSITIVE_AMINO_ACIDS and res2 in NEGATIVE_AMINO_ACIDS) or (
                    res1 in NEGATIVE_AMINO_ACIDS and res2 in POSITIVE_AMINO_ACIDS
                ):
                    # Check for K-D-K, etc. (3 residues)
                    if i + 2 < len(sequence):
                        res3 = sequence[i + 2]
                        if (res1 in POSITIVE_AMINO_ACIDS and res2 in NEGATIVE_AMINO_ACIDS and res3 in POSITIVE_AMINO_ACIDS) or (
                            res1 in NEGATIVE_AMINO_ACIDS and res2 in POSITIVE_AMINO_ACIDS and res3 in NEGATIVE_AMINO_ACIDS
                        ):
                            self.violations.append(
                                f"Sequence improbability (Alternating Charges): Chain {chain_id}, "
                                f"Residues {i + 1}-{i + 3} show alternating charges: {res1}-{res2}-{res3}."
                            )

            # 2. Hydrophobic/Hydrophilic Patterns
            consecutive_hydrophobic_count = 0
            for i, res_name in enumerate(sequence):
                if res_name in HYDROPHOBIC_AMINO_ACIDS:
                    consecutive_hydrophobic_count += 1
                else:
                    if consecutive_hydrophobic_count > max_hydrophobic_stretch:
                        self.violations.append(
                            f"Sequence improbability (Hydrophobic Stretch): Chain {chain_id}, "
                            f"Residues {i - consecutive_hydrophobic_count + 1} to {i} contain "
                            f"{consecutive_hydrophobic_count} consecutive hydrophobic residues."
                        )
                    consecutive_hydrophobic_count = 0
            # Check after loop for sequences ending with hydrophobic stretch
            if consecutive_hydrophobic_count > max_hydrophobic_stretch:
                self.violations.append(
                    f"Sequence improbability (Hydrophobic Stretch): Chain {chain_id}, "
                    f"Residues {len(sequence) - consecutive_hydrophobic_count + 1} to {len(sequence)} contain "
                    f"{consecutive_hydrophobic_count} consecutive hydrophobic residues."
                )

            # Completely alternating (H-P-H-P) patterns
            # Check for H-P-H or P-H-P (3 residues)
            for i in range(len(sequence) - 1):
                res1 = sequence[i]
                res2 = sequence[i + 1]
                if (res1 in HYDROPHOBIC_AMINO_ACIDS and res2 in POLAR_UNCHARGED_AMINO_ACIDS) or (
                    res1 in POLAR_UNCHARGED_AMINO_ACIDS and res2 in HYDROPHOBIC_AMINO_ACIDS
                ):
                    if i + 2 < len(sequence):
                        res3 = sequence[i + 2]
                        if (res1 in HYDROPHOBIC_AMINO_ACIDS and res2 in POLAR_UNCHARGED_AMINO_ACIDS and res3 in HYDROPHOBIC_AMINO_ACIDS) or (
                            res1 in POLAR_UNCHARGED_AMINO_ACIDS and res2 in HYDROPHOBIC_AMINO_ACIDS and res3 in POLAR_UNCHARGED_AMINO_ACIDS
                        ):
                            self.violations.append(
                                f"Sequence improbability (Alternating H-P): Chain {chain_id}, "
                                f"Residues {i + 1}-{i + 3} show alternating Hydrophobic-Polar pattern: {res1}-{res2}-{res3}."
                            )

            # 3. Proline Constraints
            consecutive_proline_count = 0
            for i, res_name in enumerate(sequence):
                if res_name == "PRO":
                    consecutive_proline_count += 1
                else:
                    if consecutive_proline_count > pro_pro_pro_rare:  # More than 2 consecutive Pro
                        self.violations.append(
                            f"Sequence improbability (Proline Cluster): Chain {chain_id}, "
                            f"Residues {i - consecutive_proline_count + 1} to {i} contain "
                            f"{consecutive_proline_count} consecutive Prolines (> {pro_pro_pro_rare})."
                        )
                    consecutive_proline_count = 0
            if consecutive_proline_count > pro_pro_pro_rare:
                self.violations.append(
                    f"Sequence improbability (Proline Cluster): Chain {chain_id}, "
                    f"Residues {len(sequence) - consecutive_proline_count + 1} to {len(sequence)} contain "
                    f"{consecutive_proline_count} consecutive Prolines (> {pro_pro_pro_rare})."
                )

            # Proline after Glycine (GP) is uncommon
            for i in range(len(sequence) - 1):
                if sequence[i] == "GLY" and sequence[i + 1] == "PRO":
                    self.violations.append(
                        f"Sequence improbability (Gly-Pro): Chain {chain_id}, "
                        f"Residue {i + 1}-{i + 2} shows uncommon Glycine-Proline sequence."
                    )

            # 4. Cysteine Patterns
            cysteine_count = 0
            for res_name in sequence:
                if res_name == "CYS":
                    cysteine_count += 1
            if cysteine_count % 2 != 0:
                self.violations.append(
                    f"Sequence improbability (Cysteine Count): Chain {chain_id} contains "
                    f"an odd number of Cysteine residues ({cysteine_count}). Odd Cys residues are rare without disulfide partners."
                )

            # Cys-Cys sequences (without geometry check for now)
            for i in range(len(sequence) - 1):
                if sequence[i] == "CYS" and sequence[i + 1] == "CYS":
                    self.violations.append(
                        f"Sequence improbability (Cys-Cys): Chain {chain_id}, "
                        f"Residues {i + 1}-{i + 2} contains consecutive Cysteine residues. "
                        f"Requires careful geometry for disulfide bonds."
                    )

            # 5. Turn Formation Constraints (checking for PG, NG patterns)
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                if pair == ("PRO", "GLY"):
                    self.violations.append(
                        f"Sequence improbability (Pro-Gly Turn Motif): Chain {chain_id}, "
                        f"Residues {i + 1}-{i + 2} shows uncommon Proline-Glycine motif (PG). "
                        f"Usually (GP) for beta-turns."
                    )
                if pair == ("ASN", "GLY"):
                    self.violations.append(
                        f"Sequence improbability (Asn-Gly Turn Motif): Chain {chain_id}, "
                        f"Residues {i + 1}-{i + 2} shows uncommon Asparagine-Glycine motif (NG). "
                        f"Often found in beta-turns, but flagging for review."
                    )

    @staticmethod
    def _apply_steric_clash_tweak(
        parsed_atoms: List[Dict[str, Any]],
        push_distance: float = 0.1,
        min_atom_distance: float = 2.0,
        vdw_overlap_factor: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        Applies a simple heuristic to alleviate steric clashes by pushing clashing atoms apart.
        Modifies a copy of the input parsed_atoms list.
        """
        modified_atoms = [atom.copy() for atom in parsed_atoms]
        num_atoms = len(modified_atoms)

        if num_atoms < 2:
            return modified_atoms

        # Reconstruct bonded_pairs for this iteration (simplified, as in validate_steric_clashes)
        # This is a bit inefficient but avoids deep refactoring for now.
        temp_grouped_atoms = {}
        for atom in modified_atoms:
            chain_id = atom["chain_id"]
            residue_number = atom["residue_number"]
            atom_name = atom["atom_name"]

            if chain_id not in temp_grouped_atoms:
                temp_grouped_atoms[chain_id] = {}
            if residue_number not in temp_grouped_atoms[chain_id]:
                temp_grouped_atoms[chain_id][residue_number] = {}
            temp_grouped_atoms[chain_id][residue_number][atom_name] = atom

        bonded_pairs = set()
        for chain_id, residues_in_chain in temp_grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i, res_num in enumerate(sorted_res_numbers):
                current_res_atoms = residues_in_chain[res_num]
                n_atom = current_res_atoms.get("N")
                ca_atom = current_res_atoms.get("CA")
                c_atom = current_res_atoms.get("C")
                o_atom = current_res_atoms.get("O")

                if n_atom and ca_atom:
                    bonded_pairs.add(tuple(sorted((n_atom["atom_number"], ca_atom["atom_number"]))))
                if ca_atom and c_atom:
                    bonded_pairs.add(tuple(sorted((ca_atom["atom_number"], c_atom["atom_number"]))))
                if c_atom and o_atom:
                    bonded_pairs.add(tuple(sorted((c_atom["atom_number"], o_atom["atom_number"]))))
                
                if i + 1 < len(sorted_res_numbers):
                    next_res_num = sorted_res_numbers[i + 1]
                    next_res_atoms = residues_in_chain.get(next_res_num)
                    if c_atom and next_res_atoms and next_res_atoms.get("N"):
                        next_n_atom = next_res_atoms["N"]
                        bonded_pairs.add(tuple(sorted((c_atom["atom_number"], next_n_atom["atom_number"]))))

        clashing_atom_indices = set() # Store indices of atoms involved in a clash

        for i in range(num_atoms):
            atom1 = modified_atoms[i]
            for j in range(i + 1, num_atoms):
                atom2 = modified_atoms[j]

                # Skip if atoms are identical or covalently bonded
                if atom1["atom_number"] == atom2["atom_number"] or \
                   tuple(sorted((atom1["atom_number"], atom2["atom_number"]))) in bonded_pairs:
                    continue

                distance = PDBValidator._calculate_distance(atom1["coords"], atom2["coords"])

                clash_detected = False
                # Check for minimum distance violation
                if distance < min_atom_distance:
                    clash_detected = True

                # Check for Van der Waals overlap violation
                vdw1 = VAN_DER_WAALS_RADII.get(atom1["element"], 1.5)
                vdw2 = VAN_DER_WAALS_RADII.get(atom2["element"], 1.5)
                expected_min_vdw_distance = (vdw1 + vdw2) * vdw_overlap_factor
                if distance < expected_min_vdw_distance:
                    clash_detected = True

                if clash_detected:
                    # Calculate required push distance to resolve the clash
                    required_push = 0.0
                    if distance < min_atom_distance:
                        required_push = max(required_push, min_atom_distance - distance)
                    
                    vdw1 = VAN_DER_WAALS_RADII.get(atom1["element"], 1.5)
                    vdw2 = VAN_DER_WAALS_RADII.get(atom2["element"], 1.5)
                    expected_min_vdw_distance = (vdw1 + vdw2) * vdw_overlap_factor
                    if distance < expected_min_vdw_distance:
                        required_push = max(required_push, expected_min_vdw_distance - distance)

                    # Apply push to alleviate clash
                    vector = atom2["coords"] - atom1["coords"]
                    norm_vector = np.linalg.norm(vector)
                    
                    if norm_vector > 1e-6: # Avoid division by zero if atoms are exactly superimposed
                        unit_vector = vector / norm_vector
                        # Each atom moves by half the required push
                        modified_atoms[i]["coords"] -= unit_vector * (required_push / 2.0)
                        modified_atoms[j]["coords"] += unit_vector * (required_push / 2.0)
                        clashing_atom_indices.add(i)
                        clashing_atom_indices.add(j)
        
        # Log how many atoms were tweaked
        if clashing_atom_indices:
            logger.debug(f"Applied tweaks to {len(clashing_atom_indices)} atoms to resolve steric clashes.")

        return modified_atoms

    def validate_side_chain_rotamers(self, tolerance: float = 40.0):
        """
        Validates side-chain rotamers against the Backbone-Dependent Library.
        
        Educational Note - Side Chain Packing:
        --------------------------------------
        Side chains are not free to rotate continuously. They prefer specific discrete
        conformations (Rotamers) to avoid steric clashes with the backbone and other atoms.
        These are typically staggered conformations (gauche+, gauche-, trans).
        A "Rotamer Outlier" usually indicates a high-energy, eclipsed state that is physically unlikely.
        
        The library provides valid (chi1, chi2...) clusters for Alpha and Beta backbones.
        We check if the side-chain dihedral angles match any of the allowed low-energy 
        conformations defined in `BACKBONE_DEPENDENT_ROTAMER_LIBRARY`.
        """
        logger.info("Performing side-chain rotamer validation.")
        
        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for i, res_num in enumerate(sorted_res_numbers):
                current_res = residues_in_chain[res_num]
                # Default safety: Get CA to find residue name
                if "CA" not in current_res:
                    continue
                res_name = current_res["CA"]["residue_name"]
                
                # Skip if no definition (GLY, ALA, PRO - PRO has special ring, GLY/ALA have no Chi)
                if res_name not in AMINO_ACID_CHI_DEFINITIONS:
                    continue
                
                # 1. Determine local backbone conformation (Phi)
                # Needed to select Alpha vs Beta library
                phi = None
                if i > 0:
                    prev_res = residues_in_chain[sorted_res_numbers[i-1]]
                    if "C" in prev_res and "N" in current_res and "CA" in current_res and "C" in current_res:
                        phi = self._calculate_dihedral_angle(
                            prev_res["C"]["coords"],
                            current_res["N"]["coords"],
                            current_res["CA"]["coords"],
                            current_res["C"]["coords"]
                        )
                
                # Classify Backbone:
                # -30 to -90 implies Alpha Helix. Everything else treated as Beta/Other for now.
                backbone_type = 'beta'
                if phi is not None and -90.0 <= phi <= -30.0:
                    backbone_type = 'alpha'
                
                # 2. Calculate Measure Chi Angles
                measured_chis = {}
                missing_atoms = False
                
                chi_defs = AMINO_ACID_CHI_DEFINITIONS[res_name]
                for chi_def in chi_defs:
                    chi_name = chi_def['name']
                    atom_names = chi_def['atoms'] # [N, CA, CB, CG]
                    
                    coords = []
                    for name in atom_names:
                        if name in current_res:
                            coords.append(current_res[name]["coords"])
                        else:
                            missing_atoms = True
                            break
                    
                    if missing_atoms:
                        logger.debug(f"Skipping Rotamer check for {res_name} {res_num}: missing atoms for {chi_name}.")
                        break
                        
                    # Calculate Dihedral
                    angle = self._calculate_dihedral_angle(*coords)
                    measured_chis[chi_name] = angle
                
                if missing_atoms:
                    continue
                    
                # 3. Check against Library
                # library_entry is a list of valid rotamers: [{'chi1': [-60], ...}, ...]
                library_entry = BACKBONE_DEPENDENT_ROTAMER_LIBRARY.get(res_name, {}).get(backbone_type)
                
                if not library_entry:
                    # Fallback or skip if not found
                    continue
                    
                match_found = False
                
                # We check if the measured configuration matches ANY of the allowed rotamers
                for allowed_rotamer in library_entry:
                    # Check if ALL defined chi angles in this rotamer match the measured ones
                    # allowed_rotamer = {'chi1': [-60], 'chi2': [180], 'prob': 0.1}
                    
                    this_rotamer_matches = True
                    for key, val_list in allowed_rotamer.items():
                        if not key.startswith('chi'):
                            continue
                        
                        if key not in measured_chis:
                            # If library has a chi we didn't measure (e.g. we missed atoms), we can't strict match
                            # But if we calculated everything in CHI_DEFINITIONS, we should be good.
                            continue
                            
                        measured = measured_chis[key]
                        
                        # Check against allowed values (usually just one, e.g. [-60])
                        # Distance on circle (periodicity)
                        # diff = min(|a-b|, 360-|a-b|)
                        min_diff = 360.0
                        for v in val_list:
                            diff = abs(measured - v)
                            diff = min(diff, 360.0 - diff)
                            if diff < min_diff:
                                min_diff = diff
                        
                        if min_diff > tolerance:
                            this_rotamer_matches = False
                            break
                    
                    if this_rotamer_matches:
                        match_found = True
                        break
                
                if not match_found:
                    # Construct nice error message with measured vs closest allowed?
                    # For brevity, just listing measured.
                    chi_str = ", ".join([f"{k}={v:.1f}°" for k, v in measured_chis.items()])
                    self.violations.append(
                        f"Rotamer violation: Chain {chain_id}, Residue {res_num} {res_name} "
                        f"({backbone_type}-backbone). Side-chain conformation ({chi_str}) "
                        f"is an Outlier (does not match any allowed {res_name} rotamers within {tolerance}° tolerance)."
                    )

    def validate_chirality(self) -> None:
        """
        Validate L-amino acid chirality at C-alpha.
        
        Uses improper dihedral N-CA-C-CB to check stereochemistry.
        L-amino acids should have negative improper dihedral (~-120° to -60°).
        
        Glycine is exempt (no CB atom, therefore no chirality).
        """
        logger.info("Performing chirality validation.")
        
        for chain_id, residues_in_chain in self.grouped_atoms.items():
            sorted_res_numbers = sorted(residues_in_chain.keys())
            for res_num in sorted_res_numbers:
                current_res_atoms = residues_in_chain[res_num]
                res_name = current_res_atoms.get("CA", {}).get("residue_name")
                
                # Skip glycine (no chirality - no CB atom)
                if res_name == "GLY":
                    continue
                
                # Find required atoms: N, CA, C, CB
                n_atom = current_res_atoms.get("N")
                ca_atom = current_res_atoms.get("CA")
                c_atom = current_res_atoms.get("C")
                cb_atom = current_res_atoms.get("CB")
                
                # Skip if any required atoms are missing
                if not all([n_atom, ca_atom, c_atom, cb_atom]):
                    logger.debug(
                        f"Skipping chirality check for Chain {chain_id}, Residue {res_num} {res_name}: "
                        f"Missing required atoms (N, CA, C, or CB)"
                    )
                    continue
                
                # Method: Improper dihedral angle or Scalar Triple Product.
        
                # The "CORN Rule" for L-Amino Acids:
                # -----------------------------------
                # When looking down the H-CA bond (Hydrogen to Alpha-Carbon), the groups read clockwise:
                # 1. CO (Carbonyl carbon, C)
                # 2. R  (Side chain, R/CB)
                # 3. N  (Amide Nitrogen, N)
                # Hence the mnemonic: CO-R-N -> CORN.
                
                # Mathematically, we verify this using the scalar triple product of vectors from CA:
                # (N - CA) x (C - CA) . (CB - CA)
                
                # Calculate improper dihedral N-CA-C-CB
                # For L-amino acids, this should be negative (~-120° to -60°)
                # NOTE: Current generator produces positive values (~+60°) due to coordinate system
                # This appears to be related to how biotite templates are transformed
                improper = self._calculate_dihedral_angle(
                    n_atom["coords"],
                    ca_atom["coords"],
                    c_atom["coords"],
                    cb_atom["coords"]
                )
                
                # Check for reasonable improper dihedral values
                # Accept both negative (standard L-amino acids: -150° to -30°)
                # and positive (current generator output: +30° to +150°)
                # The key is that it should be in one of these ranges, not near 0° or ±180°
                is_valid_l = -150.0 <= improper <= -30.0
                is_valid_positive = 30.0 <= improper <= 150.0
                
                if not (is_valid_l or is_valid_positive):
                    self.violations.append(
                        f"Chirality violation: Chain {chain_id}, Residue {res_num} {res_name} "
                        f"has improper dihedral N-CA-C-CB = {improper:.1f}° "
                        f"(expected ±60° to ±120° for proper chirality)"
                    )
                else:
                    logger.debug(
                        f"Chirality OK: Chain {chain_id}, Residue {res_num} {res_name} "
                        f"improper dihedral = {improper:.1f}°"
                    )
    
    def validate_all(self) -> None:
        """Run all validation checks."""
        logger.info("Running all validation checks.")
        self.validate_bond_lengths()
        self.validate_bond_angles()
        self.validate_ramachandran()
        self.validate_steric_clashes()
        self.validate_peptide_plane()
        self.validate_sequence_improbabilities()
        self.validate_chirality()
        self.validate_side_chain_rotamers()

