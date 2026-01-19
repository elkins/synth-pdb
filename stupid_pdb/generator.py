import random
import numpy as np
import logging
from datetime import datetime
from .data import (
    STANDARD_AMINO_ACIDS,
    ONE_TO_THREE_LETTER_CODE,
    AMINO_ACID_ATOMS,
    AMINO_ACID_FREQUENCIES,
    BOND_LENGTH_N_CA,
    BOND_LENGTH_CA_C,
    BOND_LENGTH_C_O,
    ANGLE_N_CA_C,
    ANGLE_CA_C_N,
    ANGLE_CA_C_O,
    BOND_LENGTH_C_N,
    ANGLE_C_N_CA,
)

from stupid_pdb.validator import PDBValidator # Temporary import for debugging

# Convert angles to radians for numpy trigonometric functions
ANGLE_N_CA_C_RAD = np.deg2rad(ANGLE_N_CA_C)
ANGLE_CA_C_N_RAD = np.deg2rad(ANGLE_CA_C_N)
ANGLE_CA_C_O_RAD = np.deg2rad(ANGLE_CA_C_O)

# Ideal Ramachandran angles for a generic alpha-helix
PHI_ALPHA_HELIX = -57.0
PSI_ALPHA_HELIX = -47.0
# Ideal Omega for trans peptide bond
OMEGA_TRANS = 180.0

logger = logging.getLogger(__name__)


def get_current_date_pdb_format() -> str:
    """
    Returns the current date formatted for PDB HEADER record (DD-MON-YY).
    """
    return datetime.now().strftime("%d-%b-%y").upper()


# This constant is used in test_generator.py for coordinate calculations.
CA_DISTANCE = (
    3.8  # Approximate C-alpha to C-alpha distance in Angstroms for a linear chain
)

PDB_ATOM_FORMAT = "ATOM  {atom_number: >5} {atom_name: <4}{alt_loc: <1}{residue_name: >3} {chain_id: <1}{residue_number: >4}{insertion_code: <1}   {x_coord: >8.3f}{y_coord: >8.3f}{z_coord: >8.3f}{occupancy: >6.2f}{temp_factor: >6.2f}          {element: >2}{charge: >2}"

# Helper function to create a minimal PDB ATOM line
def create_atom_line(
    atom_number: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
    alt_loc: str = "",
    insertion_code: str = ""
) -> str:
    return (
        f"ATOM  {atom_number: >5} {atom_name: <4}{alt_loc: <1}{residue_name: >3} {chain_id: <1}"
        f"{residue_number: >4}{insertion_code: <1}   "
        f"{x: >8.3f}{y: >8.3f}{z: >8.3f}{1.00: >6.2f}{0.00: >6.2f}          "
        f"{element: >2}  "
    )

def _position_atom_3d_from_internal_coords(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    bond_length: float,
    bond_angle_deg: float,
    dihedral_angle_deg: float,
) -> np.ndarray:
    """
    Calculates the 3D coordinates of a new atom (P4) given the coordinates of three
    preceding atoms (P1, P2, P3) and the internal coordinates:
    - bond_length: distance P3-P4
    - bond_angle_deg: angle P2-P3-P4 in degrees
    - dihedral_angle_deg: dihedral angle P1-P2-P3-P4 in degrees

    This function uses a standard method for constructing Cartesian coordinates from
    internal coordinates.
    """
    bond_angle_rad = np.deg2rad(bond_angle_deg)
    dihedral_angle_rad = np.deg2rad(dihedral_angle_deg)

    # Vectors of the reference bonds
    vec_p3_p2 = p2 - p3  # Vector from P3 to P2
    vec_p2_p1 = p1 - p2  # Vector from P2 to P1

    norm_p3_p2 = np.linalg.norm(vec_p3_p2)
    norm_p2_p1 = np.linalg.norm(vec_p2_p1)

    # Handle degenerate cases (collinear points)
    if norm_p3_p2 < 1e-6 or norm_p2_p1 < 1e-6:
        logger.warning(
            "Degenerate bond vector encountered in dihedral calculation. Placing linearly for P4."
        )
        if norm_p3_p2 > 1e-6:
            return p3 + (vec_p3_p2 / norm_p3_p2) * bond_length
        else:
            return p3 + np.array([bond_length, 0, 0])


    # Define an orthonormal basis (e1, e2, e3) for constructing P4
    # e1: Unit vector along P3 -> P2
    e1 = vec_p3_p2 / norm_p3_p2

    # e3: Unit vector normal to the P1-P2-P3 plane
    # The order of cross product determines the direction of the normal.
    # It should be cross(vec_p2_p1, vec_p3_p2)
    e3_vec_unnormalized = np.cross(vec_p2_p1, e1) # Corrected cross product order
    norm_e3_vec = np.linalg.norm(e3_vec_unnormalized)

    if norm_e3_vec < 1e-6: # P1, P2, P3 are collinear
        logger.warning(
            "P1, P2, P3 are collinear, normal vector to plane is ill-defined. Using arbitrary perpendicular for e3."
        )
        if np.isclose(e1[0], 0) and np.isclose(e1[1], 0):
            e3 = np.array([1.0, 0.0, 0.0])
        else:
            e3 = np.array([-e1[1], e1[0], 0.0])
        e3 /= np.linalg.norm(e3)
    else:
        e3 = e3_vec_unnormalized / norm_e3_vec

    # e2: Unit vector in the P1-P2-P3 plane, orthogonal to e1 and e3
    e2 = np.cross(e3, e1)

    # Now, calculate the P3->P4 vector components in this (e1, e2, e3) basis
    # The bond angle is between P2-P3 and P3-P4.
    # The dihedral angle is between plane P1-P2-P3 and P2-P3-P4.

    # P3->P4 vector components
    x_comp = -bond_length * np.cos(bond_angle_rad)
    y_comp = bond_length * np.sin(bond_angle_rad) * np.cos(-dihedral_angle_rad) # Apply sign flip here
    z_comp = bond_length * np.sin(bond_angle_rad) * np.sin(-dihedral_angle_rad) # And here

    # Transform these components back to global coordinates
    global_p4_coords = p3 + \
                       (x_comp * e1) + \
                       (y_comp * e2) + \
                       (z_comp * e3)

    # For debugging, verify the dihedral angle formed
    actual_dihedral = PDBValidator._calculate_dihedral_angle(p1, p2, p3, global_p4_coords)
    logger.debug(f"P1: {p1}, P2: {p2}, P3: {p3}")
    logger.debug(f"Bond Length: {bond_length}, Bond Angle: {bond_angle_deg}, Dihedral Angle: {dihedral_angle_deg}")
    logger.debug(f"e1: {e1}, e2: {e2}, e3: {e3}")
    logger.debug(f"x_comp: {x_comp}, y_comp: {y_comp}, z_comp: {z_comp}")
    logger.debug(f"Calculated dihedral for P4: {actual_dihedral:.2f}° (Expected: {dihedral_angle_deg:.2f}°)")
    logger.debug(f"norm_e3_vec: {norm_e3_vec}")

    return global_p4_coords


def _generate_random_amino_acid_sequence(
    length: int, use_plausible_frequencies: bool = False
) -> list[str]:
    """
    Generates a random amino acid sequence of a given length.
    If `use_plausible_frequencies` is True, uses frequencies from AMINO_ACID_FREQUENCIES.
    Otherwise, uses a uniform random distribution.
    """
    if length is None or length <= 0:
        return []

    if use_plausible_frequencies:
        amino_acids = list(AMINO_ACID_FREQUENCIES.keys())
        weights = list(AMINO_ACID_FREQUENCIES.values())
        return random.choices(amino_acids, weights=weights, k=length)
    else:
        return [random.choice(STANDARD_AMINO_ACIDS) for _ in range(length)]


def _resolve_sequence(
    length: int, user_sequence_str: str = None, use_plausible_frequencies: bool = False
) -> list[str]:
    """
    Resolves the amino acid sequence, either by parsing a user-provided sequence
    or generating a random one.
    """
    if user_sequence_str:
        user_sequence_str_upper = user_sequence_str.upper()
        if "-" in user_sequence_str_upper:
            # Assume 3-letter code format like 'ALA-GLY-VAL'
            amino_acids = [aa.upper() for aa in user_sequence_str_upper.split("-")]
            for aa in amino_acids:
                if aa not in STANDARD_AMINO_ACIDS:
                    raise ValueError(f"Invalid 3-letter amino acid code: {aa}")
            return amino_acids
        elif (
            len(user_sequence_str_upper) == 3
            and user_sequence_str_upper in STANDARD_AMINO_ACIDS
        ):
            # It's a single 3-letter amino acid code
            return [user_sequence_str_upper]
        else:
            # Assume 1-letter code format like 'AGV'
            amino_acids = []
            for one_letter_code in user_sequence_str_upper:
                if one_letter_code not in ONE_TO_THREE_LETTER_CODE:
                    raise ValueError(
                        f"Invalid 1-letter amino acid code: {one_letter_code}"
                    )
                amino_acids.append(ONE_TO_THREE_LETTER_CODE[one_letter_code])
            return amino_acids
    else:
        return _generate_random_amino_acid_sequence(
            length, use_plausible_frequencies=use_plausible_frequencies
        )


def generate_pdb_content(
    length: int = None,
    full_atom: bool = False,
    sequence_str: str = None,
    use_plausible_frequencies: bool = False,
) -> str:
    """
    Generates PDB content for a linear peptide chain.
    """
    sequence = _resolve_sequence(
        length=length,
        user_sequence_str=sequence_str,
        use_plausible_frequencies=use_plausible_frequencies,
    )

    if not sequence:
        if sequence_str is not None and len(sequence_str) == 0:
            raise ValueError("Provided sequence string cannot be empty.")
        raise ValueError(
            "Length must be a positive integer when no sequence is provided and no valid sequence string is given."
        )

    pdb_lines = []

    current_date = get_current_date_pdb_format()
    sequence_length = len(sequence)

    # PDB Header
    pdb_lines.append(
        f"HEADER    PEPTIDE           {current_date}    "
    )  # Classification, date, and ID code
    pdb_lines.append(f"TITLE     GENERATED LINEAR PEPTIDE OF LENGTH {sequence_length}")
    pdb_lines.append(
        "REMARK 1  This PDB file was generated by the CLI 'stupid-pdb' tool."
    )
    pdb_lines.append(
        "REMARK 2  It represents a simplified model of a linear peptide chain."
    )
    pdb_lines.append(
        "REMARK 2  Coordinates are idealized and do not reflect real-world physics."
    )
    pdb_lines.append(
        f"COMPND    MOL_ID: 1; MOLECULE: LINEAR PEPTIDE; CHAIN: A; LENGTH: {sequence_length};"
    )
    pdb_lines.append("SOURCE    ENGINEERED; SYNTHETIC CONSTRUCT;")
    pdb_lines.append("KEYWDS    PEPTIDE, LINEAR, GENERATED, THEORETICAL MODEL")
    pdb_lines.append("EXPDTA    THEORETICAL MODEL")
    pdb_lines.append("AUTHOR    STUPID PDB")
    pdb_lines.append("MODEL        1")

    atom_count = 1

    # Initial coordinates for the first CA atom
    current_ca_coords = np.array([0.0, 0.0, 0.0])

    for i, aa_3l_code in enumerate(sequence):
        residue_number = i + 1

        if not full_atom:
            # Only generate CA atom if full_atom is False
            pdb_lines.append(
                PDB_ATOM_FORMAT.format(
                    atom_number=atom_count,
                    atom_name="CA",
                    alt_loc="",
                    residue_name=aa_3l_code,
                    chain_id="A",
                    residue_number=residue_number,
                    insertion_code="",
                    x_coord=current_ca_coords[0],
                    y_coord=current_ca_coords[1],
                    z_coord=current_ca_coords[2],
                    occupancy=1.00,
                    temp_factor=0.00,
                    element="C",
                    charge="",
                )
            )
            atom_count += 1
            # For CA-only generation, increment current_ca_coords linearly
            current_ca_coords[0] += CA_DISTANCE
        else: # full_atom is True, use Ramachandran-guided backbone generation
            if i == 0:
                # For the first residue (i=0), set initial N, CA, C, O coordinates.
                # These are arbitrary starting points to define the initial local frame.
                # N1 at origin (0,0,0)
                n_coords = np.array([0.0, 0.0, 0.0])
                
                # CA1 relative to N1: Place along X-axis
                ca_coords = n_coords + np.array([BOND_LENGTH_N_CA, 0.0, 0.0])

                # C1 relative to CA1 and N1: Bond N1-CA1-C1 angle
                # To calculate C1: P3=CA1, P2=N1, P1 can be an arbitrary point to define a plane.
                # Let's make P1 such that the N1-CA1-C1 plane is XY, and P1 is N1 shifted in Z.
                c_coords = _position_atom_3d_from_internal_coords(
                    p1=n_coords + np.array([0.0, 0.0, 1.0]), # P1: Arbitrary point not collinear with N1,CA1
                    p2=n_coords, # P2: N1
                    p3=ca_coords, # P3: CA1
                    bond_length=BOND_LENGTH_CA_C,
                    bond_angle_deg=ANGLE_N_CA_C, # Angle N1-CA1-C1
                    dihedral_angle_deg=0.0 # Arbitrary dihedral for initial placement
                )
                
                # O1 relative to C1 and CA1
                # To calculate O1: P3=C1, P2=CA1, P1=N1.
                # Dihedral N1-CA1-C1-O1 for trans-carbonyl
                o_coords = _position_atom_3d_from_internal_coords(
                    p1=n_coords, # P1: N1
                    p2=ca_coords, # P2: CA1
                    p3=c_coords, # P3: C1
                    bond_length=BOND_LENGTH_C_O,
                    bond_angle_deg=ANGLE_CA_C_O, # Angle CA1-C1-O1
                    dihedral_angle_deg=180.0 # Dihedral N1-CA1-C1-O1 (trans)
                )
                
                # Store these as the 'previous' atoms for the next residue's N
                prev_n_coords = n_coords
                prev_ca_coords = ca_coords
                prev_c_coords = c_coords
                
            else: # Subsequent residues (i > 0)
                # Calculate N(i) (P4)
                # Dihedral: N(i-1)-CA(i-1)-C(i-1)-N(i) (OMEGA_TRANS)
                # P1: N(i-1) (prev_n_coords)
                # P2: CA(i-1) (prev_ca_coords)
                # P3: C(i-1) (prev_c_coords)
                n_coords = _position_atom_3d_from_internal_coords(
                    p1=prev_n_coords,
                    p2=prev_ca_coords,
                    p3=prev_c_coords,
                    bond_length=BOND_LENGTH_C_N,
                    bond_angle_deg=ANGLE_CA_C_N, # Angle CA(i-1)-C(i-1)-N(i)
                    dihedral_angle_deg=OMEGA_TRANS,
                )

                # Calculate CA(i) (P4)
                # Dihedral: CA(i-1)-C(i-1)-N(i)-CA(i) (PHI_ALPHA_HELIX)
                # P1: CA(i-1) (prev_ca_coords)
                # P2: C(i-1) (prev_c_coords)
                # P3: N(i) (n_coords)
                ca_coords = _position_atom_3d_from_internal_coords(
                    p1=prev_ca_coords, # P1: CA(i-1)
                    p2=prev_c_coords,  # P2: C(i-1)
                    p3=n_coords,       # P3: N(i)
                    bond_length=BOND_LENGTH_N_CA,
                    bond_angle_deg=ANGLE_C_N_CA, # Angle C(i-1)-N(i)-CA(i)
                    dihedral_angle_deg=PHI_ALPHA_HELIX,
                )

                # Calculate C(i) (P4)
                # Dihedral: C(i-1)-N(i)-CA(i)-C(i) (PSI_ALPHA_HELIX)
                # P1: C(i-1) (prev_c_coords)
                # P2: N(i) (n_coords)
                # P3: CA(i) (ca_coords)
                c_coords = _position_atom_3d_from_internal_coords(
                    p1=prev_c_coords, # P1: C(i-1)
                    p2=n_coords,      # P2: N(i)
                    p3=ca_coords,     # P3: CA(i)
                    bond_length=BOND_LENGTH_CA_C,
                    bond_angle_deg=ANGLE_N_CA_C, # Angle N(i)-CA(i)-C(i)
                    dihedral_angle_deg=PSI_ALPHA_HELIX,
                )

                # Calculate O(i) (P4)
                # Dihedral: N(i)-CA(i)-C(i)-O(i) for trans-carbonyl
                # P1: N(i)
                # P2: CA(i)
                # P3: C(i)
                o_coords = _position_atom_3d_from_internal_coords(
                    p1=n_coords, # P1: N(i)
                    p2=ca_coords, # P2: CA(i)
                    p3=c_coords, # P3: C(i)
                    bond_length=BOND_LENGTH_C_O,
                    bond_angle_deg=ANGLE_CA_C_O, # Angle CA(i)-C(i)-O(i)
                    dihedral_angle_deg=180.0, # Dihedral N(i)-CA(i)-C(i)-O(i) (trans)
                )

                # Update previous coordinates for the next iteration
                prev_n_coords = n_coords
                prev_ca_coords = ca_coords
                prev_c_coords = c_coords
            
            # Append N atom
            pdb_lines.append(
                PDB_ATOM_FORMAT.format(
                    atom_number=atom_count,
                    atom_name="N",
                    alt_loc="",
                    residue_name=aa_3l_code,
                    chain_id="A",
                    residue_number=residue_number,
                    insertion_code="",
                    x_coord=n_coords[0],
                    y_coord=n_coords[1],
                    z_coord=n_coords[2],
                    occupancy=1.00,
                    temp_factor=0.00,
                    element="N",
                    charge="",
                )
            )
            atom_count += 1

            # Append CA atom
            pdb_lines.append(
                PDB_ATOM_FORMAT.format(
                    atom_number=atom_count,
                    atom_name="CA",
                    alt_loc="",
                    residue_name=aa_3l_code,
                    chain_id="A",
                    residue_number=residue_number,
                    insertion_code="",
                    x_coord=ca_coords[0],
                    y_coord=ca_coords[1],
                    z_coord=ca_coords[2],
                    occupancy=1.00,
                    temp_factor=0.00,
                    element="C",
                    charge="",
                )
            )
            atom_count += 1

            # Append C atom
            pdb_lines.append(
                PDB_ATOM_FORMAT.format(
                    atom_number=atom_count,
                    atom_name="C",
                    alt_loc="",
                    residue_name=aa_3l_code,
                    chain_id="A",
                    residue_number=residue_number,
                    insertion_code="",
                    x_coord=c_coords[0],
                    y_coord=c_coords[1],
                    z_coord=c_coords[2],
                    occupancy=1.00,
                    temp_factor=0.00,
                    element="C",
                    charge="",
                )
            )
            atom_count += 1

            # Append O atom
            pdb_lines.append(
                PDB_ATOM_FORMAT.format(
                    atom_number=atom_count,
                    atom_name="O",
                    alt_loc="",
                    residue_name=aa_3l_code,
                    chain_id="A",
                    residue_number=residue_number,
                    insertion_code="",
                    x_coord=o_coords[0],
                    y_coord=o_coords[1],
                    z_coord=o_coords[2],
                    occupancy=1.00,
                    temp_factor=0.00,
                    element="O",
                    charge="",
                )
            )
            atom_count += 1

            # Side chain atoms
            if aa_3l_code in AMINO_ACID_ATOMS:
                for atom_data in AMINO_ACID_ATOMS[aa_3l_code]:
                    # Side chain coords are relative to CA. We'll simply add them to the new CA coords.
                    # This is a simplification and will need to be improved later for realism.
                    side_chain_coords = ca_coords + np.array(atom_data["coords"])
                    pdb_lines.append(
                        PDB_ATOM_FORMAT.format(
                            atom_number=atom_count,
                            atom_name=atom_data["name"],
                            alt_loc="",
                            residue_name=aa_3l_code,
                            chain_id="A",
                            residue_number=residue_number,
                            insertion_code="",
                            x_coord=side_chain_coords[0],
                            y_coord=side_chain_coords[1],
                            z_coord=side_chain_coords[2],
                            occupancy=1.00,
                            temp_factor=0.00,
                            element=atom_data["element"],
                            charge="",
                        )
                    )
                    atom_count += 1

    # TER and END records
    if sequence:
        last_residue_name = sequence[-1]
        last_residue_number = len(sequence)
        # Re-format TER record to match PDB specification and test expectations
        # TER serial       resName chainID resSeq
        # TER   <5 chars> <3 chars> <1 char> <4 chars>
        # The chain ID 'A' needs to be at index 21
        pdb_lines.append(
            f"TER   {atom_count: >5}      {last_residue_name: >3} A{last_residue_number: >4}"
        )



    pdb_lines.append("ENDMDL")
    pdb_lines.append("END         ")  # 10 spaces to match typical PDB END record length

    return "\n".join(pdb_lines)
