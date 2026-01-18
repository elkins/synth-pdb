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
)

# Convert angles to radians for numpy trigonometric functions
ANGLE_N_CA_C_RAD = np.deg2rad(ANGLE_N_CA_C)
ANGLE_CA_C_N_RAD = np.deg2rad(ANGLE_CA_C_N)
ANGLE_CA_C_O_RAD = np.deg2rad(ANGLE_CA_C_O)

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


def _rotate_coords_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates 2D coordinates (x, y) by a given angle in degrees around the origin (0,0).
    Assumes z-coordinate is 0 and maintains it.
    """
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(rotation_matrix, coords)


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
        else:
            # Full atom generation (N, CA, C, O, and side chain)
            # This is a simplified geometry, placing atoms in the XY plane for the first residue
            # and maintaining a linear C-alpha progression along X.

            # CA atom is at current_ca_coords
            ca_coords = current_ca_coords

            # C atom calculation: place along positive X-axis from CA
            c_coords = ca_coords + np.array([BOND_LENGTH_CA_C, 0.0, 0.0])

            # N atom calculation: forms ANGLE_N_CA_C with CA-C.
            # CA-C vector is along positive X from CA.
            # N-CA vector length is BOND_LENGTH_N_CA.
            # Place N in the XY plane such that N-CA-C angle is correct.
            n_x_offset = BOND_LENGTH_N_CA * np.cos(ANGLE_N_CA_C_RAD)
            n_y_offset = BOND_LENGTH_N_CA * np.sin(ANGLE_N_CA_C_RAD)
            n_coords = ca_coords + np.array([n_x_offset, n_y_offset, 0.0])

            # O atom calculation: forms ANGLE_CA_C_O with CA-C, with C as vertex.
            # The C-CA vector points along negative X from C.
            # C-O vector length is BOND_LENGTH_C_O.
            # Place O in the XY plane such that CA-C-O angle is correct.
            # The angle of C-O relative to the positive X-axis should be (180 - ANGLE_CA_C_O).
            o_x_offset_relative_to_C = BOND_LENGTH_C_O * np.cos(
                np.pi - ANGLE_CA_C_O_RAD
            )
            o_y_offset_relative_to_C = BOND_LENGTH_C_O * np.sin(
                np.pi - ANGLE_CA_C_O_RAD
            )
            o_coords = c_coords + np.array(
                [o_x_offset_relative_to_C, o_y_offset_relative_to_C, 0.0]
            )

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
        # Move current_ca_coords for the next residue's CA based on CA_DISTANCE
        current_ca_coords[0] += CA_DISTANCE

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
