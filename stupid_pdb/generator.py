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
    ROTAMER_LIBRARY,
)

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import io

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
logger.setLevel(logging.CRITICAL)


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
    preceding atoms (P1, P2, P3) and the internal coordinates.
    """
    bond_angle_rad = np.deg2rad(bond_angle_deg)
    dihedral_angle_rad = np.deg2rad(dihedral_angle_deg)

    a = p2 - p1
    b = p3 - p2
    c = np.cross(a, b)
    d = np.cross(c, b)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c /= np.linalg.norm(c)
    d /= np.linalg.norm(d)


    p4 = p3 + bond_length * (
        -b * np.cos(bond_angle_rad)
        + d * np.sin(bond_angle_rad) * np.cos(dihedral_angle_rad)
        + c * np.sin(bond_angle_rad) * np.sin(dihedral_angle_rad)
    )
    return p4


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
            # Assume 1-letter code format format 'AGV'
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

    pdb_header_lines = []
    current_date = get_current_date_pdb_format()
    sequence_length = len(sequence)
    pdb_header_lines.append(f"HEADER    PEPTIDE           {current_date}    ")
    pdb_header_lines.append(f"TITLE     GENERATED LINEAR PEPTIDE OF LENGTH {sequence_length}")
    pdb_header_lines.append("REMARK 1  This PDB file was generated by the CLI 'stupid-pdb' tool.")
    pdb_header_lines.append("REMARK 2  It represents a simplified model of a linear peptide chain.")
    pdb_header_lines.append("REMARK 2  Coordinates are idealized and do not reflect real-world physics.")
    pdb_header_lines.append(f"COMPND    MOL_ID: 1; MOLECULE: LINEAR PEPTIDE; CHAIN: A; LENGTH: {sequence_length};")
    pdb_header_lines.append("SOURCE    ENGINEERED; SYNTHETIC CONSTRUCT;")
    pdb_header_lines.append("KEYWDS    PEPTIDE, LINEAR, GENERATED, THEORETICAL MODEL")
    pdb_header_lines.append("EXPDTA    THEORETICAL MODEL")
    pdb_header_lines.append("AUTHOR    STUPID PDB")
    pdb_header_lines.append("MODEL        1")

    peptide = struc.AtomArray(0)
    
    # Build backbone and add side chains
    for i, res_name in enumerate(sequence):
        res_id = i + 1
        
        # Determine backbone coordinates based on previous residue or initial placement
        if i == 0:
            n_coord = np.array([0.0, 0.0, 0.0])
            ca_coord = np.array([BOND_LENGTH_N_CA, 0.0, 0.0])
            c_coord = ca_coord + np.array([BOND_LENGTH_CA_C * np.cos(np.deg2rad(180-ANGLE_N_CA_C)), BOND_LENGTH_CA_C * np.sin(np.deg2rad(180-ANGLE_N_CA_C)), 0.0])
        else:
            # Extract previous C from the already built peptide
            # Use unpadded atom names as biotite normalizes them
            prev_c_atom = peptide[(peptide.res_id == res_id - 1) & (peptide.atom_name == "C")][-1]
            prev_ca_atom = peptide[(peptide.res_id == res_id - 1) & (peptide.atom_name == "CA")][-1]
            prev_n_atom = peptide[(peptide.res_id == res_id - 1) & (peptide.atom_name == "N")][-1]

            n_coord = _position_atom_3d_from_internal_coords(
                prev_n_atom.coord, prev_ca_atom.coord, prev_c_atom.coord,
                BOND_LENGTH_C_N, ANGLE_CA_C_N, OMEGA_TRANS
            )
            ca_coord = _position_atom_3d_from_internal_coords(
                prev_ca_atom.coord, prev_c_atom.coord, n_coord,
                BOND_LENGTH_N_CA, ANGLE_C_N_CA, PHI_ALPHA_HELIX
            )
            c_coord = _position_atom_3d_from_internal_coords(
                prev_c_atom.coord, n_coord, ca_coord,
                BOND_LENGTH_CA_C, ANGLE_N_CA_C, PSI_ALPHA_HELIX
            )
        
        # Get reference residue from biotite
        # Use appropriate terminal definitions
        if i == 0: # N-terminal residue
            ref_res_template = struc.info.residue(res_name, 'N_TERM')
        elif i == len(sequence) - 1: # C-terminal residue
            ref_res_template = struc.info.residue(res_name, 'C_TERM')
        else: # Internal residue
            ref_res_template = struc.info.residue(res_name, 'INTERNAL')

        if res_name in ROTAMER_LIBRARY:
            chi1_target = ROTAMER_LIBRARY[res_name]["chi1"][0]
            
            n_template = ref_res_template[ref_res_template.atom_name == "N"][0]
            ca_template = ref_res_template[ref_res_template.atom_name == "CA"][0]
            cb_template = ref_res_template[ref_res_template.atom_name == "CB"][0]
            cg_template = ref_res_template[ref_res_template.atom_name == "CG"][0]
            
            bond_length_cb_cg = np.linalg.norm(cg_template.coord - cb_template.coord)
            angle_ca_cb_cg = _calculate_angle(ca_template.coord, cb_template.coord, cg_template.coord)

            cg_coord = _position_atom_3d_from_internal_coords(
                n_template.coord, ca_template.coord, cb_template.coord,
                bond_length_cb_cg, angle_ca_cb_cg, chi1_target
            )
            ref_res_template.coord[ref_res_template.atom_name == "CG"][0] = cg_coord
            
        # Extract N, CA, C from ref_res_template
        # Ensure these atoms are present in the template. Some templates might not have N or C (e.g., non-standard)
        template_backbone_n = ref_res_template[ref_res_template.atom_name == "N"]
        template_backbone_ca = ref_res_template[ref_res_template.atom_name == "CA"]
        template_backbone_c = ref_res_template[ref_res_template.atom_name == "C"]

        # Filter out empty AtomArrays for robustness
        mobile_atoms = []
        if template_backbone_n.array_length() > 0:
            mobile_atoms.append(template_backbone_n)
        if template_backbone_ca.array_length() > 0:
            mobile_atoms.append(template_backbone_ca)
        if template_backbone_c.array_length() > 0:
            mobile_atoms.append(template_backbone_c)
        
        if not mobile_atoms:
            raise ValueError(f"Reference residue template for {res_name} is missing N, CA, or C atoms needed for superimposition.")

        mobile_backbone_from_template = struc.array(mobile_atoms)

        # Create the 'target' structure for superimposition from the *constructed* coordinates
        target_backbone_constructed = struc.array([
            struc.Atom(n_coord, atom_name="N", res_id=res_id, res_name=res_name, element="N", hetero=False),
            struc.Atom(ca_coord, atom_name="CA", res_id=res_id, res_name=res_name, element="C", hetero=False),
            struc.Atom(c_coord, atom_name="C", res_id=res_id, res_name=res_name, element="C", hetero=False)
        ])
        
        # Perform superimposition
        _, transformation = struc.superimpose(mobile_backbone_from_template, target_backbone_constructed)
        
        # Apply transformation to the entire reference residue template
        transformed_res = ref_res_template
        transformed_res.coord = transformation.apply(transformed_res.coord)
        
        # Set residue ID and name for the transformed residue
        transformed_res.res_id[:] = res_id
        transformed_res.res_name[:] = res_name
        transformed_res.chain_id[:] = "A" # Ensure chain ID is set
        
        # Append the transformed residue to the peptide
        peptide += transformed_res
    
    # After all residues are added, ensure global chain_id is 'A' (redundant if already set above, but good safeguard)
    peptide.chain_id = np.array(["A"] * peptide.array_length(), dtype="U1")
    
    # Assign sequential atom_id to all atoms in the peptide AtomArray
    peptide.atom_id = np.arange(1, peptide.array_length() + 1)

    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(peptide)
    
    string_io = io.StringIO()
    pdb_file.write(string_io)
    
    # Construct the final PDB content including header and footer
    header_content = "\n".join(pdb_header_lines) # The pdb_lines already contain the header
    
    # Biotite's PDBFile.write() will write ATOM records, which can be 78 or 80 chars.
    # It also handles TER records between chains, but not necessarily at the end of a single chain.
    atomic_and_ter_content = string_io.getvalue()

    # Manually add TER record if biotite doesn't add one at the end of the last chain.
    # Check if the last record written by biotite is an ATOM/HETATM, if so, add TER manually.
    last_line = atomic_and_ter_content.strip().splitlines()[-1]
    if last_line.startswith("ATOM") or last_line.startswith("HETATM"):
        # Get last atom details from the peptide AtomArray
        last_atom = peptide[-1]
        ter_atom_num = peptide.array_length() + 1 # TER serial number is last ATOM serial + 1
        ter_res_name = last_atom.res_name
        ter_chain_id = last_atom.chain_id
        ter_res_num = last_atom.res_id
        ter_record = f"TER   {ter_atom_num: >5}      {ter_res_name: >3} {ter_chain_id: <1}{ter_res_num: >4}".ljust(80)
        atomic_and_ter_content = atomic_and_ter_content.strip() + "\n" + ter_record + "\n"


    # Ensure each line is 80 characters by padding with spaces if necessary
    padded_atomic_and_ter_content_lines = []
    for line in atomic_and_ter_content.splitlines():

        if len(line) < 80:
            padded_atomic_and_ter_content_lines.append(line.ljust(80))
        else:
            padded_atomic_and_ter_content_lines.append(line)
    
    # Join with newline and then strip any trailing whitespace from the overall block
    final_atomic_content_block = "\n".join(padded_atomic_and_ter_content_lines).strip()
    
    # Final assembly of content
    final_pdb_content_lines = pdb_header_lines.copy()
    final_pdb_content_lines.append(final_atomic_content_block)
    
    # ENDMDL and END are part of the full PDB format, usually coming after the atom/ter records
    final_pdb_content_lines.append("ENDMDL")
    final_pdb_content_lines.append("END         ") # 10 spaces for standard PDB END

    return "\n".join(final_pdb_content_lines)