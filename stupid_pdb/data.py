from typing import List, Dict, Set, Any

"""
This module contains data definitions for the stupid_pdb package, starting
with the 20 standard amino acids and their atomic configurations.
"""

# The 20 standard amino acids represented by their 3-letter codes.
# This list is used to randomly select amino acids for the sequence.
STANDARD_AMINO_ACIDS: List[str] = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "VAL",
    "SER",
    "THR",
    "TRP",
    "TYR",
]

# --- Amino Acid Frequencies (Approximate percentages in proteins) ---
# Source: Based on general protein composition data (e.g., from D. M. Smith, The Encyclopedia of Life Sciences, 2001)
# These are normalized to sum to 1.0
AMINO_ACID_FREQUENCIES: Dict[str, float] = {
    "ALA": 0.081,  # Alanine
    "ARG": 0.051,  # Arginine
    "ASN": 0.038,  # Asparagine
    "ASP": 0.054,  # Aspartic Acid
    "CYS": 0.019,  # Cysteine
    "GLU": 0.063,  # Glutamic Acid
    "GLN": 0.038,  # Glutamine
    "GLY": 0.073,  # Glycine
    "HIS": 0.023,  # Histidine
    "ILE": 0.055,  # Isoleucine
    "LEU": 0.091,  # Leucine
    "LYS": 0.059,  # Lysine
    "MET": 0.018,  # Methionine
    "PHE": 0.039,  # Phenylalanine
    "PRO": 0.052,  # Proline
    "SER": 0.062,  # Serine
    "THR": 0.054,  # Threonine
    "TRP": 0.014,  # Tryptophan
    "TYR": 0.032,  # Tyrosine
    "VAL": 0.069,  # Valine
}

# Mapping for 1-letter to 3-letter amino acid codes
# Standard IUPAC codes
ONE_TO_THREE_LETTER_CODE: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# --- Standard Bond Lengths and Angles (Approximations) ---

# Values are in Angstroms for bond lengths and degrees for angles

# These are simplified averages and will not create perfectly accurate structures.


# Peptide bond geometry

BOND_LENGTH_N_CA: float = 1.458  # N-Calpha

BOND_LENGTH_CA_C: float = 1.525  # Calpha-C

BOND_LENGTH_C_N: float = 1.329  # C-N (peptide bond)

BOND_LENGTH_C_O: float = 1.231  # C=O (carbonyl)


ANGLE_CA_C_N: float = 116.2  # Calpha-C-N

ANGLE_C_N_CA: float = 121.7  # C-N-Calpha

ANGLE_N_CA_C: float = 110.0  # N-Calpha-C (tetrahedral approx)

ANGLE_CA_C_O: float = 120.8  # Calpha-C=O


# Side chain geometry (approximate)

BOND_LENGTH_CA_CB: float = 1.53  # Calpha-Cbeta (typical)

BOND_LENGTH_C_H: float = 1.08  # C-H (typical)

BOND_LENGTH_N_H: float = 1.01  # N-H (typical)


# Van der Waals radii in Angstroms (approximate values)

# Source: Wikipedia, various chemistry texts.

# These are simplified values for common protein atoms.

VAN_DER_WAALS_RADII: Dict[str, float] = {
    "H": 1.20,  # Hydrogen
    "C": 1.70,  # Carbon
    "N": 1.55,  # Nitrogen
    "O": 1.52,  # Oxygen
    "S": 1.80,  # Sulfur
}

# Amino acid properties for sequence improbability checks
CHARGED_AMINO_ACIDS: Set[str] = {
    "ARG",
    "HIS",
    "LYS",
    "ASP",
    "GLU",
}  # K, R, H (positive); D, E (negative)
POSITIVE_AMINO_ACIDS: Set[str] = {"ARG", "HIS", "LYS"}
NEGATIVE_AMINO_ACIDS: Set[str] = {"ASP", "GLU"}

HYDROPHOBIC_AMINO_ACIDS: Set[str] = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
HYDROPHILIC_AMINO_ACIDS: Set[str] = {
    "ARG",
    "ASN",
    "ASP",
    "GLN",
    "GLU",
    "HIS",
    "LYS",
    "SER",
    "THR",
}  # Contains charged ones too
POLAR_UNCHARGED_AMINO_ACIDS: Set[str] = {
    "ASN",
    "GLN",
    "SER",
    "THR",
    "CYS",
    "TYR",
}  # CYS, TYR can be ambivalent

# --- Atomic Definitions for Each Amino Acid ---
# Each amino acid defines its atoms relative to the C-alpha (CA) position (0,0,0)
# for side chain atoms. Backbone atoms (N, C, O) will be placed based on previous
# residue geometry in the generator.
# Format: {'name': 'ATOM_NAME', 'element': 'ELEMENT_SYMBOL', 'coords': [x, y, z]}
# For simplicity, coords are relative to CA, assuming CA is at (0,0,0) for side chain placement.
# Backbone atoms will have special handling.

AMINO_ACID_ATOMS: Dict[str, List[Dict[str, Any]]] = {
    "ALA": [
        # Backbone (N, CA, C, O handled by generator)
        {"name": "CB", "element": "C", "coords": [1.4, 0.0, 0.0]},  # Placeholder for CB
        {"name": "HB1", "element": "H", "coords": [2.0, 0.8, 0.0]},
        {"name": "HB2", "element": "H", "coords": [2.0, -0.8, 0.0]},
        {"name": "HB3", "element": "H", "coords": [1.0, 0.0, 0.8]},
    ],
    "ARG": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "NE", "element": "N", "coords": [5.5, 0.0, 0.0]},
        {"name": "CZ", "element": "C", "coords": [6.6, 0.0, 0.0]},
        {"name": "NH1", "element": "N", "coords": [7.7, 0.8, 0.0]},
        {"name": "NH2", "element": "N", "coords": [7.7, -0.8, 0.0]},
    ],
    "ASN": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "OD1", "element": "O", "coords": [3.5, 0.8, 0.0]},
        {"name": "ND2", "element": "N", "coords": [3.5, -0.8, 0.0]},
    ],
    "ASP": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "OD1", "element": "O", "coords": [3.5, 0.8, 0.0]},
        {"name": "OD2", "element": "O", "coords": [3.5, -0.8, 0.0]},
    ],
    "CYS": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "SG", "element": "S", "coords": [2.9, 0.0, 0.0]},
    ],
    "GLU": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "OE1", "element": "O", "coords": [4.9, 0.8, 0.0]},
        {"name": "OE2", "element": "O", "coords": [4.9, -0.8, 0.0]},
    ],
    "GLN": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "OE1", "element": "O", "coords": [4.9, 0.8, 0.0]},
        {"name": "NE2", "element": "N", "coords": [4.9, -0.8, 0.0]},
    ],
    "GLY": [
        # Glycine has no beta carbon, only alpha-hydrogens
        {"name": "HA1", "element": "H", "coords": [0.7, 0.7, 0.0]},
        {"name": "HA2", "element": "H", "coords": [0.7, -0.7, 0.0]},
    ],
    "HIS": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "ND1", "element": "N", "coords": [3.5, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.5, -1.0, 0.0]},
        {"name": "CE1", "element": "C", "coords": [4.5, 1.0, 0.0]},
        {"name": "NE2", "element": "N", "coords": [4.5, -1.0, 0.0]},
    ],
    "ILE": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG1", "element": "C", "coords": [2.5, 1.0, 0.0]},
        {"name": "CG2", "element": "C", "coords": [2.5, -1.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.5, 1.0, 0.0]},
    ],
    "LEU": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
    ],
    "LYS": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "CE", "element": "C", "coords": [5.7, 0.0, 0.0]},
        {"name": "NZ", "element": "N", "coords": [7.0, 0.0, 0.0]},
    ],
    "MET": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "SD", "element": "S", "coords": [4.3, 0.0, 0.0]},
        {"name": "CE", "element": "C", "coords": [5.7, 0.0, 0.0]},
    ],
    "PHE": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
        {"name": "CE1", "element": "C", "coords": [4.9, 1.0, 0.0]},
        {"name": "CE2", "element": "C", "coords": [4.9, -1.0, 0.0]},
        {"name": "CZ", "element": "C", "coords": [5.9, 0.0, 0.0]},
    ],
    "PRO": [
        # Proline is special, its N is part of a ring. Simplified here.
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.0, 0.0, 0.0]},
    ],
    "SER": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "OG", "element": "O", "coords": [2.9, 0.0, 0.0]},
    ],
    "THR": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "OG1", "element": "O", "coords": [2.5, 1.0, 0.0]},
        {"name": "CG2", "element": "C", "coords": [2.5, -1.0, 0.0]},
    ],
    "TRP": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
        {"name": "NE1", "element": "N", "coords": [4.9, 1.0, 0.0]},
        {"name": "CE2", "element": "C", "coords": [4.9, -1.0, 0.0]},
        {"name": "CE3", "element": "C", "coords": [5.9, -1.0, 0.0]},
        {"name": "CZ2", "element": "C", "coords": [5.9, 1.0, 0.0]},
        {"name": "CZ3", "element": "C", "coords": [6.9, -1.0, 0.0]},
        {"name": "CH2", "element": "C", "coords": [6.9, 1.0, 0.0]},
    ],
    "TYR": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
        {"name": "CE1", "element": "C", "coords": [4.9, 1.0, 0.0]},
        {"name": "CE2", "element": "C", "coords": [4.9, -1.0, 0.0]},
        {"name": "CZ", "element": "C", "coords": [5.9, 0.0, 0.0]},
        {"name": "OH", "element": "O", "coords": [6.9, 0.0, 0.0]},
    ],
    "VAL": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG1", "element": "C", "coords": [2.5, 1.0, 0.0]},
        {"name": "CG2", "element": "C", "coords": [2.5, -1.0, 0.0]},
    ],
}

ROTAMER_LIBRARY: Dict[str, Dict[str, List[float]]] = {
    "LEU": {
        "chi1": [-60.0, 180.0],
    }
}

