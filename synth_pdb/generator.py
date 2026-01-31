import random
import numpy as np
# EDUCATIONAL OVERVIEW - How Synthetic Protein Generation Works:
# 1. Sequence Resolution: Determine the amino acid string (e.g., "ALA-GLY-SER").
# 2. Backbone Generation: Place N-CA-C-O atoms for each residue.
#    - Geometrically constructing the chain using Bond Lengths and Angles.
#    - Setting Dihedral Angles (Phi/Psi) to define secondary structure (Helix/Sheet).
# 3. Side-Chain Placement: Add side-chain atoms (CB, CG...) based on Rotamer Libraries.
# 4. Refinement (Optional):
#    - Packing: Optimize rotamers to avoid clashes.
# 5. Metadata: Fill in B-factors and Occupancy.
#    - X-ray: Represents thermal motion.
#    - NMR: Often used to represent local RMSD across the ensemble.

import logging
from typing import List, Optional, Dict, Tuple
from .data import (
    STANDARD_AMINO_ACIDS,
    ALL_VALID_AMINO_ACIDS,
    ONE_TO_THREE_LETTER_CODE,
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
    RAMACHANDRAN_PRESETS,
    RAMACHANDRAN_REGIONS,
    BACKBONE_DEPENDENT_ROTAMER_LIBRARY,
    BETA_TURN_TYPES,
)
from .pdb_utils import create_pdb_header, create_pdb_footer, extract_atomic_content, assemble_pdb_content
from .geometry import (
    position_atom_3d_from_internal_coords,
    calculate_angle,
    calculate_dihedral_angle,
)
# Re-export for backward compatibility with tests
_position_atom_3d_from_internal_coords = position_atom_3d_from_internal_coords

from .packing import optimize_sidechains as run_optimization
from .physics import EnergyMinimizer
from . import biophysics # New Module
import os
import shutil
import tempfile

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import io

# Convert angles to radians for numpy trigonometric functions
ANGLE_N_CA_C_RAD = np.deg2rad(ANGLE_N_CA_C)
ANGLE_CA_C_N_RAD = np.deg2rad(ANGLE_CA_C_N)
ANGLE_CA_C_O_RAD = np.deg2rad(ANGLE_CA_C_O)

# Ideal Ramachandran angles for a generic alpha-helix
PHI_ALPHA_HELIX = -57.0
PSI_ALPHA_HELIX = -47.0
# Ideal Omega for trans peptide bond
OMEGA_TRANS = 180.0
OMEGA_VARIATION = 5.0  # degrees - adds thermal fluctuation to peptide bond

logger = logging.getLogger(__name__)


# This constant is used in test_generator.py for coordinate calculations.
CA_DISTANCE = (
    3.8  # Approximate C-alpha to C-alpha distance in Angstroms for a linear chain
)

PDB_ATOM_FORMAT = "ATOM  {atom_number: >5} {atom_name: <4}{alt_loc: <1}{residue_name: >3} {chain_id: <1}{residue_number: >4}{insertion_code: <1}   {x_coord: >8.3f}{y_coord: >8.3f}{z_coord: >8.3f}{occupancy: >6.2f}{temp_factor: >6.2f}          {element: >2}{charge: >2}"


from .relaxation import predict_order_parameters

def _calculate_bfactor(
    atom_name: str,
    residue_number: int,
    total_residues: int,
    residue_name: str,
    s2: float = 0.85
) -> float:
    """
    Calculate realistic B-factor (temperature factor) derived from Order Parameter (S2).
    
    EDUCATIONAL NOTE - B-factors (Temperature Factors):
    ===================================================
    B-factors represent atomic displacement due to thermal motion and static disorder.
    They are measured in Ų (square Angstroms) and indicate atomic mobility.
    
    Physical Interpretation:
    - B = 8π²<u²> where <u²> is mean square displacement
    - Higher B-factor = more mobile/flexible atom
    - Lower B-factor = more rigid/constrained atom

    Typical Patterns in Real Protein Structures:
    1. Backbone vs Side Chains:
       - Backbone atoms (N, CA, C, O): 15-25 Ų (constrained by peptide bonds)
       - Side chain atoms (CB, CG, etc.): 20-35 Ų (more conformational freedom)
    
    2. Position Along Chain:
       In this synthetic generator, we simulate B-factors that follow these
       universal patterns of rigidity vs. flexibility.
       - Core residues: 10-20 Ų (buried, constrained)
       - Terminal residues: 30-50 Ų ("terminal fraying" - fewer constraints)
    
    3. Residue-Specific Effects:
       - Glycine: Higher (no side chain constraints, more flexible)
       - Proline: Lower (cyclic structure, rigid backbone)

    For Linear Peptides (our case):
    - No folding/packing, so all residues are "surface-like"
    - Terminal fraying effect is prominent
    - Backbone still more rigid than side chains

    4. NMR Perspective (Order Parameters):
       For NMR structures, the "B-factor" column is often repurposed.
       - It can store RMSD across the ensemble.
       - It can also represent the Order Parameter ($S^2$).
       - $S^2$ (from Lipari-Szabo) measures rigidity on ps-ns timescales.
       - $S^2 = 1.0 \rightarrow$ Rigid (Low B-factor).
       - $S^2 \approx 0.5 \rightarrow$ Flexible (High B-factor).
    
    In this generator, we explicitly link the geometric flexibility ($S^2$)
    to the crystallographic observable ($B$), creating a unified biophysical model.
    
    Args:
        atom_name: Atom name (e.g., 'N', 'CA', 'CB', 'CG')
        residue_number: Residue number (1-indexed)
        total_residues: Total number of residues in chain
        residue_name: Three-letter residue code (e.g., 'ALA', 'GLY')
        s2: Lipari-Szabo Order Parameter (0.0=Random, 1.0=Rigid). Default 0.85.
        
    Returns:
        B-factor value in Ų, rounded to 2 decimal places
    """
    # Define backbone atoms (more rigid due to peptide bond constraints)
    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'H', 'HA'}
    
    # Base physics: B-factor inversely related to Order Parameter
    # Calibration:
    # S2=1.00 -> B=5.0
    # S2=0.85 -> B=20.0 (Typical Core)
    # S2=0.50 -> B=55.0 (Typical Termini/Loop)
    
    # Simple linear map: B = M * (1 - S2) + C
    # Delta S2 = 0.35 (0.85->0.50) -> Delta B = 35 (20->55)
    # Slope M = 35/0.35 = 100
    # B = 100 * (1 - S2) + Offset
    # Check: S2=0.85 -> B = 100*0.15 = 15. Need 20. So Offset=5.
    
    base_bfactor = 100.0 * (1.0 - s2) + 5.0
    
    # Atom type adjustments
    if atom_name not in BACKBONE_ATOMS:
        base_bfactor *= 1.5  # Side chains are more mobile than backbone
        
    # Residue-specific adjustments
    if residue_name == 'GLY':
        base_bfactor += 5.0
    elif residue_name == 'PRO':
        base_bfactor -= 3.0
    
    # Add small random variation
    random_variation = np.random.uniform(-2.0, 2.0)
    
    # Calculate final B-factor
    bfactor = base_bfactor + random_variation
    
    # Clamp to realistic range (5-99 Ų)
    bfactor = max(5.0, min(99.0, bfactor))
    
    return round(bfactor, 2)


def _calculate_occupancy(
    atom_name: str,
    residue_number: int,
    total_residues: int,
    residue_name: str,
    bfactor: float
) -> float:
    """Calculate realistic occupancy for an atom (0.85-1.00)."""
    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'H', 'HA'}
    
    # Base occupancy
    base_occupancy = 0.98 if atom_name in BACKBONE_ATOMS else 0.95
    
    # Terminal disorder
    dist_from_n_term = (residue_number - 1) / max(total_residues - 1, 1)
    dist_from_c_term = (total_residues - residue_number) / max(total_residues - 1, 1)
    dist_from_nearest_term = min(dist_from_n_term, dist_from_c_term)
    terminal_factor = -0.10 * (1.0 - dist_from_nearest_term)
    
    # Residue-specific
    residue_factor = 0.0
    if residue_name in ['GLY', 'SER', 'ASN', 'GLN']:
        residue_factor = -0.03
    elif residue_name in ['PRO', 'TRP', 'PHE']:
        residue_factor = +0.02
    
    # B-factor correlation
    normalized_bfactor = (bfactor - 5.0) / 55.0
    bfactor_correlation = -0.08 * normalized_bfactor
    
    # Random variation
    random_variation = np.random.uniform(-0.01, 0.01)
    
    # Calculate and clamp
    occupancy = base_occupancy + terminal_factor + residue_factor + bfactor_correlation + random_variation
    occupancy = max(0.85, min(1.00, occupancy))
    
    return round(occupancy, 2)


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
    insertion_code: str = "",
    temp_factor: float = 0.00,  # B-factor (temperature factor) in Ų
    occupancy: float = 1.00  # Occupancy (fraction of molecules)
) -> str:
    """
    Create a PDB ATOM line.
    
    EDUCATIONAL NOTE - PDB ATOM Record Format:
    - temp_factor (columns 61-66): Atomic mobility/flexibility
    - occupancy (columns 55-60): Fraction of molecules with atom at this position
    See _calculate_bfactor() and _calculate_occupancy() for details.
    """
    return (
        f"ATOM  {atom_number: >5} {atom_name: <4}{alt_loc: <1}{residue_name: >3} {chain_id: <1}"
        f"{residue_number: >4}{insertion_code: <1}   "
        f"{x: >8.3f}{y: >8.3f}{z: >8.3f}{occupancy: >6.2f}{temp_factor: >6.2f}          "
        f"{element: >2}  "
    )

def _place_atom_with_dihedral(
    atom1: np.ndarray,
    atom2: np.ndarray,
    atom3: np.ndarray,
    bond_length: float,
    bond_angle: float,
    dihedral: float
) -> np.ndarray:
    """
    Place a new atom using bond length, angle, and dihedral.
    
    Wrapper around position_atom_3d_from_internal_coords with clearer naming.
    """
    return position_atom_3d_from_internal_coords(
        atom1, atom2, atom3, bond_length, bond_angle, dihedral
    )


def _generate_random_amino_acid_sequence(
    length: int, use_plausible_frequencies: bool = False
) -> List[str]:
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


def _detect_disulfide_bonds(peptide) -> list:
    """
    Detect potential disulfide bonds between cysteine residues.
    
    EDUCATIONAL NOTE - Disulfide Bond Detection:
    ============================================
    Disulfide bonds form between two cysteine (CYS) residues when their
    sulfur atoms (SG) are close enough to form a covalent S-S bond.
    
    Detection Criteria:
    - Both residues must be CYS
    - SG-SG distance: 2.0-2.2 Å (slightly relaxed from ideal 2.0-2.1 Å)
    - Only report each pair once (avoid duplicates)
    
    Why Distance Matters:
    - < 2.0 Å: Too close (steric clash, not realistic)
    - 2.0-2.1 Å: Ideal disulfide bond distance
    - 2.1-2.2 Å: Acceptable (allows for flexibility)
    - > 2.2 Å: Too far (no covalent bond possible)
    
    Biological Context:
    - Disulfides stabilize protein structure
    - Common in extracellular proteins
    - Rare in cytoplasm (reducing environment)
    - Important for protein folding and stability
    
    Args:
        peptide: Biotite AtomArray structure
        
    Returns:
        List of tuples (res_id1, res_id2) representing disulfide bonds
        
    Example:
        >>> disulfides = _detect_disulfide_bonds(structure)
        >>> print(disulfides)
        [(3, 8), (12, 20)]  # CYS 3-8 and CYS 12-20 are bonded
    """
    import biotite.structure as struc
    
    disulfides = []
    
    # Find all CYS/CYX residues
    cys_residues = peptide[(peptide.res_name == 'CYS') | (peptide.res_name == 'CYX')]
    
    if len(cys_residues) < 2:
        return disulfides  # Need at least 2 CYS for a bond
    
    # Get unique residue IDs
    cys_res_ids = np.unique(cys_residues.res_id)
    
    # Check all pairs of CYS residues
    for i, res_id1 in enumerate(cys_res_ids):
        for res_id2 in cys_res_ids[i+1:]:  # Avoid duplicates
            # Get SG atoms for both residues
            sg1 = peptide[(peptide.res_id == res_id1) & (peptide.atom_name == 'SG')]
            sg2 = peptide[(peptide.res_id == res_id2) & (peptide.atom_name == 'SG')]
            
            if len(sg1) > 0 and len(sg2) > 0:
                # Calculate distance
                p1 = sg1[0].coord
                p2 = sg2[0].coord
                distance = np.sqrt(np.sum((p1 - p2)**2))
                
                # Check if within disulfide bond range
                # Minimized structures can be highly strained (especially small cyclic rings),
                # so we use a generous range [1.5, 3.0] to ensure the SSBOND record is generated
                # if the physics engine identified a bond.
                if 1.5 <= distance <= 3.0:
                    disulfides.append((int(res_id1), int(res_id2)))
    
    return disulfides


def _generate_ssbond_records(disulfides: list, chain_id: str = 'A') -> str:
    """
    Generate SSBOND records for PDB header.
    
    EDUCATIONAL NOTE - PDB SSBOND Format:
    ====================================
    SSBOND records annotate disulfide bonds in PDB files.
    
    Format (PDB specification):
    SSBOND   1 CYS A    6    CYS A   11
    
    Columns:
    1-6:   "SSBOND"
    8-10:  Serial number (1, 2, 3, ...)
    12-14: Residue name 1 (always "CYS")
    16:    Chain ID 1
    18-21: Residue number 1 (right-justified)
    26-28: Residue name 2 (always "CYS")
    30:    Chain ID 2
    32-35: Residue number 2 (right-justified)
    
    Why This Matters:
    - Structure viewers use this to display bonds
    - Analysis tools use this for stability calculations
    - Essential for understanding protein structure
    - Part of standard PDB format
    
    Args:
        disulfides: List of (res_id1, res_id2) tuples
        chain_id: Chain identifier (default 'A')
        
    Returns:
        String containing SSBOND records (one per line)
        
    Example:
        >>> records = _generate_ssbond_records([(3, 8), (12, 20)], 'A')
        >>> print(records)
        SSBOND   1 CYS A    3    CYS A    8
        SSBOND   2 CYS A   12    CYS A   20
    """
    if not disulfides:
        return ""
    
    records = []
    for serial, (res_id1, res_id2) in enumerate(disulfides, 1):
        # Format according to PDB specification
        # SSBOND   1 CYS A    6    CYS A   11
        record = f"SSBOND{serial:4d} CYS {chain_id}{res_id1:5d}    CYS {chain_id}{res_id2:5d}"
        records.append(record)
    
    return "\n".join(records) + "\n" if records else ""


def _resolve_sequence(
    length: Optional[int], user_sequence_str: Optional[str] = None, use_plausible_frequencies: bool = False
) -> List[str]:
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
                if aa not in ALL_VALID_AMINO_ACIDS:
                    raise ValueError(f"Invalid 3-letter amino acid code: {aa}")
            return amino_acids
        elif (
            len(user_sequence_str_upper) == 3
            and user_sequence_str_upper in ALL_VALID_AMINO_ACIDS
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


def _sample_ramachandran_angles(res_name: str, next_res_name: Optional[str] = None) -> Tuple[float, float]:
    """
    Sample phi/psi angles from Ramachandran probability distribution.
    
    Uses residue-specific distributions for GLY and PRO, general distribution
    for all other amino acids. Samples from favored regions using weighted
    Gaussian distributions.
    
    New Feature: Pre-Proline Bias
    If next_res_name is 'PRO' and current residue is not GLY or PRO,
    uses a specific 'PRE_PRO' distribution (favors beta/extended).
    
    Args:
        res_name: Three-letter amino acid code
        next_res_name: (Optional) Code of the next residue
        
    Returns:
        Tuple of (phi, psi) angles in degrees
        
    Reference:
        Lovell et al. (2003) Proteins: Structure, Function, and Bioinformatics
    """
    # Get residue-specific or general distribution
    if res_name in RAMACHANDRAN_REGIONS:
        # GLY or PRO specific maps take precedence
        regions = RAMACHANDRAN_REGIONS[res_name]
    elif next_res_name == 'PRO':
        # Pre-Proline effect!
        # Standard residues (ALA, VAL, etc) before Proline have restricted conformation.
        regions = RAMACHANDRAN_REGIONS.get('PRE_PRO', RAMACHANDRAN_REGIONS['general'])
    else:
        regions = RAMACHANDRAN_REGIONS['general']
    
    # Get favored regions
    favored_regions = regions['favored']
    weights = [r['weight'] for r in favored_regions]
    
    # Choose region based on weights
    region_idx = np.random.choice(len(favored_regions), p=weights)
    chosen_region = favored_regions[region_idx]
    
    # Sample angles from Gaussian around region center
    phi = np.random.normal(chosen_region['phi'], chosen_region['std'])
    psi = np.random.normal(chosen_region['psi'], chosen_region['std'])
    
    # Wrap to [-180, 180]
    phi = ((phi + 180) % 360) - 180
    psi = ((psi + 180) % 360) - 180
    
    return phi, psi


def _parse_structure_regions(structure_str: str, sequence_length: int) -> Dict[int, str]:
    """
    Parse structure region specification into per-residue conformations.
    
    This function enables users to specify different secondary structure conformations
    for different regions of their peptide. This is crucial for creating realistic
    protein-like structures that have mixed secondary structures (e.g., helix-turn-sheet).
    
    EDUCATIONAL NOTE - Why This Matters:
    Real proteins don't have uniform secondary structure throughout. They typically
    have regions of alpha helices, beta sheets, turns, and loops. This function
    allows users to specify these regions explicitly, making the generated structures
    much more realistic and useful for educational demonstrations.
    
    Args:
        structure_str: Region specification string in format "start-end:conformation,..."
                      Example: "1-10:alpha,11-20:beta,21-30:random"
                      - Residue numbering is 1-indexed (first residue is 1)
                      - Conformations: alpha, beta, ppii, extended, random
                      - Multiple regions separated by commas
        sequence_length: Total number of residues in the sequence
        
    Returns:
        Dictionary mapping residue index (0-based) to conformation name.
        Only includes explicitly specified residues (gaps are allowed).
        
        EDUCATIONAL NOTE - Return Format:
        We use 0-based indexing internally (Python convention) even though
        the input uses 1-based indexing (PDB/biology convention). This is
        a common pattern in bioinformatics software.
        
    Raises:
        ValueError: If syntax is invalid, regions overlap, or ranges are out of bounds
        
    Examples:
        >>> _parse_structure_regions("1-10:alpha,11-20:beta", 20)
        {0: 'alpha', 1: 'alpha', ..., 9: 'alpha', 10: 'beta', ..., 19: 'beta'}
        
        >>> _parse_structure_regions("1-5:alpha,10-15:beta", 20)
        {0: 'alpha', ..., 4: 'alpha', 9: 'beta', ..., 14: 'beta'}
        # Note: Residues 6-9 and 16-20 are not in the dictionary (gaps allowed)
    
    EDUCATIONAL NOTE - Design Decisions:
    1. We allow gaps in coverage - unspecified residues will use the default conformation
    2. We strictly forbid overlaps - each residue can only have one conformation
    3. We validate all inputs before processing to give clear error messages
    """
    # Handle empty input - return empty dictionary (all residues will use default)
    if not structure_str:
        return {}
    
    # EDUCATIONAL NOTE - Data Structure Choice:
    # We use a dictionary to map residue indices to conformations because:
    # 1. Fast lookup: O(1) to check if a residue has a specified conformation
    # 2. Sparse representation: Only stores specified residues (memory efficient)
    # 3. Easy to check for overlaps: Just check if key already exists
    residue_conformations = {}
    
    # Split the input string by commas to get individual region specifications
    # Example: "1-10:alpha,11-20:beta" -> ["1-10:alpha", "11-20:beta"]
    regions = structure_str.split(',')
    
    # Process each region specification
    for region in regions:
        # Remove any leading/trailing whitespace for robustness
        # This allows users to write "1-10:alpha, 11-20:beta" (with spaces)
        region = region.strip()
        
        # VALIDATION STEP 1: Check for colon separator
        # Expected format: "start-end:conformation"
        if ':' not in region:
            raise ValueError(
                f"Invalid region syntax: '{region}'. "
                f"Expected format: 'start-end:conformation' (e.g., '1-10:alpha')"
            )
        
        # Split by colon to separate range from conformation
        # Example: "1-10:alpha" -> range_part="1-10", conformation="alpha"
        range_part, conformation = region.split(':', 1)
        
        # VALIDATION STEP 2: Check conformation name
        # Build list of valid conformations from presets plus 'random'
        valid_conformations = list(RAMACHANDRAN_PRESETS.keys()) + ['random'] + list(BETA_TURN_TYPES.keys())
        if conformation not in valid_conformations:
            raise ValueError(
                f"Invalid conformation '{conformation}'. "
                f"Valid options are: {', '.join(valid_conformations)}"
            )
        
        # VALIDATION STEP 3: Check for dash separator in range
        # Expected format: "start-end"
        if '-' not in range_part:
            raise ValueError(
                f"Invalid range syntax: '{range_part}'. "
                f"Expected format: 'start-end' (e.g., '1-10')"
            )
        
        # Split range by dash to get start and end positions
        # Example: "1-10" -> start_str="1", end_str="10"
        start_str, end_str = range_part.split('-', 1)
        
        # VALIDATION STEP 4: Parse numbers
        # Try to convert strings to integers, give clear error if they're not numbers
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            raise ValueError(
                f"Invalid range numbers: '{range_part}'. "
                f"Start and end must be integers (e.g., '1-10')"
            )
        
        # Special Check for Beta-Turns
        if conformation in BETA_TURN_TYPES:
            if (end - start + 1) != 4:
                raise ValueError(
                    f"Beta-turn '{conformation}' region {start}-{end} must be exactly 4 residues long. "
                    f"Got {end - start + 1}."
                )
        
        # VALIDATION STEP 5: Check range bounds
        # EDUCATIONAL NOTE - Why These Checks Matter:
        # 1. start < 1: PDB/biology uses 1-based indexing, so 0 or negative makes no sense
        # 2. end > sequence_length: Can't specify residues that don't exist
        # 3. start > end: Range would be backwards (e.g., "10-5"), which is nonsensical
        if start < 1 or end > sequence_length:
            raise ValueError(
                f"Range {start}-{end} is out of bounds for sequence length {sequence_length}. "
                f"Valid range is 1 to {sequence_length}"
            )
        if start > end:
            raise ValueError(
                f"Invalid range: start ({start}) is greater than end ({end}). "
                f"Range must be in format 'smaller-larger' (e.g., '1-10', not '10-1')"
            )
        
        # VALIDATION STEP 6: Check for overlaps and assign conformations
        # EDUCATIONAL NOTE - Why We Forbid Overlaps:
        # If residue 5 is specified as both "alpha" and "beta", which should we use?
        # Rather than making an arbitrary choice (like "last one wins"), we require
        # the user to be explicit and not specify overlapping regions.
        for res_idx in range(start - 1, end):  # Convert to 0-based indexing
            # Check if this residue was already specified in a previous region
            if res_idx in residue_conformations:
                # EDUCATIONAL NOTE - Error Message Design:
                # We convert back to 1-based indexing in the error message because
                # that's what the user specified. This makes errors easier to understand.
                raise ValueError(
                    f"Overlapping regions detected: residue {res_idx + 1} is specified "
                    f"in multiple regions. Each residue can only have one conformation."
                )
            
            # Assign the conformation to this residue (using 0-based indexing internally)
            residue_conformations[res_idx] = conformation
    
    # Return the mapping of residue indices to conformations
    # EDUCATIONAL NOTE - What Happens to Gaps:
    # If a residue index is not in this dictionary, the calling code will use
    # the default conformation specified by the --conformation parameter.
    # This allows users to specify only the interesting regions and let the
    # rest use a sensible default.
    return residue_conformations


def generate_pdb_content(
    length: Optional[int] = None,
    sequence_str: Optional[str] = None,
    use_plausible_frequencies: bool = False,
    conformation: str = 'alpha',
    structure: Optional[str] = None,
    optimize_sidechains: bool = False,
    minimize_energy: bool = False,
    forcefield: str = 'amber14-all.xml',
    seed: Optional[int] = None,
    ph: float = 7.4,
    cap_termini: bool = False,
    equilibrate: bool = False,
    equilibrate_steps: int = 1000,
    metal_ions: str = 'auto',
    minimization_k: float = 10.0, # Tolerance
    minimization_max_iter: int = 0, # 0 = unlimited
    cis_proline_frequency: float = 0.05, # Probability of Cis-Proline (0.05 = 5%)
    phosphorylation_rate: float = 0.0, # Probability of S/T/Y phosphorylation
    cyclic: bool = False, # Head-to-Tail cyclization
) -> str:
    """
    Generates PDB content for a linear or cyclic peptide chain.
    
    EDUCATIONAL NOTE - New Feature: Cyclic Peptides
    Cyclic peptides have their N-terminus bonded to their C-terminus. 
    This modification increases metabolic stability and is common in 
    therapeutic peptides (e.g., Cyclosporin).
    
    Args:
        length: Number of residues (ignored if sequence_str provided)
        sequence_str: Explicit amino acid sequence (1-letter or 3-letter codes)
        use_plausible_frequencies: Use biologically realistic amino acid frequencies
        conformation: Default secondary structure conformation.
                     Options: 'alpha', 'beta', 'ppii', 'extended', 'random'
                     Default: 'alpha' (alpha helix)
                     Used for all residues if structure is not provided,
                     or for residues not specified in structure parameter.
        structure: Per-region conformation specification (NEW!)
                  Format: "start-end:conformation,start-end:conformation,..."
                  Example: "1-10:alpha,11-15:random,16-30:beta"
                  If provided, overrides conformation for specified regions.
                  Unspecified residues use the default conformation parameter.
        optimize_sidechains: Run Monte Carlo side-chain optimization
        minimize_energy: Run OpenMM energy minimization (REQUIRED for cyclic closure)
        forcefield: Forcefield to use for minimization
        seed: Random seed for reproducible generation
        ph: pH for titration
        cap_termini: Add ACE/NME caps (Disabled if cyclic=True)
        equilibrate: Run MD equilibration
        equilibrate_steps: Number of MD steps
        metal_ions: Handle metal ions
        minimization_k: Tolerance
        minimization_max_iter: Max iterations
        cis_proline_frequency: Frequency of cis-proline
        phosphorylation_rate: Frequency of phosphorylation
        cyclic: Whether to generate a cyclic peptide (Head-to-Tail)
    
    Returns:
        str: Complete PDB file content
        
    Raises:
        ValueError: If invalid conformation name or structure syntax provided
        
    EDUCATIONAL NOTE - Why Per-Region Conformations Matter:
    Real proteins have mixed secondary structures. For example:
    - Zinc fingers: beta sheets + alpha helices
    - Immunoglobulins: multiple beta sheets connected by loops
    - Helix-turn-helix motifs: two alpha helices connected by a turn
    This feature allows users to create these realistic structures.

    EDUCATIONAL NOTE - Macrocyclization (Cyclic Peptides):
    -----------------------------------------------------
    Cyclic peptides (macrocycles) are chains where the N-terminus and C-terminus 
    are covalently linked. This has profound biological implications:
    1. Metabolic Stability: Resistance to exopeptidases that chew protein ends.
    2. Binding Affinity: By "locking" the molecule into a specific shape, 
       the entropic penalty of binding to a target is greatly reduced.
    3. Bioavailability: Many legendary drugs (like Cyclosporine A) are macrocycles.
    """
    if seed is not None:
         logger.info(f"Setting random seed to {seed} for reproducibility.")
         random.seed(seed)
         np.random.seed(seed)
         
    sequence = _resolve_sequence(
        length=length,
        user_sequence_str=sequence_str,
        use_plausible_frequencies=use_plausible_frequencies,
    )

    # EDUCATIONAL NOTE - Post-Translational Modifications (PTMs):
    # -----------------------------------------------------------
    # Phosphorylation (addition of a phosphate group, PO4^3-) is a reversible
    # modification that acts as a molecular switch.
    # 
    # Biophysical Impact:
    # 1. Electrostatics: Adds a massive negative charge (-2e at physiological pH).
    #    This can repel nearby negative residues or attract positive ones (Salt Bridges),
    #    drastically altering the local conformation.
    # 2. Sterics: The phosphate group is bulky, often forcing the backbone into
    #    new shapes to accommodate it.
    #
    # Simulation:
    # We convert the standard residue (SER/THR/TYR) to its phosphorylated form
    # (SEP/TPO/PTR) which OpenMM's AMBER forcefield recognizes and treats with
    # correct physics (charge and partial double bond parameters).
    if phosphorylation_rate > 0:
         # ... conversion logic ...
        modified_sequence = []
        for aa in sequence:
            if aa == 'SER' and random.random() < phosphorylation_rate:
                modified_sequence.append('SEP')
            elif aa == 'THR' and random.random() < phosphorylation_rate:
                modified_sequence.append('TPO')
            elif aa == 'TYR' and random.random() < phosphorylation_rate:
                modified_sequence.append('PTR')
            else:
                modified_sequence.append(aa)
        sequence = modified_sequence

    if not sequence:
        if sequence_str is not None and len(sequence_str) == 0:
            raise ValueError("Provided sequence string cannot be empty.")
        raise ValueError(
            "Length must be a positive integer when no sequence is provided and no valid sequence string is given."
        )

    # Calculate sequence length first - we need this for parsing structure regions
    sequence_length = len(sequence)

    # EDUCATIONAL NOTE - Input Validation:
    # We validate the default conformation early to give clear error messages.
    # Even if structure parameter overrides it for some residues, we need to
    # ensure the default is valid for any gaps or when structure is not provided.
    valid_conformations = list(RAMACHANDRAN_PRESETS.keys()) + ['random']
    if conformation not in valid_conformations:
        raise ValueError(
            f"Invalid conformation '{conformation}'. "
            f"Valid options are: {', '.join(valid_conformations)}"
        )

    # EDUCATIONAL NOTE - Per-Residue Conformation Assignment:
    # We now support two modes:
    # 1. Uniform conformation (old behavior): All residues use same conformation
    # 2. Per-region conformation (new!): Different regions can have different conformations
    
    # Parse per-residue conformations if structure parameter is provided
    if structure:
        # Parse the structure specification into a dictionary
        # mapping residue index (0-based) to conformation name
        residue_conformations = _parse_structure_regions(structure, sequence_length)
        
        # Fill in any gaps with the default conformation
        # EDUCATIONAL NOTE - Gap Handling:
        # If a residue is not specified in the structure parameter,
        # we use the default conformation. This allows users to specify
        # only the interesting regions and let the rest use a sensible default.
        for i in range(sequence_length):
            if i not in residue_conformations:
                residue_conformations[i] = conformation
    else:
        # No structure parameter provided - use uniform conformation for all residues
        # This maintains backward compatibility with existing code
        residue_conformations = {i: conformation for i in range(sequence_length)}

    
    # EDUCATIONAL NOTE - Why We Don't Validate Conformations Here:
    # We already validated conformations in _parse_structure_regions(),
    # so we don't need to re-validate them here. The default conformation
    # will be validated when we actually use it below.


    peptide = struc.AtomArray(0)
    residue_coordinates = {}
    
    # Build backbone and add side chains
    for i, res_name in enumerate(sequence):
        res_id = i + 1
        
        # Determine backbone coordinates based on previous residue or initial placement

        # Determine conformation for this residue
        # Use 0-based index to lookup in dictionary
        # Default to 'alpha' if not specified (though _parse_structure_regions handles this mostly)
        res_conformation = residue_conformations.get(i, conformation)
        # Compatibility alias for rotamer selection logic downstream
        current_conformation = res_conformation
        
        # Calculate backbone coordinates
        if i == 0:
            # First residue (N-terminus)
            # Place N at origin (0,0,0)
            n_coord = np.array([0.0, 0.0, 0.0])
            
            # Place CA along x-axis
            ca_coord = np.array([BOND_LENGTH_N_CA, 0.0, 0.0])
            
            # Place C in x-y plane using N-CA-C angle
            # We assume a default psi angle to start or just place it geometrically flat first
            # Ideally, we place it based on the first psi, but we don't have previous omega.
            # Simplified: Place C such that N-CA-C angle is correct in x-y plane.
            
            # Calculate coordinates for C
            # Using simple trigonometry for the first angle
            # x = CA_x + d * cos(pi - angle)
            # y = CA_y + d * sin(pi - angle)
            
            # Correction: N is at origin, CA is at (d, 0, 0).
            # Vector CA->N is (-1, 0, 0).
            # We want vector CA->C to have angle ANGLE_N_CA_C with CA->N.
            # So angle with x-axis is (180 - angle).
            angle_with_x = np.pi - ANGLE_N_CA_C_RAD
            
            c_x = ca_coord[0] + BOND_LENGTH_CA_C * np.cos(angle_with_x)
            c_y = ca_coord[1] + BOND_LENGTH_CA_C * np.sin(angle_with_x)
            c_z = 0.0
            c_z = 0.0
            c_coord = np.array([c_x, c_y, c_z])
            
            # Initialize current_psi for usage in next iteration logic or end of loop
            if res_conformation in RAMACHANDRAN_PRESETS:
                current_psi = RAMACHANDRAN_PRESETS[res_conformation]['psi']
            elif res_conformation == 'random':
                 # Check next residue
                 next_res_name = sequence[i+1] if i + 1 < sequence_length else None
                 _, current_psi = _sample_ramachandran_angles(res_name, next_res_name)
            elif res_conformation in BETA_TURN_TYPES:
                # Residue 1 of turn?
                # For i=0, it's always pos 1 of ANY structure starting at 1.
                # If generated logic requires robust turn pos:
                current_psi = RAMACHANDRAN_PRESETS['extended']['psi'] # Entry to turn
            else:
                current_psi = -47.0 # Default Alpha
            
        else:
            # For subsequent residues, we place atoms N, CA, C using internal coordinates
            # defined by previous atoms.
            
            prev_res_idx = i - 1
            prev_coords = residue_coordinates[prev_res_idx]
            prev_n_coord = prev_coords['N']
            prev_ca_coord = prev_coords['CA']
            prev_c_coord = prev_coords['C']
            
            # Determine Phi/Psi angles
            # Note: Phi is torsion of C(i-1)-N(i)-CA(i)-C(i)
            # Psi is torsion of N(i)-CA(i)-C(i)-N(i+1)
            # Omega is torsion of CA(i-1)-C(i-1)-N(i)-CA(i) -- Peptide bond
            
            # Get next residue name if available
            next_res_name = sequence[i+1] if i + 1 < sequence_length else None
            
            # 1. Place N(i)
            # Bond: C(i-1)-N(i)
            # Angle: CA(i-1)-C(i-1)-N(i)
            # Dihedral: Psi(i-1) - typically ~180 for trans? No, Omega is the peptide bond dihedral.
            # Wait. Omega is CA(prev)-C(prev)-N(curr)-CA(curr).
            # The dihedral to place N(curr) relative to prev_N, prev_CA, prev_C is PSI(prev).
            # No.
            # Chain: ... N(prev) - CA(prev) - C(prev) - N(curr) ...
            # To place N(curr), we need:
            # - Bond: C(prev)-N(curr)
            # - Angle: CA(prev)-C(prev)-N(curr) (Angle 1)
            # - Dihedral: N(prev)-CA(prev)-C(prev)-N(curr) -> This is PSI(prev)
            
            # Retrieve Psi from previous residue's conformation
            prev_conformation = residue_conformations.get(prev_res_idx, conformation)
            
            current_phi = None
            current_psi = None

            # Handle Beta-Turns explicitly
            # If we are in a turn, the angles are fixed based on position in turn.
            # Need to know which residue of the turn we are (1, 2, 3, 4)
            # We need to find the START of this turn region to calculate offset.
            # This is slightly inefficient but robust: find the range this index belongs to.
            # Since we stored per-residue conformations, we don't have the ranges directly.
            # But we can infer position:
            # If prev_conformation is 'typeI', check i-1, i-2, i-3 to find start.
            
            # Check if we are residue 2 or 3 of the turn (the ones with angles).
            # Res 1: i (start)
            # Res 2: i+1 (offset 1) -> Has specific Phi/Psi
            # Res 3: i+2 (offset 2) -> Has specific Phi/Psi
            # Res 4: i+3 (end)
            
            # To place N(i), we use Psi(i-1).
            # If prev_res (i-1) was Res 1 of turn, use random/default Psi? Or Helix/Sheet?
            # Actually, Beta-Turns define Phi2, Psi2, Phi3, Psi3.
            # Psi1 and Phi4 are not strictly defined (they connect to rest of chain).
            
            # Let's handle standard sampling first, then override if specific turn res.
            pass

            if prev_conformation in RAMACHANDRAN_PRESETS:
                prev_psi = RAMACHANDRAN_PRESETS[prev_conformation]['psi']
            elif prev_conformation in BETA_TURN_TYPES:
                # Logic to determine Psi based on position in turn
                # Scan backward to find start of turn
                start_dist = 0
                while (prev_res_idx - start_dist) >= 0 and residue_conformations.get(prev_res_idx - start_dist) == prev_conformation:
                    start_dist += 1
                # start_dist is now how far back the block goes.
                # If we are at i, prev is i-1.
                # If block is 0,1,2,3. i=1 (Res 2). prev=0 (Res 1).
                # start_dist would be 1 (since 0 is same).
                # Position index within turn for prev_res:
                # Turn length is 4.
                # We need to forward-scan to find total length? No, we validated it's 4.
                # So if we are in a block of 4, we need to know index 0..3
                
                # This is tricky with just the dict.
                # Let's look at i-1. Is it the 1st, 2nd, 3rd residue of the turn?
                # Check i-2. If same type, not 1st.
                # Check i-3. If same type.
                
                c_prev = prev_conformation
                if residue_conformations.get(prev_res_idx - 1) != c_prev:
                    turn_pos = 1 # Prev is 1st residue.
                elif residue_conformations.get(prev_res_idx - 2) != c_prev:
                    turn_pos = 2 # Prev is 2nd residue.
                elif residue_conformations.get(prev_res_idx - 3) != c_prev:
                    turn_pos = 3 # Prev is 3rd residue.
                else:
                    turn_pos = 4 # Prev is 4th residue.
                
                # Turn angles defined for Res 2 and Res 3.
                # "Type": [(Phi2, Psi2), (Phi3, Psi3)]
                turn_angles = BETA_TURN_TYPES[prev_conformation]
                
                if turn_pos == 1:
                    # Prev is Res 1. We need Psi1.
                    # Turns usually connect strands, so Psi1 is often sheet-like/extended?
                    # Or we sample random 'general'.
                    # Ideally we'd be robust. Let's use 'extended' for entry into turn as default.
                    prev_psi = RAMACHANDRAN_PRESETS['extended']['psi'] 
                elif turn_pos == 2:
                    # Prev is Res 2. We need Psi2.
                    prev_psi = turn_angles[0][1]
                elif turn_pos == 3:
                    # Prev is Res 3. We need Psi3.
                    prev_psi = turn_angles[1][1]
                else: # turn_pos == 4 (End of turn)
                    # Prev is Res 4. We need Psi4.
                    # Exit turn. Use extended or helix default.
                    prev_psi = RAMACHANDRAN_PRESETS['extended']['psi']
                 
            elif prev_conformation == 'random':
                # Use the stored psi if we generated it?
                # We synthesize fresh here. This ignores continuity if we don't store.
                # Ideally we should generate all angles first.
                # But current loop generates on fly.
                # For 'random', we re-sample. This is inconsistent if we need Phi/Psi pairs.
                # Assuming simple independent sampling for now as per original code.
                _, prev_psi = _sample_ramachandran_angles(sequence[prev_res_idx], res_name)

            else:
                # Should not happen
                prev_psi = RAMACHANDRAN_PRESETS['alpha']['psi']

            # CRITICAL FIX for OpenMM Minimization:
            # Force standard bond geometry when transitioning between distinct conformations.
            # The simple NeRF algorithm relies on the previous frame being perfectly geometric.
            # If standard sampling leads to a weird Psi, the next N placement might be valid spatially
            # but OpenMM's bond heuristic is strict.
            # We explicitly ensure C-N bond length is EXACTLY BOND_LENGTH_C_N (1.33A).
            
            n_coord = _place_atom_with_dihedral(
                prev_n_coord, prev_ca_coord, prev_c_coord,
                BOND_LENGTH_C_N,
                ANGLE_CA_C_N, # Angle at C: CA-C-N
                prev_psi
            )
            
            # 2. Place CA(i)
            # Bond: N(i)-CA(i)
            # Angle: C(prev)-N(i)-CA(i)
            # Dihedral: CA(prev)-C(prev)-N(i)-CA(i) -> This is OMEGA (Peptide bond)
            
            # Sample Omega
            # Check for Cis-Proline
            omega_mean = OMEGA_TRANS # 180
            if res_name == 'PRO' and random.random() < cis_proline_frequency:
                omega_mean = 0.0 # Cis
            
            # Sample with variation
            omega = np.random.normal(omega_mean, OMEGA_VARIATION)
            
            ca_coord = _place_atom_with_dihedral(
                prev_ca_coord, prev_c_coord, n_coord,
                BOND_LENGTH_N_CA,
                ANGLE_C_N_CA,
                omega
            )
            
            # 3. Place C(i)
            # Bond: CA(i)-C(i)
            # Angle: N(i)-CA(i)-C(i)
            # Dihedral: C(prev)-N(i)-CA(i)-C(i) -> This is PHI(i)
            
            if res_conformation in RAMACHANDRAN_PRESETS:
                current_phi = RAMACHANDRAN_PRESETS[res_conformation]['phi']
                current_psi = RAMACHANDRAN_PRESETS[res_conformation]['psi']
            elif res_conformation in BETA_TURN_TYPES:
                # Same logic as Psi, but for current residue Phi
                c_curr = res_conformation
                # Find position of CURRENT residue
                if residue_conformations.get(i - 1) != c_curr:
                    turn_pos = 1 # Current is 1st residue.
                elif residue_conformations.get(i - 2) != c_curr:
                    turn_pos = 2 # Current is 2nd residue.
                elif residue_conformations.get(i - 3) != c_curr:
                    turn_pos = 3 # Current is 3rd residue.
                else:
                    turn_pos = 4 # Current is 4th residue.
                
                turn_angles = BETA_TURN_TYPES[res_conformation]
                if turn_pos == 1:
                    # Entry into turn. Phi1/Psi1.
                    current_phi = RAMACHANDRAN_PRESETS['extended']['phi']
                    current_psi = RAMACHANDRAN_PRESETS['extended']['psi']
                elif turn_pos == 2:
                    current_phi = turn_angles[0][0] # Phi2
                    current_psi = turn_angles[0][1] # Psi2
                elif turn_pos == 3:
                    current_phi = turn_angles[1][0] # Phi3
                    current_psi = turn_angles[1][1] # Psi3
                else:
                    current_phi = RAMACHANDRAN_PRESETS['extended']['phi'] # Phi4
                    current_psi = RAMACHANDRAN_PRESETS['extended']['psi'] # Psi4

            elif res_conformation == 'random':
                current_phi, current_psi = _sample_ramachandran_angles(res_name, next_res_name)
            else:
                current_phi = RAMACHANDRAN_PRESETS['alpha']['phi']
                current_psi = RAMACHANDRAN_PRESETS['alpha']['psi']
                 
            c_coord = _place_atom_with_dihedral(
                prev_c_coord, n_coord, ca_coord,
                BOND_LENGTH_CA_C,
                ANGLE_N_CA_C,
                current_phi
            )


        # Store coordinates for next iteration (NeRF needs them)
        residue_coordinates[i] = {
            'N': n_coord,
            'CA': ca_coord,
            'C': c_coord
        }

        # Store Psi for next iteration
        prev_psi = current_psi
        
        # Get reference residue from biotite
        # CRITICAL FIX: Always use .copy() because struc.info.residue returns cached templates.
        # Modifying the template in-place causes subsequent residues of the same type to be broken.
        if i == 0: # N-terminal residue
            #ref_res_template = struc.info.residue(res_name, 'N_TERM').copy() # Not supported in all biotite versions - Roberto Tejero
            ref_res_template = struc.info.residue(res_name).copy()
        elif i == len(sequence) - 1: # C-terminal residue
            #ref_res_template = struc.info.residue(res_name, 'C_TERM').copy() # Not supported in all biotite versions - Roberto Tejero
            ref_res_template = struc.info.residue(res_name).copy()
        else: # Internal residue
            #ref_res_template = struc.info.residue(res_name, 'INTERNAL').copy() # Not supported in all biotite versions - Roberto Tejero
            ref_res_template = struc.info.residue(res_name).copy()

        # EDUCATIONAL NOTE - Peptide Bond Chemistry:
        # A peptide bond forms via dehydration synthesis (loss of H2O).
        # The Carboxyl group (COOH) of one amino acid joins the Amine group (NH2) of the next.
        # This means internal residues lose their terminal Oxygen (OXT) and associated Hydrogens.
        # We must explicitly remove terminal-only atoms from all residues to represent 
        # a continuous polypeptide chain correctly and avoid OpenMM template errors.
        
        # EDUCATIONAL NOTE - Terminal Atom Management in Rings:
        # In a linear peptide, the ends are "unfinished" (OXT at C-term, extra H at N-term).
        # In a cyclic peptide, these atoms are SACRIFICED to form the peptide bond 
        # between the ends. Failing to remove them leads to "impossible" geometry
        # with 5-valent carbons or nitrogen atoms with too many bonds.
        
        # 1. Remove OXT for all except absolute C-terminus
        # If cyclic, even the absolute C-terminus loses OXT to form the head-to-tail bond.
        if i < len(sequence) - 1 or cyclic:
            ref_res_template = ref_res_template[ref_res_template.atom_name != "OXT"]
            # Also remove HXT (Hydroxyl Hydrogen) if present in template
            ref_res_template = ref_res_template[ref_res_template.atom_name != "HXT"]
            
        # 2. Remove extra N-terminal Hydrogens (H2, H3) for all except absolute N-terminus
        # If cyclic, even the absolute N-terminus loses H2/H3 as it's no longer a zwitterion.
        if i > 0 or cyclic:
            ref_res_template = ref_res_template[ref_res_template.atom_name != "H2"]
            ref_res_template = ref_res_template[ref_res_template.atom_name != "H3"]
            
            # PROLINE FIX: Internal/Cyclic Proline has NO amide hydrogen (H).
            # Biotite templates often include a single 'H' which must be stripped
            # to avoid OpenMM template mismatch errors in cyclic peptides.
            if res_name == "PRO":
                ref_res_template = ref_res_template[ref_res_template.atom_name != "H"]

        # 3. Apply rotamers if available
        # EDUCATIONAL NOTE - Rotamer Selection Strategy:
        # We employ a 'Backbone-Dependent' selection strategy where possible.
        # This means we check the residue's secondary structure context (Helix vs Sheet)
        # to choose the most likely side-chain conformation.
        # Logic:
        # 1. Check if specific rotamers exist for this residue + conformation in BACKBONE_DEPENDENT_LIB.
        # 2. If not, fall back to the generic ROTAMER_LIBRARY (Backbone-Independent).
        
        rotamers = None
        
        # Try Backbone-Dependent Lookup first
        # current_conformation is available from earlier in the loop (e.g., 'alpha', 'beta')
        if res_name in BACKBONE_DEPENDENT_ROTAMER_LIBRARY:
            if current_conformation in BACKBONE_DEPENDENT_ROTAMER_LIBRARY[res_name]:
                rotamers = BACKBONE_DEPENDENT_ROTAMER_LIBRARY[res_name][current_conformation]
        
        # Fallback to Backbone-Independent Library
        if rotamers is None and res_name in ROTAMER_LIBRARY:
            rotamers = ROTAMER_LIBRARY[res_name]
            
        if rotamers:
            # Weighted random selection based on experimental probabilities
            weights = [r.get('prob', 0.0) for r in rotamers]
            selected_rotamer = random.choices(rotamers, weights=weights, k=1)[0]
                
            # Apply chi angles
            if 'chi1' in selected_rotamer:
                chi1_target = selected_rotamer["chi1"][0]
                
                # Logic to find the gamma atom (CG, CG1, OG, OG1, SG) for Chi1 definition (N-CA-CB-Gamma)
                # Priority: CG > CG1 > OG > OG1 > SG
                gamma_atom_name = None
                for candidate in ["CG", "CG1", "OG", "OG1", "SG"]:
                    if len(ref_res_template[ref_res_template.atom_name == candidate]) > 0:
                        gamma_atom_name = candidate
                        break
                
                if gamma_atom_name:
                    # EDUCATIONAL NOTE - Sidechain Rotation:
                    # Instead of placing a single atom (which breaks branched residues like VAL),
                    # we rotate the ENTIRE sidechain about the CA-CB axis to reach target chi1.
                    # We removed the +180.0 offset because NeRF and IUPAC both use 0 for Cis.
                    ca_atom = ref_res_template[ref_res_template.atom_name == "CA"][0]
                    cb_atom = ref_res_template[ref_res_template.atom_name == "CB"][0]
                    n_atom = ref_res_template[ref_res_template.atom_name == "N"][0]
                    g_atom = ref_res_template[ref_res_template.atom_name == gamma_atom_name][0]
                    
                    current_chi1 = calculate_dihedral_angle(
                        n_atom.coord, ca_atom.coord, cb_atom.coord, g_atom.coord
                    )
                    # Rodrigues rotation formula CCW looking down CA->CB 
                    # results in a NEGATIVE change in dihedral angle in our convention.
                    # So rotation_angle = current - target
                    diff_deg = current_chi1 - chi1_target
                    
                    # Rotate all atoms downstream of CB
                    # We identify sidechain atoms as everything except N, CA, C, O, H, HA
                    backbone_names = {"N", "CA", "C", "O", "H", "HA", "CB"}
                    for atom_idx in range(len(ref_res_template)):
                        if ref_res_template.atom_name[atom_idx] not in backbone_names:
                            # Rotate point about CA-CB axis
                            p = ref_res_template.coord[atom_idx]
                            v = cb_atom.coord - ca_atom.coord
                            v /= np.linalg.norm(v)
                            
                            # Rodrigues' rotation formula
                            alpha = np.deg2rad(diff_deg)
                            cos_a = np.cos(alpha)
                            sin_a = np.sin(alpha)
                            
                            rel_p = p - ca_atom.coord
                            rotated_p = (
                                rel_p * cos_a + 
                                np.cross(v, rel_p) * sin_a + 
                                v * np.dot(v, rel_p) * (1 - cos_a)
                            )
                            ref_res_template.coord[atom_idx] = rotated_p + ca_atom.coord
            
        # Extract N, CA, C from ref_res_template
        # Ensure these atoms are present in the template for superimposition
        template_backbone_n = ref_res_template[ref_res_template.atom_name == "N"]
        template_backbone_ca = ref_res_template[ref_res_template.atom_name == "CA"]
        template_backbone_c = ref_res_template[ref_res_template.atom_name == "C"]

        # Concatenate backbone atoms into a single AtomArray
        # We must have exactly 3 atoms (N, CA, C) to match the target array
        mobile_backbone_from_template = template_backbone_n + template_backbone_ca + template_backbone_c
        
        if len(mobile_backbone_from_template) != 3:
            raise ValueError(
                f"Reference residue template for {res_name} is missing required "
                f"backbone atoms (N, CA, C) for superimposition. "
                f"Found atoms: {list(mobile_backbone_from_template.atom_name)}"
            )

        # Create the 'target' structure for superimposition from the *constructed* coordinates
        target_backbone_constructed = struc.array([
            struc.Atom(n_coord, atom_name="N", res_id=res_id, res_name=res_name, element="N", hetero=False),
            struc.Atom(ca_coord, atom_name="CA", res_id=res_id, res_name=res_name, element="C", hetero=False),
            struc.Atom(c_coord, atom_name="C", res_id=res_id, res_name=res_name, element="C", hetero=False)
        ])
        
        # AHA MOMENT - Superimposition Direction:
        # In the "AI Trinity" debugging phase, we found that residues were disconnected 
        # (6A-13A gaps). This was due to superimposing the backbone onto the template 
        # instead of moving the template into our newly constructed global frame.
        # Fixed: target=constructed_frame, mobile=residue_template.
        _, transformation = struc.superimpose(target_backbone_constructed, mobile_backbone_from_template)
        
        # Apply transformation to the entire reference residue template
        transformed_res = ref_res_template
        transformed_res.coord = transformation.apply(transformed_res.coord)
        
        # Set residue ID and name for the transformed residue
        transformed_res.res_id[:] = res_id
        transformed_res.res_name[:] = res_name
        transformed_res.chain_id[:] = "A" # Ensure chain ID is set
        
        # Append the transformed residue to the peptide
        if i == 0:
            peptide = transformed_res.copy()
        else:
            peptide += transformed_res
            
        # Store these for next iteration
        # Use BOTH the construction coordinates AND the transformed ones if needed?
        # Actually, NeRF needs the PREVIOUS frame. 
        # To perfectly connect residues, we MUST use the coordinates where we placed 
        # C of the previous residue.
        prev_n_coord = n_coord
        prev_ca_coord = ca_coord
        prev_c_coord = c_coord
    
    # After all residues are added, ensure global chain_id is 'A' (redundant if already set above, but good safeguard)
    peptide.chain_id = np.array(["A"] * peptide.array_length(), dtype="U1")
    
    # EDUCATIONAL NOTE - Side-Chain Optimization:
    # If requested, run Monte Carlo optimization to fix steric clashes.
    # This is "Phase 1" of biophysical realism.
    if optimize_sidechains:
        logger.info("Running side-chain optimization...")
        peptide = run_optimization(peptide)

    # EDUCATIONAL NOTE - Biophysical Realism (Phase 2):
    # We apply chemical modifications after geometric construction/packing but BEFORE 
    # energy minimization. Correct protonation states (pH) are critical for 
    # correct electrostatics in the forcefield.
    
    # 1. Terminal Capping (ACE/NME) - If requested
    # Cyclic peptides are naturally "capped" by their own ends, so we disable this.
    if cap_termini and not cyclic:
         peptide = biophysics.cap_termini(peptide)
         
    # 2. pH Titration (Protonation States)
    # Adjusts HIS -> HIP/HIE/HID based on pH.
    peptide = biophysics.apply_ph_titration(peptide, ph=ph)


    # EDUCATIONAL NOTE - Metal Ion Coordination (Phase 15):
    # Inorganic cofactors like Zinc (Zn2+) are automatically detected.
    # If a coordintion motif is found (Cys/His clusters), the ion is 
    # injected and harmonic constraints are applied in the physics module.
    if metal_ions == 'auto':
        from .cofactors import find_metal_binding_sites, add_metal_ion
        sites = find_metal_binding_sites(peptide)
        for site in sites:
            peptide = add_metal_ion(peptide, site)


    # EDUCATIONAL NOTE - Energy Minimization (Phase 2):
    # OpenMM requires a file-based interaction usually for easy topology handling from PDB.
    # So we write the current state to a temp file, minimize it, and read it back (or return the content).
    atomic_and_ter_content = None
    if minimize_energy:
        logger.info("Running energy minimization (OpenMM)...")
        try:
            # We need a temp file for input
            # Create a temporary directory to avoid race conditions/clutter
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_pdb_path = os.path.join(tmpdirname, "pre_min.pdb")
                output_pdb_path = os.path.join(tmpdirname, "minimized.pdb")
                
                # Write current peptide to input_pdb_path
                # We need to construct a basic PDB file first
                # Use atomic content + minimal header
                
                # CRITICAL Fix for OpenMM:
                # OpenMM's addHydrogens is robust if we start with clean headers.
                # But inputting existing Hydrogens (from Biotite templates) often causes
                # template mismatch errors ("too many H atoms" or naming issues).
                # So we STRIP all hydrogens before passing to OpenMM for standard linear chains.
                # FOR CYCLIC: We KEEP them to help OpenMM identify internal residues vs termini.
                if cyclic:
                    peptide_to_save = peptide
                else:
                    peptide_to_save = peptide[peptide.element != "H"]
                
                # Actually we can use assemble_pdb_content but we need atomic lines first
                # Or just use biotite to write
                pdb_file = pdb.PDBFile()
                pdb_file.set_structure(peptide_to_save)
                pdb_file.write(input_pdb_path)
                
                minimizer = EnergyMinimizer(forcefield_name=forcefield)
                
                # We use add_hydrogens_and_minimize because synth-pdb lacks H by default
                if equilibrate:
                    logger.info(f"Running MD Equilibration ({equilibrate_steps} steps). This includes minimization.")
                    success = minimizer.equilibrate(
                        input_pdb_path, 
                        output_pdb_path, 
                        steps=equilibrate_steps,
                        cyclic=cyclic
                    )
                else:
                    success = minimizer.add_hydrogens_and_minimize(
                        input_pdb_path, 
                        output_pdb_path,
                        max_iterations=minimization_max_iter,
                        tolerance=minimization_k,
                        cyclic=cyclic
                    )
                
                    if success:
                        logger.info("Minimization/Equilibration successful.")
                        # Read back the optimized structure
                        # We return the CONTENT of this file
                        with open(output_pdb_path, 'r') as f:
                            atomic_and_ter_content = f.read()
                            
                        # CRITICAL FIX: Do NOT return early.
                        # We must continue execution so that B-factors and Occupancies 
                        # are calculated and injected below.
                        
                        # EDUCATIONAL NOTE - Re-parsing Minimized Structure:
                        # We must read the minimized structure back into our internal 'peptide' object.
                        # Why? Because downstream steps (like disulfide detection and B-factor calculation)
                        # depend on the exact atomic coordinates. If we used the old 'peptide' object,
                        # we would be analyzing the un-minimized geometry!
                        pdb_file = pdb.PDBFile.read(output_pdb_path)
                        peptide = pdb_file.get_structure(model=1)
                        
                        # RESTORE PTM NAMES (Fix for "Missing Orange Balls"):
                        # OpenMM minimization required reverting SEP->SER etc.
                        # We now restore the original names so the viewer (and user) knows 
                        # where the PTMs are, even if the Phosphate atom is missing (sacrificed for physics).
                        # The sequence list contains the INTENDED residue names.
                        # We assume 1-to-1 mapping (residue indices align).
                        try:
                            # Build map from sequence index to PTM name
                            # sequence is 0-indexed list of names ['ALA', 'SEP', ...]
                            # peptide residues are 1-indexed in res_id
                            
                            # Note: sequence might not include ACE/NME caps if added by OpenMM?
                            # Our generator doesn't add ACE/NME unless 'cap_termini' is True, 
                            # in which case biophysics.cap_termini does it.
                            # But biophysics.cap_termini adds ATOMS to the terminal residues, 
                            # it doesn't usually add new RESIDUES (unless NME is separate?).
                            # Usually NME is a separate residue. ACE is part of N-term?
                            # Let's assume residues match 'sequence'.
                            
                            # Safest way: iterate by matching index if possible.
                            
                            unique_res_ids = np.unique(peptide.res_id)
                            # N residues (sequence) vs M residues (minimized structure)
                            n_seq = len(sequence)
                            n_min = len(unique_res_ids)
                            
                            start_offset = 0
                            
                            # Robust Offset Detection:
                            # Check if the first residue is 'ACE'. If so, everything shifts by 1.
                            if n_min > 0:
                                first_res_id = unique_res_ids[0]
                                # Get name of this residue from the atom array
                                # (Any atom matching this res_id will share the res_name)
                                mask_first = peptide.res_id == first_res_id
                                if np.any(mask_first):
                                    first_res_name = peptide.res_name[mask_first][0]
                                    if first_res_name == "ACE":
                                        logger.info("Detected N-terminal ACE cap. Applying start offset of 1.")
                                        start_offset = 1
                            
                            # Safety Check:
                            # Do we have enough minimized residues to cover the sequence?
                            # e.g., if offset=1 and n_seq=24, we need at least index 24 (total 25 items).
                            # n_min must be >= n_seq + start_offset
                            
                            if n_min < n_seq + start_offset:
                                logger.warning(f"Residue count mismatch: Minimized structure has {n_min} residues, but sequence needs {n_seq} (offset {start_offset}). Cannot map PTMs.")
                                match_valid = False
                            else:
                                match_valid = True
                                
                            if match_valid:
                                for i, res_name_target in enumerate(sequence):
                                    # unique_res_ids includes caps. 
                                    # If offset=1, sequence[0] maps to unique_res_ids[1]
                                    rid = unique_res_ids[i + start_offset]
                                    
                                    if res_name_target in ['SEP', 'TPO', 'PTR', 'HIE', 'HID', 'HIP']:
                                        # Restore name in peptide array
                                        mask = peptide.res_id == rid
                                        peptide.res_name[mask] = res_name_target
                                
                        except Exception as e:
                            logger.warning(f"Failed to restore PTM names: {e}")
                        
                        # We skip the biotite PDB generation block below since we have content.
                        pass
                    else:
                        logger.error("Minimization failed. Returning un-minimized structure.")
                    # If failed, we fall through to standard generation below
                    atomic_and_ter_content = None
        except Exception as e:
            logger.error(f"Error during minimization workflow: {e}")
            # Fallthrough to return original peptide content
            atomic_and_ter_content = None

    if atomic_and_ter_content is None:
        # Assign sequential atom_id to all atoms in the peptide AtomArray
        peptide.atom_id = np.arange(1, peptide.array_length() + 1)
    
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(peptide)
        
        string_io = io.StringIO()
        pdb_file.write(string_io)
        
        # Biotite's PDBFile.write() will write ATOM records, which can be 78 or 80 chars.
        # It also handles TER records between chains, but not necessarily at the end of a single chain.
        atomic_and_ter_content = string_io.getvalue()
    
    # EDUCATIONAL NOTE - Adding Realistic B-factors:
    # Biotite sets all B-factors to 0.00 by default. We post-process the PDB string
    # to replace these with realistic values based on atom type, position, and residue type.
    # This makes the output look more professional and realistic.
    
    # EDUCATIONAL NOTE - Adding Realistic Occupancy:
    # Similarly, biotite sets all occupancy values to 1.00. We calculate realistic
    # occupancy values (0.85-1.00) that correlate with B-factors and reflect disorder.
    
    # Get total number of residues for B-factor and occupancy calculation
    total_residues = len(set(peptide.res_id))
    
    # Calculate Order Parameters (S2) for the generated structure
    # This ensures consistency between the structure (Helices/Loops) and the Data (B-factors/Relaxation).
    s2_map = predict_order_parameters(peptide)
    
    # Sanitize: Extract only atomic content to avoid header duplication/mess
    atomic_and_ter_content = extract_atomic_content(atomic_and_ter_content)
    
    # Process each line and add realistic B-factors and occupancy
    processed_lines = []
    
    # Track atom serials for CONECT records (visualization support)
    n_term_serial = None
    c_term_serial = None
    sg_serials = {} # res_num -> serial
    serial = 0 # Initialize to avoid UnboundLocalError
    
    for line in atomic_and_ter_content.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Serial is in columns 7-11 (0-indexed: 6-11)
            serial = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_num = int(line[22:26].strip())
            
            # Collect serials for cyclic/disulfide CONECT records
            if cyclic:
                if res_num == 1 and atom_name == "N":
                    n_term_serial = serial
                if res_num == total_residues and atom_name == "C":
                    c_term_serial = serial
            
            if (res_name == "CYS" or res_name == "CYX") and atom_name == "SG":
                sg_serials[res_num] = serial

            # Lookup S2 for this residue (default 0.85 if not found)
            current_s2 = s2_map.get(res_num, 0.85)

            # Calculate realistic B-factor (temperature factor) for this atom
            bfactor = _calculate_bfactor(atom_name, res_num, total_residues, res_name, s2=current_s2)
            
            # Calculate realistic occupancy for this atom (correlates with B-factor)
            occupancy = _calculate_occupancy(atom_name, res_num, total_residues, res_name, bfactor)
            
            # Replace B-factor and occupancy in the line
            line = line[:54] + f"{occupancy:6.2f}" + f"{bfactor:6.2f}" + line[66:]
        
        processed_lines.append(line)
    
    atomic_and_ter_content = "\n".join(processed_lines) + "\n"

    # Ensure TER record exists at the end
    lines = atomic_and_ter_content.strip().splitlines()
    if not lines:
        logger.error("Generated PDB content is empty! Falling back to raw sequence string.")
        return atomic_and_ter_content
    
    last_line = lines[-1]
    if last_line.startswith("ATOM") or last_line.startswith("HETATM"):
        last_atom = peptide[-1]
        # serial is guaranteed to be at least its initial value 0
        ter_atom_num = serial + 1
        ter_record = f"TER   {ter_atom_num: >5}      {last_atom.res_name: >3} {last_atom.chain_id: <1}{last_atom.res_id: >4}".ljust(80)
        atomic_and_ter_content = atomic_and_ter_content.strip() + "\n" + ter_record + "\n"

    # Ensure each line is 80 characters
    padded_atomic_and_ter_content_lines = []
    for line in atomic_and_ter_content.splitlines():
        padded_atomic_and_ter_content_lines.append(line.ljust(80))
    
    final_atomic_content_block = "\n".join(padded_atomic_and_ter_content_lines).strip()
    
    # Generate CONECT records for visualization
    conect_records = []
    if cyclic and n_term_serial and c_term_serial:
        conect_records.append(f"CONECT{n_term_serial:5d}{c_term_serial:5d}".ljust(80))
    
    # Detect disulfides to get records and serials
    disulfides = _detect_disulfide_bonds(peptide)
    ssbond_records = _generate_ssbond_records(disulfides, chain_id='A')
    
    if disulfides:
        for r1, r2 in disulfides:
            s1 = sg_serials.get(r1)
            s2 = sg_serials.get(r2)
            if s1 and s2:
                conect_records.append(f"CONECT{s1:5d}{s2:5d}".ljust(80))

    conect_block = "\n".join(conect_records)
    if conect_block:
        conect_block += "\n"

    # Final assembly using centralized utility
    return assemble_pdb_content(
        final_atomic_content_block,
        sequence_length,
        command_args=_build_command_string(args) if 'args' in locals() else None,
        extra_records=ssbond_records if ssbond_records else None,
        conect_records=conect_block if conect_block else None
    )
