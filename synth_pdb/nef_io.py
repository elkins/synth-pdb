"""
NEF (NMR Exchange Format) I/O module for synth-pdb.
Handles writing synthetic NMR data to valid NEF files.
"""

import logging
import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

def write_nef_file(
    filename: str,
    sequence: str,
    restraints: List[Dict],
    system_name: str = "synth-pdb-project"
) -> None:
    """
    Write a minimal valid NEF file containing sequence and distance restraints.
    
    Args:
        filename: Output filepath.
        sequence: Amino acid sequence string (1-letter code).
        restraints: List of restraint dicts from nmr.calculate_synthetic_noes.
        system_name: Name for the NEF saveframe.
    """
    logger.info(f"Writing NEF file to {filename}...")
    
    # NEF uses NMR-STAR syntax.
    # We will construct it manually to avoid dependencies, as our scope is limited.
    
    # 1. Header
    nc = "data_" + system_name + "\n"
    nc += "\n"
    nc += "_nef_nmr_meta_data.nef_format_version 1.1\n"
    nc += f"_nef_nmr_meta_data.creation_date {datetime.datetime.now().isoformat()}\n"
    nc += "_nef_nmr_meta_data.program_name synth-pdb\n"
    nc += "\n"
    
    # 2. Sequence (nef_sequence)
    # We need to expand 1-letter code to residues
    nc += "save_nef_sequence\n"
    nc += "   _nef_sequence.sf_category nef_sequence\n"
    nc += "   _nef_sequence.sf_framecode nef_sequence\n"
    nc += "\n"
    nc += "   loop_\n"
    nc += "      _nef_sequence.chain_code\n"
    nc += "      _nef_sequence.sequence_code\n"
    nc += "      _nef_sequence.residue_name\n"
    nc += "      _nef_sequence.residue_type\n" # protein
    
    # Mapping 1-letter to 3-letter (reuse data.py if possible, or simple map)
    # We'll do a quick local map or import proper one.
    # Importing from data for robustness
    from .data import STANDARD_AMINO_ACIDS, ONE_TO_THREE_LETTER_CODE
    
    # Invert mapping for 1->3
    # Actually ONE_TO_THREE_LETTER_CODE is {1: 3}
    one_to_three = ONE_TO_THREE_LETTER_CODE
    
    for i, char in enumerate(sequence):
        res_num = i + 1
        res_name = one_to_three.get(char, "UNK")
        nc += f"      A {res_num} {res_name} protein\n"
        
    nc += "   stop_\n"
    nc += "save_\n"
    nc += "\n"
    
    # 3. Distance Restraints (nef_distance_restraint_list)
    nc += "save_synthetic_noes\n"
    nc += "   _nef_distance_restraint_list.sf_category nef_distance_restraint_list\n"
    nc += "   _nef_distance_restraint_list.sf_framecode synthetic_noes\n"
    nc += "   _nef_distance_restraint_list.restraint_origin synthetic\n"
    nc += "\n"
    nc += "   loop_\n"
    nc += "      _nef_distance_restraint.index\n"
    nc += "      _nef_distance_restraint.restraint_id\n" # ID within list
    nc += "      _nef_distance_restraint.chain_code_1\n"
    nc += "      _nef_distance_restraint.sequence_code_1\n"
    nc += "      _nef_distance_restraint.residue_name_1\n"
    nc += "      _nef_distance_restraint.atom_name_1\n"
    nc += "      _nef_distance_restraint.chain_code_2\n"
    nc += "      _nef_distance_restraint.sequence_code_2\n"
    nc += "      _nef_distance_restraint.residue_name_2\n"
    nc += "      _nef_distance_restraint.atom_name_2\n"
    nc += "      _nef_distance_restraint.target_value\n"
    nc += "      _nef_distance_restraint.upper_limit\n"
    nc += "      _nef_distance_restraint.lower_limit\n"
    nc += "      _nef_distance_restraint.weight\n"
    
    for i, r in enumerate(restraints):
        idx = i + 1
        # Convert atom names to NEF standard?
        # NEF usually follows IUPAC. Biotite should output IUPAC-ish.
        # Naming is tricky (HB2 vs 2HB). We output what we have.
        
        row = f"      {idx} {idx} "
        row += f"{r['chain_1']} {r['residue_index_1']} {r['res_name_1']} {r['atom_name_1']} "
        row += f"{r['chain_2']} {r['residue_index_2']} {r['res_name_2']} {r['atom_name_2']} "
        row += f"{r['actual_distance']:.3f} {r['upper_limit']:.3f} {r['lower_limit']:.3f} 1.0\n"
        nc += row

    nc += "   stop_\n"
    nc += "save_\n"
    
    with open(filename, "w") as f:
        f.write(nc)
    
    logger.info(f"Successfully wrote {len(restraints)} restraints to {filename}.")
