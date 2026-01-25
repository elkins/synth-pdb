"""
Visualization module for synth-pdb.
Generates scripts for external molecular viewers (PyMOL, ChimeraX).
"""

import logging
from typing import List, Dict
from .nef_io import read_nef_restraints

logger = logging.getLogger(__name__)

def generate_pymol_script(
    pdb_file: str,
    restraints: List[Dict],
    output_pml: str
) -> None:
    """
    Generate a PyMOL script (.pml) to visualize NEF restraints on a PDB structure.
    
    Args:
        pdb_file: Path to the PDB structure file (relative to where script runs).
        restraints: List of restraint dictionaries (from nmr.calculate_synthetic_noes or read_nef_restraints).
        output_pml: Output path for the PyMOL script.
    """
    logger.info(f"Generating PyMOL script: {output_pml}")
    
    if not restraints:
        logger.warning("No restraints provided. Script will only load structure.")
    
    script = []
    
    # 1. Load Structure
    # PyMOL 'load' command
    script.append(f"load {pdb_file}, structure")
    script.append("hide everything")
    script.append("show cartoon")
    script.append("color spectrum")
    script.append("set cartoon_transparency, 0.2") # Make cartoon transparent to see bonds
    
    # 2. Draw Restraints
    # PyMOL 'distance' command: distance name, selection1, selection2
    script.append("\n# Synthetic NOE Restraints")
    
    for i, r in enumerate(restraints):
        idx = i + 1
        name = f"noe_{idx}"
        
        # Selection syntax: (chain A and resi 10 and name CA)
        # Support both 'seq_1' (NEF Reader) and 'residue_index_1' (NMR Calc) keys
        c1 = r.get('chain_1', 'A')
        s1 = r.get('seq_1', r.get('residue_index_1'))
        a1 = r.get('atom_1', r.get('atom_name_1'))
        
        c2 = r.get('chain_2', 'A')
        s2 = r.get('seq_2', r.get('residue_index_2'))
        a2 = r.get('atom_2', r.get('atom_name_2'))
        
        if s1 is None or s2 is None:
             logger.warning(f"Skipping restraint {idx}: Missing sequence number.")
             continue
             
        sel1 = f"(chain {c1} and resi {s1} and name {a1})"
        sel2 = f"(chain {c2} and resi {s2} and name {a2})"
        
        # Draw distance object
        script.append(f"distance {name}, {sel1}, {sel2}")
    
    # 3. Styling
    script.append("\n# Styling Restraints")
    script.append("group noes, noe_*")    # Group them
    script.append("hide labels, noes")    # Hide distance value labels
    script.append("color grey50, noes")   # Color dashes grey
    script.append("set dash_gap, 0")      # Solid lines (optional, or keeping dashes)
    script.append("set dash_radius, 0.05") # Thinner lines
    script.append("center")
    
    # Write to file
    with open(output_pml, 'w') as f:
        f.write("\n".join(script))
        
    logger.info(f"Generated PyMOL script with {len(restraints)} restraints.")
