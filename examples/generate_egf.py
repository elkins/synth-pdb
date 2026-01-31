#!/usr/bin/env python3

import os
import sys
from synth_pdb.main import main as synth_main

def generate_egf_example():
    """
    Generate a Human Epidermal Growth Factor (EGF) protein example.
    Sequence: NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR (53 aa)
    Disulfides: C6-C20, C14-C31, C33-C42
    """
    egf_seq = "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR"
    output_file = "egf_protein.pdb"
    
    # We use the CLI interface programmatically for simplicity
    # Use random conformation as a starting point for minimization to find disulfides
    # We add --minimize to trigger OpenMM and disulfide detection
    # We add --cap-termini for biophysical realism
    # We add --gen-shifts and --gen-relax to show off the NMR features
    
    print(f"Generating EGF protein with sequence: {egf_seq}")
    
    sys.argv = [
        "synth_pdb",
        "--sequence", egf_seq,
        "--output", output_file,
        "--conformation", "random",
        "--minimize",
        "--cap-termini",
        "--gen-shifts",
        "--gen-relax",
        "--validate",
        "--log-level", "DEBUG"
    ]
    
    try:
        synth_main()
        print(f"\nSuccessfully generated EGF protein: {output_file}")
        print("Note: EGF has 3 critical disulfide bonds (C6-C20, C14-C31, C33-C42).")
        print("The minimized structure in 'egf_protein.pdb' should have these detected and modeled.")
        
    except Exception as e:
        print(f"Error generating EGF: {e}")

if __name__ == "__main__":
    generate_egf_example()
