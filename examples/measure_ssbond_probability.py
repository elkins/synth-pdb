#!/usr/bin/env python3

import sys
import os
import random
import logging
import numpy as np
import biotite.structure.io.pdb as pdb
import io

# Disable logging to keep output clean
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("synth_pdb")
logger.setLevel(logging.ERROR)

from synth_pdb.generator import generate_pdb_content

def measure_probability(sequence, n_trials=100):
    print(f"Measuring SSBOND formation probability for sequence: {sequence}")
    print(f"Configuration: minimized=True, conformation=random, trials={n_trials}")
    print("-" * 60)
    
    ssbond_hits = 0
    
    for i in range(n_trials):
        try:
            # Generate content
            # valid arguments are: sequence_str, conformation, optimize_sidechains, minimize_energy
            content = generate_pdb_content(
                sequence_str=sequence, 
                conformation="random", 
                optimize_sidechains=True, 
                minimize_energy=True
            )
            
            # Check for SSBOND
            if "SSBOND" in content:
                ssbond_hits += 1
                sys.stdout.write("x") # simple progress bar
            else:
                sys.stdout.write(".")
            sys.stdout.flush()
            
        except Exception as e:
            # If generation fails (rare), just count as failure
            sys.stdout.write("E")
            sys.stdout.flush()
            
    print("\n" + "-" * 60)
    print(f"Trials: {n_trials}")
    print(f"Successes: {ssbond_hits}")
    print(f"Probability: {ssbond_hits/n_trials * 100:.1f}%")

if __name__ == "__main__":
    # Test a few candidates
    print('Testing for SSBOND in candidate sequences')

    # 'CGGC' - short flexible loop
    measure_probability("CGGC", n_trials=30)
    
    # 'CGC' - very tight turn
    measure_probability("CGC", n_trials=30)
    
    # 'CGGGC' - longer flexible loop
    measure_probability("CGGGC", n_trials=30)
