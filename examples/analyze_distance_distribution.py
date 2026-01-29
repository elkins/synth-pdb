#!/usr/bin/env python3

import sys
import os
import numpy as np
import logging
import io
import matplotlib.pyplot as plt
import biotite.structure.io.pdb as pdb

# Disable logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("synth_pdb")
logger.setLevel(logging.ERROR)

from synth_pdb.generator import generate_pdb_content

def analyze_distances(sequence, n_trials=100):
    print(f"Analyzing CYS-CYS distances for: {sequence} ({n_trials} trials)")
    distances = []
    
    for _ in range(n_trials):
        try:
            # Generate un-minimized content to see "raw" random sampling
            content = generate_pdb_content(
                sequence_str=sequence, 
                conformation="random", 
                optimize_sidechains=True, 
                minimize_energy=False # Don't minimize yet
            )
            
            pdb_file = pdb.PDBFile.read(io.StringIO(content))
            structure = pdb_file.get_structure(model=1)
            
            cys_res = structure[structure.res_name == "CYS"]
            sgs = cys_res[cys_res.atom_name == "SG"]
            
            if len(sgs) >= 2:
                # Just take first pair
                d = np.linalg.norm(sgs[0].coord - sgs[1].coord)
                distances.append(d)
                
        except Exception:
            pass

    distances = np.array(distances)
    print(f"Min: {distances.min():.2f}, Max: {distances.max():.2f}, Mean: {distances.mean():.2f}")
    print(f"Count < 2.5 A: {np.sum(distances < 2.5)}")
    print(f"Count < 5.0 A: {np.sum(distances < 5.0)}")
    print(f"Count < 8.0 A: {np.sum(distances < 8.0)}")
    print(f"Count < 10.0 A: {np.sum(distances < 10.0)}")

if __name__ == "__main__":
    analyze_distances("CGGC", n_trials=100)
