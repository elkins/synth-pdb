#!/usr/bin/env python3

import numpy as np
import biotite.structure.io.pdb as pdb
import io
import sys
import os

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import EnergyMinimizer

def create_distanced_cysteines(filename, distance):
    # Generates a peptide and manually sets distance to 'distance' Angstroms
    content = generate_pdb_content(sequence_str="CGGC", conformation="extended", optimize_sidechains=False)
    pdb_file = pdb.PDBFile.read(io.StringIO(content))
    structure = pdb_file.get_structure(model=1)
    
    sg1 = structure[(structure.res_id == 1) & (structure.atom_name == "SG")][0]
    sg4 = structure[(structure.res_id == 4) & (structure.atom_name == "SG")][0]
    target_pos = sg1.coord + np.array([0.0, 0.0, distance])
    shift_vector = target_pos - sg4.coord
    mask_res4 = (structure.res_id == 4)
    structure.coord[mask_res4] += shift_vector
    
    pdb_out = pdb.PDBFile()
    pdb_out.set_structure(structure)
    pdb_out.write(filename)
    return filename

def test_steering():
    input_file = "test_steering_input.pdb"
    output_file = "test_steering_output.pdb"
    
    # Test 6.0 Angstroms (Steering test)
    print("Testing Steering from 6.0 Angstroms...")
    create_distanced_cysteines(input_file, 6.0)
    
    # We rely on physics.py having the logic. 
    # Initially the threshold was 2.5 in physics.py.
    # So this will always FAIL (no bond) with threshold at 2.5.
    # Threshold was updated to 5.0 and it succeeds.
 
    minimizer = EnergyMinimizer()
    minimizer.add_hydrogens_and_minimize(input_file, output_file)
    
    # Check results
    final_pdb = pdb.PDBFile.read(output_file)
    final_struct = final_pdb.get_structure(model=1)
    sg1 = final_struct[(final_struct.res_id == 1) & (final_struct.atom_name == "SG")][0]
    sg4 = final_struct[(final_struct.res_id == 4) & (final_struct.atom_name == "SG")][0]
    dist = np.linalg.norm(sg1.coord - sg4.coord)
    print(f"Final Distance: {dist:.3f} A")
    
    if dist < 2.5:
        print("SUCCESS: Bond formed!")
    else:
        print("FAILURE: Bond did not form (as expected if threshold < 5.0)")

if __name__ == "__main__":
    test_steering()
