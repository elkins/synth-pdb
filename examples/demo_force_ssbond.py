#!/usr/bin/env python3

import numpy as np
import biotite.structure.io.pdb as pdb
import io
import sys
import os

# Add the parent directory to path so we can import synth_pdb if running from examples/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import EnergyMinimizer

def run_demo():
    print("DEMO: Creating a Peptide with a Disulfide Bond")
    print("----------------------------------------------")
    
    # 1. Generate a simple peptide "CYS-GLY-GLY-CYS" (CGGC)
    #    The loop is short, so it's easy to bend.
    print("1. Generating initial backbone for sequence 'CGGC' (extended)...")
    content = generate_pdb_content(sequence_str="CGGC", conformation="extended", optimize_sidechains=False)
    
    # Load into Biotite structure
    pdb_file = pdb.PDBFile.read(io.StringIO(content))
    structure = pdb_file.get_structure(model=1)
    
    # 2. Manually manipulate coordinates to bring CYS 1 and CYS 4 close.
    #    In a real scenario, this would happen during folding or loop closure.
    #    Here we "mock" a folded state to test the physics engine's response.
    
    # Let's say we want to bond CYS 1 (N-term) and CYS 4 (C-term).
    cys1_mask = (structure.res_id == 1)
    cys4_mask = (structure.res_id == 4)
    
    # Get SG coordinates
    sg1 = structure[cys1_mask & (structure.atom_name == "SG")][0]
    
    # Move Residue 4 so its SG is ~2.05 A away from SG1
    # We'll just perform a rigid translation of Res 4 for simplicity.
    # This distorts the backbone between 3 and 4, but the minimizer handles that.
    sg4 = structure[cys4_mask & (structure.atom_name == "SG")][0]
    
    target_pos = sg1.coord + np.array([0.0, 0.0, 2.05]) # Place 2.05 A away in Z
    shift_vector = target_pos - sg4.coord
    
    structure.coord[cys4_mask] += shift_vector
    
    # Verify initial distance
    sg1 = structure[(structure.res_id == 1) & (structure.atom_name == "SG")][0]
    sg4 = structure[(structure.res_id == 4) & (structure.atom_name == "SG")][0]
    dist = np.linalg.norm(sg1.coord - sg4.coord)
    print(f"   Modified Distance (CYS1-CYS4): {dist:.3f} A")
    
    # Save input
    input_file = "demo_ssbond_input.pdb"
    output_file = "demo_ssbond_final.pdb"
    
    pdb_out = pdb.PDBFile()
    pdb_out.set_structure(structure)
    pdb_out.write(input_file)
    print(f"2. Saved modified structure to {input_file}")
    
    # 3. Run Physics minimization
    print("3. Running Energy Minimization (OpenMM)...")
    print("   (This will detect the proximal cysteines, patch them as CYX, and add a bond)")
    
    minimizer = EnergyMinimizer()
    if not minimizer.add_hydrogens_and_minimize(input_file, output_file):
        print("Error: Minimization failed.")
        return

    # 4. Analyze Results
    print(f"4. Minimization complete. Saved to {output_file}")
    
    final_pdb = pdb.PDBFile.read(output_file)
    final_struct = final_pdb.get_structure(model=1)
    
    sg1 = final_struct[(final_struct.res_id == 1) & (final_struct.atom_name == "SG")][0]
    sg4 = final_struct[(final_struct.res_id == 4) & (final_struct.atom_name == "SG")][0]
    final_dist = np.linalg.norm(sg1.coord - sg4.coord)
    
    print(f"   Final Distance (CYS1-CYS4): {final_dist:.3f} A")
    
    # Check for SSBOND record in the raw text
    with open(output_file, 'r') as f:
        lines = f.readlines()
        ssbonds = [l for l in lines if l.startswith("SSBOND")]
        if ssbonds:
            print(f"   Found SSBOND Record:\n   {ssbonds[0].strip()}")
        else:
            print("   Warning: SSBOND record not found in header (Generator might need update to write it from topology)")
            # Note: The generator.py logic writes SSBONDs, but here we ran physics.py directly.
            # physics.py writes the file using OpenMM -> PDB. OpenMM's PDBFile writer usually handles SSBONDs if they exist in topology?
            # Actually, OpenMM PDBFile.writeFile DOES write CONECT records but SSBOND records are not standardly written by it 
            # unless we preserve the header or add it manually.
            # However, the physical bond exists!

if __name__ == "__main__":
    run_demo()
