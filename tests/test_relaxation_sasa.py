
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.relaxation import predict_order_parameters

def create_buried_exposed_system():
    """
    Creates a simple system with a 'buried' residue and an 'exposed' residue.
    
    Structure:
    - Residue 1 (Buried): Surrounded by other atoms.
    - Residue 2 (Exposed): Far away, isolated.
    """
    # Create a dense cluster around Res 1
    # We use simpler atoms (CA) for SASA calculation test
    atoms = []
    
    # Residue 1: Center of a cluster
    atoms.append(struc.Atom([0,0,0], atom_name="CA", res_name="ALA", res_id=1, chain_id="A", element="C"))
    # Surround it with "shielding" atoms (Res 3, 4, 5, 6, 7, 8)
    # Positions +/- 3.5 Angstroms in X, Y, Z
    offsets = [
        [3.5, 0, 0], [-3.5, 0, 0],
        [0, 3.5, 0], [0, -3.5, 0],
        [0, 0, 3.5], [0, 0, -3.5]
    ]
    for i, off in enumerate(offsets):
        res_id = 10 + i
        atoms.append(struc.Atom(off, atom_name="CA", res_name="ALA", res_id=res_id, chain_id="A", element="C"))
        
    # Residue 2: Far away (Exposed)
    atoms.append(struc.Atom([50,0,0], atom_name="CA", res_name="ALA", res_id=2, chain_id="A", element="C"))
    
    # Needs to be an AtomArray
    array = struc.array(atoms)
    
    # Note: predict_order_parameters also relies on secondary structure.
    # Our mock has no backbone angles, so get_secondary_structure will return 'coil' for all.
    # This is perfect! If both are 'coil', the only difference in S2 should come from SASA.
    
    return array

def test_sasa_modulation_of_s2():
    """
    Test that a buried residue has a higher S2 (more rigid) than an exposed residue
    when both have the same secondary structure (coil).
    """
    structure = create_buried_exposed_system()
    
    # Calculate S2
    # We suppress warnings because our mock structure is weird (no bonds, missing atoms)
    # but Biotite SASA works on point clouds if vdw_radii are handled.
    # synth_pdb might need to handle the atomic radii lookup.
    
    # Let's see if it crashes or works.
    s2_map = predict_order_parameters(structure)
    
    # Check values
    s2_buried = s2_map[1]
    s2_exposed = s2_map[2]
    
    print(f"S2 Buried: {s2_buried}")
    print(f"S2 Exposed: {s2_exposed}")
    
    # Buried should be significantly higher (more rigid) than exposed
    # Default Coil S2 is 0.65
    # Buried should approach 0.85
    # Exposed should stay near 0.65 (or lower?)
    
    # Assert significant difference (> 0.05)
    assert s2_buried > s2_exposed + 0.05, \
        f"Buried residue S2 ({s2_buried}) should be higher than Exposed S2 ({s2_exposed})"

def test_sasa_integration_runs():
    """
    Test that the function runs on a somewhat normal peptide without crashing.
    """
    # Create simple peptide
    # We need a small backbone to avoid secondary structure crashes if any
    atom1 = struc.Atom([0,0,0], atom_name="CA", res_name="GLY", res_id=1, element="C")
    atom2 = struc.Atom([3,0,0], atom_name="CA", res_name="GLY", res_id=2, element="C")
    structure = struc.array([atom1, atom2])
    
    s2_map = predict_order_parameters(structure)
    assert len(s2_map) == 2
