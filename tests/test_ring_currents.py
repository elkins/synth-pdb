
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.chemical_shifts import predict_chemical_shifts, RANDOM_COIL_SHIFTS

def create_mock_phe_system():
    """
    Creates a mock system with a PHE residue at the origin and 
    isolated protons at specific geometric positions to test ring currents.
    """
    # 1. Create a PHE residue (Ring in XY plane, centered at origin)
    # Simplified geometry for testing
    phe_atoms = [
        # Backbone (needed for identification)
        ("N", [0, 10, 0], "N"),
        ("CA", [0, 10, 0], "C"),
        ("C", [0, 10, 0], "C"),
        
        # Ring Atoms (Benzene-like roughly)
        # Center is 0,0,0
        ("CG",  [0.0, 1.40, 0.0], "C"),
        ("CD1", [1.21, 0.70, 0.0], "C"),
        ("CD2", [-1.21, 0.70, 0.0], "C"),
        ("CE1", [1.21, -0.70, 0.0], "C"),
        ("CE2", [-1.21, -0.70, 0.0], "C"),
        ("CZ",  [0.0, -1.40, 0.0], "C"),
    ]
    
    atoms = []
    for name, coord, element in phe_atoms:
        atom = struc.Atom(coord, atom_name=name, res_name="PHE", res_id=1, chain_id="A", element=element)
        atoms.append(atom)
        
    # 2. Add a Probe Proton directly ABOVE the ring (Shielded Zone)
    # Position: (0, 0, 3.0) - On the normal vector
    # We call it "HA" of residue 2 (ALA) for simplicity
    probe_shielded = struc.Atom([0.0, 0.0, 3.0], atom_name="HA", res_name="ALA", res_id=2, chain_id="A", element="H")
    atoms.append(probe_shielded)
    
    # 3. Add a Probe Proton in the PLANE of the ring (Deshielded Zone)
    # Position: (0, 4.0, 0.0) - Extended from CG
    probe_deshielded = struc.Atom([0.0, 4.0, 0.0], atom_name="HA", res_name="ALA", res_id=3, chain_id="A", element="H")
    atoms.append(probe_deshielded)
    
    array = struc.array(atoms)
    
    # Needs to be able to identify Secondary structure, so let's mock it to 'coil' for all
    # The current implementation of get_secondary_structure needs backbone atoms.
    # We provided Backbone for PHE. ALA 2 and 3 are just floating protons, which might break SS detection.
    # But chemical_shifts.py handles missing SS by defaulting to coil.
    
    return array

def test_ring_current_shielding():
    """
    Test that a proton placed ABOVE an aromatic ring is SHIELDED (Shift decreases).
    """
    system = create_mock_phe_system()
    
    # Predict shifts
    # Suppress RuntimeWarning from biotite due to missing backbone in mock system
    import warnings
    from unittest.mock import patch
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Mock random normal to return 0.0 (remove noise)
        with patch("numpy.random.normal", return_value=0.0):
            shifts = predict_chemical_shifts(system)
    
    # Get Baseline (Random Coil) for ALA HA
    base_shift = RANDOM_COIL_SHIFTS["ALA"]["HA"]
    
    # Get Shift for Residue 2 (Shielded Probe)
    # Note: dictionary structure is [chain][res_id][atom]
    probe_shift = shifts["A"][2]["HA"]
    
    print(f"Base ALA HA: {base_shift}")
    print(f"Shielded Probe: {probe_shift}")
    
    # Ring Current Effect should be NEGATIVE (Upfield Shift)
    # Typically -0.5 to -1.5 ppm at 3.0 Angstroms
    # We assert it is significantly lower than baseline
    # The current code (without RC) will just return base_shift +/- noise (0.15)
    
    # Ensure it's not just noise
    assert probe_shift < (base_shift - 0.2), \
        f"Probe above ring should be shielded! (Got {probe_shift}, Base {base_shift})"

def test_ring_current_deshielding():
    """
    Test that a proton placed IN THE PLANE of an aromatic ring is DESHIELDED (Shift increases).
    """
    system = create_mock_phe_system()
    import warnings
    from unittest.mock import patch
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with patch("numpy.random.normal", return_value=0.0):
            shifts = predict_chemical_shifts(system)
    
    base_shift = RANDOM_COIL_SHIFTS["ALA"]["HA"]
    
    # Residue 3 is the in-plane probe
    probe_shift = shifts["A"][3]["HA"]
    
    print(f"Base ALA HA: {base_shift}")
    print(f"Deshielded Probe: {probe_shift}")
    
    # Ring Current Effect should be POSITIVE (Downfield Shift)
    assert probe_shift > (base_shift + 0.2), \
        f"Probe in ring plane should be deshielded! (Got {probe_shift}, Base {base_shift})"
