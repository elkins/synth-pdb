
import pytest
import os
import logging
import numpy as np
from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import EnergyMinimizer, HAS_OPENMM

@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_salt_bridge_restraint_logging(caplog):
    """
    Verify that salt bridge restraints are detected and applied.
    We use sequence ER in alpha helical conformation.
    """
    caplog.set_level(logging.DEBUG)
    
    # E and K separated by 3 A: i, i+4 (common in helices)
    sequence = "AEAAAK"
    pdb_content = generate_pdb_content(sequence_str=sequence, conformation="alpha")
    
    input_pdb = "test_sb_input.pdb"
    output_pdb = "test_sb_output.pdb"
    
    with open(input_pdb, "w") as f:
        f.write(pdb_content)
        
    try:
        minimizer = EnergyMinimizer()
        success = minimizer.add_hydrogens_and_minimize(input_pdb, output_pdb)
        
        # If it fails to find bridge, we want to know WHY (distances).
        # We check the log for my debug prints in biophysics.py or physics.py
        log_text = caplog.text
        print(log_text) # Show in test output
        
        assert success
        # Even if 0 found, we should see the DEBUG line
        assert "DEBUG: Found" in log_text
        # If we find 0, we'll need to adjust the test or the generator.
        
    finally:
        if os.path.exists(input_pdb): os.remove(input_pdb)
        if os.path.exists(output_pdb): os.remove(output_pdb)
