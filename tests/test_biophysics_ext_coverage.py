import pytest
import numpy as np
import os
import tempfile
import biotite.structure as struc
from synth_pdb.docking import DockingPrep
from synth_pdb.contact import compute_contact_map
from synth_pdb.distogram import calculate_distogram

from synth_pdb.generator import generate_pdb_content
from synth_pdb.distogram import export_distogram

def test_docking_prep_pqr():
    # Use generator with minimization to ensure OpenMM-compatible atom sets
    pdb_content = generate_pdb_content(sequence_str="ALA-GLY-SER", minimize_energy=True, cap_termini=True)
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp_pdb:
        tmp_pdb.write(pdb_content)
        tmp_pdb.flush()
        
        with tempfile.NamedTemporaryFile(suffix=".pqr") as tmp_pqr:
            prep = DockingPrep()
            # This should now succeed with a real structure
            success = prep.write_pqr(tmp_pdb.name, tmp_pqr.name)
            assert success
            assert os.path.exists(tmp_pqr.name)

def test_contact_map_noe_method():
    atoms = struc.array([
        struc.Atom([0,0,0], atom_name="CA", res_name="ALA", res_id=1, chain_id="A"),
        struc.Atom([1,0,0], atom_name="CB", res_name="ALA", res_id=1, chain_id="A"),
        struc.Atom([10,0,0], atom_name="CA", res_name="ALA", res_id=2, chain_id="A"),
        struc.Atom([11,0,0], atom_name="CB", res_name="ALA", res_id=2, chain_id="A")
    ])
    # Test NOE method (uses CB)
    cmap = compute_contact_map(atoms, method="noe")
    assert cmap.shape == (2, 2)
    
    # Test invalid method
    with pytest.raises(ValueError, match="Method must be 'ca' or 'noe'"):
        compute_contact_map(atoms, method="invalid")

def test_distogram_edge_cases():
    atoms = struc.array([
        struc.Atom([0,0,0], atom_name="CA", res_name="ALA", res_id=1, chain_id="A"),
        struc.Atom([10,0,0], atom_name="CA", res_name="ALA", res_id=2, chain_id="A")
    ])
    disto = calculate_distogram(atoms, method="ca")
    assert disto.shape == (2, 2)
    assert np.allclose(disto[0, 1], 10.0)
    
    # Test CB method
    disto_cb = calculate_distogram(atoms, method="cb")
    assert disto_cb.shape == (2, 2) # Should fall back to CA if CB missing
    
    # Test unknown method
    disto_unk = calculate_distogram(atoms, method="unknown")
    assert disto_unk.shape == (2, 2)

    # Test export
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test.json")
        export_distogram(disto, json_path, fmt="json")
        assert os.path.exists(json_path)
        
        csv_path = os.path.join(tmpdir, "test.csv")
        export_distogram(disto, csv_path, fmt="csv")
        assert os.path.exists(csv_path)
        
        npz_path = os.path.join(tmpdir, "test.npz")
        export_distogram(disto, npz_path, fmt="npz")
        assert os.path.exists(npz_path)
        
        with pytest.raises(ValueError, match="Unknown format: invalid"):
            export_distogram(disto, "test.txt", fmt="invalid")
