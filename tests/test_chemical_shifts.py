
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.chemical_shifts import predict_chemical_shifts, calculate_csi, RANDOM_COIL_SHIFTS, SECONDARY_SHIFTS
from synth_pdb.generator import generate_pdb_content
from synth_pdb.pdb_utils import extract_atomic_content, assemble_pdb_content
import biotite.structure.io.pdb as pdb_io
import io

@pytest.fixture
def mock_alpha_structure(mocker):
    """Mock structure that returns Alpha Helix angles when measured."""
    # Create a dummy structure
    structure = struc.AtomArray(10)
    structure.res_id = np.array([1]*5 + [2]*5) # 2 residues
    structure.res_name = np.array(["ALA"]*10)
    structure.chain_id = np.array(["A"]*10)
    
    # Mock dihedral_backbone function
    # return (phi, psi, omega)
    # Alpha: -57, -47
    phi = np.array([np.nan, -60.0, -60.0, -60.0, -60.0]) # 5 residues/angles
    psi = np.array([-50.0, -50.0, -50.0, -50.0, np.nan])
    omega = np.array([180.0]*5)
    
    mocker.patch("biotite.structure.dihedral_backbone", return_value=(np.radians(phi), np.radians(psi), np.radians(omega)))
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1])) # Indices

    return structure

def test_helix_trends(mocker):
    """Test using mocked angles."""
    # Setup Mock
    phi = np.array([np.nan, -60.0, -60.0, -60.0, -60.0])
    psi = np.array([-50.0, -50.0, -50.0, -50.0, np.nan])
    omega = np.array([180.0]*5)
    
    mocker.patch("biotite.structure.dihedral_backbone", return_value=(np.radians(phi), np.radians(psi), np.radians(omega)))
    
    # Needs a real-ish structure to iterate over
    # Create a minimal one: 5 Residues of ALA
    # We need atom arrays to iterate
    # Create dummy structure
    structure = struc.AtomArray(1) # size doesn't matter if we mock get_residue_starts
    
    # Mock resid iteration
    # 5 residues
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2, 3, 4]))
    
    # We need structure slices to have .res_name, .chain_id, .res_id
    # We can mock the __getitem__ or just make structure valid-ish
    # Easier: make real structure of length 5 res
    structure = struc.AtomArray(5) # 1 atom per res
    structure.res_name = np.array(["ALA"] * 5)
    structure.chain_id = np.array(["A"] * 5)
    structure.res_id = np.array([1, 2, 3, 4, 5])
    
    # Run
    shifts = predict_chemical_shifts(structure)
    
    # Residue 2 (Index 2 in Py, ID 3) should have Alpha angles (-60, -50)
    # Index 2: Phi[2]=-60, Psi[2]=-50. Matches Alpha criteria.
    res3 = shifts['A'][3]
    rc = RANDOM_COIL_SHIFTS['ALA']
    
    # Should contain secondary structure offset
    # CA offset +3.1
    # 52.5 + 3.1 = 55.6
    assert res3['CA'] > rc['CA'] + 1.0

def test_sheet_trends(mocker):
    """Test using mocked angles for Beta Sheet."""
    # Beta: -120, 120
    phi = np.array([np.nan, -130.0, -130.0, -130.0, -130.0])
    psi = np.array([130.0, 130.0, 130.0, 130.0, np.nan])
    omega = np.array([180.0]*5)
    
    mocker.patch("biotite.structure.dihedral_backbone", return_value=(np.radians(phi), np.radians(psi), np.radians(omega)))
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2, 3, 4]))
    
    structure = struc.AtomArray(5)
    structure.res_name = np.array(["VAL"] * 5)
    structure.chain_id = np.array(["A"] * 5)
    structure.res_id = np.array([1, 2, 3, 4, 5])
    
    shifts = predict_chemical_shifts(structure)
    
    # Residue 3
    res3 = shifts['A'][3]
    rc = RANDOM_COIL_SHIFTS['VAL']
    
    # CA offset -1.5
    assert res3['CA'] < rc['CA'] - 0.5

def test_structure_returns_dict(mock_alpha_structure):
    """Test that function returns dictionary with expected keys."""
    shifts = predict_chemical_shifts(mock_alpha_structure)
    assert isinstance(shifts, dict)
    assert 'A' in shifts # Chain A
    assert 1 in shifts['A'] # Residue 1
    
    # Check atoms exist
    res1 = shifts['A'][1]
    for atom in ["N", "CA", "CB"]: # ALA has CB
        assert atom in res1

def test_glycine_shifts():
    """Test Glycine has no CB."""
    # Create a structure with Glycine
    content = generate_pdb_content(sequence_str="GGG", conformation="alpha")
    pdb_file = pdb_io.PDBFile.read(io.StringIO(content))
    structure = pdb_file.get_structure(model=1)
    
    shifts = predict_chemical_shifts(structure)
    res2_gly = shifts['A'][2]
    
    assert "CA" in res2_gly
    assert "CB" not in res2_gly
    assert "HA" in res2_gly

def test_proline_shifts():
    """Test Proline has no Amide N/H."""
    content = generate_pdb_content(sequence_str="APA", conformation="alpha")
    pdb_file = pdb_io.PDBFile.read(io.StringIO(content))
    structure = pdb_file.get_structure(model=1)
    
    shifts = predict_chemical_shifts(structure)
    res2_pro = shifts['A'][2]
    
    assert "N" not in res2_pro
    assert "H" not in res2_pro
    assert "CA" in res2_pro

def test_csi_calculation():
    """Test that CSI correctly subtracts Random Coil values."""
    # 1. Setup Mock Structure (needed for ResName lookup)
    structure = struc.AtomArray(1)
    structure.res_id = np.array([10])
    structure.res_name = np.array(["ALA"])
    structure.atom_name = np.array(["CA"])
    structure.chain_id = np.array(["A"])
    
    # 2. Setup Shifts (Simulate a Helix)
    # ALA Random Coil CA = 52.5
    # Helix Offset = +3.1 -> Predicted = 55.6
    shifts = {
        "A": {
            10: {"CA": 56.5} # +4.0 deviation
        }
    }
    
    # 3. Calculate
    csi = calculate_csi(shifts, structure)
    
    # 4. Verify
    # Delta = 56.5 - 52.5 = 4.0
    assert "A" in csi
    assert 10 in csi["A"]
    delta = csi["A"][10]
    assert pytest.approx(delta, 0.1) == 4.0
