import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.geometry import reconstruct_sidechain, calculate_dihedral_angle, calculate_angle
from synth_pdb.generator import generate_pdb_content
import biotite.structure.io.pdb as pdb
import io

def test_reconstruct_sidechain_basic():
    """Test sidechain reconstruction for a simple residue (CYS)."""
    # Generate a simple CYS-containing peptide
    pdb_content = generate_pdb_content(sequence_str="AC", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    # CYS chi1 targets
    res_id = 2
    target_chi1 = 60.0
    rotamer = {'chi1': [target_chi1]}
    
    # Capture original coordinates
    orig_coords = peptide.coord[peptide.res_id == res_id].copy()
    
    # Run reconstruction
    reconstruct_sidechain(peptide, res_id, rotamer)
    
    # Check that coordinates changed
    new_coords = peptide.coord[peptide.res_id == res_id]
    assert not np.array_equal(orig_coords, new_coords), "Sidechain coordinates did not change after reconstruction."
    
    # Verify new Chi1
    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    sg_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "SG")].coord[0]
    
    # Biotite dihedral returns radians
    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, sg_coord))
    
    # Wrap to match target_chi1 (usually -180 to 180)
    if calc_chi1 < 0: calc_chi1 += 360
    if target_chi1 < 0: target_chi1 += 360
    
    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1

def test_reconstruct_sidechain_branched_val():
    """Test sidechain reconstruction for branched residue (VAL)."""
    pdb_content = generate_pdb_content(sequence_str="AV", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    res_id = 2
    target_chi1 = 180.0
    rotamer = {'chi1': [target_chi1]}
    
    reconstruct_sidechain(peptide, res_id, rotamer)
    
    # Valine has CG1 and CG2. Chi1 is N-CA-CB-CG1.
    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    cg1_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CG1")].coord[0]
    
    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, cg1_coord))
    
    if calc_chi1 < 0: calc_chi1 += 360
    if target_chi1 < 0: target_chi1 += 360
    
    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1

def test_reconstruct_sidechain_branched_leu():
    """Test sidechain reconstruction for residue with CG (LEU)."""
    pdb_content = generate_pdb_content(sequence_str="AL", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    res_id = 2
    target_chi1 = 60.0 # Common staggered
    rotamer = {'chi1': [target_chi1]}
    
    reconstruct_sidechain(peptide, res_id, rotamer)
    
    n_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")].coord[0]
    ca_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")].coord[0]
    cb_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")].coord[0]
    cg_coord = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CG")].coord[0]
    
    calc_chi1 = np.rad2deg(struc.dihedral(n_coord, ca_coord, cb_coord, cg_coord))
    
    if calc_chi1 < 0: calc_chi1 += 360
    if target_chi1 < 0: target_chi1 += 360
    
    assert pytest.approx(calc_chi1, abs=0.5) == target_chi1

def test_reconstruct_sidechain_invalid_residue():
    """Test error handling for non-existent residue."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    with pytest.raises(ValueError):
        reconstruct_sidechain(peptide, 999, {'chi1': [60.0]})

def test_reconstruct_sidechain_missing_backbone():
    """Test graceful failure when backbone atoms are missing."""
    pdb_content = generate_pdb_content(sequence_str="A", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    # Delete CA atom
    peptide = peptide[peptide.atom_name != "CA"]
    
    # Should log warning and return None
    result = reconstruct_sidechain(peptide, 1, {'chi1': [60.0]})
    assert result is None

def test_calculate_angle():
    """Test the calculate_angle utility."""
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    angle = calculate_angle(p1, p2, p3)
    assert pytest.approx(angle) == 90.0

def test_calculate_dihedral_angle():
    """Test the calculate_dihedral_angle utility."""
    p1 = np.array([1.0, 1.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.0])
    p4 = np.array([1.0, 0.0, 0.0])
    # This should be a 180 degree planar trans-like dihedral
    # Wait, let's check the vectors.
    # v1 = p1-p2 = [1, 0, 0]
    # v2 = p2-p3 = [0, 1, 0]
    # v3 = p3-p4 = [-1, 0, 0]
    # Normal n1 = v1 x v2 = [0, 0, 1]
    # Normal n2 = v2 x v3 = [0, 0, 1]
    # Angle between normals is 0?
    dihedral = calculate_dihedral_angle(p1, p2, p3, p4)
    # Actually, let's use a simpler 90 degree one.
    p1 = np.array([1.0, 1.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.0])
    p4 = np.array([0.0, 0.0, 1.0])
    # v1 = [1, 0, 0], v2 = [0, 1, 0] -> n1 = [0, 0, 1]
    # v2 = [0, 1, 0], v3 = [0, 0, -1] -> n2 = [1, 0, 0]
    # n1 dot n2 = 0 -> 90 degrees
    dihedral = calculate_dihedral_angle(p1, p2, p3, p4)
    assert pytest.approx(abs(dihedral)) == 90.0

def test_reconstruct_sidechain_unknown_residue(caplog, monkeypatch):
    """Test warning for unknown residue."""
    import biotite.structure as struc
    import biotite.structure.info as info
    
    def mock_residue(res_name):
        raise KeyError(f"Unknown residue {res_name}")
        
    monkeypatch.setattr(info, "residue", mock_residue)
    
    # Create a dummy structure
    atom = struc.Atom(
        res_id=1, res_name="XXX", atom_name="CA", coord=[0,0,0], chain_id="A"
    )
    peptide = struc.array([atom])
    peptide += struc.array([
        struc.Atom(res_id=1, res_name="XXX", atom_name="N", coord=[-1,0,0], chain_id="A"),
        struc.Atom(res_id=1, res_name="XXX", atom_name="C", coord=[0,1,0], chain_id="A")
    ])
    
    reconstruct_sidechain(peptide, 1, {'chi1': [60.0]})
    assert "Unknown residue XXX" in caplog.text

def test_reconstruct_sidechain_no_chi1():
    """Test that it returns early if chi1 is missing."""
    pdb_content = generate_pdb_content(sequence_str="AC", conformation="alpha")
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    peptide = pdb_file.get_structure(model=1)
    
    orig_coords = peptide.coord.copy()
    reconstruct_sidechain(peptide, 2, {'chi2': [180.0]}) # No chi1
    assert np.array_equal(orig_coords, peptide.coord)

def test_position_atom_3d_consistency():
    """Test position_atom_3d_from_internal_coords returns consistent values."""
    from synth_pdb.geometry import position_atom_3d_from_internal_coords
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.5, 0.0, 0.0])
    p3 = np.array([2.0, 1.0, 0.0])
    
    # Internal coords for a regular geometry
    b_len = 1.5
    b_ang = 110.0
    dihedral = 180.0
    
    pos = position_atom_3d_from_internal_coords(p1, p2, p3, b_len, b_ang, dihedral)
    assert pos.shape == (3,)
    assert not np.any(np.isnan(pos))
