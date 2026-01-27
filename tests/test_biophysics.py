
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.biophysics import find_salt_bridges

# Try to import the module (will fail initially)
try:
    from synth_pdb import biophysics
except ImportError:
    biophysics = None

def create_his_peptide():
    """Creates a simple ALA-HIS-ALA peptide."""
    # Mocking structure with minimal atoms for renaming test
    atoms = struc.AtomArray(3)
    atoms.res_name = np.array(["ALA", "HIS", "ALA"])
    atoms.res_id = np.array([1, 2, 3])
    atoms.chain_id = np.array(["A", "A", "A"])
    atoms.atom_name = np.array(["CA", "CA", "CA"])
    return atoms

class TestBiophysics:

    def test_module_exists(self):
        if biophysics is None:
            pytest.fail("synth_pdb.biophysics module not found")

    def test_ph_titration_low_ph(self):
        """Test HIS -> HIP conversion at low pH."""
        if biophysics is None:
            pytest.skip("Module not implemented")
            
        atoms = create_his_peptide()
        
        # Apply pH 5.0 (Acidic)
        titrated = biophysics.apply_ph_titration(atoms, ph=5.0)
        
        # Check renaming
        assert titrated.res_name[1] == "HIP"
        # Others untouched
        assert titrated.res_name[0] == "ALA"

    def test_ph_titration_high_ph(self):
        """Test HIS -> HIE/HID conversion at physiological pH."""
        if biophysics is None:
            pytest.skip("Module not implemented")
            
        atoms = create_his_peptide()
        
        # Apply pH 7.4
        titrated = biophysics.apply_ph_titration(atoms, ph=7.4)
        
        # Should be HIE or HID, or remain HIS if standard.
        # Ideally we want explicit states.
        # Let's assert it's NOT HIP.
        assert titrated.res_name[1] in ["HIE", "HID", "HIS"]
        assert titrated.res_name[1] != "HIP"

    def test_cap_termini_functionality(self):
        """Test ACE and NME addition."""
        if biophysics is None:
            pytest.skip("Module not implemented")

        # Create 2-residue peptide with backbone atoms
        # ALA-ALA
        # We need N, CA, C coords to avoid IndexError in biophysics.py
        # Using dummy coords
        n1 = struc.Atom([0,0,0], atom_name="N", res_id=1, res_name="ALA", element="N")
        ca1 = struc.Atom([1.4,0,0], atom_name="CA", res_id=1, res_name="ALA", element="C")
        c1 = struc.Atom([2.0,1.2,0], atom_name="C", res_id=1, res_name="ALA", element="C")
        
        n2 = struc.Atom([2.8,1.2,0], atom_name="N", res_id=2, res_name="ALA", element="N")
        ca2 = struc.Atom([3.5,2.4,0], atom_name="CA", res_id=2, res_name="ALA", element="C")
        c2 = struc.Atom([4.5,2.4,1.2], atom_name="C", res_id=2, res_name="ALA", element="C")
        
        atoms = struc.array([n1, ca1, c1, n2, ca2, c2])
        atoms.chain_id = np.array(["A"]*6)
        
        capped = biophysics.cap_termini(atoms)
        
        # Check for ACE
        assert "ACE" in capped.res_name
        ace_atoms = capped[capped.res_name == "ACE"]
        assert len(ace_atoms) == 3 # C, O, CH3
        # Check ACE geometry exists (not 0,0,0 unless inputs were)
        # Inputs were close to 0 but distinct.
        
        # Check for NME
        assert "NME" in capped.res_name
        nme_atoms = capped[capped.res_name == "NME"]
        assert len(nme_atoms) == 2 # N, CH3
        
        # Check total length
        # original 6 + 3 ACE + 2 NME = 11
        assert len(capped) == 11

def test_find_salt_bridges_simple():
    """
    Test detection of a simple salt bridge between ASP and LYS.
    """
    # Create simple structure: ASP 1 and LYS 2
    # ASP OD1 - LYS NZ distance ~3.5 A
    
    atoms = []
    # ASP 1
    atoms.append(struc.Atom([0, 0, 0], res_id=1, res_name="ASP", atom_name="OD1", element="O"))
    atoms.append(struc.Atom([1, 0, 0], res_id=1, res_name="ASP", atom_name="OD2", element="O"))
    atoms.append(struc.Atom([0, 1, 0], res_id=1, res_name="ASP", atom_name="CG", element="C")) 
    
    # LYS 2
    # NZ within 3.5 of OD1
    atoms.append(struc.Atom([3.5, 0, 0], res_id=2, res_name="LYS", atom_name="NZ", element="N"))
    atoms.append(struc.Atom([4.5, 0, 0], res_id=2, res_name="LYS", atom_name="CE", element="C"))
    
    structure = struc.array(atoms)
    
    bridges = find_salt_bridges(structure, cutoff=4.0)
    
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge["res_ia"] == 1
    assert bridge["res_ib"] == 2
    assert "NZ" in [bridge["atom_a"], bridge["atom_b"]]
    # It might pick OD1 or OD2 depending on which is closer (TDD logic chooses closest)
    assert any(x in [bridge["atom_a"], bridge["atom_b"]] for x in ["OD1", "OD2"])

def test_find_salt_bridges_cutoff():
    """
    Test that salt bridges outside the cutoff are ignored.
    """
    atoms = []
    # ASP 1
    atoms.append(struc.Atom([0, 0, 0], res_id=1, res_name="ASP", atom_name="OD1", element="O"))
    # LYS 2 (5.0 A away)
    atoms.append(struc.Atom([5.0, 0, 0], res_id=2, res_name="LYS", atom_name="NZ", element="N"))
    
    structure = struc.array(atoms)
    bridges = find_salt_bridges(structure, cutoff=4.0)
    assert len(bridges) == 0

def test_find_salt_bridges_arg_glu():
    """
    Test detection between GLU and ARG.
    """
    atoms = []
    # GLU 5 - OE1
    atoms.append(struc.Atom([10, 10, 10], res_id=5, res_name="GLU", atom_name="OE1", element="O"))
    # ARG 10 - NH1 (within 3 A)
    atoms.append(struc.Atom([10, 10, 13], res_id=10, res_name="ARG", atom_name="NH1", element="N"))
    
    structure = struc.array(atoms)
    bridges = find_salt_bridges(structure, cutoff=4.0)
    
    assert len(bridges) == 1
    assert bridges[0]["res_ia"] == 5
    assert bridges[0]["res_ib"] == 10

def test_find_salt_bridges_ignore_same_residue():
    """
    Ensure we don't detect fake bridges within the same residue.
    """
    atoms = []
    # HIS 1 (if we misconfigure, it might think ND1 and NE2 are a bridge)
    atoms.append(struc.Atom([0, 0, 0], res_id=1, res_name="HIS", atom_name="ND1", element="N"))
    atoms.append(struc.Atom([1, 0, 0], res_id=1, res_name="HIS", atom_name="NE2", element="N"))
    
    structure = struc.array(atoms)
    bridges = find_salt_bridges(structure, cutoff=4.0)
    assert len(bridges) == 0

def test_apply_ph_titration():
    """Test Histidine protonation states at different pH."""
    from synth_pdb.biophysics import apply_ph_titration
    
    atoms = [
        struc.Atom([0,0,0], res_id=1, res_name="HIS", atom_name="CA", element="C"),
        struc.Atom([0,0,0], res_id=2, res_name="HIS", atom_name="CA", element="C")
    ]
    structure = struc.array(atoms)
    
    # pH 5.0 -> HIP
    structure = apply_ph_titration(structure, ph=5.0)
    assert np.all(structure.res_name == "HIP")
    
    # pH 8.0 -> HIE or HID
    structure.res_name[:] = "HIS"
    structure = apply_ph_titration(structure, ph=8.0)
    assert np.all(np.isin(structure.res_name, ["HIE", "HID"]))

def test_cap_termini():
    """Test that capping adds ACE and NME residues."""
    from synth_pdb.biophysics import cap_termini
    from synth_pdb.generator import generate_pdb_content
    import io
    import biotite.structure.io.pdb as pdb
    
    # Generate simple peptide
    pdb_content = generate_pdb_content(length=5)
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    structure = pdb_file.get_structure(model=1)
    
    # Original length
    original_res_ids = sorted(list(set(structure.res_id)))
    
    # Cap it
    capped = cap_termini(structure)
    capped_res_names = list(set(capped.res_name))
    
    assert "ACE" in capped_res_names
    assert "NME" in capped_res_names
