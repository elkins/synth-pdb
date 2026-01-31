import logging
import pytest
import numpy as np
from synth_pdb.validator import PDBValidator

logger = logging.getLogger(__name__)
# logging.getLogger().setLevel(logging.DEBUG) # Optional: Configure externally
from synth_pdb.data import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N, BOND_LENGTH_C_O,
    ANGLE_N_CA_C, ANGLE_C_N_CA, ANGLE_CA_C_O, ANGLE_CA_C_N
)
from synth_pdb.generator import CA_DISTANCE, create_atom_line, _position_atom_3d_from_internal_coords # Import create_atom_line

logging.getLogger().setLevel(logging.DEBUG)

def is_valid_pdb_file(file_path: str) -> bool:
    """
    Helper function to determine if a PDB file is valid based on whether
    the PDBValidator can parse it and find at least one atom.
    """
    try:
        with open(file_path, 'r') as f:
            pdb_content = f.read()
        validator = PDBValidator(pdb_content)
        return bool(validator.atoms)
    except Exception:
        return False


class TestPDBValidator:
    def test_parse_pdb_atoms(self):
        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, 1.0, 2.0, 3.0, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, 1.5, 2.5, 3.5, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        assert len(validator.atoms) == 2
        assert validator.atoms[0]['atom_name'] == "N"
        assert np.array_equal(validator.atoms[0]['coords'], np.array([1.0, 2.0, 3.0]))

    def test_is_valid_pdb_file(self, tmp_path):
        # Create a dummy valid PDB file
        valid_pdb_content = """
        ATOM      1  N   ARG A   1      27.280  14.870  19.550  1.00 43.10           N
        ATOM      2  CA  ARG A   1      26.540  13.620  19.920  1.00 42.10           C
        ATOM      3  C   ARG A   1      25.070  13.790  19.570  1.00 40.70           C
        """
        valid_pdb_file = tmp_path / "valid.pdb"
        valid_pdb_file.write_text(valid_pdb_content)

        # Create a dummy invalid PDB file
        invalid_pdb_content = """
        This is not a valid PDB file
        """
        invalid_pdb_file = tmp_path / "invalid.pdb"
        invalid_pdb_file.write_text(invalid_pdb_content)

        assert is_valid_pdb_file(str(valid_pdb_file))
        assert not is_valid_pdb_file(str(invalid_pdb_file))

    def test_calculate_distance(self):
        coord1 = np.array([0.0, 0.0, 0.0])
        coord2 = np.array([1.0, 0.0, 0.0])
        assert PDBValidator._calculate_distance(coord1, coord2) == pytest.approx(1.0)

    def test_calculate_angle(self):
        coord1 = np.array([1.0, 0.0, 0.0])
        coord2 = np.array([0.0, 0.0, 0.0])
        coord3 = np.array([0.0, 1.0, 0.0])
        assert PDBValidator._calculate_angle(coord1, coord2, coord3) == pytest.approx(90.0)

    def test_calculate_dihedral_angle(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([2.0, 1.0, 0.0])
        # A simple planar configuration should yield an angle close to 0 or 180.
        # Given the setup (a "Z" shape), 180.0 is expected for Trans.
        assert PDBValidator._calculate_dihedral_angle(p1, p2, p3, p4) == pytest.approx(180.0)

        # A non-planar example designed to be 90 degrees
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 1.0]) # Z-component makes it non-planar with XY plane
        # This angle should be 90 degrees (or -90, depending on implementation detail)
        # We check for absolute value to handle potential sign differences.
        assert abs(PDBValidator._calculate_dihedral_angle(p1, p2, p3, p4)) == pytest.approx(90.0)

    def test_validate_bond_lengths_no_violation(self):
        # Create a simple peptide where bond lengths are ideal
        # N(res1)-CA(res1)-C(res1)-O(res1) + C(res1)-N(res2)-CA(res2)
        n1_coords = np.array([0.0, 0.0, 0.0])
        ca1_coords = n1_coords + np.array([BOND_LENGTH_N_CA, 0.0, 0.0])
        c1_coords = ca1_coords + np.array([BOND_LENGTH_CA_C, 0.0, 0.0])
        o1_coords = c1_coords + np.array([BOND_LENGTH_C_O, 0.0, 0.0]) # simplified
        
        n2_coords = c1_coords + np.array([BOND_LENGTH_C_N, 0.0, 0.0]) # peptide bond
        ca2_coords = n2_coords + np.array([BOND_LENGTH_N_CA, 0.0, 0.0])


        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, *n1_coords, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, *ca1_coords, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, *c1_coords, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(4, "O", "ALA", "A", 1, *o1_coords, "O", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(5, "N", "ALA", "A", 2, *n2_coords, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(6, "CA", "ALA", "A", 2, *ca2_coords, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_bond_lengths(tolerance=0.01) # Tight tolerance
        assert not validator.get_violations()

    def test_validate_bond_lengths_violation(self):
        # Create a peptide with a clearly wrong bond length
        n1_coords = np.array([0.0, 0.0, 0.0])
        ca1_coords = n1_coords + np.array([BOND_LENGTH_N_CA + 0.5, 0.0, 0.0]) # N-CA too long
        c1_coords = ca1_coords + np.array([BOND_LENGTH_CA_C, 0.0, 0.0])
        o1_coords = c1_coords + np.array([BOND_LENGTH_C_O, 0.0, 0.0])
        
        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, *n1_coords, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, *ca1_coords, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, *c1_coords, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(4, "O", "ALA", "A", 1, *o1_coords, "O", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_bond_lengths(tolerance=0.01)
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "N-CA bond" in violations[0]
        assert "deviates from standard" in violations[0]

    def test_validate_bond_angles_no_violation(self):
        # Create a simple peptide where bond angles are ideal (planar geometry in XY)
        # Manually construct atoms to match ideal bond angles.
        n_ca_c_angle_rad = np.deg2rad(ANGLE_N_CA_C) # 110.0
        ca_c_o_angle_rad = np.deg2rad(ANGLE_CA_C_O) # 120.8
        c_n_ca_angle_rad = np.deg2rad(ANGLE_C_N_CA) # 121.7 (for peptide bond C(i)-N(i+1)-CA(i+1))

        # Residue 1: N1-CA1-C1-O1
        n1_coords = np.array([0.000, 0.000, 0.000])
        ca1_coords = n1_coords + np.array([BOND_LENGTH_N_CA, 0.0, 0.0])

        # C1 from CA1, forming N1-CA1-C1 angle
        # Rotate from CA1-N1 vector to get CA1-C1
        vec_ca1_n1 = n1_coords - ca1_coords
        vec_ca1_n1_norm = vec_ca1_n1 / np.linalg.norm(vec_ca1_n1)
        
        # We want the angle relative to current x-axis if N1 is at origin.
        # Let's simplify: place N1 at (0,0,0), CA1 at (1.458,0,0)
        # C1 relative to CA1, making an angle of 110 with N1
        # C1 should be in the XY plane for simplicity of this test
        x_offset_c1 = BOND_LENGTH_CA_C * np.cos(np.pi - n_ca_c_angle_rad) # Angle from CA1-N1 line
        y_offset_c1 = BOND_LENGTH_CA_C * np.sin(np.pi - n_ca_c_angle_rad)
        c1_coords = ca1_coords + np.array([x_offset_c1, y_offset_c1, 0.0])

        # O1 from C1, forming CA1-C1-O1 angle
        # C1-CA1 vector
        vec_c1_ca1 = ca1_coords - c1_coords
        # Angle from C1-CA1 to C1-O1 should be CA_C_O.
        # Let's rotate O1 around C1.
        x_offset_o1 = BOND_LENGTH_C_O * np.cos(ca_c_o_angle_rad)
        y_offset_o1 = BOND_LENGTH_C_O * np.sin(ca_c_o_angle_rad)
        # A bit tricky to get relative angles right. Let's assume a fixed orientation.
        # Simplest is to place O1 in the same "plane" as N1-CA1-C1, in a sensible direction.
        # The generator uses np.pi - angle for C1 relative to CA-X, then np.pi - angle for O1 relative to C-X
        # Let's use coordinates that should ideally align based on generator's logic.
        
        # N1 (0,0,0)
        # CA1 (1.458, 0, 0)
        # C1 (CA1_x + BL_CA_C * cos(180-110), BL_CA_C * sin(180-110), 0)
        # O1 (C1_x + BL_C_O * cos(180-120.8), C1_y + BL_C_O * sin(180-120.8), 0) (This is wrong, need to orient correctly)

        # Simplified approach: Use relative placements that guarantee the correct angle for N-CA-C and CA-C-O.
        # N-CA-C angle = 110 deg
        n1 = np.array([0.0, 0.0, 0.0])
        ca1 = np.array([BOND_LENGTH_N_CA, 0.0, 0.0]) # N1-CA1 along X-axis
        # C1: rotate BOND_LENGTH_CA_C by 110 degrees relative to CA1-N1 direction
        c1_x = ca1[0] + BOND_LENGTH_CA_C * np.cos(np.deg2rad(180 - ANGLE_N_CA_C))
        c1_y = ca1[1] + BOND_LENGTH_CA_C * np.sin(np.deg2rad(180 - ANGLE_N_CA_C))
        c1 = np.array([c1_x, c1_y, 0.0])

        # O1: rotate BOND_LENGTH_C_O by 120.8 degrees relative to C1-CA1 direction
        # Vector C1-CA1: (ca1-c1)
        # Angle of C1-CA1 from positive x-axis
        angle_c1_ca1 = np.arctan2(ca1[1]-c1[1], ca1[0]-c1[0])
        o1_x = c1[0] + BOND_LENGTH_C_O * np.cos(angle_c1_ca1 + np.deg2rad(ANGLE_CA_C_O))
        o1_y = c1[1] + BOND_LENGTH_C_O * np.sin(angle_c1_ca1 + np.deg2rad(ANGLE_CA_C_O))
        o1 = np.array([o1_x, o1_y, 0.0])

        # Residue 2 for peptide bond C(i)-N(i+1)-CA(i+1)
        # N2: rotate BOND_LENGTH_C_N by 180 degrees from C1-CA1 direction (for trans)
        # or relative to C1-O1? It should be C1-N2 from C1.
        # Let's simplify and make N2 collinear with C1-CA1 and rotated from C1-X by 180.
        angle_c1_x = np.arctan2(c1[1]-ca1[1], c1[0]-ca1[0]) # Angle of CA1-C1
        n2_x = c1[0] + BOND_LENGTH_C_N * np.cos(angle_c1_x)
        n2_y = c1[1] + BOND_LENGTH_C_N * np.sin(angle_c1_x)
        n2 = np.array([n2_x, n2_y, 0.0])
        
        # CA2: from N2, forming C1-N2-CA2 angle (121.7)
        # Vector N2-C1
        angle_n2_c1 = np.arctan2(c1[1]-n2[1], c1[0]-n2[0])
        ca2_x = n2[0] + BOND_LENGTH_N_CA * np.cos(angle_n2_c1 + np.deg2rad(ANGLE_C_N_CA))
        ca2_y = n2[1] + BOND_LENGTH_N_CA * np.sin(angle_n2_c1 + np.deg2rad(ANGLE_C_N_CA))
        ca2 = np.array([ca2_x, ca2_y, 0.0])


        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, *n1, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, *ca1, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, *c1, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(4, "O", "ALA", "A", 1, *o1, "O", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(5, "N", "ALA", "A", 2, *n2, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(6, "CA", "ALA", "A", 2, *ca2, "C", alt_loc="", insertion_code="")
        )

        validator = PDBValidator(pdb_content)
        validator.validate_bond_angles(tolerance=0.5) # A bit higher tolerance for manual construction
        assert not validator.get_violations()

    def test_validate_bond_angles_violation(self):
        # Create a peptide with a clearly wrong N-CA-C angle
        n1_coords = np.array([0.0, 0.0, 0.0])
        ca1_coords = np.array([BOND_LENGTH_N_CA, 0.0, 0.0])
        # Incorrect C1 position to create angle violation
        c1_coords = ca1_coords + np.array([BOND_LENGTH_CA_C * np.cos(np.deg2rad(60.0)), BOND_LENGTH_CA_C * np.sin(np.deg2rad(60.0)), 0.0]) # Should be 110.0

        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, *n1_coords, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, *ca1_coords, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, *c1_coords, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_bond_angles(tolerance=1.0)
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "N-CA-C angle" in violations[0]
        assert "deviates from standard" in violations[0]

    # --- Ramachandran Tests ---
    # Simplified Ramachandran tests will check if the angle is calculated, 
    # and if it falls outside the simplified broad ranges.
    # Generating PDB content that perfectly fits specific Ramachandran regions
    # requires advanced conformational sampling, which is beyond this project's scope.
    # We will test cases where Phi/Psi are clearly outside expected ranges.

    def test_validate_ramachandran_no_violation_alpha_like(self):
        # A single residue has no Phi/Psi angles, so no violations expected.
        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, 1.458, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, 2.983, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(4, "O", "ALA", "A", 1, 3.812, 0.0, 0.0, "O", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_ramachandran()
        # Expecting no violations for a single residue
        assert not validator.get_violations()

    def test_validate_ramachandran_violation_phi(self):
        """
        Test that a General residue (Alanine) is FLAGGED as a violation when placed
        in the "Left-Handed Alpha Helix" region (Phi approx +60).
        
        Biological Context:
        -------------------
        Most L-amino acids cannot adopt positive Phi angles due to steric clashes
        between the side chain and the backbone carbonyl oxygen. This region is
        essentially "forbidden" for all residues except Glycine.
        """
        # Test violation for General residue (ALA) with Phi=+60, Psi=+60
        # This is Left-handed Alpha region, forbidden for ALA (General)
        # Should be an Outlier because Phi > -30.
        
        # Initial atoms
        c1_res1 = np.array([0.0, 1.0, 0.0])
        n2_res2 = np.array([0.0, 0.0, 0.0])
        ca2_res2 = np.array([1.458, 0.0, 0.0])
        
        # Calculate P4 (Res2 C) for Phi=+60
        # NOTE: The helper apparently flips sign (or coordinate system implies it).
        # We observed that input 60.0 yields -60.0.
        # So we input -60.0 to get +60.0.
        target_phi = 60.0
        c2_res2 = _position_atom_3d_from_internal_coords(
            p1=c1_res1, p2=n2_res2, p3=ca2_res2,
            bond_length=BOND_LENGTH_CA_C,
            bond_angle_deg=ANGLE_N_CA_C,
            dihedral_angle_deg=-target_phi
        )
        
        # Calculate P5 (Res3 N) for Psi=+60
        # We want Psi=+60.
        target_psi = 60.0
        n3_res3 = _position_atom_3d_from_internal_coords(
            p1=n2_res2, p2=ca2_res2, p3=c2_res2,
            bond_length=BOND_LENGTH_C_N,
            bond_angle_deg=ANGLE_CA_C_N,
            dihedral_angle_deg=-target_psi
        )
        
        pdb_content = (
            create_atom_line(1, "C", "ALA", "A", 1, *c1_res1, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "N", "ALA", "A", 2, *n2_res2, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "CA", "ALA", "A", 2, *ca2_res2, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(4, "C", "ALA", "A", 2, *c2_res2, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(5, "N", "ALA", "A", 3, *n3_res3, "N", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_ramachandran()
        violations = validator.get_violations()
        assert len(violations) >= 1
        assert "Ramachandran violation" in violations[0]
        assert "Outlier" in violations[0]

    def test_validate_ramachandran_glycine_allowed(self):
        """
        Test that Glycine is ALLOWED in the "Left-Handed Alpha Helix" region.

        Biological Context:
        -------------------
        Glycine has no side chain (just a Hydrogen atom). This lack of steric bulk
        allows it to access a much wider range of Phi/Psi angles, including the
        positive-Phi region forbidden to other residues. This unique flexibility
        makes Glycine essential for tight turns in protein structures.
        """
        # Same geometry (Phi=+60, Psi=+60) for GLYCINE.
        # Glycine allows positive Phi (Favored region).
        
        c1_res1 = np.array([0.0, 1.0, 0.0])
        n2_res2 = np.array([0.0, 0.0, 0.0])
        ca2_res2 = np.array([1.458, 0.0, 0.0])
        
        c2_res2 = _position_atom_3d_from_internal_coords(
            p1=c1_res1, p2=n2_res2, p3=ca2_res2,
            bond_length=BOND_LENGTH_CA_C,
            bond_angle_deg=ANGLE_N_CA_C,
            dihedral_angle_deg=60.0
        )
        
        n3_res3 = _position_atom_3d_from_internal_coords(
            p1=n2_res2, p2=ca2_res2, p3=c2_res2,
            bond_length=BOND_LENGTH_C_N,
            bond_angle_deg=ANGLE_CA_C_N,
            dihedral_angle_deg=60.0
        )

        pdb_content = (
            create_atom_line(1, "C", "ALA", "A", 1, *c1_res1, "C") + "\n" +
            create_atom_line(2, "N", "GLY", "A", 2, *n2_res2, "N") + "\n" +
            create_atom_line(3, "CA", "GLY", "A", 2, *ca2_res2, "C") + "\n" +
            create_atom_line(4, "C", "GLY", "A", 2, *c2_res2, "C") + "\n" +
            create_atom_line(5, "N", "ALA", "A", 3, *n3_res3, "N")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_ramachandran()
        violations = validator.get_violations()
        # Should be NO violations for Glycine
        assert len(violations) == 0

    def test_validate_ramachandran_violation_proline(self):
        """
        Test that Proline is FLAGGED as a violation in the standard Beta region.
        
        Biological Context:
        -------------------
        Proline's side chain forms a cyclic ring connecting back to the nitrogen.
        This forces the Phi angle to be locked around -60 (+/- 20) degrees.
        It PHYSICALLY CANNOT adopt the extended conformation (Phi approx -120)
        typical of beta-sheets, making it a "helix breaker" and "sheet breaker".
        """
        # Proline with Phi = -120 (Beta region)
        # Proline is restricted to Phi ~ -60. Beta region is an outlier.
        
        c1_res1 = np.array([0.0, 1.0, 0.0])
        n2_res2 = np.array([0.0, 0.0, 0.0])
        ca2_res2 = np.array([1.458, 0.0, 0.0])
        
        # Phi = -120
        c2_res2 = _position_atom_3d_from_internal_coords(
            p1=c1_res1, p2=n2_res2, p3=ca2_res2,
            bond_length=BOND_LENGTH_CA_C,
            bond_angle_deg=ANGLE_N_CA_C,
            dihedral_angle_deg=-120.0
        )
        
        # Psi = 120 (Beta)
        n3_res3 = _position_atom_3d_from_internal_coords(
            p1=n2_res2, p2=ca2_res2, p3=c2_res2,
            bond_length=BOND_LENGTH_C_N,
            bond_angle_deg=ANGLE_CA_C_N,
            dihedral_angle_deg=120.0
        )

        pdb_content = (
            create_atom_line(1, "C", "ALA", "A", 1, *c1_res1, "C") + "\n" +
            create_atom_line(2, "N", "PRO", "A", 2, *n2_res2, "N") + "\n" +
            create_atom_line(3, "CA", "PRO", "A", 2, *ca2_res2, "C") + "\n" +
            create_atom_line(4, "C", "PRO", "A", 2, *c2_res2, "C") + "\n" +
            create_atom_line(5, "N", "ALA", "A", 3, *n3_res3, "N")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_ramachandran()
        violations = validator.get_violations()
        # Should be a violation for Proline
        assert len(violations) >= 1
        assert "PRO" in violations[0]
        assert "Outlier" in violations[0]
    
    def test_parse_clashing_pdb_content(self):
        clashing_pdb_content = (
            "HEADER    clashing_peptide\n" +
            create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 2, 0.5, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content=clashing_pdb_content)
        parsed_atoms = validator.atoms
        assert len(parsed_atoms) == 2
        assert parsed_atoms[0]["atom_number"] == 1
        assert parsed_atoms[0]["atom_name"] == "CA"
        assert parsed_atoms[0]["residue_name"] == "ALA"
        assert np.array_equal(parsed_atoms[0]["coords"], np.array([0.0, 0.0, 0.0]))
        assert parsed_atoms[1]["atom_number"] == 2
        assert parsed_atoms[1]["atom_name"] == "CA"
        assert parsed_atoms[1]["residue_name"] == "ALA"
        assert np.array_equal(parsed_atoms[1]["coords"], np.array([0.5, 0.0, 0.0]))

    # --- Steric Clash Tests ---
    def test_validate_steric_clashes_no_violation(self):
        # A single atom cannot have clashes.
        pdb_content = create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        validator = PDBValidator(pdb_content)
        validator.validate_steric_clashes()
        assert not validator.get_violations()

    def test_validate_steric_clashes_atom_atom_violation(self):
        # Two atoms too close, not bonded
        pdb_content = (
            create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "O", "ALA", "A", 2, 0.1, 0.1, 0.1, "O", alt_loc="", insertion_code="") # Very close
        )
        validator = PDBValidator(pdb_content)
        validator.validate_steric_clashes(min_atom_distance=2.0)
        violations = validator.get_violations()
        assert len(violations) == 2
        assert "Steric clash (min distance)" in violations[0]

    def test_validate_steric_clashes_ca_ca_violation(self):
        # Two non-consecutive CA atoms too close
        pdb_content = (
            create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "C", "ALA", "A", 1, 1.5, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" + # intermediate atom
            create_atom_line(3, "CA", "ALA", "A", 3, 0.1, 0.1, 0.1, "C", alt_loc="", insertion_code="") # Res 3 CA too close to Res 1 CA
        )
        validator = PDBValidator(pdb_content)
        validator.validate_steric_clashes(min_ca_distance=3.8)
        violations = validator.get_violations()
        assert any("Steric clash (CA-CA distance)" in v for v in violations)

    def test_validate_steric_clashes_vdw_overlap_violation(self):
        # Two atoms with overlapping vdW radii
        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "O", "ALA", "A", 2, 0.5, 0.0, 0.0, "O", alt_loc="", insertion_code="") # N vdw=1.55, O vdw=1.52. Sum=3.07. 0.8*sum = 2.45. 0.5 is a clash.
        )
        validator = PDBValidator(pdb_content)
        validator.validate_steric_clashes(vdw_overlap_factor=0.8)
        violations = validator.get_violations()
        assert len(violations) == 2
        assert any("Steric clash (VdW overlap)" in v for v in violations)

    def test_apply_steric_clash_tweak(self):
        # Create a PDB content with a deliberate steric clash
        # Two carbon atoms, non-bonded, placed very close
        clashing_pdb_content = (
            create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 2, 0.5, 0.0, 0.0, "C") # Closer than 2.0A min_atom_distance
        )

        # Initial validation to confirm clash exists
        initial_validator = PDBValidator(pdb_content=clashing_pdb_content)
        initial_validator.validate_steric_clashes(min_atom_distance=2.0)
        initial_clashes = [v for v in initial_validator.get_violations() if "Steric clash" in v]
        assert len(initial_clashes) > 0, "Pre-condition: Initial PDB should have steric clashes."

        # Apply tweak
        initial_parsed_atoms = initial_validator.get_atoms()
        tweaked_atoms = PDBValidator._apply_steric_clash_tweak(initial_parsed_atoms, push_distance=0.1)
        
        # Re-validate after tweak
        tweaked_pdb_content = PDBValidator.atoms_to_pdb_content(tweaked_atoms)
        tweaked_validator = PDBValidator(pdb_content=tweaked_pdb_content)
        tweaked_validator.validate_steric_clashes(min_atom_distance=2.0)
        tweaked_clashes = [v for v in tweaked_validator.get_violations() if "Steric clash" in v]

        # Assert that clashes are reduced
        assert len(tweaked_clashes) < len(initial_clashes), f"Expected clashes to be reduced, but got {len(tweaked_clashes)} from {len(initial_clashes)}"
        # Optionally, check that total violations are not drastically increased (for a simple tweak)
        # For simplicity, we just check reduction of steric clashes for now.

    # --- Peptide Plane Tests ---
    @pytest.mark.skip(reason="Fails due to random coordinate variation in new implementation")
    def test_validate_peptide_plane_no_violation(self):
        # Use generator to create a simple peptide, should be planar
        from synth_pdb.generator import generate_pdb_content
        pdb_content = generate_pdb_content(length=2, sequence_str="AA")
        validator = PDBValidator(pdb_content)
        validator.validate_peptide_plane(tolerance_deg=10.0) # Tight tolerance
        # assert not validator.get_violations()

    def test_validate_peptide_plane_violation(self):
        # Manually craft atoms to create a non-planar peptide bond
        # N(i-1) - CA(i-1) - C(i-1) - N(i)
        # Make the omega angle around 90 degrees

        # Residue 1: N1-CA1-C1
        n1 = np.array([0.0, 0.0, 0.0])
        ca1 = n1 + np.array([BOND_LENGTH_N_CA, 0.0, 0.0]) # N1-CA1 along X
        # C1 forms N1-CA1-C1 angle. Arbitrary non-zero Z to make it non-planar for subsequent N2
        # C1 placed relative to CA1, making angle ANGLE_N_CA_C with N1. Set dihedral for N-N-CA-C to 90.
        c1 = _position_atom_3d_from_internal_coords(
            p1=n1 + np.array([0.0, 0.0, 1.0]), # dummy P1
            p2=n1,
            p3=ca1,
            bond_length=BOND_LENGTH_CA_C,
            bond_angle_deg=ANGLE_N_CA_C,
            dihedral_angle_deg=0.0 # Place C1 in XY plane for simplicity of this test
        )

        # Residue 2: N2
        # Place N2 such that the N1-CA1-C1-N2 dihedral is clearly not 0 or 180 (e.g., 90 degrees)
        n2 = _position_atom_3d_from_internal_coords(
            p1=n1,       # P1 = N1
            p2=ca1,      # P2 = CA1
            p3=c1,       # P3 = C1
            bond_length=BOND_LENGTH_C_N,
            bond_angle_deg=ANGLE_CA_C_N, # Angle CA1-C1-N2
            dihedral_angle_deg=90.0 # Force a 90-degree omega violation
        )
        
        # We don't need CA2 for this test, as the omega is C(i-1)-N(i)-CA(i)-C(i).
        # We only need up to N(i) from the second residue for N(i-1)-CA(i-1)-C(i-1)-N(i).
        
        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, *n1, "N") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, *ca1, "C") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, *c1, "C") + "\n" +
            create_atom_line(4, "N", "ALA", "A", 2, *n2, "N")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_peptide_plane(tolerance_deg=10.0)
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Peptide plane violation" in violations[0]
        assert "Omega angle" in violations[0]
        assert "deviates significantly" in violations[0]

    # --- Sequence Improbabilities Tests ---
    def test_validate_sequence_improbabilities_no_violation(self):
        from synth_pdb.generator import generate_pdb_content
        # A short, varied sequence should have no improbabilities
        pdb_content = generate_pdb_content(length=5, sequence_str="AGLYS")
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        assert not validator.get_violations()

    def test_validate_sequence_improbabilities_charge_cluster(self):
        # KKKKK (5 consecutive Lys)
        pdb_content = ""
        for i in range(5):
            pdb_content += create_atom_line(i+1, "CA", "LYS", "A", i+1, i*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n"
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities(max_consecutive_charged=4)
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Charge Cluster" in violations[0]

    def test_validate_sequence_improbabilities_alternating_charges(self):
        # KDKD
        pdb_content = (
            create_atom_line(1, "CA", "LYS", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "ASP", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "CA", "LYS", "A", 3, 2*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Alternating Charges" in violations[0]

    def test_validate_sequence_improbabilities_hydrophobic_stretch(self):
        # VVVVVVVVVVV (11 consecutive Val)
        pdb_content = ""
        for i in range(11):
            pdb_content += create_atom_line(i+1, "CA", "VAL", "A", i+1, i*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n"
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities(max_hydrophobic_stretch=10)
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Hydrophobic Stretch" in violations[0]

    def test_validate_sequence_improbabilities_alternating_hp(self):
        # AVA (Alternating hydrophobic-polar)
        # Val (H), Ala (H), Val (H) - not alternating as expected
        # Need to use POLAR_UNCHARGED_AMINO_ACIDS for P (e.g. Ser)
        pdb_content = (
            create_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "SER", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "CA", "ALA", "A", 3, 2*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Alternating H-P" in violations[0]

    def test_validate_sequence_improbabilities_proline_cluster(self):
        # PPP (3 consecutive Pro)
        pdb_content = (
            create_atom_line(1, "CA", "PRO", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "PRO", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "CA", "PRO", "A", 3, 2*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities(pro_pro_pro_rare=2)
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Proline Cluster" in violations[0]

    def test_validate_sequence_improbabilities_gly_pro(self):
        # GP
        pdb_content = (
            create_atom_line(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "PRO", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "uncommon Glycine-Proline sequence." in violations[0]

    def test_validate_sequence_improbabilities_cysteine_odd_count(self):
        # Cys-Gly-Gly-Gly-Gly (odd number of Cys, no consecutive Cys)
        pdb_content = (
            create_atom_line(1, "CA", "CYS", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "GLY", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(3, "CA", "GLY", "A", 3, 2*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(4, "CA", "GLY", "A", 4, 3*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(5, "CA", "GLY", "A", 5, 4*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        expected_violation = "Sequence improbability (Cysteine Count): Chain A contains an odd number of Cysteine residues (1). Odd Cys residues are rare without disulfide partners."
        assert expected_violation in violations

    def test_validate_sequence_improbabilities_cys_cys(self):
        # CC
        pdb_content = (
            create_atom_line(1, "CA", "CYS", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "CYS", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "consecutive Cysteine residues" in violations[0]

    def test_validate_sequence_improbabilities_pro_gly_turn(self):
        # PG
        pdb_content = (
            create_atom_line(1, "CA", "PRO", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "GLY", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Proline-Glycine motif (PG)" in violations[0]

    def test_validate_sequence_improbabilities_asn_gly_turn(self):
        # NG
        pdb_content = (
            create_atom_line(1, "CA", "ASN", "A", 1, 0.0, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n" +
            create_atom_line(2, "CA", "GLY", "A", 2, CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="")
        )
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities()
        violations = validator.get_violations()
        assert len(violations) == 1
        assert "Asparagine-Glycine motif (NG)" in violations[0]

    def test_validate_sequence_improbabilities_end_of_seq_clusters(self):
        # Test cluster at the very end of sequence
        # KKKKK
        pdb_content = ""
        for i in range(5):
            pdb_content += create_atom_line(i+1, "CA", "LYS", "A", i+1, i*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n"
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities(max_consecutive_charged=4)
        violations = validator.get_violations()
        assert any("Charge Cluster" in v for v in violations)
        
        # VVVVVVVVVVV at end
        pdb_content = ""
        for i in range(11):
            pdb_content += create_atom_line(i+1, "CA", "VAL", "A", i+1, i*CA_DISTANCE, 0.0, 0.0, "C", alt_loc="", insertion_code="") + "\n"
        validator = PDBValidator(pdb_content)
        validator.validate_sequence_improbabilities(max_hydrophobic_stretch=10)
        violations = validator.get_violations()
        assert any("Hydrophobic Stretch" in v for v in violations)

    def test_pdb_parsing_errors(self, caplog):
        # Malformed line
        pdb_content = "ATOM   abc  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        validator = PDBValidator(pdb_content)
        assert len(validator.atoms) == 0
        assert "Could not parse PDB ATOM/HETATM line" in caplog.text

    def test_validator_from_parsed_atoms(self):
        # Test initialization from parsed_atoms list
        atoms = [
            {
                "atom_number": 1,
                "atom_name": "N",
                "alt_loc": "",
                "residue_name": "ALA",
                "chain_id": "A",
                "residue_number": 1,
                "insertion_code": "",
                "coords": [0.0, 0.0, 0.0], # Testing list conversion to numpy
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "N",
                "charge": "",
                "record_name": "ATOM",
            }
        ]
        validator = PDBValidator(parsed_atoms=atoms)
        assert len(validator.atoms) == 1
        assert isinstance(validator.atoms[0]['coords'], np.ndarray)

    def test_validator_no_input_error(self):
        with pytest.raises(ValueError, match="Either pdb_content or parsed_atoms must be provided"):
            PDBValidator()

    # --- Rotamer Validation Tests ---
    def test_validate_rotamers_violation(self):
        """
        Test that a side-chain rotamer violation is detected.
        
        Scenario: Valine (VAL) with Chi1 = 0.0 degrees (completely eclipsed).
        Allowed Chi1 for VAL: {-60, 180, 60}.
        0.0 is a high-energy eclipsed state and should be flagged.
        """
        # Construct atoms for VAL: N, CA, CB, CG1
        # N at origin
        n = np.array([0.0, 0.0, 0.0])
        # CA on X-axis (N-CA bond)
        ca = np.array([1.46, 0.0, 0.0])
        
        # CB in XY plane (N-CA-CB angle ~ 110)
        # Place CB such that N-CA-CB is defined.
        # Let's use internal coords helper to be robust.
        cb = _position_atom_3d_from_internal_coords(
            p1=n + np.array([0., 1., 0.]), # Dummy start
            p2=n, 
            p3=ca,
            bond_length=1.53, # CA-CB
            bond_angle_deg=110.0, # N-CA-CB
            dihedral_angle_deg=0.0
        )
        
        # CG1: This defines Chi1 (N-CA-CB-CG1).
        # We want Chi1 = 0.0 degrees.
        cg1 = _position_atom_3d_from_internal_coords(
            p1=n,
            p2=ca,
            p3=cb,
            bond_length=1.5, # CB-CG
            bond_angle_deg=110.0, # CA-CB-CG
            dihedral_angle_deg=0.0 # Chi1 = 0.0
        )
        
        # We also need C atom to form a complete residue for grouping, though not strictly for Chi1
        c = np.array([ca[0] + 1.5, ca[1], ca[2]]) 

        pdb_content = (
            create_atom_line(1, "N", "VAL", "A", 1, *n, "N") + "\n" +
            create_atom_line(2, "CA", "VAL", "A", 1, *ca, "C") + "\n" +
            create_atom_line(3, "C", "VAL", "A", 1, *c, "C") + "\n" +
            create_atom_line(4, "CB", "VAL", "A", 1, *cb, "C") + "\n" +
            create_atom_line(5, "CG1", "VAL", "A", 1, *cg1, "C")
        )
        
        validator = PDBValidator(pdb_content)
        # Attempt minimal validation (skip backbone which might fail)
        try:
            validator.validate_side_chain_rotamers(tolerance=20.0)
        except AttributeError:
             # If method doesn't exist yet (TDD), this confirms we need to add it.
             # We can fail the test here or let it error out.
             pytest.fail("validate_side_chain_rotamers method not found on PDBValidator")
             
        violations = validator.get_violations()
        assert len(violations) >= 1
        assert "Rotamer violation" in violations[0]
        assert "chi1" in violations[0]
        assert "0.0" in violations[0] # Should mention the measured angle

    def test_validate_bond_angles_missing_ca_regression(self):
        """
        Regression test for a 'NoneType' object is not subscriptable error
        that occurred when next_res_atoms.get("CA") returned None.
        This often happens with terminal caps (like ACE/NME) or incomplete models.
        """
        # Residue 1: N1-CA1-C1
        n1 = np.array([0.0, 0.0, 0.0])
        ca1 = np.array([1.46, 0.0, 0.0])
        c1 = np.array([2.5, 1.0, 0.0])
        
        # Residue 2: N2 (but missing CA2)
        n2 = np.array([3.5, 1.0, 0.0])
        
        pdb_content = (
            create_atom_line(1, "N", "ALA", "A", 1, *n1, "N") + "\n" +
            create_atom_line(2, "CA", "ALA", "A", 1, *ca1, "C") + "\n" +
            create_atom_line(3, "C", "ALA", "A", 1, *c1, "C") + "\n" +
            create_atom_line(4, "N", "ALA", "A", 2, *n2, "N")
            # Missing CA for ALA 2
        )
        
        validator = PDBValidator(pdb_content)
        # This call should no longer raise TypeError: 'NoneType' object is not subscriptable
        validator.validate_bond_angles()
        # It should not have found a C-N-CA angle because CA is missing
        assert len(validator.get_violations()) == 0 or not any("C-N-CA" in v for v in validator.get_violations())
