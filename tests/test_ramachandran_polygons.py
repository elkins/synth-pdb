
import pytest
from synth_pdb.validator import PDBValidator
from synth_pdb.data import RAMACHANDRAN_POLYGONS

class TestRamachandranPolygons:
    """
    Unit tests for the High-Resolution Ramachandran Polygons.
    Tests the geometry logic directly without needing full PDB generation.
    """

    def test_alpha_helix_is_favored_general(self):
        """Standard Alpha Helix (-60, -45) should be in General/Favored."""
        validator = PDBValidator(parsed_atoms=[]) # Dummy init
        
        # Point: (Phi, Psi)
        point = (-60.0, -47.0) 
        
        # Check against General -> Favored
        polygons = RAMACHANDRAN_POLYGONS["General"]["Favored"]
        is_inside = False
        for poly in polygons:
            if validator._is_point_in_polygon(point, poly):
                is_inside = True
                break
        
        assert is_inside, f"Alpha Helix {point} should be in General Favored region."

    def test_beta_sheet_is_favored_general(self):
        """Standard Beta Sheet (-120, 120) should be in General/Favored."""
        validator = PDBValidator(parsed_atoms=[])
        point = (-135.0, 135.0) # Typical Beta
        
        polygons = RAMACHANDRAN_POLYGONS["General"]["Favored"]
        is_inside = any(validator._is_point_in_polygon(point, poly) for poly in polygons)
        assert is_inside, f"Beta Sheet {point} should be in General Favored region."

    def test_left_handed_alpha_is_forbidden_general(self):
        """Left-handed Alpha (60, 45) should NOT be in General/Favored."""
        validator = PDBValidator(parsed_atoms=[])
        point = (60.0, 45.0)
        
        polygons = RAMACHANDRAN_POLYGONS["General"]["Favored"]
        is_inside = any(validator._is_point_in_polygon(point, poly) for poly in polygons)
        assert not is_inside, f"Left-Handed Alpha {point} must NOT be in General Favored."

    def test_glycine_allows_left_handed_alpha(self):
        """Glycine SHOULD allow Left-handed Alpha (60, 45)."""
        validator = PDBValidator(parsed_atoms=[])
        point = (60.0, 45.0) # Alpha-L center ish
        
        polygons = RAMACHANDRAN_POLYGONS["GLY"]["Favored"]
        is_inside = any(validator._is_point_in_polygon(point, poly) for poly in polygons)
        assert is_inside, f"Glycine should allow positive Phi/Psi {point}."

    def test_proline_is_restricted(self):
        """Proline should disallow generic negative Phi like -120."""
        validator = PDBValidator(parsed_atoms=[])
        
        # Allowed Proline: (-60, -30) or (-60, 150)
        good_point = (-60.0, -30.0)
        polygons = RAMACHANDRAN_POLYGONS["PRO"]["Favored"]
        assert any(validator._is_point_in_polygon(good_point, p) for p in polygons), \
            "Proline should allow (-60, -30)"
            
        # Forbidden Proline: (-120, 120) -> Standard Beta but Proline phi can't do -120
        bad_point = (-120.0, 120.0)
        assert not any(validator._is_point_in_polygon(bad_point, p) for p in polygons), \
            "Proline should NOT allow Phi=-120"

    def test_boundary_conditions(self):
        """Test points exactly on vertices or edges (simple check)."""
        validator = PDBValidator(parsed_atoms=[])
        # A vertex of the Alpha region: (-70, -30)
        point = (-70.0, -30.0) 
        # Note: Point-in-polygon logic for boundaries can be tricky.
        # Ray casting algorithm usually counts boundary as inside or outside depending on implementation.
        # Here we just want to ensure it doesn't crash.
        polygons = RAMACHANDRAN_POLYGONS["General"]["Favored"]
        # Just ensure it runs
        validator._is_point_in_polygon(point, polygons[0])
