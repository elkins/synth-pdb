"""
Tests for per-region conformation control.

Following TDD methodology - RED PHASE.
These tests should FAIL initially until we implement the per-region conformation feature.
"""
import pytest
from synth_pdb.generator import generate_pdb_content, _parse_structure_regions
from synth_pdb.validator import PDBValidator


class TestPerRegionConformations:
    """Test suite for per-region conformation control."""
    
    def test_parse_simple_structure(self):
        """Test parsing simple structure specification."""
        result = _parse_structure_regions("1-10:alpha,11-20:beta", 20)
        assert len(result) == 20
        assert all(result[i] == 'alpha' for i in range(10))
        assert all(result[i] == 'beta' for i in range(10, 20))
    
    def test_parse_three_regions(self):
        """Test parsing three regions."""
        result = _parse_structure_regions("1-5:alpha,6-10:beta,11-15:extended", 15)
        assert len(result) == 15
        assert all(result[i] == 'alpha' for i in range(5))
        assert all(result[i] == 'beta' for i in range(5, 10))
        assert all(result[i] == 'extended' for i in range(10, 15))
    
    def test_parse_with_random(self):
        """Test parsing with random conformation."""
        result = _parse_structure_regions("1-5:alpha,6-10:random", 10)
        assert result[0] == 'alpha'
        assert result[5] == 'random'
    
    def test_parse_invalid_syntax_no_colon(self):
        """Test that invalid syntax (no colon) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid region syntax"):
            _parse_structure_regions("1-10alpha", 20)
    
    def test_parse_invalid_syntax_no_dash(self):
        """Test that invalid syntax (no dash) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid range syntax"):
            _parse_structure_regions("110:alpha", 20)
    
    def test_parse_invalid_conformation(self):
        """Test that invalid conformation name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid conformation"):
            _parse_structure_regions("1-10:invalid_conf", 20)
    
    def test_parse_overlapping_regions(self):
        """Test that overlapping regions raise ValueError."""
        with pytest.raises(ValueError, match="Overlapping regions"):
            _parse_structure_regions("1-10:alpha,5-15:beta", 20)
    
    def test_parse_out_of_bounds_high(self):
        """Test that out-of-bounds ranges raise ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            _parse_structure_regions("1-25:alpha", 20)
    
    def test_parse_out_of_bounds_low(self):
        """Test that start < 1 raises ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            _parse_structure_regions("0-10:alpha", 20)
    
    def test_parse_invalid_range_order(self):
        """Test that start > end raises ValueError."""
        with pytest.raises(ValueError, match="Invalid range"):
            _parse_structure_regions("10-5:alpha", 20)
    
    def test_parse_gaps_allowed(self):
        """Test that gaps in coverage are allowed."""
        result = _parse_structure_regions("1-5:alpha,10-15:beta", 20)
        # Residues 6-9 and 16-20 are not specified
        assert len(result) == 11  # 5 alpha (1-5) + 6 beta (10-15) = 11 specified residues
        assert all(result[i] == 'alpha' for i in range(5))
        assert all(result[i] == 'beta' for i in range(9, 15))
    
    def test_generate_mixed_structure(self):
        """Test generating structure with mixed conformations."""
        pdb = generate_pdb_content(
            sequence_str="AAAAAAAAAAAAAAAAAAAA",  # Fixed 20-residue sequence
            structure="1-10:alpha,11-20:beta"
        )
        assert "ATOM" in pdb
        
        # Validate it's a valid structure
        validator = PDBValidator(pdb)
        atoms = validator.atoms
        assert len(atoms) > 0
        
        # Verify we have 20 CA atoms (one per residue)
        ca_atoms = [a for a in atoms if a['atom_name'].strip() == 'CA']
        assert len(ca_atoms) == 20
    
    def test_structure_overrides_conformation(self):
        """Test that --structure parameter overrides --conformation."""
        # Use fixed sequence to ensure deterministic comparison
        pdb_with_structure = generate_pdb_content(
            sequence_str="AAAAAAAAAA",  # Fixed sequence
            conformation='beta',  # This should be overridden
            structure='1-10:alpha'
        )
        pdb_alpha_only = generate_pdb_content(
            sequence_str="AAAAAAAAAA",  # Same fixed sequence
            conformation='alpha'
        )
        # They should produce identical structures
        assert pdb_with_structure == pdb_alpha_only
    
    def test_structure_with_gaps_uses_default(self):
        """Test that gaps in structure specification use default conformation."""
        pdb = generate_pdb_content(
            length=15,
            conformation='extended',  # Default for unspecified regions
            structure='1-5:alpha,11-15:beta'
        )
        # Residues 6-10 should use 'extended' conformation
        assert "ATOM" in pdb
    
    def test_backward_compatibility_no_structure(self):
        """Test that existing --conformation still works when structure not provided."""
        pdb = generate_pdb_content(length=10, conformation='beta')
        assert "ATOM" in pdb
        
        # Should be valid
        validator = PDBValidator(pdb)
        assert len(validator.atoms) > 0
    
    def test_structure_with_sequence(self):
        """Test structure parameter works with explicit sequence."""
        pdb = generate_pdb_content(
            sequence_str="ACDEFGHIKLMNPQRSTVWY",
            structure="1-5:alpha,6-10:beta,11-20:extended"
        )
        assert "ATOM" in pdb
        
        # Verify we have 20 residues
        validator = PDBValidator(pdb)
        ca_atoms = [a for a in validator.atoms if a['atom_name'].strip() == 'CA']
        assert len(ca_atoms) == 20
    
    def test_parse_structure_regions_function_exists(self):
        """Test that _parse_structure_regions function exists."""
        from synth_pdb.generator import _parse_structure_regions
        assert callable(_parse_structure_regions)
    
    def test_generate_pdb_content_accepts_structure_parameter(self):
        """Test that generate_pdb_content accepts structure parameter."""
        import inspect
        sig = inspect.signature(generate_pdb_content)
        assert 'structure' in sig.parameters
