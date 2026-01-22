"""
Tests for conformational diversity feature.
Following TDD methodology - these tests should fail initially.
"""
import pytest
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator
from synth_pdb.data import RAMACHANDRAN_PRESETS
import numpy as np


class TestConformationalDiversity:
    """Test suite for conformational diversity feature."""
    
    def test_ramachandran_presets_exist(self):
        """Test that RAMACHANDRAN_PRESETS dictionary exists in data.py."""
        assert 'alpha' in RAMACHANDRAN_PRESETS
        assert 'beta' in RAMACHANDRAN_PRESETS
        assert 'extended' in RAMACHANDRAN_PRESETS
        assert 'ppii' in RAMACHANDRAN_PRESETS
    
    def test_ramachandran_preset_structure(self):
        """Test that each preset has phi and psi keys."""
        for conformation, angles in RAMACHANDRAN_PRESETS.items():
            assert 'phi' in angles, f"{conformation} missing phi"
            assert 'psi' in angles, f"{conformation} missing psi"
            assert isinstance(angles['phi'], (int, float))
            assert isinstance(angles['psi'], (int, float))
    
    def test_alpha_helix_angles(self):
        """Test alpha helix preset has correct angles."""
        alpha = RAMACHANDRAN_PRESETS['alpha']
        assert alpha['phi'] == pytest.approx(-57.0, abs=1.0)
        assert alpha['psi'] == pytest.approx(-47.0, abs=1.0)
    
    def test_beta_sheet_angles(self):
        """Test beta sheet preset has correct angles."""
        beta = RAMACHANDRAN_PRESETS['beta']
        assert beta['phi'] == pytest.approx(-135.0, abs=5.0)
        assert beta['psi'] == pytest.approx(135.0, abs=5.0)
    
    def test_generate_with_alpha_conformation(self):
        """Test generating PDB with alpha helix conformation."""
        pdb_content = generate_pdb_content(
            length=5,
            sequence_str="AAAAA",
            conformation='alpha'
        )
        
        # Validate structure
        validator = PDBValidator(pdb_content)
        validator.validate_ramachandran()
        violations = validator.get_violations()
        
        # Alpha helix should have relatively few Ramachandran violations
        # (The validator has strict thresholds, so some violations are expected
        # even for idealized structures)
        rama_violations = [v for v in violations if 'Ramachandran' in v]
        assert len(rama_violations) < 10, \
            f"Alpha helix should have minimal Ramachandran violations, got {len(rama_violations)}"
    
    def test_generate_with_beta_conformation(self):
        """Test generating PDB with beta sheet conformation."""
        pdb_content = generate_pdb_content(
            length=5,
            sequence_str="AAAAA",
            conformation='beta'
        )
        
        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content
        
        # Validate it parses correctly
        validator = PDBValidator(pdb_content)
        atoms = validator._parse_pdb_atoms(pdb_content)
        assert len(atoms) > 0
    
    def test_generate_with_extended_conformation(self):
        """Test generating PDB with extended conformation."""
        pdb_content = generate_pdb_content(
            length=5,
            sequence_str="AAAAA",
            conformation='extended'
        )
        
        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content
    
    def test_generate_with_ppii_conformation(self):
        """Test generating PDB with polyproline II conformation."""
        pdb_content = generate_pdb_content(
            length=5,
            sequence_str="PPPPP",
            conformation='ppii'
        )
        
        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content
    
    def test_generate_with_random_conformation(self):
        """Test generating PDB with random conformation sampling."""
        # Generate two structures with random conformations
        pdb1 = generate_pdb_content(length=10, conformation='random')
        pdb2 = generate_pdb_content(length=10, conformation='random')
        
        # Both should be valid
        assert "ATOM" in pdb1
        assert "ATOM" in pdb2
        
        # They should be different (with high probability)
        # This is a probabilistic test, but with 10 residues the chance of identical
        # random structures is vanishingly small
        assert pdb1 != pdb2, "Random conformations should produce different structures"
    
    def test_default_conformation_is_alpha(self):
        """Test that default conformation (when not specified) is alpha helix."""
        pdb_default = generate_pdb_content(length=5, sequence_str="AAAAA")
        pdb_alpha = generate_pdb_content(length=5, sequence_str="AAAAA", conformation='alpha')
        
        # Default should match explicit alpha
        assert pdb_default == pdb_alpha
    
    def test_invalid_conformation_raises_error(self):
        """Test that invalid conformation name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid conformation"):
            generate_pdb_content(length=5, conformation='invalid_name')
    
    def test_conformation_affects_backbone_geometry(self):
        """Test that different conformations produce different backbone geometries."""
        alpha_pdb = generate_pdb_content(length=10, sequence_str="A"*10, conformation='alpha')
        beta_pdb = generate_pdb_content(length=10, sequence_str="A"*10, conformation='beta')
        
        # Parse CA coordinates
        alpha_validator = PDBValidator(alpha_pdb)
        beta_validator = PDBValidator(beta_pdb)
        
        alpha_atoms = alpha_validator._parse_pdb_atoms(alpha_pdb)
        beta_atoms = beta_validator._parse_pdb_atoms(beta_pdb)
        
        # Get CA atoms
        alpha_ca = [a for a in alpha_atoms if a['atom_name'].strip() == 'CA']
        beta_ca = [a for a in beta_atoms if a['atom_name'].strip() == 'CA']
        
        # Calculate end-to-end distance
        alpha_distance = np.linalg.norm(
            np.array(alpha_ca[-1]['coords']) - np.array(alpha_ca[0]['coords'])
        )
        beta_distance = np.linalg.norm(
            np.array(beta_ca[-1]['coords']) - np.array(beta_ca[0]['coords'])
        )
        
        # Beta sheets are more extended than alpha helices
        # So end-to-end distance should be greater for beta
        assert beta_distance > alpha_distance, \
            f"Beta sheet ({beta_distance:.2f}Å) should be more extended than alpha helix ({alpha_distance:.2f}Å)"
