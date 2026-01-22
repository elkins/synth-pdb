"""
Tests for chirality validation.

Following TDD methodology - RED PHASE.
These tests should FAIL initially until we implement the chirality validation.
"""
import pytest
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


class TestChiralityValidation:
    """Test suite for L-amino acid chirality validation."""
    
    def test_l_amino_acid_chirality_passes(self):
        """Test that correctly generated L-amino acids pass chirality check."""
        # Generate a structure with various amino acids
        pdb = generate_pdb_content(length=5, sequence_str="ACDEF")
        validator = PDBValidator(pdb)
        validator.validate_chirality()
        
        # Should have no chirality violations for correctly generated L-amino acids
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0, \
            f"Expected no chirality violations, but got: {chirality_violations}"
    
    def test_glycine_exempt_from_chirality(self):
        """Test that glycine (no CB) is exempt from chirality validation."""
        # Glycine has no CB atom, so it has no chirality
        pdb = generate_pdb_content(length=5, sequence_str="GGGGG")
        validator = PDBValidator(pdb)
        validator.validate_chirality()
        
        # Should have no violations (GLY is exempt)
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0
    
    def test_mixed_sequence_chirality(self):
        """Test chirality validation on mixed sequence including GLY."""
        # Mix of amino acids including glycine
        pdb = generate_pdb_content(sequence_str="AGVGIGPG")
        validator = PDBValidator(pdb)
        validator.validate_chirality()
        
        # Should pass for all correctly generated residues
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0
    
    def test_all_amino_acids_chirality(self):
        """Test chirality validation on all 20 standard amino acids."""
        # All standard amino acids
        pdb = generate_pdb_content(sequence_str="ACDEFGHIKLMNPQRSTVWY")
        validator = PDBValidator(pdb)
        validator.validate_chirality()
        
        # Should pass for all (GLY is exempt)
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0
    
    def test_chirality_method_exists(self):
        """Test that validate_chirality method exists on PDBValidator."""
        pdb = generate_pdb_content(length=3)
        validator = PDBValidator(pdb)
        
        # Method should exist
        assert hasattr(validator, 'validate_chirality'), \
            "PDBValidator should have validate_chirality method"
        
        # Should be callable
        assert callable(validator.validate_chirality), \
            "validate_chirality should be callable"
    
    def test_chirality_included_in_validate_all(self):
        """Test that chirality validation is included in validate_all()."""
        pdb = generate_pdb_content(length=5, sequence_str="ACDEF")
        validator = PDBValidator(pdb)
        
        # Clear any existing violations
        validator.violations = []
        
        # Run validate_all
        validator.validate_all()
        
        # Chirality should have been checked (no violations expected for correct structure)
        # We can't directly test if it was called, but we can verify no chirality violations
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0
    
    def test_proline_chirality(self):
        """Test that proline (cyclic structure) still has correct chirality."""
        # Proline has a cyclic structure but still has a chiral C-alpha
        pdb = generate_pdb_content(sequence_str="PPPPP")
        validator = PDBValidator(pdb)
        validator.validate_chirality()
        
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0
    
    def test_terminal_residues_chirality(self):
        """Test chirality validation on N-terminal and C-terminal residues."""
        # Terminal residues have different atom sets but should still have correct chirality
        pdb = generate_pdb_content(sequence_str="AVL")
        validator = PDBValidator(pdb)
        validator.validate_chirality()
        
        chirality_violations = [v for v in validator.get_violations() if 'Chirality' in v or 'chirality' in v]
        assert len(chirality_violations) == 0
