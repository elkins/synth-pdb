import pytest
import numpy as np
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator

def test_d_amino_acid_generation_basic():
    """
    Test that a sequence with D-amino acids can be parsed and generated.
    The sequence 'D-ALA-ALA' should result in a 2-residue peptide.
    """
    # This should fail if D- prefix is not recognized
    sequence = "D-ALA-ALA"
    pdb_content = generate_pdb_content(sequence_str=sequence, length=2)
    
    assert "DAL" in pdb_content
    # We expect D-ALA residues to still be labeled as 'ALA' in the PDB file 
    # (as 'DALA' is not a standard PDB name, though some models use it).
    # However, the validator should know they are 'D'.
    
    validator = PDBValidator(pdb_content)
    # At this point, the validator doesn't know it's a D-amino acid 
    # because the PDB residue name is just 'ALA'.
    # PROPOSAL: We might want to mark them in the PDB or pass metadata.
    # Standard PDB usually uses DAL for D-Alanine, but many tools stay with ALA.
    # Let's see how our validator handles it.

def test_d_amino_acid_chirality_inversion():
    """
    Test that D-ALA has inverted chirality compared to L-ALA.
    """
    l_pdb = generate_pdb_content(sequence_str="ALA", length=1)
    d_pdb = generate_pdb_content(sequence_str="D-ALA", length=1)
    
    l_val = PDBValidator(l_pdb)
    d_val = PDBValidator(d_pdb)
    
    # We need to extract the improper dihedral N-CA-C-CB
    def get_improper(validator):
        atoms = validator.atoms
        n = [a for a in atoms if a['atom_name'] == "N"][0]
        ca = [a for a in atoms if a['atom_name'] == "CA"][0]
        c = [a for a in atoms if a['atom_name'] == "C"][0]
        cb = [a for a in atoms if a['atom_name'] == "CB"][0]
        return validator._calculate_dihedral_angle(n['coords'], ca['coords'], c['coords'], cb['coords'])

    l_improper = get_improper(l_val)
    d_improper = get_improper(d_val)
    
    # D-amino acids should have the opposite sign of L-amino acids
    # L-ALA is ~ +34 (in our generator)
    # D-ALA should be ~ -34
    assert np.sign(l_improper) != np.sign(d_improper)
    assert abs(l_improper - (-d_improper)) < 5.0

def test_d_amino_acid_validation():
    """
    Test that the validator correctly identifies D-amino acids if specified.
    """
    # We need a way to tell the validator that a residue is D-chiral.
    # One way is to check the Remark or have a special residue name in the validator's internal state.
    # For now, let's assume if we specify "D-ALA" in the generator, 
    # it stays "D-ALA" in the AtomArray for validation if we pass it directly.
    
    sequence = "D-ALA-ALA"
    pdb_content = generate_pdb_content(sequence_str=sequence, length=2)
    
    validator = PDBValidator(pdb_content)
    validator.validate_chirality()
    
    violations = validator.get_violations()
    chirality_violations = [v for v in violations if "Chirality violation" in v]
    
    # This should pass without violations if implemented correctly
    assert len(chirality_violations) == 0
