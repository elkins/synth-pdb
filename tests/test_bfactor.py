"""
Tests for B-factor (temperature factor) assignment.

EDUCATIONAL NOTE - What are B-factors?
B-factors (temperature factors) represent atomic displacement due to:
1. Thermal motion (atoms vibrating)
2. Static disorder (multiple conformations)
3. Positional uncertainty

Typical ranges:
- Well-ordered atoms: 10-20 Ų
- Average atoms: 20-30 Ų
- Flexible regions: 30-50 Ų
- Very disordered: >50 Ų

Patterns in real proteins:
- Backbone < side chains (backbone more rigid)
- Core < surface (core more constrained)
- Middle < termini (termini more mobile)
"""

import pytest
import numpy as np
from synth_pdb.generator import _calculate_bfactor, generate_pdb_content


class TestBfactorCalculation:
    """Test suite for B-factor assignment."""
    
    def test_bfactor_function_exists(self):
        """Test that B-factor calculation function exists."""
        # This will fail until we implement the function
        assert callable(_calculate_bfactor)
    
    def test_backbone_vs_sidechain_bfactors(self):
        """
        Test that backbone atoms have lower B-factors than side chain atoms.
        
        EDUCATIONAL NOTE:
        Backbone atoms (N, CA, C, O) are constrained by peptide bonds and
        have less conformational freedom than side chain atoms.
        """
        # Middle residue to avoid terminal effects
        residue_num = 10
        total_residues = 20
        
        # Backbone atom
        b_backbone = _calculate_bfactor('CA', residue_num, total_residues, 'ALA')
        
        # Side chain atom
        b_sidechain = _calculate_bfactor('CB', residue_num, total_residues, 'ALA')
        
        # Side chains should generally have higher B-factors
        # Using average over multiple calls to account for random variation
        backbone_avg = np.mean([
            _calculate_bfactor('CA', residue_num, total_residues, 'ALA')
            for _ in range(10)
        ])
        sidechain_avg = np.mean([
            _calculate_bfactor('CB', residue_num, total_residues, 'ALA')
            for _ in range(10)
        ])
        
        assert sidechain_avg > backbone_avg
    
    def test_terminal_residues_higher_bfactors(self):
        """
        Test that terminal residues (Low S2) have higher B-factors than middle residues (High S2).
        
        EDUCATIONAL NOTE:
        Terminal residues are more mobile because they have fewer constraints.
        This is captured by lower Order Parameters (S2).
        """
        total_residues = 20
        
        # N-terminus (Low S2)
        b_n_term = _calculate_bfactor('CA', 1, total_residues, 'ALA', s2=0.45)
        
        # Middle (High S2)
        b_middle = _calculate_bfactor('CA', 10, total_residues, 'ALA', s2=0.85)
        
        # C-terminus (Low S2)
        b_c_term = _calculate_bfactor('CA', 20, total_residues, 'ALA', s2=0.45)
        
        # Average over multiple calls
        n_term_avg = np.mean([
            _calculate_bfactor('CA', 1, total_residues, 'ALA', s2=0.45)
            for _ in range(10)
        ])
        middle_avg = np.mean([
            _calculate_bfactor('CA', 10, total_residues, 'ALA', s2=0.85)
            for _ in range(10)
        ])
        c_term_avg = np.mean([
            _calculate_bfactor('CA', 20, total_residues, 'ALA', s2=0.45)
            for _ in range(10)
        ])
        
        # Termini should have higher B-factors
        assert n_term_avg > middle_avg
        assert c_term_avg > middle_avg

    def test_glycine_higher_bfactor(self):
        """
        Test that glycine has higher B-factors than other residues.
        
        EDUCATIONAL NOTE:
        Glycine has no side chain (only H as CB), giving it more conformational
        freedom and higher mobility than other amino acids.
        """
        # Average over multiple calls
        gly_avg = np.mean([
            _calculate_bfactor('CA', 10, 20, 'GLY', s2=0.85)
            for _ in range(10)
        ])
        ala_avg = np.mean([
            _calculate_bfactor('CA', 10, 20, 'ALA', s2=0.85)
            for _ in range(10)
        ])
        
        assert gly_avg > ala_avg
    
    def test_proline_lower_bfactor(self):
        """
        Test that proline has lower B-factors than other residues.
        
        EDUCATIONAL NOTE:
        Proline's cyclic structure (side chain connects back to backbone N)
        restricts backbone flexibility, making it more rigid.
        """
        # Average over multiple calls
        pro_avg = np.mean([
            _calculate_bfactor('CA', 10, 20, 'PRO', s2=0.85)
            for _ in range(10)
        ])
        ala_avg = np.mean([
            _calculate_bfactor('CA', 10, 20, 'ALA', s2=0.85)
            for _ in range(10)
        ])
        
        assert pro_avg < ala_avg

    def test_bfactor_realistic_range(self):
        """
        Test that B-factors are in realistic range (5-100 Ų).
        
        EDUCATIONAL NOTE:
        Typical B-factors in crystal structures:
        - Very rigid: 5-15 Ų (Core helices/sheets)
        - Normal: 15-30 Ų
        - Flexible: 30-50 Ų (Loops)
        - Highly Disordered: 50-99 Ų (Termini/Unstructured tails)
        
        In our Model-Free simulation, termini with S2 ~ 0.45 can reach 60-80 Ų.
        """
        # Test various scenarios
        test_cases = [
            ('CA', 1, 20, 'ALA'),   # N-terminus backbone
            ('CB', 1, 20, 'ALA'),   # N-terminus sidechain
            ('CA', 10, 20, 'ALA'),  # Middle backbone
            ('CB', 10, 20, 'ALA'),  # Middle sidechain
            ('CA', 20, 20, 'ALA'),  # C-terminus backbone
            ('CA', 10, 20, 'GLY'),  # Glycine
            ('CA', 10, 20, 'PRO'),  # Proline
        ]
        
        for atom_name, res_num, total_res, res_name in test_cases:
            bfactor = _calculate_bfactor(atom_name, res_num, total_res, res_name)
            assert 5.0 <= bfactor <= 100.0, \
                f"B-factor {bfactor} out of range for {res_name} {atom_name}"

    def test_bfactor_in_generated_pdb(self):
        """
        Test that generated PDB files contain realistic B-factors.
        
        EDUCATIONAL NOTE:
        B-factors appear in columns 61-66 of ATOM records in PDB format.
        They should not all be 0.00 (which indicates missing data).
        """
        pdb_content = generate_pdb_content(length=10, conformation='alpha')
        
        # Extract B-factors from ATOM lines
        bfactors = []
        MIN_BFACTOR = 5.0
        MAX_BFACTOR = 100.0 # Updated for Model-Free physics (Termini can be highly flexible)
        
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM'):
                # B-factor is in columns 61-66 (0-indexed: 60-66)
                bfactor_str = line[60:66].strip()
                if bfactor_str:
                    b_factor = float(bfactor_str)
                    bfactors.append(b_factor)
                    assert b_factor >= MIN_BFACTOR, f"B-factor {b_factor} too low"
                    assert b_factor <= MAX_BFACTOR, f"B-factor {b_factor} too high"
        
        # Should have B-factors for all atoms
        assert len(bfactors) > 0, "No B-factors found in PDB"
        
        # Should not all be 0.00
        assert not all(b == 0.0 for b in bfactors), \
            "All B-factors are 0.00 (not realistic)"
        
        # Should be in realistic range
        assert all(5.0 <= b <= 100.0 for b in bfactors), \
            f"Some B-factors out of range: {bfactors}"
        
        # Should show variation (not all the same)
        assert len(set(bfactors)) > 1, \
            "All B-factors are identical (should vary)"
    
    def test_bfactor_variation_along_chain(self):
        """
        Test that B-factors vary along the peptide chain.
        
        EDUCATIONAL NOTE:
        In real structures, B-factors typically show gradients:
        - Increase toward termini
        - Vary between backbone and side chains
        - Show local variations due to structure
        """
        pdb_content = generate_pdb_content(length=20, conformation='alpha')
        
        # Extract CA B-factors by residue number
        ca_bfactors = {}
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                res_num = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                ca_bfactors[res_num] = bfactor
        
        # Should have B-factors for all residues
        assert len(ca_bfactors) == 20
        
        # Terminal residues should generally have higher B-factors than middle
        # (averaged to account for random variation)
        n_term_avg = np.mean([ca_bfactors[i] for i in range(1, 4)])
        middle_avg = np.mean([ca_bfactors[i] for i in range(9, 13)])
        c_term_avg = np.mean([ca_bfactors[i] for i in range(18, 21)])
        
        # At least one terminus should be higher than middle
        assert (n_term_avg > middle_avg) or (c_term_avg > middle_avg), \
            "Termini should have higher B-factors than middle"
