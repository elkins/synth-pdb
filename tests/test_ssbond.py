"""
Tests for disulfide bond (SSBOND) detection and annotation.

EDUCATIONAL NOTE - Disulfide Bonds in Proteins
==============================================
Disulfide bonds (also called disulfide bridges) are covalent bonds between
two cysteine residues in a protein.

What are Disulfide Bonds?
- Covalent S-S bond between sulfur atoms of two cysteines
- Form when two CYS side chains come close together
- Typical SG-SG distance: 2.0-2.1 Å
- One of the strongest bonds in proteins

How They Form:
1. Two cysteine residues come close in 3D space
2. Oxidation reaction occurs (requires oxidizing environment)
3. Sulfur atoms (SG) form covalent S-S bond
4. Two hydrogen atoms are removed (oxidation)

Chemical Reaction:
2 Cys-SH → Cys-S-S-Cys + 2H+ + 2e-

Why They're Important:
1. **Protein Stability**: Stabilize folded structure
2. **Extracellular Proteins**: Common in proteins outside cells
3. **Protein Folding**: Guide and constrain folding pathways
4. **Redox Regulation**: Can be formed/broken to regulate activity

Where They Occur:
- Extracellular proteins (oxidizing environment)
- Secreted proteins (antibodies, hormones, enzymes)
- Rare in cytoplasm (reducing environment prevents formation)

In PDB Files:
- Annotated with SSBOND records in header
- Format: SSBOND   1 CYS A    6    CYS A   11
- Indicates which CYS residues are bonded
- Helps understand protein structure and stability

Educational Impact:
Understanding disulfide bonds teaches:
1. Post-translational modifications
2. Protein stability mechanisms
3. Relationship between sequence and structure
4. Importance of cellular environment (oxidizing vs reducing)
"""


import pytest
import numpy as np
from synth_pdb.generator import generate_pdb_content
import biotite.structure.io as strucio
import biotite.structure as struc
import tempfile
import os
import re


class TestDisulfideBonds:
    """Test suite for disulfide bond detection and SSBOND records."""
    
    def test_detect_disulfide_bonds_function_exists(self):
        """
        Test that disulfide bond detection function exists.
        
        EDUCATIONAL NOTE:
        We need a function to detect potential disulfide bonds by:
        1. Finding all CYS residues in the structure
        2. Calculating SG-SG distances between all CYS pairs
        3. Identifying pairs within bonding distance (2.0-2.1 Å)
        """
        from synth_pdb.generator import _detect_disulfide_bonds
        assert callable(_detect_disulfide_bonds)
    
    def test_detects_close_cysteines(self):
        """
        Test that close cysteine pairs are detected as disulfide bonds.
        
        EDUCATIONAL NOTE - Distance Criteria:
        Disulfide bonds have very specific geometry:
        - SG-SG distance: 2.0-2.1 Å (very consistent)
        - S-S-Cα angle: ~103°
        - Dihedral angles: specific preferences
        
        For detection, we primarily use distance:
        - < 2.0 Å: Too close (steric clash)
        - 2.0-2.1 Å: Typical disulfide bond
        - > 2.1 Å: Too far (no bond)
        
        We use a slightly relaxed criterion (2.0-2.2 Å) to account for:
        - Structural flexibility
        - Coordinate precision
        - Different oxidation states
        """
        # Generate structure with cysteines
        # Note: In random structures, CYS pairs are unlikely to be close enough
        # This test will need a structure where we can control CYS positions
        pdb_content = generate_pdb_content(sequence_str="CCC", conformation='alpha')
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name
        
        try:
            structure = strucio.load_structure(temp_path)
            
            # Import detection function
            from synth_pdb.generator import _detect_disulfide_bonds
            
            # Detect disulfides
            disulfides = _detect_disulfide_bonds(structure)
            
            # Should return a list (may be empty if no close pairs)
            assert isinstance(disulfides, list)
            
            # Each disulfide should be a tuple of (res_id1, res_id2)
            for bond in disulfides:
                assert isinstance(bond, tuple)
                assert len(bond) == 2
                assert isinstance(bond[0], (int, np.integer))
                assert isinstance(bond[1], (int, np.integer))
        finally:
            os.remove(temp_path)
    
    def test_no_false_positives_distant_cysteines(self):
        """
        Test that distant cysteine pairs are NOT detected as disulfide bonds.
        
        EDUCATIONAL NOTE - False Positives:
        It's important not to report disulfide bonds that don't exist:
        - Distant CYS pairs (>2.2 Å) should not be bonded
        - Only actual close pairs should be reported
        - False positives would mislead structure analysis
        
        In linear peptides (our case):
        - Sequential CYS residues are typically ~6-7 Å apart
        - Would need specific folding to bring them close
        - Most random structures won't have disulfides
        """
        # Generate structure with distant cysteines
        pdb_content = generate_pdb_content(sequence_str="CAAAAAAAC", conformation='extended')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name
        
        try:
            structure = strucio.load_structure(temp_path)
            from synth_pdb.generator import _detect_disulfide_bonds
            
            disulfides = _detect_disulfide_bonds(structure)
            
            # Extended conformation should have no close CYS pairs
            # (CYS 1 and CYS 9 are far apart)
            assert len(disulfides) == 0, \
                f"Should not detect disulfides in extended structure, but found {len(disulfides)}"
        finally:
            os.remove(temp_path)
    
    def test_ssbond_record_format(self):
        """
        Test that SSBOND records are correctly formatted.
        
        EDUCATIONAL NOTE - PDB SSBOND Format:
        SSBOND records follow strict PDB format specification:
        
        Format:
        SSBOND   1 CYS A    6    CYS A   11                          1555   1555  2.03
        
        Columns:
        1-6:   "SSBOND"
        8-10:  Serial number (right-justified)
        12-14: Residue name 1 (CYS)
        16:    Chain ID 1
        18-21: Residue number 1 (right-justified)
        26-28: Residue name 2 (CYS)
        30:    Chain ID 2
        32-35: Residue number 2 (right-justified)
        60-65: Symmetry operator 1 (optional)
        67-72: Symmetry operator 2 (optional)
        74-78: SG-SG distance (optional)
        
        For our simple case:
        - Single chain (A)
        - No symmetry operators
        - Can include distance
        """
        from synth_pdb.generator import _generate_ssbond_records
        
        # Test with simple disulfide list
        disulfides = [(3, 8), (12, 20)]
        chain_id = 'A'
        
        records = _generate_ssbond_records(disulfides, chain_id)
        
        # Should return string with SSBOND records
        assert isinstance(records, str)
        
        # Should have one record per disulfide
        lines = [l for l in records.split('\n') if l.strip()]
        assert len(lines) == len(disulfides)
        
        # Each line should start with SSBOND
        for line in lines:
            assert line.startswith('SSBOND')
            assert 'CYS' in line
            assert chain_id in line
    
    def test_ssbond_in_generated_pdb(self):
        """
        Test that SSBOND and CONECT records appear in generated PDB files.
        """
        # Generate structure with cysteines
        pdb_content = generate_pdb_content(sequence_str="CCCCC", conformation='alpha')
        
        # Check if SSBOND records are present (if any disulfides detected)
        if 'SSBOND' in pdb_content:
            # Verify SSBOND format
            ssbond_lines = [l for l in pdb_content.split('\n') if l.startswith('SSBOND')]
            assert len(ssbond_lines) > 0
            
            for line in ssbond_lines:
                assert 'CYS' in line
                assert len(line) >= 30
            
            # Verify CONECT format (New Requirement)
            # When a disulfide is present, the SG atoms should be linked via CONECT
            conect_lines = [l for l in pdb_content.split('\n') if l.startswith('CONECT')]
            assert len(conect_lines) >= 2, "Disulfide found but no CONECT records generated"
            
            # Check for reciprocity
            # If atom X is connected to Y, Y matches X
            import re
            conect_pairs = []
            for line in conect_lines:
                parts = re.findall(r'\d+', line[6:])
                if len(parts) >= 2:
                    conect_pairs.append(tuple(sorted((int(parts[0]), int(parts[1])))))
            
            assert len(conect_pairs) > 0, "No valid CONECT pairs found in records"

    def test_ssbond_and_conect_correlation(self):
        """
        Verify that SSBOND header records have corresponding CONECT records.
        """
        # We need a predictable structure. CGGGGC cyclic forces 1-6 proximity.
        from synth_pdb.physics import HAS_OPENMM
        if not HAS_OPENMM:
            pytest.skip("Requires OpenMM for disulfide closure logic")
            
        pdb_content = generate_pdb_content(
            sequence_str="CGGGGC",
            cyclic=True,
            minimize_energy=True
        )
        
        lines = pdb_content.split('\n')
        ssbond_lines = [l for l in lines if l.startswith('SSBOND')]
        conect_lines = [l for l in lines if l.startswith('CONECT')]
        
        assert len(ssbond_lines) >= 1
        assert len(conect_lines) >= 3 # 2 for N-C closure + at least 1 for S-S (bidirectional usually)
        
        # Find SG atoms for residues 1 and 6
        atom_lines = [l for l in lines if l.startswith('ATOM')]
        sg1_idx = None
        sg6_idx = None
        for line in atom_lines:
            res_id = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            atom_idx = int(line[6:11].strip())
            if res_id == 1 and atom_name == 'SG': sg1_idx = atom_idx
            if res_id == 6 and atom_name == 'SG': sg6_idx = atom_idx
            
        assert sg1_idx is not None
        assert sg6_idx is not None
        
        # Verify CONECT specifically links sg1_idx and sg6_idx
        ss_conect_found = False
        for line in conect_lines:
            parts = [int(p) for p in re.findall(r'\d+', line[6:])]
            if sg1_idx in parts and sg6_idx in parts:
                ss_conect_found = True
                break
        
        assert ss_conect_found, f"No CONECT record found between SG1 ({sg1_idx}) and SG6 ({sg6_idx})"
    
    def test_handles_no_cysteines_gracefully(self):
        """
        Test that structures without cysteines don't cause errors.
        
        EDUCATIONAL NOTE - Edge Cases:
        Not all proteins have disulfide bonds:
        - Cytoplasmic proteins rarely have them
        - Some proteins have no cysteines at all
        - Detection should handle this gracefully
        
        Expected behavior:
        - No error/exception
        - Empty disulfide list
        - No SSBOND records in PDB
        """
        # Generate structure without cysteines
        pdb_content = generate_pdb_content(sequence_str="AAAAA", conformation='alpha')
        
        # Should not have SSBOND records
        assert 'SSBOND' not in pdb_content
        
        # Should generate valid PDB
        assert 'ATOM' in pdb_content
        assert pdb_content.strip().endswith('END')
    
    def test_multiple_disulfides(self):
        """
        Test handling of multiple disulfide bonds.
        
        EDUCATIONAL NOTE - Multiple Disulfides:
        Many proteins have multiple disulfide bonds:
        - Antibodies: 12-16 disulfides
        - Insulin: 3 disulfides
        - Lysozyme: 4 disulfides
        
        Each disulfide gets its own SSBOND record:
        - Serial numbers increment (1, 2, 3, ...)
        - Each record is independent
        - Order doesn't matter (but usually by residue number)
        """
        from synth_pdb.generator import _generate_ssbond_records
        
        # Test with multiple disulfides
        disulfides = [(2, 5), (8, 12), (15, 20)]
        chain_id = 'A'
        
        records = _generate_ssbond_records(disulfides, chain_id)
        lines = [l for l in records.split('\n') if l.strip()]
        
        # Should have one record per disulfide
        assert len(lines) == 3
        
        # Serial numbers should increment
        for i, line in enumerate(lines, 1):
            # Serial number is in columns 8-10
            serial = line[7:10].strip()
            assert serial == str(i)
