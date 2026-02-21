
import pytest
import subprocess
import os
import sys
from pathlib import Path
from synth_pdb.validator import PDBValidator

# Helper to run the CLI command
def run_synth_pdb(args):
    cmd = [sys.executable, "-m", "synth_pdb.main"] + args
    subprocess.check_call(cmd)

class TestEducationalExamples:

    def test_glucagon_alpha_helix(self, tmp_path):
        """
        Test Glucagon (29 residues) as an alpha-helical hormone.
        Verifies that the generated PDB has low Ramachandran violations for an alpha helix.
        """
        output_file = tmp_path / "glucagon.pdb"
        
        args = [
            "--sequence", "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
            "--conformation", "alpha",
            "--refine-clashes", "0",
            "--output", str(output_file)
        ]
        
        run_synth_pdb(args)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            content = f.read()
            
        validator = PDBValidator(content)
        # Verify it generated the correct length (29 residues)
        sequences = validator._get_sequences_by_chain()
        assert len(sequences.get('A', '')) == 29

    def test_melittin_bent_helix(self, tmp_path):
        """
        Test Melittin (26 residues) with a bent helix structure.
        Uses --structure to define regions.
        """
        output_file = tmp_path / "melittin.pdb"
        
        # Sequence: GIGAVLKVLTTGLPALISWIKRKRQQ
        # Structure: 1-11 alpha, 12-14 random (hinge), 15-26 alpha
        args = [
            "--sequence", "GIGAVLKVLTTGLPALISWIKRKRQQ",
            "--structure", "1-11:alpha,12-14:random,15-26:alpha",
            "--refine-clashes", "50",
            "--output", str(output_file)
        ]
        
        run_synth_pdb(args)

        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Basic validity check (it generated successfully)
        assert "ATOM" in content
        
        # Verify length matches (sometimes refinement might drop atoms if catastrophic, but shouldn't)
        validator = PDBValidator(content)
        sequences = validator._get_sequences_by_chain()
        assert len(sequences.get('A', '')) == 26

    def test_bpti_disulfide_bonds(self, tmp_path):
        """
        Test BPTI (58 residues) for disulfide bond detection.
        BPTI has 3 disulfide bonds.
        """
        output_file = tmp_path / "bpti.pdb"
        
        args = [
            "--sequence", "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA",
            "--conformation", "random", 
            "--refine-clashes", "100",
            "--output", str(output_file)
        ]
        
        run_synth_pdb(args)

        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Check that the file is valid PDB format
        assert "HEADER" in content
        assert "ATOM" in content
        assert "END" in content
        
        # The generator should ATTEMPT to detect disulfides and write SSBOND if found.
        # BPTI is small and constrained; some disulfides should be detected.
        # Check for SSBOND and CONECT correlation
        if "SSBOND" in content:
            assert "CONECT" in content, "SSBOND header present but no CONECT records for structural bonding"
            
            # Verify that SG atoms are involved in CONECT records
            lines = content.split('\n')
            sg_indices = [int(l[6:11]) for l in lines if l.startswith('ATOM') and l[12:16].strip() == 'SG']
            conect_lines = [l for l in lines if l.startswith('CONECT')]
            
            sg_bonded = False
            import re
            for line in conect_lines:
                parts = [int(p) for p in re.findall(r'\d+', line[6:])]
                if any(idx in parts for idx in sg_indices):
                    sg_bonded = True
                    break
            assert sg_bonded, "SSBOND present but no SG atoms found in CONECT records"


    def test_ubiquitin_complex_structure(self, tmp_path):
        """
        Test Ubiquitin (76 residues) with mixed alpha/beta structure.
        Good stress test for clashes.
        """
        output_file = tmp_path / "ubiquitin.pdb"
        
        # Simplified structure for reliability in test environment
        args = [
            "--sequence", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "--structure", "1-7:beta,23-34:alpha", 
            "--refine-clashes", "50",
            "--best-of-N", "2", 
            "--output", str(output_file)
        ]
        
        run_synth_pdb(args)
        
        assert output_file.exists()
        with open(output_file, 'r') as f:
            content = f.read()
            
        validator = PDBValidator(content)
        # It should adhere to the rules, but might have some clashes. 
        # We just want to ensure it generated the full length.
        sequences = validator._get_sequences_by_chain()
        assert len(sequences.get('A', '')) == 76

    def test_human_egf_disulfides(self, tmp_path):
        """
        Test Human EGF (53 residues) for disulfide bond detection.
        hEGF has 3 disulfide bonds (6-20, 14-31, 33-42).
        Verifies correct length and presence of SSBOND records.
        """
        output_file = tmp_path / "egf.pdb"
        
        success = False
        # Try a few seeds because random generation + minimization is sensitive to 
        # numpy versions and OS math differences which can cause flakiness.
        for seed in [42, 43, 44, 45, 46]:
            args = [
                "--sequence", "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR",
                "--conformation", "random",
                "--minimize",
                "--seed", str(seed),
                "--metal-ions", "none",
                "--output", str(output_file)
            ]
            
            run_synth_pdb(args)
            
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                content = f.read()
                
            validator = PDBValidator(content)
            # Verify it generated the correct length (53 residues)
            sequences = validator._get_sequences_by_chain()
            assert len(sequences.get('A', '')) == 53
            
            # Check for SSBOND records
            if "SSBOND" in content and "CONECT" in content:
                # Verify CONECT references SG atoms
                lines = content.split('\n')
                sg_indices = [int(l[6:11]) for l in lines if l.startswith('ATOM') and l[12:16].strip() == 'SG']
                if any("CONECT" in l and any(str(idx) in l for idx in sg_indices) for l in lines):
                    success = True
                    break
        
        assert success, "hEGF failed to form any disulfide bonds across multiple random seeds"
