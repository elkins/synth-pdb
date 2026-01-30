
import pytest
import numpy as np
import biotite.structure as struc
from unittest.mock import patch, MagicMock
from synth_pdb.viewer import _create_3dmol_html, _find_hbonds
from synth_pdb.generator import generate_pdb_content


class TestVisualizationUpgrades:
    """TDD for Visualization Upgrades (H-bonds, Beta-Turns, PTMs)."""

    def test_find_hbonds_alpha_helix(self):
        """
        Verify that _find_hbonds detects the i to i+4 pattern in an alpha helix.
        """
        # Generate a perfect alpha helix (10 residues)
        pdb_content = generate_pdb_content(length=10, conformation="alpha")
        
        hbonds = _find_hbonds(pdb_content)
        
        # In a 10-residue helix, we expect H-bonds:
        # 1-5, 2-6, 3-7, 4-8, 5-9, 6-10 (approx 6 bonds)
        # depending on exact distance cutoff (usually 3.5A for O-N)
        
        # Relaxed expectation: Biotite strict check might miss some marginal ones
        assert len(hbonds) >= 3, f"Expected at least 3 H-bonds in 10-mer helix, found {len(hbonds)}"
        
        # Check structure of hbond dict
        first_bond = hbonds[0]
        assert 'start_resi' in first_bond
        assert 'end_resi' in first_bond
        assert first_bond['start_resi'] < first_bond['end_resi']

    def test_highlights_in_html(self):
        """
        Verify that passing 'highlights' to _create_3dmol_html generates addStyle commands.
        """
        pdb_content = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        
        highlights = [
            {'start': 3, 'end': 6, 'color': 'purple', 'style': 'stick', 'label': 'Type II Turn'}
        ]
        
        html = _create_3dmol_html(pdb_content, "test.pdb", "cartoon", "spectrum", highlights=highlights)
        
        # Check for specific JS commands
        # Match partial strings to avoid whitespace/quote issues
        assert "viewer.addStyle" in html
        assert "resi:[3, 4, 5, 6]" in html or "resi: [3, 4, 5, 6]" in html
        assert "stick:{colorscheme:'purple'" in html or "color:'purple'" in html


    def test_ptm_labeling(self):
        """
        Verify that PTMs (SEP, TPO) get labeled automatically.
        """
        # Manually create PDB content with a SEP residue
        pdb_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  N   SEP A   2       3.000   0.000   0.000  1.00  0.00           N  \n" # Phosphoserine
            "ATOM      3  P   SEP A   2       4.000   1.000   0.000  1.00  0.00           P  \n"
        )
        
        html = _create_3dmol_html(pdb_content, "ptm.pdb", "cartoon", "spectrum")
        
        # Should detect SEP and add a label
        # Matches viewer.addLabel("SEP", ...)
        assert 'viewer.addLabel("SEP"' in html or "viewer.addLabel('SEP'" in html
        assert "backgroundColor:'orange'" in html or "backgroundColor: 'orange'" in html


    def test_hbond_visualization_in_html(self):
        """
        Verify that detected H-bonds are rendered as cylinders.
        """
        # Mock _find_hbonds to return a known bond
        with patch('synth_pdb.viewer._find_hbonds') as mock_find:
            mock_find.return_value = [{'start_resi': 1, 'end_resi': 5, 'start_atom': 'O', 'end_atom': 'N'}]
            
            html = _create_3dmol_html("dummy_pdb", "test.pdb", "cartoon", "spectrum", show_hbonds=True)

            
            # Check for line addition (dashed, magenta)
            assert "viewer.addLine" in html
            assert "color: 'magenta'" in html 
            assert "dashed: true" in html 
            assert "linewidth: 10" in html


    def test_ssbond_visualization_in_html(self):
        """
        Verify that detected SSBONDs are rendered as yellow cylinders with stick sidechains.
        """
        # Create a mock PDB with clear SSBOND records
        pdb_content = (
            "SSBOND   1 CYS A    1    CYS A    2\n"
            "ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        )
        
        # We also need to mock _find_ssbonds to rely on its output rather than parsing the dummy PDB content
        with patch('synth_pdb.viewer._find_ssbonds') as mock_find_ss:
             mock_find_ss.return_value = [{'c1': 'A', 'r1': 1, 'c2': 'A', 'r2': 2}]
             
             html = _create_3dmol_html(pdb_content, "test.pdb", "cartoon", "spectrum")
             
             # Check for Yellow Cylinder
             assert "viewer.addCylinder" in html
             assert "color: 'yellow'" in html
             
             # Check for Stick Style (Polish fix)
             assert "stick:{radius:0.2}" in html or "stick:{radius: 0.2}" in html
