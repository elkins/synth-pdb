"""
Tests for 3D molecular viewer functionality.
"""
import pytest
from synth_pdb.viewer import view_structure_in_browser, _create_3dmol_html
from synth_pdb.generator import generate_pdb_content


class TestMolecularViewer:
    """Test suite for 3D molecular visualization."""
    
    def test_create_3dmol_html(self):
        """Test HTML generation for 3Dmol.js viewer."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum")
        
        # Check essential components
        assert "<!DOCTYPE html>" in html
        assert "3Dmol.js" in html or "3Dmol" in html
        assert "test.pdb" in html
        assert "synth-pdb" in html
        assert "cartoon" in html
        assert "spectrum" in html
        
    def test_html_escaping(self):
        """Test that PDB data is properly escaped in HTML."""
        # PDB data with special characters that need escaping
        pdb_data = "ATOM  `test\\data`"
        html = _create_3dmol_html(pdb_data, "test.pdb", "stick", "chain")
        
        # Should escape backticks and backslashes
        assert "\\`" in html or "\\\\`" in html  # Escaped backtick
        assert "\\\\" in html  # Escaped backslash
        
    def test_different_styles(self):
        """Test HTML generation with different visualization styles."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        
        for style in ["cartoon", "stick", "sphere", "line"]:
            html = _create_3dmol_html(pdb_data, "test.pdb", style, "spectrum")
            assert style in html
            
    def test_different_colors(self):
        """Test HTML generation with different color schemes."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        
        for color in ["spectrum", "chain", "ss", "white"]:
            html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", color)
            assert color in html
    
    def test_viewer_with_generated_structure(self):
        """Test viewer HTML generation with actual generated PDB."""
        # Generate a simple structure
        pdb = generate_pdb_content(sequence_str="ACDEFG")
        
        # Create HTML
        html = _create_3dmol_html(pdb, "generated.pdb", "cartoon", "spectrum")
        
        # Should contain PDB atoms
        assert "ATOM" in html
        assert "ALA" in html or "CYS" in html  # Should have amino acid names
        
    def test_html_has_controls(self):
        """Test that HTML includes interactive controls."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum")
        
        # Check for control buttons
        assert "Reset View" in html
        assert "Toggle Spin" in html
        assert "Style:" in html
        assert "Color:" in html
        
    def test_html_has_instructions(self):
        """Test that HTML includes user instructions."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum")
        
        # Check for instructions
        assert "rotate" in html.lower() or "drag" in html.lower()
        assert "zoom" in html.lower()

    def test_feature_toggles_and_restraint_logic(self):
        """Test that Ghost Mode and Restraints logic are present in HTML."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        
        # Create a dummy restraint to ensure logic is populated
        dummy_restraints = [{'chain_1': 'A', 'residue_index_1': 1, 'atom_name_1': 'HA',
                             'chain_2': 'A', 'residue_index_2': 2, 'atom_name_2': 'H', 'dist': 3.5}]
        
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum", restraints=dummy_restraints)
        
        # Check for Ghost Mode UI and Logic
        assert "Ghost Mode" in html
        assert "ghostMode = false" in html
        assert "toggleGhost" in html
        assert "opacity: opacityVal" in html  # Logic for applying transparency
        
        # Check for Restraints UI and Logic
        assert "Restraints" in html
        assert "showRestraints = true" in html
        assert "toggleRestraints" in html
        assert "drawRestraints" in html
        assert "viewer.addCylinder" in html
        
        # Check that proper keys are generated in the JS array
        assert "c1:'A'" in html
        assert "d:3.5" in html

    def test_view_structure_in_browser(self, mocker):
        """Test the main entry point for the browser-based viewer."""
        # Mock dependencies to avoid side effects
        mock_webbrowser = mocker.patch("webbrowser.open")
        
        # For tempfile, we need to mock it carefully because of the context manager
        mock_temp = mocker.patch("tempfile.NamedTemporaryFile")
        mock_instance = mock_temp.return_value.__enter__.return_value
        mock_instance.name = "/tmp/fake_viewer.html"
        
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        
        # Call function
        view_structure_in_browser(pdb_data, "test.pdb")
        
        # Verify interactions
        assert mock_temp.called
        assert mock_instance.write.called
        mock_webbrowser.assert_called_once_with("file:///tmp/fake_viewer.html")

    def test_view_structure_in_browser_error(self, mocker):
        """Test error handling in view_structure_in_browser."""
        mocker.patch("synth_pdb.viewer._create_3dmol_html", side_effect=ValueError("Test Error"))
        
        with pytest.raises(ValueError, match="Test Error"):
            view_structure_in_browser("pdb", "test.pdb")

    def test_ssbond_parsing(self):
        """Test parsing of SSBOND records from PDB."""
        from synth_pdb.viewer import _find_ssbonds
        pdb_data = "SSBOND   1 CYS A    3    CYS A   10\n"
        ssbonds = _find_ssbonds(pdb_data)
        assert len(ssbonds) == 1
        assert ssbonds[0]['r1'] == 3
        assert ssbonds[0]['r2'] == 10
        
        # Test malformed
        bad_pdb = "SSBOND   1 CYS A  XXXX   CYS A   10\n"
        assert _find_ssbonds(bad_pdb) == []

    def test_conect_parsing(self):
        """Test parsing of CONECT records from PDB."""
        from synth_pdb.viewer import _find_conects
        pdb_data = "CONECT    1    5\nCONECT    5    1\n"
        conects = _find_conects(pdb_data)
        # Should avoid duplicates
        assert len(conects) == 1
        assert (1, 5) in conects or (5, 1) in conects

    def test_hbond_fallback(self):
        """Test H-bond detection fallback when Biotite strict mode fails or finds nothing."""
        # Simple Ala-Ala-Ala-Ala-Ala (not a helix but we want to trigger geometric fallback)
        # We need N and O atoms
        pdb_data = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.200   2.300   0.000  1.00  0.00           O
ATOM      5  N   ALA A   5       2.000   1.400   3.000  1.00  0.00           N
"""
        from synth_pdb.viewer import _find_hbonds
        hbonds = _find_hbonds(pdb_data)
        # N at (2, 1.4, 3) and O at (1.2, 2.3, 0) -> dist approx 3.1
        assert len(hbonds) >= 1

    def test_highlight_styles_and_labels(self):
        """Test different highlight styles and label generation."""
        pdb_data = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N"
        highlights = [
            {'start': 1, 'end': 1, 'color': 'red', 'style': 'cartoon', 'label': 'HOTSPOT'},
            {'start': 2, 'end': 2, 'color': 'blue', 'style': 'stick'}
        ]
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum", highlights=highlights)
        
        # Cartoon highlight
        assert "{cartoon:{color:'red'}}" in html
        # Stick highlight
        assert "stick:{colorscheme:'blue'" in html
        # Label
        assert "addLabel('HOTSPOT'" in html

    def test_ptm_labels(self):
        """Test detection and labeling of PTM residues."""
        pdb_data = "ATOM      1  N   SEP A   5       0.000   0.000   0.000  1.00  0.00           N"
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum")
        assert 'viewer.addLabel("SEP"' in html

    def test_integration_ssbond_and_conect(self):
        """Test that SSBOND and CONECT records influence the HTML."""
        pdb_data = "SSBOND   1 CYS A    3    CYS A   10\nCONECT    1    5\n"
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum")
        
        assert "Detected Disulfide Bonds" in html
        assert "Detected CONECT Records" in html
        assert "resi:3" in html
        assert "resi:10" in html

    def test_integration_hbonds(self):
        """Test that H-bond visualization is triggered."""
        pdb_data = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      4  O   ALA A   1       1.200   2.300   0.000  1.00  0.00           O
ATOM      5  N   ALA A   5       2.000   1.400   3.000  1.00  0.00           N
"""
        html = _create_3dmol_html(pdb_data, "test.pdb", "cartoon", "spectrum", show_hbonds=True)
        assert "Detected Backbone H-Bonds" in html
        assert "viewer.addCylinder" in html
