
import pytest
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock, patch
from synth_pdb.physics import EnergyMinimizer
from synth_pdb.generator import generate_pdb_content

class TestCoverageGaps:

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_ptm_atom_stripping_and_translation(self, mock_app):
        """
        Verify that PTMs (SEP, TPO, PTR) are translated to standard residues
        and their extra atoms (P, O1P, etc.) are stripped.
        """
        minimizer = EnergyMinimizer()
        
        pdb_lines = [
            "ATOM    100  N   SEP A  10      11.111  22.222  33.333  1.00  0.00           N  ",
            "ATOM    101  CA  SEP A  10      11.111  22.222  33.333  1.00  0.00           C  ",
            "ATOM    102  P   SEP A  10      11.111  22.222  33.333  1.00  0.00           P  ", 
            "ATOM    103  O1P SEP A  10      11.111  22.222  33.333  1.00  0.00           O  ", 
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tf:
            tf.writelines([l + "\n" for l in pdb_lines])
            input_path = tf.name
            
        try:
            with patch("tempfile.NamedTemporaryFile") as mock_tf_class:
                mock_tf = MagicMock()
                mock_tf.__enter__.return_value.name = "intercepted.pdb"
                mock_tf_class.return_value = mock_tf
                
                mock_app.PDBFile.side_effect = Exception("Stop execution after strip check")
                
                try:
                    minimizer._run_simulation(input_path, "out.pdb")
                except Exception as e:
                    if str(e) != "Stop execution after strip check": raise

                written_lines = mock_tf.__enter__.return_value.writelines.call_args[0][0]
                
                for line in written_lines:
                    if "SEP" in line:
                        pytest.fail(f"Residue name 'SEP' should have been translated: {line}")
                    if " P " in line or "O1P" in line:
                        pytest.fail(f"PTM atom should have been stripped: {line}")
                
                assert any("SER" in line and " N " in line for line in written_lines)
                assert any("SER" in line and " CA " in line for line in written_lines)
                
        finally:
            if os.path.exists(input_path): os.unlink(input_path)

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    @patch("synth_pdb.physics.mm")
    def test_health_check_nan_handling(self, mock_mm, mock_app):
        """
        Verify that NaNs in energy or positions cause simulation to fail.
        """
        # Patch ForceField constructor to avoid real file loading
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        
        minimizer = EnergyMinimizer()
        
        # Mocking the PDB loading
        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb
        
        # Mock Topology with atoms
        mock_topo = MagicMock()
        mock_atom = MagicMock()
        mock_topo.atoms.return_value = [mock_atom]
        mock_pdb.topology = mock_topo
        
        # Mock Positions
        mock_pos = MagicMock()
        mock_pos.__len__.return_value = 1
        mock_pos.value_in_unit.return_value = [[1,1,1]]
        mock_pdb.positions = mock_pos
        
        # Mock Modeller
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology = mock_topo
        mock_modeller.positions = mock_pos
        
        # Mock Simulation and Context
        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        mock_context = MagicMock()
        mock_sim.context = mock_context
        
        # Health Check: Return NaN energy
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value.value_in_unit.return_value = np.nan
        mock_state.getPositions.return_value = mock_pos
        mock_context.getState.return_value = mock_state
        
        # 1. NaN Energy check
        assert minimizer._run_simulation("dummy.pdb", "out.pdb") is None
        
        # 2. NaN Position check
        mock_state.getPotentialEnergy.return_value.value_in_unit.return_value = 100.0 # Valid energy
        mock_nan_pos = MagicMock()
        mock_nan_pos.__len__.return_value = 1
        mock_nan_pos.value_in_unit.return_value = [[np.nan, 1, 1]]
        mock_state.getPositions.return_value = mock_nan_pos
        assert minimizer._run_simulation("dummy.pdb", "out.pdb") is None

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_health_check_high_energy_warning(self, mock_app, caplog):
        """
        Verify that extremely high energy triggers a warning but allows success.
        """
        import logging
        caplog.set_level(logging.WARNING)
        
        mock_ff = MagicMock()
        mock_app.ForceField.return_value = mock_ff
        mock_ff.createSystem.return_value = MagicMock()
        
        minimizer = EnergyMinimizer()
        
        # Same mocks as above
        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb
        mock_topo = MagicMock()
        mock_atom = MagicMock()
        mock_topo.atoms.return_value = [mock_atom]
        mock_pdb.topology = mock_topo
        mock_pos = MagicMock()
        mock_pos.__len__.return_value = 1
        mock_pos.value_in_unit.return_value = [[1,1,1]]
        mock_pdb.positions = mock_pos
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology = mock_topo
        mock_modeller.positions = mock_pos
        mock_sim = MagicMock()
        mock_app.Simulation.return_value = mock_sim
        
        # Return high energy
        mock_state = MagicMock()
        mock_state.getPotentialEnergy.return_value.value_in_unit.return_value = 1e9 
        mock_state.getPositions.return_value = mock_pos
        mock_sim.context.getState.return_value = mock_state
        
        with patch("synth_pdb.physics.app.PDBFile.writeFile"):
            assert minimizer._run_simulation("dummy.pdb", "out.pdb") is not None
            assert "High Potential Energy" in caplog.text

    @patch("synth_pdb.generator._detect_disulfide_bonds", return_value=[])
    def test_generator_ace_offset_mapping(self, mock_detect):
        """
        Verify that generator.py correctly maps PTM names even with ACE cap offsets.
        """
        min_pdb_content = (
            "ATOM      1  CH3 ACE A   0       0.000   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      2  C   ACE A   0       1.000   1.000   1.000  1.00  0.00           C  \n"
            "ATOM      3  O   ACE A   0       2.000   2.000   2.000  1.00  0.00           O  \n"
            "ATOM      4  N   GLY A   1       3.000   3.000   3.000  1.00  0.00           N  \n"
            "ATOM      5  CA  GLY A   1       4.000   4.000   4.000  1.00  0.00           C  \n"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tf:
            tf.write(min_pdb_content)
            out_pdb_path = tf.name

        try:
            with patch("synth_pdb.generator.EnergyMinimizer") as mock_min_class:
                mock_min = mock_min_class.return_value
                def side_effect(input_path, output_path, **kwargs):
                    with open(output_path, 'w') as f:
                        f.write(min_pdb_content)
                    return True
                mock_min.add_hydrogens_and_minimize.side_effect = side_effect
                
                content = generate_pdb_content(
                    sequence_str="SEP",
                    minimize_energy=True,
                    forcefield="amber14-all.xml"
                )
                
                assert "SEP" in content
                assert "ACE" in content
        finally:
            if os.path.exists(out_pdb_path): os.unlink(out_pdb_path)
