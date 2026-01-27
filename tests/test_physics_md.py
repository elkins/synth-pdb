import pytest
from unittest.mock import MagicMock, patch
import sys
import importlib
from synth_pdb import physics

class TestMDEquilibration:

    def test_equilibrate_calls_simulation_step(self, mocker, tmp_path):
        """Test that equilibrate runs minimization AND simulation steps."""
        
        # 1. Prepare Mocks
        openmm_mock = MagicMock()
        app_mock = MagicMock()
        unit_mock = MagicMock()
        
        # Setup specific return values
        mock_simulation = MagicMock()
        app_mock.Simulation.return_value = mock_simulation
        
        mock_pdb_file = MagicMock()
        mock_pdb_file.topology.atoms.return_value = [MagicMock()]
        mock_pdb_file.topology.residues.return_value = [MagicMock()]
        app_mock.PDBFile.return_value = mock_pdb_file
        
        # Configure Fake objects for Energy return (avoid MagicMock format issues)
        class FakeQuantity:
            def value_in_unit(self, u):
                return 100.0
        class FakeState:
            def getPotentialEnergy(self):
                return FakeQuantity()
            def getPositions(self):
                # Provide a list-like object for length checks
                return [FakeQuantity()] * 10
        
        mock_modeller = MagicMock()
        mock_modeller.positions = [FakeQuantity()] * 10
        mock_modeller.topology.atoms.return_value = [MagicMock()]
        mock_modeller.topology.residues.return_value = [MagicMock()]
        app_mock.Modeller.return_value = mock_modeller
        
        mock_simulation = MagicMock()
        mock_simulation.context.getState.return_value = FakeState()
        app_mock.Simulation.return_value = mock_simulation

        # Link openmm.app to app_mock explicitly
        openmm_mock.app = app_mock
        openmm_mock.unit = unit_mock

        # 2. Patch sys.modules safely using patch.dict
        # We need to map 'openmm', 'openmm.app', 'openmm.unit'
        modules_to_patch = {
            "openmm": openmm_mock,
            "openmm.app": app_mock,
            "openmm.unit": unit_mock
        }
        
        with patch.dict(sys.modules, modules_to_patch):
            # 3. Reload physics inside the patch context so it imports the mocks
            importlib.reload(physics)
            
            # 4. Run Test Logic
            # Instantiate minimizer (which now uses mocked openmm)
            minimizer = physics.EnergyMinimizer()
            
            # Check mocks were used (EnergyMinimizer checks HAS_OPENMM)
            # We must ensure HAS_OPENMM is True. physics.py determines it on import.
            # Since imports succeeded (mocks), HAS_OPENMM should be True.
            
            if not hasattr(minimizer, "equilibrate"):
                 pytest.fail("equilibrate method not implemented")

            input_pdb = str(tmp_path / "input.pdb")
            output_pdb = str(tmp_path / "output.pdb")
            with open(input_pdb, "w") as f:
                f.write("ATOM")
            
            # success = minimizer.equilibrate(input_pdb, output_pdb, steps=1000)
            
            success = minimizer.equilibrate(input_pdb, output_pdb, steps=1000)
            
            assert success is True
            print("MOCK CALLS:", mock_simulation.mock_calls)
            mock_simulation.step.assert_called_with(1000)
            
        # 5. RESTORE Step: Reload physics OUTSIDE the patch to restore real OpenMM
        # This is critical to prevent Segfaults in subsequent tests
        importlib.reload(physics)

    def test_equilibrate_missing_openmm(self):
        """Test fallback if OpenMM is missing."""
        # Force HAS_OPENMM = False
        # We can't strictly modify the module variable easily if we reload it,
        # but we can patch the attribute on the module object.
        with patch.object(physics, "HAS_OPENMM", False):
            minimizer = physics.EnergyMinimizer()
            success = minimizer.equilibrate("dummy", "dummy")
            assert success is False
