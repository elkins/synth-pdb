
import pytest
from unittest.mock import MagicMock, patch
import sys
import synth_pdb.physics

class TestPhysicsCoverage:

    def test_missing_openmm_dependency(self):
        """
        Test that methods return gracefully when OpenMM is not installed.
        """
        # Mock HAS_OPENMM = False
        with patch("synth_pdb.physics.HAS_OPENMM", False):
            minimizer = synth_pdb.physics.EnergyMinimizer()
            
            # Should fail/return False gracefully
            assert minimizer.minimize("dummy.pdb", "out.pdb") is False
            assert minimizer.equilibrate("dummy.pdb", "out.pdb") is False
            assert minimizer.add_hydrogens_and_minimize("dummy.pdb", "out.pdb") is False

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_forcefield_loading_error(self, mock_app):
        """
        Test that init handles ForceField loading errors.
        """
        # ForceField constructor raises Exception
        mock_app.ForceField.side_effect = Exception("XML file missing")
        mock_app.OBC2 = "OBC2" # Needed for defaults

        with pytest.raises(Exception, match="XML file missing"):
            synth_pdb.physics.EnergyMinimizer()

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_simulation_failure(self, mock_app):
        """
        Test general simulation failure (e.g., bad topology).
        """
        # Set up a working minimizer mock
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Mock PDBFile to fail
        mock_app.PDBFile.side_effect = Exception("Corrupt PDB")
        
        # Should return False and catch exception
        assert minimizer._run_simulation("bad.pdb", "out.pdb") is False

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_hetatm_restoration_logic(self, mock_app, caplog):
        """
        Test the specific logic for preserving ZN ions during hydrogen checking.
        The "AI Trinity" logic: identifying non-protein atoms, storing them, 
        and restoring them after addHydrogens.
        """
        import logging
        caplog.set_level(logging.INFO)
        
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Initialize mock objects explicitly to avoid NameErrors in partial edits
        mock_res_ala = MagicMock()
        mock_res_zn = MagicMock()
        
        # Mock PDBFile and Topology
        mock_pdb = MagicMock()
        mock_topology = MagicMock()
        mock_pdb.topology = mock_topology
        mock_positions = [1, 2, 3] # Dummy list
        mock_pdb.positions = mock_positions
        
        mock_app.PDBFile.return_value = mock_pdb
        
        # Mock Modeller
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology = mock_topology
        mock_modeller.positions = mock_positions # Initially same
        
        # Setup residues in topology:
        # 1. Protein residue (ALA)
        # 2. Zinc Ion (ZN) - The target of our test
        
        mock_res_ala = MagicMock()
        mock_res_ala.name = "ALA"
        
        mock_res_zn = MagicMock()
        mock_res_zn.name = "ZN"
        # Setup atoms for ZN residue
        mock_atom_zn = MagicMock()
        mock_atom_zn.name = "ZN"
        mock_atom_zn.element = "Zn" 
        mock_atom_zn.index = 0
        mock_res_zn.atoms.return_value = [mock_atom_zn]
        
        # Initial residues loop
        mock_topology.residues.return_value = [mock_res_ala, mock_res_zn]
        
        # Mock addHydrogens behavior
        # After addHydrogens is called, we simulate the Modeller losing the ZN residue
        def side_effect_add_hydrogens(*args, **kwargs):
             mock_topology.residues.side_effect = None 
             # Only ALA is left after hydrogen addition
             # Reset atoms iterator to only return ALA atoms
             mock_topology.residues.return_value = [mock_res_ala] 
             mock_topology.atoms.side_effect = lambda: iter([MagicMock()]) 
             
             def side_effect_add_atom(*args, **kwargs):
                  pass 
             mock_topology.addAtom.side_effect = side_effect_add_atom
             return
        mock_modeller.addHydrogens.side_effect = side_effect_add_hydrogens

        # Setup Mock Residues
        mock_res_ala.name = "ALA"
        mock_res_zn.name = "ZN"

        # Setup atoms for ZN - CRITICAL for loops over res.atoms()
        # We must ensure atoms() returns an iterator yielding our atom
        mock_atom_zn = MagicMock()
        mock_atom_zn.name = "ZN"
        mock_atom_zn.element = "Zn"
        mock_atom_zn.index = 0
        mock_res_zn.atoms.return_value = [mock_atom_zn]

        # Mock dependencies for restoring HETATM
        mock_topology.addChain.return_value = "new_chain"
        mock_topology.addResidue.return_value = "new_res"
        
        # Mock internal imports
        mock_biotite = MagicMock()
        mock_biotite_structure = MagicMock()
        mock_biotite_pdb_module = MagicMock()
        mock_biotite_pdb_file = MagicMock()
        mock_biotite_struc = MagicMock()
        
        mock_biotite_pdb_module.PDBFile = mock_biotite_pdb_file
        mock_biotite_pdb_file.read.return_value.get_structure.return_value = MagicMock() 

        # IMPORTANT: ensure Modeller.topology.residues() returns [ALA, ZN] initially
        # Use simple list return value for residues() 
        mock_topology.residues.return_value = [mock_res_ala, mock_res_zn]
        # Use lambda for atoms() to return fresh iterator every time
        mock_topology.atoms.side_effect = lambda: iter([MagicMock(), mock_atom_zn])

        mock_cofactors = MagicMock()
        mock_cofactors.find_metal_binding_sites.return_value = [] 
        
        mock_biophysics = MagicMock()
        mock_biophysics.find_salt_bridges.return_value = [] 
        
        # Mock Simulation
        mock_simulation = MagicMock()
        mock_app.Simulation.return_value = mock_simulation
        mock_state = MagicMock()
        mock_state.getPositions.return_value = [1, 2, 3] 
        mock_simulation.context.getState.return_value = mock_state
        mock_simulation.topology = mock_topology 

        # Patch sys.modules
        with patch.dict(sys.modules, {
            "biotite": mock_biotite,
            "biotite.structure": mock_biotite_structure,
            "biotite.structure.io": MagicMock(),
            "biotite.structure.io.pdb": mock_biotite_pdb_module,
            "synth_pdb.cofactors": mock_cofactors,
            "synth_pdb.biophysics": mock_biophysics
        }):
             # Run internal simulation method
             minimizer._run_simulation("dummy.pdb", "out.pdb", add_hydrogens=True)
        
        # Verifications
        # 1. Did we detect ZN and try to restore it?
        # Check logs for "Restoring lost HETATM: ZN"
        assert "Restoring lost HETATM: ZN" in caplog.text
        
        # 2. Did we call topology.addResidue("ZN", ...) ?
        mock_topology.addResidue.assert_called_with("ZN", "new_chain")
        
        # 3. Did we call topology.addAtom("ZN", ...) ?
        mock_topology.addAtom.assert_called_with("ZN", "Zn", "new_res")

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    def test_minimize_calls_run_simulation(self):
        """
        Test that minimize() correctly calls _run_simulation with add_hydrogens=False.
        """
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Mock the internal _run_simulation method
        with patch.object(minimizer, '_run_simulation', return_value=True) as mock_run:
            result = minimizer.minimize("in.pdb", "out.pdb", max_iterations=50, tolerance=5.0)
            
            assert result is True
            mock_run.assert_called_once_with("in.pdb", "out.pdb", max_iterations=50, tolerance=5.0, add_hydrogens=False, cyclic=False)

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_zero_atoms_in_topology(self, mock_app, caplog):
        """
        Test that _run_simulation returns False if topology has 0 atoms.
        """
        import logging
        caplog.set_level(logging.ERROR)
        
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Mock PDB with empty atoms list
        mock_pdb = MagicMock()
        mock_topology = MagicMock()
        mock_pdb.topology = mock_topology
        mock_pdb.positions = []
        
        mock_app.PDBFile.return_value = mock_pdb
        mock_topology.atoms.return_value = iter([]) # Empty iterator
        
        # Mock Modeller to also return empty atoms
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology.atoms.return_value = iter([])
        
        result = minimizer._run_simulation("empty.pdb", "out.pdb", add_hydrogens=True)
        
        assert result is False
        assert "Topology has 0 atoms" in caplog.text

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_empty_positions_from_openmm(self, mock_app, caplog):
        """
        Test that we catch empty positions returned by OpenMM state.
        """
        import logging
        caplog.set_level(logging.ERROR)
        
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Setup successful initialization but empty positions at end
        mock_pdb = MagicMock()
        mock_app.PDBFile.return_value = mock_pdb
        mock_pdb.topology.atoms.return_value = iter([MagicMock()]) # Not empty initially
        
        mock_modeller = MagicMock() # Modeller
        mock_app.Modeller.return_value = mock_modeller
        # IMPORTANT: Use lambda to return NEW iterator on every call
        mock_modeller.topology.atoms.side_effect = lambda: iter([MagicMock()])
        
        # Simulation setup
        mock_simulation = MagicMock()
        mock_app.Simulation.return_value = mock_simulation
        
        # State returns empty positions
        mock_state = MagicMock()
        mock_state.getPositions.return_value = [] 
        mock_simulation.context.getState.return_value = mock_state
        
        # Need to mock forcefield creation to avoid real file loading
        minimizer.forcefield = MagicMock()
        
        # Mock internal dependencies to avoid import errors or side effects
        with patch.dict(sys.modules, {
            "biotite": MagicMock(), 
            "synth_pdb.cofactors": MagicMock(),
            "synth_pdb.biophysics": MagicMock()
        }):
             result = minimizer._run_simulation("test.pdb", "out.pdb", add_hydrogens=True)
             
        assert result is False
        assert "OpenMM returned empty positions" in caplog.text

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    @patch("synth_pdb.physics.mm")
    def test_salt_bridge_force_application(self, mock_mm, mock_app, caplog):
        """
        Test that salt bridge forces are applied when bridges are detected.
        """
        import logging
        caplog.set_level(logging.DEBUG)
        
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Mock PDB & Topology
        mock_pdb = MagicMock()
        mock_topology = MagicMock()
        mock_pdb.topology = mock_topology
        mock_pdb.positions = [1, 2, 3]
        mock_app.PDBFile.return_value = mock_pdb
        
        # Setup Atoms: Residue 1 (LYS), Residue 10 (ASP)
        atom1 = MagicMock(); atom1.index = 0; atom1.name = "NZ"; atom1.residue.id = "1"; atom1.residue.name = "LYS"
        atom2 = MagicMock(); atom2.index = 1; atom2.name = "OD1"; atom2.residue.id = "10"; atom2.residue.name = "ASP"
        
        mock_topology.atoms.return_value = iter([atom1, atom2]) # Used in Modeller init
        
        # Mock Modeller to return same atoms
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology.atoms.side_effect = lambda: iter([atom1, atom2])
        mock_modeller.topology.residues.return_value = [] # No ion checks needed
        
        # Mock find_salt_bridges return value
        mock_biophysics = MagicMock()
        mock_biophysics.find_salt_bridges.return_value = [
            {"res_ia": 1, "atom_a": "NZ", "res_ib": 10, "atom_b": "OD1", "distance": 3.0}
        ]
        
        # Mock ForceField & System
        minimizer.forcefield = MagicMock()
        mock_system = MagicMock()
        minimizer.forcefield.createSystem.return_value = mock_system
        
        # Mock Simulation
        mock_simulation = MagicMock()
        mock_app.Simulation.return_value = mock_simulation
        mock_simulation.context.getState.return_value.getPositions.return_value = [1]
        
        # Mock Dependencies
        with patch.dict(sys.modules, {
            "biotite": MagicMock(),
            "biotite.structure": MagicMock(),
            "biotite.structure.io.pdb": MagicMock(),
            "synth_pdb.cofactors": MagicMock(),
            "synth_pdb.biophysics": mock_biophysics
        }):
            minimizer._run_simulation("test.pdb", "out.pdb", add_hydrogens=True)
            
        # Verify Force Creation
        mock_mm.CustomBondForce.assert_any_call("0.5*k_sb*(r-r0)^2")
        
        # Verify Force was added to system
        # We expect addForce to be called twice (once for coordination (empty), once for salt bridges)
        # Actually coordination force is only added if restraints > 0.
        # But we mocked mm.CustomBondForce, so we check if system.addForce was called with the result
        assert mock_system.addForce.called
        
        # Check logs for debug message
        assert "Found 1 salt bridges" in caplog.text



    @patch("synth_pdb.physics.HAS_OPENMM", True)
    def test_wrapper_methods(self):
        """
        Test utility wrappers add_hydrogens_and_minimize and equilibrate steps.
        """
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Test add_hydrogens_and_minimize
        with patch.object(minimizer, '_run_simulation', return_value=True) as mock_run:
            minimizer.add_hydrogens_and_minimize("in.pdb", "out.pdb")
            mock_run.assert_called_with("in.pdb", "out.pdb", add_hydrogens=True, max_iterations=0, tolerance=10.0, cyclic=False)
            
        # Test equilibrate with steps
        # This requires mocking _run_simulation internals to verify simulation.step(steps) is called
        # But _run_simulation handles that logic. We just need to check if _run_simulation
        # was called with correct equilibration_steps arg by the wrapper
        with patch.object(minimizer, '_run_simulation', return_value=True) as mock_run:
            minimizer.equilibrate("in.pdb", "out.pdb", steps=500)
            mock_run.assert_called_with("in.pdb", "out.pdb", add_hydrogens=True, equilibration_steps=500, cyclic=False)

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_create_system_exception_fallback(self, mock_app, caplog):
        """
        Test that we fallback if createSystem fails (e.g., due to implicit solvent issues).
        """
        minimizer = synth_pdb.physics.EnergyMinimizer()
        
        # Standard Setup
        mock_pdb = MagicMock()
        mock_pdb.topology.atoms.return_value = iter([MagicMock()]) 
        mock_app.PDBFile.return_value = mock_pdb
        
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.topology.atoms.side_effect = lambda: iter([MagicMock()]) # Fresh iterator
        mock_modeller.positions = [0]
        
        mock_simulation = MagicMock()
        mock_app.Simulation.return_value = mock_simulation
        mock_simulation.context.getState.return_value.getPositions.return_value = [1]
        
        minimizer.forcefield = MagicMock()
        
        # Mock createSystem to fail ONCE then succeed
        # First call has implicitSolvent arg
        def create_system_side_effect(*args, **kwargs):
            if 'implicitSolvent' in kwargs:
                raise Exception("Implicit solvent not supported")
            return MagicMock() # Second call succeeds
            
        minimizer.forcefield.createSystem.side_effect = create_system_side_effect
        
        with patch.dict(sys.modules, {"biotite": MagicMock(), "synth_pdb.cofactors": MagicMock(), "synth_pdb.biophysics": MagicMock()}):
            result = minimizer._run_simulation("test.pdb", "out.pdb", add_hydrogens=True)
            
        assert result is True
        # Verify createSystem was called twice
        assert minimizer.forcefield.createSystem.call_count == 2


