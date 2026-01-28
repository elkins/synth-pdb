
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

