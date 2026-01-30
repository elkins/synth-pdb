
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.generator import generate_pdb_content
from synth_pdb.data import BETA_TURN_TYPES

class TestBetaTurns:
    """
    TDD Test Suite for Beta-Turn Geometries.
    """

    def _get_backbone_angles(self, pdb_content):
        import io
        pdb_file = struc.io.pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        phi, psi, omega = struc.dihedral_backbone(structure)
        return np.degrees(phi), np.degrees(psi)


    def test_type_ii_turn_dihedrals(self):
        """
        Request a Type II turn at residues 2-5 (indices 1-4).
        Turn residues: i, i+1, i+2, i+3.
        Defined angles for i+1 (Res 3) and i+2 (Res 4).
        """
        # Sequence: A-A-G-A-A (5 residues)
        # Turn at 2-5? No, a turn is 4 residues.
        # Let's say residues 1-4 are the turn.
        # i=1, i+1=2, i+2=3, i+3=4.
        # Angels define Res 2 and Res 3.
        
        # We need a longer sequence to avoid end effects?
        # Let's use 10 residues. Turn at 3-6.
        # i=3, i+1=4, i+2=5, i+3=6.
        # Target angles for Res 4 and Res 5.
        
        seq = "AAAAAAAAAA" # All Ala
        # Structure string: "1-2:alpha,3-6:typeII,7-10:alpha"
        
        # NOTE: generator structure uses 1-based indexing.
        content = generate_pdb_content(
            sequence_str=seq,
            structure="3-6:typeII" # Unspecified defaults to alpha/random
        )
        
        phi, psi = self._get_backbone_angles(content)
        
        # Check Res 4 (Index 3)
        # Type II: i+1 = (-60, 120)
        p4, ps4 = phi[3], psi[3]
        target_p4, target_ps4 = BETA_TURN_TYPES['typeII'][0]
        
        assert np.isclose(p4, target_p4, atol=5.0), f"Res 4 Phi {p4} != {target_p4}"
        assert np.isclose(ps4, target_ps4, atol=5.0), f"Res 4 Psi {ps4} != {target_ps4}"
        
        # Check Res 5 (Index 4)
        # Type II: i+2 = (80, 0)
        p5, ps5 = phi[4], psi[4]
        target_p5, target_ps5 = BETA_TURN_TYPES['typeII'][1]
        
        assert np.isclose(p5, target_p5, atol=5.0), f"Res 5 Phi {p5} != {target_p5}"
        assert np.isclose(ps5, target_ps5, atol=5.0), f"Res 5 Psi {ps5} != {target_ps5}"

    def test_invalid_turn_length(self):
        """Turn regions MUST be exactly 4 residues long."""
        with pytest.raises(ValueError, match="must be.*4 residues"):
             generate_pdb_content(sequence_str="AAAAA", structure="1-3:typeI") # 3 residues
             
    def test_multiple_turns(self):
        """Two turns in one peptide."""
        seq = "AAAAAAAAAAAA" # 12 residues
        # 1-4: Type I
        # 9-12: Type II
        content = generate_pdb_content(
            sequence_str=seq,
            structure="1-4:typeI,9-12:typeII"
        )
        phi, psi = self._get_backbone_angles(content)
        
        # Check Type I at Res 2 (Index 1)
        p2, ps2 = phi[1], psi[1]
        t1_p, t1_ps = BETA_TURN_TYPES['typeI'][0] 
        assert np.isclose(p2, t1_p, atol=2.0)
        
        # Check Type II at Res 10 (Index 9)
        p10, ps10 = phi[9], psi[9]
        t2_p, t2_ps = BETA_TURN_TYPES['typeII'][0]
        assert np.isclose(p10, t2_p, atol=2.0)
