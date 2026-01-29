
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.generator import generate_pdb_content

class TestCisProline:
    """
    TDD Test Suite for Cis-Proline Isomerization.
    
    EDUCATIONAL NOTE - Peptide Bond Isomerism (Cis vs Trans)
    --------------------------------------------------------
    The peptide bond (C-N) has partial double-bond character, restricting rotation.
    This leads to two planar isomers defined by the Omega (ω) angle:

    1. Trans (ω ≈ 180°):
       The side chains of adjacent residues are on opposite sides.
       This is energetically favored (~99.7% of bonds) because it minimizes steric clashes.

    2. Cis (ω ≈ 0°):
       The side chains are on the same side. This creates significant steric clash
       for most amino acids, so it is extremely rare.

    3. The Proline Exception:
       Proline has a unique cyclic side chain fused to the backbone Nitrogen.
       This makes the energy difference between Trans and Cis much smaller.
       - Non-Proline: ~0.03% Cis
       - X-Proline:   ~5.00% Cis

    Why it matters for NMR:
    Cis-Proline introduces "minor states" in protein structures that are often
    essential for function (e.g., in enzyme active sites) and are visible in NMR
    spectra as distinct peak sets. A realistic generator MUST sample this!

    This suite verifies that:
    1. We default to Trans.
    2. We can force Cis conformation via flags/probability.
    3. The implemented geometry engine correctly builds the 3D structure for cis-proline.
    """

    def _get_omega_angles(self, pdb_content):
        """Helper to extract omega angles from generated PDB content."""
        import biotite.structure.io.pdb as pdb
        import io
        
        file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = file.get_structure(model=1)
        
        # Calculate backbone dihedrals (phi, psi, omega)
        # Omega is the angle between CA_i, C_i, N_{i+1}, CA_{i+1}
        phi, psi, omega = struc.dihedral_backbone(structure)
        
        # Remove NaNs (first/last residues) and convert to degrees
        omega_deg = np.degrees(omega)
        omega_deg = omega_deg[~np.isnan(omega_deg)]
        return omega_deg

    def test_default_is_trans(self):
        """Test that by default, peptide bonds are Trans (near +/- 180)."""
        # Poly-Proline sequence
        seq = "PPPPPPPPPP"
        # Force 0% cis to ensure deterministic Trans for this test
        content = generate_pdb_content(sequence_str=seq, cis_proline_frequency=0.0)
        omegas = self._get_omega_angles(content)
        
        # Check that all are near 180 (abs > 150)
        is_trans = np.abs(omegas) > 150
        assert np.all(is_trans), f"Default generation produced non-trans bonds: {omegas}"

    def test_force_cis_proline(self):
        """
        Test that we can force Cis-Proline generation.
        """
        seq = "APA" # ALA-PRO-ALA. The A-P bond should be cis.
        
        content = generate_pdb_content(
            sequence_str=seq,
            cis_proline_frequency=1.0 # 100% probability
        )
            
        omegas = self._get_omega_angles(content)
        
        # omegas has NaNs removed. 
        # For len 3 ("APA"), original omegas are [NaN, w1, w2].
        # Filtered omegas are [w1, w2].
        # w1 is bond A-P (Residue 1).
        
        omega_pro = omegas[0] # The first valid omega is A-P
        
        # Check for CIS (near 0, between -30 and +30)
        assert abs(omega_pro) < 30.0, f"Expected Cis-Proline (omega~0), got {omega_pro}"

    def test_cis_only_on_proline(self):
        """Test that --cis-proline-frequency does NOT affect non-proline residues."""
        seq = "AAAAA" # No prolines
        
        try:
            content = generate_pdb_content(
                sequence_str=seq,
                cis_proline_frequency=1.0
            )
        except TypeError:
             pytest.fail("Argument not implemented yet")

        omegas = self._get_omega_angles(content)
        
        # All should still be Trans
        is_trans = np.abs(omegas) > 150
        assert np.all(is_trans), "Non-proline residues became cis!"
