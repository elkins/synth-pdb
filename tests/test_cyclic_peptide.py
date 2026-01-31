
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import HAS_OPENMM

class TestCyclicPeptides:
    """
    Test suite for Head-to-Tail Cyclic Peptide verification.
    Follows TDD: These tests define the expected behavior before implementation.
    """

    @pytest.mark.skipif(not HAS_OPENMM, reason="Cyclization requires OpenMM physics engine")
    def test_cyclic_geometry_closure(self):
        """
        Verify that a 5-residue Poly-Glycine ring is actually closed.
        
        Physics Goal:
        The distance between N-terminus (N) and C-terminus (C) should be ~1.33 Angstroms
        (standard peptide bond length), not >3.0 Angstroms (open chain).
        """
        # 1. Generate Cyclic Peptide (Expected API)
        # We use Glycine because it's flexible and easiest to cyclize.
        pdb_content = generate_pdb_content(
            sequence_str="GGGGG",
            cyclic=True,        # <--- New Flag
            minimize_energy=True # Required for closure
        )
        
        # 2. Parse
        import io
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        # 3. Identify Termini
        # Note: In a cyclic peptide, residue 1 is bonded to residue N.
        n_term_res = structure[structure.res_id == 1]
        c_term_res = structure[structure.res_id == 5]
        
        # Get Backbone Atoms
        n_atom = n_term_res[n_term_res.atom_name == "N"][0]
        c_atom = c_term_res[c_term_res.atom_name == "C"][0]
        
        # 4. Measure Distance
        dist = np.linalg.norm(n_atom.coord - c_atom.coord)
        
        # 5. Assert Closure
        # Bond length ~ 1.33 A. Allow some tolerance for strain (1.45 A max).
        print(f"Cyclic Bond Distance (N1-C5): {dist:.3f} A")
        assert dist < 1.45, f"Ring invalid! N-C distance is {dist:.3f} A (Expected ~1.33 A)"

    @pytest.mark.skipif(not HAS_OPENMM, reason="Cyclization requires OpenMM physics engine")
    def test_cyclic_removes_termini(self):
        """
        Verify that cyclic peptides do NOT have terminal capping groups.
        A cyclic peptide has no start or end, so:
        - N-terminus should NOT have H1/H2/H3 (only H).
        - C-terminus should NOT have OXT.
        """
        pdb_content = generate_pdb_content(
            sequence_str="GGGG",
            cyclic=True,
            minimize_energy=True
        )
        
        # Check for forbidden atoms
        assert "OXT" not in pdb_content, "Cyclic peptide should not have C-terminal Oxygen (OXT)"
        assert "H1 " not in pdb_content, "Cyclic peptide should not have N-terminal protons (H1)"


    def test_cyclic_with_proline_minimization(self):
        """Test that cyclic peptides containing Proline can be minimized without crashing."""
        seq = "GPGPG"
        try:
            pdb_content = generate_pdb_content(sequence_str=seq, cyclic=True, minimize_energy=True)
            assert "PRO" in pdb_content
        except Exception as e:
            pytest.fail(f"Minimization failed for cyclic PRO-peptide: {e}")

    def test_cyclic_conect_generation(self):
        """Verify that CONECT records are generated for cyclic peptides."""
        seq = "GGGGG"
        pdb_content = generate_pdb_content(sequence_str=seq, cyclic=True)
        
        # Find CONECT records
        lines = [line for line in pdb_content.splitlines() if line.startswith("CONECT")]
        assert len(lines) > 0, "No CONECT records found for cyclic peptide"
        
        # Check that it connects the N-term (atom 1 usually) and C-term
        # Actually we should look for the specific serials from the start/end
        # Finding atom 1 (N) and last atom (C)
        atomic_lines = [l for l in pdb_content.splitlines() if l.startswith("ATOM")]
        n_serial = int(atomic_lines[0][6:11].strip())
        
        # Find the last 'C' atom serial (not the OXT/O)
        c_serial = None
        last_res_num = len(seq)
        for l in reversed(atomic_lines):
            rnum = int(l[22:26].strip())
            aname = l[12:16].strip()
            if rnum == last_res_num and aname == 'C':
                c_serial = int(l[6:11].strip())
                break
        
        assert c_serial is not None, "Could not find last C atom serial"
        
        # Verify a CONECT line contains both
        found_link = False
        for line in lines:
            if f"{n_serial:5d}" in line and f"{c_serial:5d}" in line:
                found_link = True
                break
        assert found_link, f"CONECT record missing for link {n_serial} <-> {c_serial}"
