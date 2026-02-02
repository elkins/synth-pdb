
import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import HAS_OPENMM

class TestTopologicalComplexity:
    """Verifies that complex topologies (Cyclic + Disulfide) work correctly together."""

    @pytest.mark.skipif(not HAS_OPENMM, reason="Cyclization requires OpenMM physics engine")
    def test_cyclic_with_disulfide(self):
        """
        Verify a peptide that is both cyclic AND has a disulfide bond.
        Sequence: C G G G G C
        """
        seq = "CGGGGC"
        pdb_content = generate_pdb_content(
            sequence_str=seq,
            cyclic=True,
            minimize_energy=True
        )
        
        # 1. Verify SSBOND record (Disulfide)
        assert "SSBOND" in pdb_content, "SSBOND record missing in cyclic-disulfide peptide"
        
        # 2. Verify CONECT records (Cyclic Backbone AND Disulfide Bridge)
        assert "CONECT" in pdb_content, "CONECT records missing in cyclic-disulfide peptide"
        
        # Verify specific CONECT indices for the disulfide bridge
        # In CGGGGC: C1 and C6 are the cysteines.
        # SG atoms are usually atoms 6 and 45 (approx based on earlier runs)
        # But we should find them dynamically.
        lines = pdb_content.split('\n')
        atom_lines = [l for l in lines if l.startswith('ATOM')]
        sg_indices = []
        for line in atom_lines:
            if line[12:16].strip() == 'SG':
                sg_indices.append(int(line[6:11].strip()))
        
        assert len(sg_indices) == 2, f"Expected 2 SG atoms, found {len(sg_indices)}"
        
        import re
        conect_lines = [l for l in lines if l.startswith('CONECT')]
        ss_conect_found = False
        for line in conect_lines:
            parts = [int(p) for p in re.findall(r'\d+', line[6:])]
            if sg_indices[0] in parts and sg_indices[1] in parts:
                ss_conect_found = True
                break
        
        assert ss_conect_found, f"Disulfide CONECT missing between {sg_indices[0]} and {sg_indices[1]}"
        
        # 3. Verify N-C Closure
        import io
        import biotite.structure.io.pdb as pdb
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        n_atom = structure[(structure.res_id == 1) & (structure.atom_name == "N")][0]
        c_atom = structure[(structure.res_id == 6) & (structure.atom_name == "C")][0]
        
        dist = np.sqrt(np.sum((n_atom.coord - c_atom.coord)**2))
        # Strained 6-mer rings can have significantly stretched bonds (~1.5A+).
        assert dist < 1.60, f"Cyclic bond failed! Dist: {dist:.3f}"
        
        # 4. Verify S-S Distance
        # In CGGGGC cyclic, 1 and 6 are close.
        sg1 = structure[(structure.res_id == 1) & (structure.atom_name == "SG")][0]
        sg6 = structure[(structure.res_id == 6) & (structure.atom_name == "SG")][0]
        ss_dist = np.sqrt(np.sum((sg1.coord - sg6.coord)**2))
        # Minimized structures can be highly strained (especially small cyclic rings),
        # matching the generator's generous range [1.5, 3.0].
        assert 1.5 <= ss_dist <= 3.0, f"Disulfide bond distance invalid in cyclic peptide! Dist: {ss_dist:.3f}"
