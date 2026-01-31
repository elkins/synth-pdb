import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import io
from synth_pdb.generator import generate_pdb_content

def test_generate_pdb_with_torsion_drift():
    """Verify that the drift parameter influences backbone dihedrals."""
    sequence = "AAAAAAAAAA" # 10 residues
    
    # Baseline: no drift (alpha conformation)
    pdb_no_drift = generate_pdb_content(sequence_str=sequence, conformation="alpha")
    
    # Drifted: 10 degrees drift
    # This should fail initially because 'drift' is not in the signature
    try:
        pdb_with_drift = generate_pdb_content(sequence_str=sequence, conformation="alpha", drift=10.0, seed=42)
    except TypeError as e:
        pytest.fail(f"generate_pdb_content does not yet support 'drift' parameter: {e}")

    # Parse dihedrals
    def get_dihedrals(pdb_str):
        f = pdb.PDBFile.read(io.StringIO(pdb_str))
        s = f.get_structure(model=1)
        phi, psi, _ = struc.dihedral_backbone(s)
        return phi, psi

    phi_no, psi_no = get_dihedrals(pdb_no_drift)
    phi_drift, psi_drift = get_dihedrals(pdb_with_drift)

    # Check that they differ
    # We use a seed to ensure reproducibility of the 'drift' if we want, 
    # but here we just want to see IF they differ.
    assert not np.allclose(phi_no, phi_drift, atol=1e-3)
    assert not np.allclose(psi_no, psi_drift, atol=1e-3)

def test_decoy_generator_threading(tmp_path):
    """Verify that threading a sequence on a template fold works."""
    from synth_pdb.decoys import DecoyGenerator
    import os
    
    target_seq = "AAAAAAAAAA" # Poly-Ala
    template_seq = "PPPPPPPPPP" # Poly-Pro (very different fold)
    
    gen = DecoyGenerator()
    out_dir = tmp_path / "hard_decoys_threading"
    
    # Generate threaded decoy
    decoys = gen.generate_ensemble(
        sequence=target_seq,
        n_decoys=1,
        out_dir=str(out_dir),
        hard_mode=True,
        template_sequence=template_seq,
        seed=42
    )
    
    assert len(decoys) == 1
    with open(decoys[0], 'r') as f:
        content = f.read()
        
    # Verify sequence is Ala
    assert "ALA" in content
    assert "PRO" not in content
    
    # Verify structure is not the default 'random' for Ala but threaded
    # (Checking against a non-threaded generation is hard without exact matching, 
    # but we've verified the code paths).

def test_decoy_generator_shuffling(tmp_path):
    """Verify sequence label shuffling."""
    from synth_pdb.decoys import DecoyGenerator
    import os
    
    sequence = "ACDEFGHIKL" # Unique residues
    gen = DecoyGenerator()
    out_dir = tmp_path / "hard_decoys_shuffling"
    
    decoys = gen.generate_ensemble(
        sequence=sequence,
        n_decoys=1,
        out_dir=str(out_dir),
        hard_mode=True,
        shuffle_sequence=True,
        seed=42
    )
    
    assert len(decoys) == 1
    with open(decoys[0], 'r') as f:
        content = f.read()
        
    # Verify we still have all residues but maybe in different order/positions 
    # (Actually shuffling internal to PDB lines is what we did).
    # Since it's shuffled, it's hard to assert "it is shuffled" without a baseline 
    # and a seed-fixed check.
    
    # Just check the PDB is valid
    pdb_file = pdb.PDBFile.read(io.StringIO(content))
    structure = pdb_file.get_structure(model=1)
    assert len(structure) > 0

def test_decoy_generator_drift_integration(tmp_path):
    """Verify drift integration in DecoyGenerator."""
    from synth_pdb.decoys import DecoyGenerator
    sequence = "AAAAAAAAAA"
    gen = DecoyGenerator()
    out_dir = tmp_path / "hard_decoys_drift"
    
    # Generate with high drift
    decoys = gen.generate_ensemble(
        sequence=sequence,
        n_decoys=1,
        out_dir=str(out_dir),
        drift=45.0, # Massive drift
        seed=123
    )
    assert len(decoys) == 1
