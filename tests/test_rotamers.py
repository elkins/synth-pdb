
import pytest
import numpy as np
from collections import Counter
import synth_pdb.generator
from synth_pdb.data import ROTAMER_LIBRARY

def get_chi1_angle(peptide, res_id):
    """
    Helper to calculate Chi1 angle for Valine (N-CA-CB-CG1) from a generated structure.
    """
    # Get atoms
    try:
        n = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")][0]
        ca = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")][0]
        cb = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")][0]
        cg1 = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CG1")][0]
    except IndexError:
        return None

    # Calculate dihedral
    from synth_pdb.geometry import calculate_dihedral_angle
    angle = calculate_dihedral_angle(n.coord, ca.coord, cb.coord, cg1.coord)
    return angle

def sample_rotamer_distribution(conformation, n_samples=50):
    """
    Generates N peptides of Length 1 (Just Valine) with a specific conformation
    and returns the list of observed Chi1 angles.
    """
    angles = []
    for i in range(n_samples):
        # Generate a single Valine
        # We use a length 3 sequence ALA-VAL-ALA to avoid terminal effects and ensure geometry
        # But for speed, let's try single AA or small peptide. 
        # NeRF needs atoms to build off, so usually index 0 is special.
        # Let's use a 3-residue chain and look at the middle one.
        pdb_content = synth_pdb.generator.generate_pdb_content(
            sequence_str="ALA-VAL-ALA",
            conformation=conformation,
            minimize_energy=False # Speed up
        )
        
        # Parse back (mocking IO or using biotite)
        # Using biotite directly is better but generate_pdb_content returns a string.
        # Let's use the internal _resolve_sequence helper? No, we need the 3D structure.
        # We can parse the string with biotite.
        import biotite.structure.io.pdb as pdb
        import io
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        
        # Get Chi1 of Residue 2 (VAL)
        angle = get_chi1_angle(structure, 2)
        if angle is not None:
            angles.append(angle)
            
    return angles

def classify_rotamer(angle):
    """
    Classify angle into g+ (60), g- (-60/300), t (180).
    """
    # Normalize to [-180, 180]
    angle = ((angle + 180) % 360) - 180
    
    if -120 < angle <= 0:
        return "g-" # around -60
    elif 0 < angle <= 120:
        return "g+" # around 60
    else:
        return "t"  # around 180

# @pytest.mark.skipif(True, reason="Feature not implemented yet - test expected to fail")
def test_rotamer_dependence_on_structure():
    """
    Verify that the rotamer distribution is different for Alpha Helix vs Beta Sheet.
    
    Biophysical Expectation for Valine (Dunbrack 2002):
    - Alpha Helix: 't' (trans, 180) is strongly disfavored (steric clash with backbone i-3, i-4).
                   'g-' (-60) is dominant.
    - Beta Sheet:  't' is much more allowed/favored than in helices.
                   'g-' is still common.
    
    If our implementation is backbone-INDEPENDENT (current state), the distributions will be STATISTICALLY IDENTICAL.
    If back-DEPENDENT, they should differ.
    """
    
    # 1. Sample Alpha Helix
    alpha_angles = sample_rotamer_distribution('alpha', n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])
    
    # 2. Sample Beta Sheet
    beta_angles = sample_rotamer_distribution('beta', n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])
    
    print(f"\nAlpha Counts: {alpha_counts}")
    print(f"Beta Counts:  {beta_counts}")
    
    # 3. Assert Difference
    # In the current backbone-independent code, both draw from the SAME prob distribution:
    # VAL: g- (70%), t (20%), g+ (10%)
    # So both should look roughly like 21, 6, 3
    
    # If we implement the feature correctly:
    # Alpha should have very low 't' count.
    # Beta might have higher 't' or different 'g-'.
    
    # For this failure test, we just check if they are identical (which they shouldn't be in reality, 
    # but WILL be in the current code).
    # Wait, random sampling might make them slightly different by chance.
    # But since the underlying probability is the same, they shouldn't be SYSTEMATICALLY different.
    
    # Let's strictly assert:
    # "The probability of 't' in Alpha should be significantly LOWER than in Beta"
    # Or simply: The code currently uses the SAME dict for both.
    
    # To check if we are using backbone-dependent stats, we can check if the function actually uses the new data.
    # But functional testing is better.
    
    # Assertion:
    # This assertion will FAIL currently because the distributions are drawn from the same pool.
    # We expect this test to fail now, and PASS after we implement the feature.
    
    # Note: To make it fail reliably now, we need to assert that they are DIFFERENT distributions.
    # Currently they are essentially the same.
    # It's hard to prove they are "same" with random sampling without large N.
    
    # Alternative:
    # We can inspect the logs or use mocks to see if a different library was accessed?
    # No, stick to functional.
    
    # Let's assert that observed Trans fraction in Alpha is distinct from Beta.
    # Currently: Alpha Trans ~ 20%, Beta Trans ~ 20%. -> Difference ~ 0.
    # Target: Alpha Trans < 5%, Beta Trans > 30%.
    
    alpha_trans_frac = alpha_counts['t'] / 30.0
    beta_trans_frac = beta_counts['t'] / 30.0
    
    print(f"Alpha Trans Fraction: {alpha_trans_frac}")
    print(f"Beta Trans Fraction: {beta_trans_frac}")
    
    # This should fail if they are surprisingly similar (which they are now)
    # or pass if we implemented the logic.
    assert abs(alpha_trans_frac - beta_trans_frac) > 0.15, \
        f"Rotamer distributions appear backbone-independent! Trans fractions (Alpha={alpha_trans_frac:.2f}, Beta={beta_trans_frac:.2f}) are too similar."

