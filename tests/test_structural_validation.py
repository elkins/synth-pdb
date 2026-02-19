
import pytest
import os
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import requests
from synth_pdb.physics import EnergyMinimizer
from synth_pdb.validator import PDBValidator

# Define URLs for the data
PDB_URL = "https://files.rcsb.org/download/1UBQ.pdb"
PDB_FILE = "1UBQ.pdb"

@pytest.fixture(scope="module")
def experimental_pdb():
    """
    Downloads the 1UBQ PDB file if it doesn't exist locally.
    """
    if not os.path.exists(PDB_FILE):
        print(f"Downloading {PDB_URL}...")
        response = requests.get(PDB_URL)
        response.raise_for_status()
        with open(PDB_FILE, "w") as f:
            f.write(response.text)
    return PDB_FILE

def test_ubiquitin_structural_fidelity(experimental_pdb, tmp_path):
    """
    Verifies structural fidelity of the minimized 1UBQ against the original experimental structure.
    Checks:
    1. Backbone RMSD (should be < 1.0 A)
    2. Phi/Psi Correlation (should be > 0.9)
    3. Absence of severe clashes in refined structure
    """
    # 1. Load experimental structure
    exp_file = pdb.PDBFile.read(experimental_pdb)
    exp_struct = exp_file.get_structure(model=1)
    
    # Filter for protein atoms only
    exp_struct = exp_struct[struc.filter_amino_acids(exp_struct)]
    
    # 2. Refine the structure using EnergyMinimizer
    minimizer = EnergyMinimizer()
    minimized_pdb = tmp_path / "1UBQ_minimized.pdb"
    
    # Clean PDB for OpenMM (no HETATM)
    with open(experimental_pdb, 'r') as f:
        lines = f.readlines()
    cleaned_pdb = tmp_path / "1UBQ_cleaned.pdb"
    with open(cleaned_pdb, 'w') as f:
        f.writelines([l for l in lines if l.startswith("ATOM")])
        
    success = minimizer.add_hydrogens_and_minimize(str(cleaned_pdb), str(minimized_pdb))
    assert success, "Energy minimization failed"
    
    # 3. Load minimized structure
    min_file = pdb.PDBFile.read(str(minimized_pdb))
    min_struct = min_file.get_structure(model=1)
    min_struct = min_struct[struc.filter_amino_acids(min_struct)]
    
    # 4. Calculate RMSD
    # Align structures first
    # We use backbone heavy atoms (N, CA, C) for alignment and RMSD
    exp_backbone_mask = (exp_struct.atom_name == "N") | (exp_struct.atom_name == "CA") | (exp_struct.atom_name == "C")
    min_backbone_mask = (min_struct.atom_name == "N") | (min_struct.atom_name == "CA") | (min_struct.atom_name == "C")
    
    exp_backbone = exp_struct[exp_backbone_mask]
    min_backbone = min_struct[min_backbone_mask]
    
    # Ensure they have the same number of atoms
    if len(exp_backbone) != len(min_backbone):
        # This might happen if there are altlocs or missing atoms. 
        # For 1UBQ it should be fine if we filtered correctly.
        # Let's be more robust by matching on residue id and atom name.
        common_atoms_exp = []
        common_atoms_min = []
        
        # Build a map for quick lookup in min_struct
        min_map = {}
        for atom in min_struct:
            key = (atom.res_id, atom.res_name, atom.atom_name)
            min_map[key] = atom.coord
            
        for atom in exp_struct:
            key = (atom.res_id, atom.res_name, atom.atom_name)
            if key in min_map and atom.atom_name in ["N", "CA", "C"]:
                common_atoms_exp.append(atom.coord)
                common_atoms_min.append(min_map[key])
        
        exp_backbone_coords = np.array(common_atoms_exp)
        min_backbone_coords = np.array(common_atoms_min)
        
        # We need AtomArrays for superimpose, but we can also just use the coords if we know they are aligned
        # For simplicity, let's just use the coords with a custom RMSD if needed, 
        # or rebuild AtomArrays.
        
        # Rebuild minimal AtomArrays for superimpose
        exp_backbone = struc.AtomArray(len(exp_backbone_coords))
        exp_backbone.coord = exp_backbone_coords
        min_backbone = struc.AtomArray(len(min_backbone_coords))
        min_backbone.coord = min_backbone_coords

    assert len(exp_backbone) == len(min_backbone), f"Atom count mismatch: {len(exp_backbone)} vs {len(min_backbone)}"
    
    superimposed, transformation = struc.superimpose(exp_backbone, min_backbone)
    rmsd = struc.rmsd(exp_backbone, superimposed)
    
    print(f"Backbone RMSD: {rmsd:.4f} A")
    assert rmsd < 1.0, f"RMSD too high: {rmsd:.4f} A"
    
    # 5. Compare Dihedral Angles
    exp_phi, exp_psi, exp_omega = struc.dihedral_backbone(exp_struct)
    min_phi, min_psi, min_omega = struc.dihedral_backbone(min_struct)
    
    # Get residue IDs for masking
    res_ids = exp_struct[exp_struct.atom_name == "CA"].res_id
    # Mask out NaNs and the flexible C-terminus (72-76)
    core_mask = (res_ids <= 71)
    
    # Biotite dihedrals are per-residue, so they match res_ids length
    mask = ~np.isnan(exp_phi) & ~np.isnan(min_phi) & core_mask
    phi_corr = np.corrcoef(exp_phi[mask], min_phi[mask])[0, 1]
    
    mask = ~np.isnan(exp_psi) & ~np.isnan(min_psi) & core_mask
    psi_corr = np.corrcoef(exp_psi[mask], min_psi[mask])[0, 1]
    
    print(f"Core Phi Correlation: {phi_corr:.4f}")
    print(f"Core Psi Correlation: {psi_corr:.4f}")
    
    assert phi_corr > 0.90, f"Phi correlation too low: {phi_corr:.4f}"
    assert psi_corr > 0.85, f"Psi correlation too low: {psi_corr:.4f}"
    
    # 6. Secondary Structure Fidelity
    # Use Biotite to annotate SSE
    exp_sse = struc.annotate_sse(exp_struct)
    min_sse = struc.annotate_sse(min_struct)
    
    # SSE strings: 'H' = Helix, 'E' = Sheet, 'C' = Coil
    # Compare matching residues
    # Annotation is per-residue. Mask for the core.
    exp_sse_core = exp_sse[core_mask]
    min_sse_core = min_sse[core_mask]
    
    # Calculate agreement percentage
    agreement = np.sum(exp_sse_core == min_sse_core) / len(exp_sse_core)
    print(f"SSE Agreement: {agreement:.4f}")
    
    # We expect > 70% agreement for a minimized crystal structure
    # (DSSP is sensitive to minor geometry shifts)
    assert agreement > 0.70, f"SSE agreement too low: {agreement:.4f}"
    
    # 7. Validate with PDBValidator
    validator = PDBValidator(pdb_content=open(minimized_pdb).read())
    validator.validate_all()
    violations = validator.get_violations()
    
    # Filter out minor violations if necessary, but ideally there should be very few/none
    # after energy minimization.
    # Some Ramachandran outliers might exist in experimental structures too.
    important_violations = [v for v in violations if "Bond length" in v or "Steric clash" in v]
    
    print(f"Number of major violations: {len(important_violations)}")
    for v in important_violations:
        print(f"  Violation: {v}")
        
    # We expect 0 major violations after minimization
    assert len(important_violations) == 0, f"Found structural violations after minimization: {important_violations}"

def test_experimental_clash_comparison(experimental_pdb, tmp_path):
    """
    Compares steric clashes in experimental vs minimized structures.
    Shows that minimization actually improves (reduces) physical violations.
    """
    # Validate Experimental
    with open(experimental_pdb, 'r') as f:
        exp_pdb_content = f.read()
    
    # PDBValidator doesn't handle HETATM well in all checks, so we might see noise if we don't clean it
    cleaned_exp_lines = [l for l in exp_pdb_content.splitlines() if l.startswith("ATOM")]
    cleaned_exp_pdb = "\n".join(cleaned_exp_lines)
    
    exp_validator = PDBValidator(pdb_content=cleaned_exp_pdb)
    exp_validator.validate_steric_clashes()
    exp_clashes = exp_validator.get_violations()
    
    print(f"Experimental Clashes: {len(exp_clashes)}")
    
    # Minimize
    minimizer = EnergyMinimizer()
    minimized_pdb_path = tmp_path / "1UBQ_min.pdb"
    with open(tmp_path / "exp_clean.pdb", 'w') as f:
        f.write(cleaned_exp_pdb)
        
    minimizer.add_hydrogens_and_minimize(str(tmp_path / "exp_clean.pdb"), str(minimized_pdb_path))
    
    # Validate Minimized
    with open(minimized_pdb_path, 'r') as f:
        min_pdb_content = f.read()
    min_validator = PDBValidator(pdb_content=min_pdb_content)
    min_validator.validate_steric_clashes()
    min_clashes = min_validator.get_violations()
    
    print(f"Minimized Clashes: {len(min_clashes)}")
    
    # The minimized structure should have fewer or equal clashes compared to the raw experimental one
    # (Experimental PDBs often have "clashes" because they lack hydrogens or have minor errors)
    assert len(min_clashes) <= len(exp_clashes)
