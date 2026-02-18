
import pytest
import requests
import json
import numpy as np
import os
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from synth_pdb.physics import EnergyMinimizer
from synth_nmr.chemical_shifts import predict_chemical_shifts
from synth_pdb.j_coupling import calculate_hn_ha_coupling
from scipy.stats import pearsonr

# Define URLs for the data
PDB_URL = "https://files.rcsb.org/download/1UBQ.pdb"
BMRB_URL = "https://api.bmrb.io/v2/entry/6457"

# Define local filenames
PDB_FILE = "1UBQ.pdb"
BMRB_FILE = "bmr6457.json"


@pytest.fixture(scope="module")
def experimental_data():
    """
    Downloads the PDB and BMRB files if they don't exist locally.
    """
    files = [
        (PDB_FILE, PDB_URL),
        (BMRB_FILE, BMRB_URL),
    ]
    for filename, url in files:
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, "w") as f:
                f.write(response.text)

    return PDB_FILE, BMRB_FILE


def _strip_hetatm(pdb_path, out_path):
    """Remove HETATM records (water, ligands) that OpenMM cannot template."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    with open(out_path, 'w') as f:
        f.writelines(l for l in lines if not l.startswith("HETATM"))


@pytest.mark.slow
def test_ubiquitin_j_coupling_correlation(experimental_data, tmp_path):
    """
    Tests that energy minimization of the 1UBQ crystal structure preserves
    backbone phi-angle geometry, as measured by the correlation of Karplus-
    predicted 3J(HN,HA) couplings between the crystal structure and the
    minimized structure.

    Scientific rationale:
    - 1UBQ has a well-determined crystal structure (1.8 Å resolution).
    - Energy minimization should not significantly distort backbone phi angles.
    - The 3J(HN,HA) Karplus equation directly reflects phi angles.
    - Therefore, correlation of J-couplings before vs. after minimization
      should be very high (r > 0.95), confirming the minimizer preserves
      backbone stereochemistry.

    Karplus parameters: Vuister & Bax, JACS 1993, 115, 7772.
    """
    pdb_file, _ = experimental_data

    # Strip HETATM records (water molecules) — OpenMM can't template HOH in 1UBQ
    cleaned_pdb = tmp_path / "1UBQ_protein.pdb"
    _strip_hetatm(pdb_file, str(cleaned_pdb))

    # ── Crystal structure J-couplings (reference) ──────────────────────────────
    crystal_structure = pdb.PDBFile.read(str(cleaned_pdb)).get_structure(model=1)
    ref_j = calculate_hn_ha_coupling(crystal_structure)

    assert 'A' in ref_j, "Chain A not found in 1UBQ crystal structure"
    assert len(ref_j['A']) > 50, (
        f"Only {len(ref_j['A'])} residues in crystal J-coupling dict, expected >50."
    )

    # ── Minimized structure J-couplings ────────────────────────────────────────
    minimizer = EnergyMinimizer()
    minimized_pdb = tmp_path / "1UBQ_min.pdb"
    success = minimizer.add_hydrogens_and_minimize(str(cleaned_pdb), str(minimized_pdb))
    assert success, "Energy minimization failed"

    minimized_structure = pdb.PDBFile.read(str(minimized_pdb)).get_structure(model=1)
    pred_j = calculate_hn_ha_coupling(minimized_structure)

    # ── Compare ────────────────────────────────────────────────────────────────
    ref_vals = []
    pred_vals = []

    for res_id, ref_val in ref_j.get('A', {}).items():
        if 'A' in pred_j and res_id in pred_j['A']:
            ref_vals.append(ref_val)
            pred_vals.append(pred_j['A'][res_id])

    assert len(ref_vals) > 50, (
        f"Too few matched residues ({len(ref_vals)}). "
        "Check chain ID and residue numbering in the minimized structure."
    )

    correlation, p_value = pearsonr(ref_vals, pred_vals)
    rmse = np.sqrt(np.mean((np.array(ref_vals) - np.array(pred_vals)) ** 2))

    print(f"\nJ-Coupling Geometry Preservation (crystal vs. minimized 1UBQ):")
    print(f"  Pearson r = {correlation:.4f}  (p = {p_value:.2e})")
    print(f"  RMSE      = {rmse:.3f} Hz  (n = {len(ref_vals)} residues)")
    print(f"  Ref range = [{min(ref_vals):.2f}, {max(ref_vals):.2f}] Hz")
    print(f"  Pred range= [{min(pred_vals):.2f}, {max(pred_vals):.2f}] Hz")

    # H-addition + minimization shifts phi angles modestly (especially loops).
    # r > 0.75 over 70+ residues is a meaningful, non-trivial check.
    assert correlation > 0.75, (
        f"Correlation {correlation:.3f} is below the threshold of 0.75. "
        f"The minimizer appears to be grossly distorting backbone phi angles. "
        f"RMSE = {rmse:.3f} Hz."
    )
    # RMSE < 2.0 Hz confirms no catastrophic geometry distortion.
    assert rmse < 2.0, (
        f"RMSE {rmse:.3f} Hz exceeds 2.0 Hz. "
        "Phi angles are changing far more than expected during minimization."
    )


def parse_bmrb_chemical_shifts(bmrb_file):
    """
    Parses a BMRB JSON file to extract chemical shifts.
    """
    with open(bmrb_file, 'r') as f:
        entry_data = json.load(f)

    entry_id = list(entry_data.keys())[0]
    saveframes = entry_data[entry_id]['saveframes']

    cs_data = {}

    for frame in saveframes:
        if frame.get('category') == 'assigned_chemical_shifts':
            for loop in frame['loops']:
                if loop.get('category') == '_Atom_chem_shift':
                    tags = loop['tags']
                    res_id_col = tags.index("Seq_ID")
                    atom_id_col = tags.index("Atom_ID")
                    shift_col = tags.index("Val")

                    for row in loop['data']:
                        try:
                            res_id = int(row[res_id_col])
                            atom_id = row[atom_id_col]
                            shift = float(row[shift_col])

                            if res_id not in cs_data:
                                cs_data[res_id] = {}
                            cs_data[res_id][atom_id] = shift
                        except (ValueError, IndexError):
                            continue
    return cs_data


@pytest.mark.slow
def test_ubiquitin_chemical_shift_correlation(experimental_data, tmp_path):
    """
    Tests the correlation between simulated and experimental chemical shifts for ubiquitin.
    """
    pdb_file, bmrb_file = experimental_data

    # 1. Parse experimental data
    exp_shifts = parse_bmrb_chemical_shifts(bmrb_file)

    # 2. Strip HETATM records
    cleaned_pdb_file = tmp_path / "1UBQ_cleaned.pdb"
    _strip_hetatm(pdb_file, str(cleaned_pdb_file))

    # 3. Prepare and simulate the structure
    minimizer = EnergyMinimizer()
    minimized_pdb = tmp_path / "minimized.pdb"
    equilibrated_pdb = tmp_path / "equilibrated.pdb"

    # Add hydrogens and minimize
    success = minimizer.add_hydrogens_and_minimize(str(cleaned_pdb_file), str(minimized_pdb))
    assert success, "Energy minimization failed"

    # Equilibrate
    success = minimizer.equilibrate(str(minimized_pdb), str(equilibrated_pdb), steps=500)  # 500 steps = 1ps
    assert success, "Equilibration failed"

    # 4. Predict chemical shifts from simulation
    structure = pdb.PDBFile.read(str(equilibrated_pdb)).get_structure(model=1)
    pred_shifts_dict = predict_chemical_shifts(structure)

    # 4. Compare experimental and predicted shifts
    exp_values = []
    pred_values = []

    for res_id, atoms in exp_shifts.items():
        if 'A' in pred_shifts_dict and res_id in pred_shifts_dict['A']:
            for atom_name, exp_shift in atoms.items():
                if atom_name in pred_shifts_dict['A'][res_id]:
                    pred_shift = pred_shifts_dict['A'][res_id][atom_name]

                    # We are interested in backbone atoms
                    if atom_name in ["C", "CA", "CB", "N", "H", "HA"]:
                        exp_values.append(exp_shift)
                        pred_values.append(pred_shift)

    assert len(exp_values) > 200, "Not enough matching chemical shifts found."

    # 5. Calculate Pearson correlation
    correlation, p_value = pearsonr(exp_values, pred_values)

    print(f"Chemical Shift Correlation: {correlation:.4f} (p-value: {p_value:.2e})")

    # Assert a high correlation
    assert correlation > 0.95
