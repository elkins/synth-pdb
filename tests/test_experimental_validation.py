
import pytest
import requests
import json
import numpy as np
import os
import biotite.structure.io.pdb as pdb
from synth_pdb.physics import EnergyMinimizer
from synth_nmr.chemical_shifts import predict_chemical_shifts
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
    if not os.path.exists(PDB_FILE):
        print(f"Downloading {PDB_URL}...")
        response = requests.get(PDB_URL)
        response.raise_for_status()
        with open(PDB_FILE, "w") as f:
            f.write(response.text)

    if not os.path.exists(BMRB_FILE):
        print(f"Downloading {BMRB_URL}...")
        response = requests.get(BMRB_URL)
        response.raise_for_status()
        with open(BMRB_FILE, "w") as f:
            f.write(response.text)
            
    return PDB_FILE, BMRB_FILE

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
    
    # 2. Clean the PDB file from HETATM records
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = [line for line in lines if not line.startswith("HETATM")]
    
    cleaned_pdb_file = tmp_path / "1UBQ_cleaned.pdb"
    with open(cleaned_pdb_file, 'w') as f:
        f.writelines(cleaned_lines)

    # 3. Prepare and simulate the structure
    minimizer = EnergyMinimizer()
    minimized_pdb = tmp_path / "minimized.pdb"
    equilibrated_pdb = tmp_path / "equilibrated.pdb"
    
    # Add hydrogens and minimize
    success = minimizer.add_hydrogens_and_minimize(str(cleaned_pdb_file), str(minimized_pdb))
    assert success, "Energy minimization failed"
    
    # Equilibrate
    success = minimizer.equilibrate(str(minimized_pdb), str(equilibrated_pdb), steps=500) # 500 steps = 1ps
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
