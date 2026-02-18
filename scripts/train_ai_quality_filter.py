
import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from synth_pdb.generator import generate_pdb_content
from synth_pdb.ai.features import extract_quality_features, get_feature_names

def generate_dataset(n_samples=200):
    """Generates a balanced dataset of Good and Bad structures."""
    print(f"Generating training data ({n_samples} samples)...")
    
    import biotite.structure.io.pdb as pdb
    import biotite.structure as struc
    import io

    # 4. Bad (Single Clash)
    # Target: 40% Good, 20% Bad (Random), 20% Bad (Distorted), 20% Bad (Clashing)
    n_good = int(n_samples * 0.4)
    n_bad_random = int(n_samples * 0.2)
    n_bad_distorted = int(n_samples * 0.2)
    n_bad_clash = n_samples - n_good - n_bad_random - n_bad_distorted
    
    X = []
    y = []
    
    print(f"Dataset Split: {n_good} Good, {n_bad_random} Random, {n_bad_distorted} Distorted, {n_bad_clash} Clashing")

    # 1. Good (Alpha Helix)
    for i in range(n_good):
        if i % 10 == 0: print(f"  Good {i}/{n_good}")
        try:
            pdb_content = generate_pdb_content(length=20, conformation='alpha', minimize_energy=False)
            X.append(extract_quality_features(pdb_content))
            y.append(1)
        except: pass

    # 2. Bad (Random)
    for i in range(n_bad_random):
        if i % 10 == 0: print(f"  Random {i}/{n_bad_random}")
        try:
            pdb_content = generate_pdb_content(length=20, conformation='random', minimize_energy=False)
            X.append(extract_quality_features(pdb_content))
            y.append(0)
        except: pass

    # 3. Bad (Distorted)
    for i in range(n_bad_distorted):
        if i % 10 == 0: print(f"  Distorted {i}/{n_bad_distorted}")
        try:
            clean = generate_pdb_content(length=20, conformation='alpha', minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb.PDBFile.read(f).get_structure(model=1)
            struc_obj.coord += np.random.normal(0, 0.5, struc_obj.coord.shape)
            f_out = io.StringIO()
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            X.append(extract_quality_features(f_out.getvalue()))
            y.append(0)
        except: pass

    # 4. Bad (Single Clash)
    for i in range(n_bad_clash):
        if i % 10 == 0: print(f"  Clashing {i}/{n_bad_clash}")
        try:
            clean = generate_pdb_content(length=20, conformation='alpha', minimize_energy=False)
            f = io.StringIO(clean)
            struc_obj = pdb.PDBFile.read(f).get_structure(model=1)
            
            # Move a backbone atom (CA) to overlap another to ensure detection
            ca_indices = [i for i, a in enumerate(struc_obj) if a.atom_name == "CA"]
            if len(ca_indices) >= 5:
                # Move CA of residue 2 to CA of residue 5 (clash!)
                struc_obj.coord[ca_indices[1]] = struc_obj.coord[ca_indices[4]]
            
            f_out = io.StringIO()
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(struc_obj)
            pdb_file.write(f_out)
            X.append(extract_quality_features(f_out.getvalue()))
            y.append(0)
        except: pass
        
    X = np.array(X)
    y = np.array(y)
    
    # Print Stats
    print("\nFeature Statistics (Mean):")
    feature_names = get_feature_names()
    print(f"{'Feature':<30} | {'Good':<10} | {'Bad':<10}")
    print("-" * 56)
    for i, name in enumerate(feature_names):
        good_mean = np.mean(X[y==1][:, i]) if len(X[y==1]) > 0 else 0
        bad_mean = np.mean(X[y==0][:, i]) if len(X[y==0]) > 0 else 0
        print(f"{name:<30} | {good_mean:<10.4f} | {bad_mean:<10.4f}")
        
    return X, y

def train_model(output_path, n_samples=200):
    X, y = generate_dataset(n_samples=n_samples)
    
    print("\nTraining RandomForest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=None, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:")
    try:
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, target_names=['Bad', 'Good'], labels=[0, 1]))
    except Exception as e:
        print(f"Evaluation warning: {e}")
    
    # Feature Importance
    print("\nFeature Importance:")
    feature_names = get_feature_names()
    importances = clf.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
        
    # Save
    print(f"\nSaving model to {output_path}...")
    joblib.dump(clf, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Quality Filter Model")
    parser.add_argument("--output", default="synth_pdb/ai/models/quality_filter_v1.joblib", help="Output path for model")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of samples to generate")
    args = parser.parse_args()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    train_model(args.output, n_samples=args.n_samples)
