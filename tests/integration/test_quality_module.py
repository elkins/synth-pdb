
import os
import unittest
import numpy as np
import tempfile
import pytest

joblib = pytest.importorskip("joblib", reason="joblib not installed; install synth-pdb[ai]")

from synth_pdb.generator import generate_pdb_content
from synth_pdb.quality.features import extract_quality_features, get_feature_names
from synth_pdb.quality.classifier import ProteinQualityClassifier
from synth_pdb.quality.interpolate import interpolate_structures

class TestQualityModules(unittest.TestCase):
    def setUp(self):
        # Generate two simple PDBs for testing
        self.pdb1_content = generate_pdb_content(length=10, sequence_str="AAAAAAAAAA", conformation="alpha")
        self.pdb2_content = generate_pdb_content(length=10, sequence_str="AAAAAAAAAA", conformation="extended")
        
        self.test_dir = tempfile.mkdtemp()
        self.pdb1_path = os.path.join(self.test_dir, "start.pdb")
        self.pdb2_path = os.path.join(self.test_dir, "end.pdb")
        
        with open(self.pdb1_path, "w") as f:
            f.write(self.pdb1_content)
        with open(self.pdb2_path, "w") as f:
            f.write(self.pdb2_content)

    def test_feature_extraction(self):
        print("\nTesting Feature Extraction...")
        features = extract_quality_features(self.pdb1_content)
        feature_names = get_feature_names()
        
        self.assertEqual(len(features), len(feature_names))
        print(f"Features extraction successful. Vector shape: {features.shape}")
        # Basic checks
        self.assertTrue(0 <= features[0] <= 100) # Rama favored %

    def test_classifier(self):
        print("\nTesting Quality Classifier...")
        # Ensure model exists (it should have been trained by previous step)
        model_path = "synth_pdb/quality/models/quality_filter_v1.joblib"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, skipping classifier test.")
            return

        clf = ProteinQualityClassifier(model_path=model_path)
        is_good, prob, feats = clf.predict(self.pdb1_content)
        
        print(f"Prediction: Good={is_good}, Prob={prob:.2f}")
        self.assertIsInstance(is_good, (bool, np.bool_))
        self.assertTrue(0 <= prob <= 1.0)
        
    def test_interpolation(self):
        print("\nTesting Interpolation...")
        out_prefix = os.path.join(self.test_dir, "morph")
        steps = 3
        
        try:
            interpolate_structures(self.pdb1_path, self.pdb2_path, steps, out_prefix)
            
            # Check outputs
            for i in range(steps + 1):
                params_path = f"{out_prefix}_{i}.pdb"
                self.assertTrue(os.path.exists(params_path), f"Output {params_path} missing")
                print(f"Generated frame {i}: {params_path}")
                
        except ImportError:
            print("Skipping interpolation test (missing dependencies?)")
        except Exception as e:
            self.fail(f"Interpolation failed: {e}")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
