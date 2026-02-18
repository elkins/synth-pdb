
import unittest
import subprocess
import sys
import os
import tempfile

class TestAICLI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.start_pdb = os.path.join(self.test_dir, "start.pdb")
        self.end_pdb = os.path.join(self.test_dir, "end.pdb")
        
        # Create dummy PDBs for interpolation test
        # We need valid PDB structure for synth-pdb to read it
        # We can use the generator to make them!
        
        # But we need to call generator from python here or just write a simple string
        pdb_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   1       1.378   2.472   0.000  1.00  0.00           O\n"
            "TER\n"
        )
        with open(self.start_pdb, "w") as f:
            f.write(pdb_content)
        with open(self.end_pdb, "w") as f:
            f.write(pdb_content) # same content is fine for basic CLI check

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def run_command(self, args):
        cmd = [sys.executable, "-m", "synth_pdb.main"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir)
        return result

    def test_cli_interpolate(self):
        # synth-pdb --mode ai --ai-op interpolate --start <p1> --end <p2> --steps 1 --output morph
        # output will be morph_0.pdb, morph_1.pdb
        
        cmd = [
            "--mode", "ai",
            "--ai-op", "interpolate",
            "--start-pdb", self.start_pdb,
            "--end-pdb", self.end_pdb,
            "--steps", "1",
            "--output", "morph" # Prefix
        ]
        
        result = self.run_command(cmd)
        
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "morph_0.pdb")))

    def test_cli_ai_filter(self):
        # synth-pdb --length 10 --ai-filter --ai-score-cutoff 0.0 --out filtered.pdb
        # Using cutoff 0.0 guarantees pass
        # Using length 5 to be fast
        
        # Need to ensure model exists or mock it? 
        # The environment should have the model from previous steps.
        # But if not, this relies on global state.
        # Ideally functional tests run in configured env.
        
        # Check if model exists first?
        if not os.path.exists("synth_pdb/ai/models/quality_filter_v1.joblib"):
             print("Skipping AI Filter CLI test because model file is missing.")
             return

        cmd = [
            "--length", "5",
            "--ai-filter",
            "--ai-score-cutoff", "0.0",
            "--output", "filtered.pdb"
        ]
        
        result = self.run_command(cmd)
        
        if result.returncode != 0:
             # It might fail if scikit-learn not installed (but we are running this test..)
             print("STDERR:", result.stderr)
             
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "filtered.pdb")))

if __name__ == '__main__':
    unittest.main()
