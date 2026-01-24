import unittest
import os

class TestDocumentationIntegrity(unittest.TestCase):
    """
    Safeguard to ensure educational notes are not accidentally removed.
    
    These tests scan the source code for specific educational content that
    must be preserved to maintain the pedagogical value of the project.
    """
    
    def setUp(self):
        # Define paths relative to this test file
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.generator_path = os.path.join(self.base_dir, 'synth_pdb', 'generator.py')
        self.bfactor_test_path = os.path.join(self.base_dir, 'tests', 'test_bfactor.py')
        self.ramachandran_test_path = os.path.join(self.base_dir, 'tests', 'test_ramachandran.py')

    def _check_file_contains(self, filepath, substrings):
        """Helper to assert file contains list of substrings."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = " ".join(f.read().split())
            
        for substring in substrings:
            normalized_substring = " ".join(substring.split())
            self.assertIn(
                normalized_substring, 
                content, 
                f"Missing educational note in {os.path.basename(filepath)}: '{substring[:50]}...'"
            )

    def test_generator_educational_notes(self):
        """Ensure generator.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - B-factors (Temperature Factors)",
            "B = 8π²<u²>",  # Physics formula
            "NMR Perspective (Order Parameters)",
            "Lipari-Szabo",
            "Typical Patterns in Real Protein Structures",
            "In this synthetic generator, we simulate B-factors that follow these universal patterns",
            "For Linear Peptides (our case):",  # Context specific note
        ]
        self._check_file_contains(self.generator_path, required_notes)

    def test_bfactor_test_educational_notes(self):
        """Ensure test_bfactor.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - What are B-factors?",
            "Backbone atoms (N, CA, C, O) are constrained by peptide bonds",
            "Glycine has no side chain",
            "Proline's cyclic structure",
            "Typical B-factors in crystal structures:",
            "In our Model-Free simulation, termini with S2 ~ 0.45",
        ]
        self._check_file_contains(self.bfactor_test_path, required_notes)

    def test_ramachandran_test_educational_notes(self):
        """Ensure test_ramachandran.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Ramachandran Plot",
            "Why Glycine is Special",
            "Why Proline is Special",
            "Left-Handed Alpha Region",
        ]
        self._check_file_contains(self.ramachandran_test_path, required_notes)
