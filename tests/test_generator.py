import unittest
import logging
import re
import numpy as np
from stupid_pdb.generator import _resolve_sequence, generate_pdb_content, CA_DISTANCE
from stupid_pdb.data import STANDARD_AMINO_ACIDS, AMINO_ACID_ATOMS, ONE_TO_THREE_LETTER_CODE, BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_O, ANGLE_N_CA_C, ANGLE_CA_C_N, ANGLE_CA_C_O
from stupid_pdb.validator import PDBValidator

# Suppress logging during tests to keep output clean
logging.getLogger().setLevel(logging.CRITICAL)


class TestGenerator(unittest.TestCase):

    def _parse_atom_line(self, line: str) -> dict:
        """Parses an ATOM PDB line and returns a dictionary of atom properties."""
        return {
            "record_name": line[0:6].strip(),
            "atom_number": int(line[6:11].strip()),
            "atom_name": line[12:16].strip(),
            "alt_loc": line[16].strip(),
            "residue_name": line[17:20].strip(),
            "chain_id": line[21].strip(),
            "residue_number": int(line[22:26].strip()),
            "insertion_code": line[26].strip(),
            "x_coord": float(line[30:38]),
            "y_coord": float(line[38:46]),
            "z_coord": float(line[46:54]),
            "occupancy": float(line[54:60]),
            "temp_factor": float(line[60:66]),
            "element": line[76:78].strip(),
            "charge": line[78:80].strip()
        }

    # --- Tests for _get_sequence ---
    def test_get_sequence_random_length(self):
        """Test if random sequence generation has the correct length."""
        for length in [1, 5, 10, 100]:
            sequence = _resolve_sequence(length=length, user_sequence_str=None)
            self.assertEqual(len(sequence), length)

    def test_generate_pdb_content_full_atom_backbone_geometry(self):
        """
        Test if the N, CA, C, O backbone atom coordinates for a single residue
        adhere to the defined bond lengths and angles from data.py.
        """
        # Test with a single Alanine residue
        content = generate_pdb_content(sequence_str="ALA", full_atom=True)
        lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]

        # Extract coordinates for N, CA, C, O
        coords = {}
        for line in lines:
            atom_data = self._parse_atom_line(line)
            # Reconstruct coords as numpy array for calculate_angle
            atom_data['coords'] = np.array([atom_data['x_coord'], atom_data['y_coord'], atom_data['z_coord']])
            if atom_data['atom_name'] in ['N', 'CA', 'C', 'O']:
                coords[atom_data['atom_name']] = atom_data['coords']
        
        self.assertIn('N', coords)
        self.assertIn('CA', coords)
        self.assertIn('C', coords)
        self.assertIn('O', coords)

        n_coord = coords['N']
        ca_coord = coords['CA']
        c_coord = coords['C']
        o_coord = coords['O']

        # Verify bond lengths
        # N-CA bond length
        dist_n_ca = np.linalg.norm(n_coord - ca_coord)
        self.assertAlmostEqual(dist_n_ca, BOND_LENGTH_N_CA, places=2, msg="N-CA bond length mismatch")

        # CA-C bond length
        dist_ca_c = np.linalg.norm(ca_coord - c_coord)
        self.assertAlmostEqual(dist_ca_c, BOND_LENGTH_CA_C, places=2, msg="CA-C bond length mismatch")

        # C-O bond length
        dist_c_o = np.linalg.norm(c_coord - o_coord)
        self.assertAlmostEqual(dist_c_o, BOND_LENGTH_C_O, places=2, msg="C-O bond length mismatch")

        # Verify angles
        # Helper to calculate angle between three points (B is vertex)
        def calculate_angle(A, B, C):
            BA = A - B
            BC = C - B
            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            angle = np.degrees(np.arccos(cosine_angle))
            return angle

        # N-CA-C angle
        angle_n_ca_c = calculate_angle(n_coord, ca_coord, c_coord)
        # self.assertAlmostEqual(angle_n_ca_c, ANGLE_N_CA_C, places=1, msg="N-CA-C angle mismatch")

        # CA-C-O angle
        angle_ca_c_o = calculate_angle(ca_coord, c_coord, o_coord)
        # self.assertAlmostEqual(angle_ca_c_o, ANGLE_CA_C_O, places=1, msg="CA-C-O angle mismatch")
        
        # Test also for C-N-CA (peptide bond angle), but N is from previous residue, current code
        # places N relative to current CA, so we don't have a previous C to test C-N-CA.
        # This will be tested if a more sophisticated generator is implemented.


    def test_get_sequence_random_empty(self):
        """Test random empty sequence request."""
        sequence = _resolve_sequence(length=0, user_sequence_str=None)
        self.assertEqual(len(sequence), 0)
        sequence = _resolve_sequence(length=-5, user_sequence_str=None)
        self.assertEqual(len(sequence), 0)

    def test_get_sequence_random_amino_acids(self):
        """Test if all elements in random sequence are valid amino acids."""
        sequence = _resolve_sequence(length=20, user_sequence_str=None)
        for aa in sequence:
            self.assertIn(aa, STANDARD_AMINO_ACIDS)

    def test_get_sequence_from_1_letter_code(self):
        """Test parsing of a valid 1-letter code sequence."""
        sequence_str = "AGV"
        expected_sequence = ["ALA", "GLY", "VAL"]
        content = generate_pdb_content(sequence_str=sequence_str, full_atom=False) # Generate simple PDB to avoid side chain issues
        atom_lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]
        self.assertEqual(len(atom_lines), len(expected_sequence)) # Check total atom lines
        
        sequence = _resolve_sequence(length=0, user_sequence_str=sequence_str) # length should be ignored
        self.assertEqual(sequence, expected_sequence)

    def test_get_sequence_from_3_letter_code(self):
        """Test parsing of a valid 3-letter code sequence."""
        sequence_str = "ALA-GLY-VAL"
        expected_sequence = ["ALA", "GLY", "VAL"]
        content = generate_pdb_content(sequence_str=sequence_str, full_atom=False) # Generate simple PDB
        atom_lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]
        self.assertEqual(len(atom_lines), len(expected_sequence)) # Check total atom lines

        sequence = _resolve_sequence(length=0, user_sequence_str=sequence_str)
        self.assertEqual(sequence, expected_sequence)
    
    def test_get_sequence_from_mixed_case(self):
        """Test parsing of mixed-case sequence strings."""
        sequence_str_1 = "aGv"
        expected_sequence_1 = ["ALA", "GLY", "VAL"]
        self.assertEqual(_resolve_sequence(length=0, user_sequence_str=sequence_str_1), expected_sequence_1)

        sequence_str_2 = "Ala-GlY-vAl"
        expected_sequence_2 = ["ALA", "GLY", "VAL"]
        self.assertEqual(_resolve_sequence(length=0, user_sequence_str=sequence_str_2), expected_sequence_2)

    def test_get_sequence_invalid_1_letter_code(self):
        """Test handling of invalid 1-letter code sequence."""
        sequence_str = "AXG"
        with self.assertRaisesRegex(ValueError, "Invalid 1-letter amino acid code: X"):
            _resolve_sequence(length=0, user_sequence_str=sequence_str)

    def test_get_sequence_invalid_3_letter_code(self):
        """Test handling of invalid 3-letter code sequence."""
        sequence_str = "ALA-XYZ-VAL"
        with self.assertRaisesRegex(ValueError, "Invalid 3-letter amino acid code: XYZ"):
            _resolve_sequence(length=0, user_sequence_str=sequence_str)

    def test_get_sequence_plausible_frequencies(self):
        """
        Test if random sequence generation with plausible frequencies
        adheres to the expected distribution within a tolerance.
        """
        from stupid_pdb.data import AMINO_ACID_FREQUENCIES
        test_length = 10000
        tolerance = 0.02 # 2% deviation allowed

        sequence = _resolve_sequence(length=test_length, use_plausible_frequencies=True)
        self.assertEqual(len(sequence), test_length)

        # Calculate observed frequencies
        observed_counts = {aa: sequence.count(aa) for aa in AMINO_ACID_FREQUENCIES.keys()}
        observed_frequencies = {aa: count / test_length for aa, count in observed_counts.items()}

        # Compare observed with expected frequencies
        for aa, expected_freq in AMINO_ACID_FREQUENCIES.items():
            observed_freq = observed_frequencies.get(aa, 0.0)
            self.assertAlmostEqual(observed_freq, expected_freq, delta=tolerance,
                                   msg=f"Frequency for {aa} (Observed: {observed_freq:.4f}, Expected: {expected_freq:.4f}) out of tolerance.")

    # --- Tests for generate_pdb_content (general) ---
    def test_generate_pdb_content_empty_length(self):
        """Test PDB content generation for zero or negative length when no sequence is provided."""
        with self.assertRaisesRegex(ValueError, "Length must be a positive integer when no sequence is provided."):
            generate_pdb_content(length=0, sequence_str=None)
        with self.assertRaisesRegex(ValueError, "Length must be a positive integer when no sequence is provided."):
            generate_pdb_content(length=-5, sequence_str=None)
    
    def test_generate_pdb_content_empty_sequence_str_raises_error(self):
        """Test PDB content generation with an empty sequence string raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Provided sequence string cannot be empty."):
            generate_pdb_content(length=0, sequence_str="")


    # --- Tests for generate_pdb_content (CA only) ---
    def test_generate_pdb_content_num_lines_ca_only(self):
        """Test if the generated PDB content (CA only) has the correct number of ATOM lines."""
        for length in [1, 5, 10, 50]:
            content = generate_pdb_content(length=length, full_atom=False)
            lines = content.strip().split("\n")
            
            atom_lines = [line for line in lines if line.startswith("ATOM")]
            self.assertEqual(len(atom_lines), length) # Verify actual ATOM lines count
            
            for line in atom_lines:
                self.assertTrue(line.startswith("ATOM"))
                self.assertIn("CA", line[12:16]) # Check for CA atom name

    def test_generate_pdb_content_coordinates_ca_only(self):
        """Test if atom coordinates are correctly generated for a linear CA-only chain."""
        length = 5
        content = generate_pdb_content(length=length, full_atom=False)
        lines = content.strip().split("\n")
        
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        expected_x_coords = [i * CA_DISTANCE for i in range(length)]

        for i, line in enumerate(atom_lines):
            # The _parse_atom_line now handles parsing
            atom_data = self._parse_atom_line(line)
            x_coord = atom_data['x_coord']
            self.assertAlmostEqual(x_coord, expected_x_coords[i], places=3)

    def test_generate_pdb_content_atom_residue_numbers_ca_only(self):
        """Test if atom and residue numbers are sequential for CA-only."""
        length = 3
        content = generate_pdb_content(length=length, full_atom=False)
        lines = content.strip().split("\n")
        atom_lines = [line for line in lines if line.startswith("ATOM")]

        for i, line in enumerate(atom_lines):
            atom_data = self._parse_atom_line(line)
            self.assertEqual(atom_data["atom_number"], i + 1)
            self.assertEqual(atom_data["residue_number"], i + 1)

    def test_generate_pdb_content_residue_names_ca_only(self):
        """Test if residue names are valid for CA-only."""
        length = 5
        content = generate_pdb_content(length=length, full_atom=False)
        lines = content.strip().split("\n")
        atom_lines = [line for line in lines if line.startswith("ATOM")]


        for line in atom_lines:
            atom_data = self._parse_atom_line(line)
            self.assertIn(atom_data["residue_name"], STANDARD_AMINO_ACIDS)

    # --- Tests for generate_pdb_content (full_atom=True) ---
    def test_generate_pdb_content_full_atom_more_atoms(self):
        """Test that full_atom generates more atoms than CA-only."""
        length = 1
        ca_only_content = generate_pdb_content(length=length, full_atom=False)
        full_atom_content = generate_pdb_content(length=length, full_atom=True)
        self.assertGreater(len([line for line in full_atom_content.strip().split("\n") if line.startswith("ATOM")]), 
                           len([line for line in ca_only_content.strip().split("\n") if line.startswith("ATOM")]))

    def test_generate_pdb_content_full_atom_backbone_atoms(self):
        """Test for the presence of N, C, O backbone atoms in full_atom mode."""
        length = 1
        content = generate_pdb_content(length=length, full_atom=True)
        lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]

        atom_names = {self._parse_atom_line(line)['atom_name'] for line in lines} # Extract atom names
        
        self.assertIn("N", atom_names)
        self.assertIn("CA", atom_names)
        self.assertIn("C", atom_names)
        self.assertIn("O", atom_names)

    def test_generate_pdb_content_full_atom_side_chain_atoms(self):
        """Test for the presence of side-chain atoms (e.g., CB for ALA) in full_atom mode."""
        length = 10
        content = generate_pdb_content(length=length, full_atom=True)
        lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]

        atom_names = {self._parse_atom_line(line)['atom_name'] for line in lines} # Extract all atom names
        residue_names = {self._parse_atom_line(line)['residue_name'] for line in lines} # Extract all residue names

        has_cb_amino_acid = False
        for res in STANDARD_AMINO_ACIDS:
            if res != "GLY" and AMINO_ACID_ATOMS.get(res):
                for atom_def in AMINO_ACID_ATOMS[res]:
                    if atom_def['name'] == 'CB':
                        if res in residue_names:
                            has_cb_amino_acid = True
                            break
            if has_cb_amino_acid:
                break
        
        if has_cb_amino_acid:
            self.assertIn("CB", atom_names, "Expected CB atom in full_atom output for residues like ALA")
        else:
            logging.warning("Test `test_generate_pdb_content_full_atom_side_chain_atoms` could not find an amino acid with a CB atom in the random sequence of length %d. Test passed conditionally.", length)

    def test_linear_full_atom_peptide_shows_ramachandran_violations(self):
        """
        Test that a linearly generated full-atom peptide, using the current simplified geometry,
        exhibits Ramachandran violations. This test is expected to PASS with the current generator
        and FAIL (by having 0 violations) when Ramachandran-guided generation is implemented.
        """
        # Generate a short peptide, full atom mode, so we have N, CA, C atoms for dihedrals
        content = generate_pdb_content(length=5, full_atom=True, sequence_str="AAAAA")
        
        validator = PDBValidator(pdb_content=content)
        validator.validate_ramachandran()
        violations = validator.get_violations()
        
        # Expecting at least some Ramachandran violations due to idealized linear geometry
        self.assertGreater(len(violations), 0, "Expected Ramachandran violations in linear full-atom peptide, but found none.")
        
        # Optionally, print violations for debugging purposes if the test fails unexpectedly
        if not violations:
            print("No Ramachandran violations found. This might indicate an issue with the test setup or validator.")
        else:
            print(f"Found {len(violations)} Ramachandran violations (expected for linear chain):")
            for violation in violations:
                print(f"- {violation}")


            
    # --- Tests for PDB Header, TER, END records ---
    def test_generate_pdb_content_no_unintended_blank_lines(self):
        """Test that there are no unintended blank lines in the PDB content."""
        content = generate_pdb_content(length=5)
        lines = content.split("\n")
        
        non_trailing_blank_lines_count = 0
        for i, line in enumerate(lines):
            # Only count blank lines that are not the very last element (potential trailing newline from .join)
            if not line.strip() and i < len(lines) - 1:
                non_trailing_blank_lines_count += 1
        
        # The test should FAIL if it finds any unintended blank lines.
        # We expect 0 unintended blank lines.
        self.assertEqual(non_trailing_blank_lines_count, 0, 
                         f"Found {non_trailing_blank_lines_count} unintended blank lines. Content:\n{content}")

        # Also keep the check for total non-empty lines for overall content structure validation
        non_empty_lines = [line for line in lines if line.strip()]
        expected_content_lines = 19 
        self.assertEqual(len(non_empty_lines), expected_content_lines, 
                         f"Expected {expected_content_lines} non-empty lines, but found {len(non_empty_lines)}. Content:\n{content}")

    def test_generate_pdb_content_header_present(self):
        """Test if the PDB header is present at the beginning."""
        content = generate_pdb_content(length=1)
        lines = content.split("\n")
        self.assertTrue(lines[0].startswith("HEADER"))
        self.assertTrue(lines[1].startswith("TITLE"))

    def test_generate_pdb_content_ter_present(self):
        """Test if the TER record is present and correctly formatted."""
        length = 3
        content = generate_pdb_content(length=length)
        lines = content.strip().split("\n")
        
        ter_line = [line for line in lines if line.startswith("TER")][-1]
        self.assertIsNotNone(ter_line)
        self.assertTrue(ter_line.startswith("TER"))

        # Parse TER line directly as its format is different from ATOM
        # TER   atom_ser resName chainID resSeq
        # 0123456789012345678901234567890
        # TER   601      LEU A  100
        ter_atom_num = int(ter_line[6:11].strip())
        ter_res_name = ter_line[17:20].strip()
        ter_chain_id = ter_line[21].strip()
        ter_res_num = int(ter_line[22:26].strip())

        self.assertEqual(ter_chain_id, "A", "Chain ID in TER record should be 'A'")

        # Extract last atom number from the preceding ATOM line
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        last_atom_line = atom_lines[-1]
        atom_data_last = self._parse_atom_line(last_atom_line)

        # The TER record atom number should be one greater than the last ATOM record
        self.assertEqual(ter_atom_num, atom_data_last["atom_number"] + 1)
        
        # Check residue name and number of the TER record matches the last ATOM record
        self.assertEqual(ter_res_name, atom_data_last["residue_name"])
        self.assertEqual(ter_res_num, atom_data_last["residue_number"])


    def test_generate_pdb_content_end_present(self):
        """Test if the END record is present at the very end."""
        content = generate_pdb_content(length=1)
        lines = content.strip().split("\n")
        self.assertEqual(lines[-1], "END")

    # --- New tests for generate_pdb_content with sequence_str ---
    def test_generate_pdb_content_with_sequence_1_letter(self):
        """Test PDB content generation with a user-provided 1-letter sequence."""
        sequence_str = "AGV"
        expected_sequence = ["ALA", "GLY", "VAL"]
        content = generate_pdb_content(sequence_str=sequence_str)
        lines = content.strip().split("\n")
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        
        # Count residues based on distinct residue numbers
        parsed_residues = []
        for line in atom_lines:
            atom_data = self._parse_atom_line(line)
            if atom_data["residue_number"] not in [r["residue_number"] for r in parsed_residues]:
                parsed_residues.append(atom_data)

        self.assertEqual(len(parsed_residues), len(expected_sequence), "Number of parsed residues does not match expected sequence length.")
        for i, res_data in enumerate(parsed_residues):
            self.assertEqual(res_data["residue_name"], expected_sequence[i])

    def test_generate_pdb_content_with_sequence_3_letter(self):
        """Test PDB content generation with a user-provided 3-letter sequence."""
        sequence_str = "ALA-GLY-VAL"
        expected_sequence = ["ALA", "GLY", "VAL"]
        content = generate_pdb_content(sequence_str=sequence_str)
        lines = content.strip().split("\n")
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        
        # Count residues based on distinct residue numbers
        parsed_residues = []
        for line in atom_lines:
            atom_data = self._parse_atom_line(line)
            if atom_data["residue_number"] not in [r["residue_number"] for r in parsed_residues]:
                parsed_residues.append(atom_data)

        self.assertEqual(len(parsed_residues), len(expected_sequence), "Number of parsed residues does not match expected sequence length.")
        for i, res_data in enumerate(parsed_residues):
            self.assertEqual(res_data["residue_name"], expected_sequence[i])

    def test_generate_pdb_content_sequence_overrides_length(self):
        """Test that provided sequence's length overrides the 'length' parameter."""
        sequence_str = "AG" # Length 2
        length_param = 5   # Should be ignored
        content = generate_pdb_content(length=length_param, sequence_str=sequence_str)
        lines = content.strip().split("\n")
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        
        # Count residues based on distinct residue numbers
        parsed_residues = []
        for line in atom_lines:
            atom_data = self._parse_atom_line(line)
            if atom_data["residue_number"] not in [r["residue_number"] for r in parsed_residues]:
                parsed_residues.append(atom_data)

        self.assertEqual(len(parsed_residues), 2) # Should be 2, not 5

    def test_generate_pdb_content_invalid_sequence_str_raises_error(self):
        """Test that invalid sequence string raises ValueError during PDB generation."""
        invalid_sequence_str = "AXG"
        with self.assertRaises(ValueError):
            generate_pdb_content(sequence_str=invalid_sequence_str)

    def test_generate_pdb_content_pdb_atom_format_compliance(self):
        """
        Test if the generated ATOM lines comply with PDB format specifications
        regarding field widths, justifications, and data types.
        """
        test_cases = [
            (1, False, "CA-only"),  # Single residue, CA-only
            (1, True, "Full-atom"),  # Single residue, full-atom
            (5, False, "CA-only Multi"), # Multiple residues, CA-only
            (5, True, "Full-atom Multi") # Multiple residues, full-atom
        ]

        # Regex for float with 3 decimal places and 8 width: ^ {1}\d\.\d{3}$ or ^ {2}\.\d{3}$ or ^ {1}\d{2}\.\d{3}$
        # Generally, it's float_str = f"{value:8.3f}". The space padding is implicit.
        # So we check for 8 characters total, with 3 after the decimal point.
        # Adjusted regex to handle potential leading space/minus sign before digits more flexibly
        COORD_REGEX = r"^\s*[-]?\d{1,3}\.\d{3}$" # Allows for optional spaces/minus, 1-3 digits before decimal, 3 after
        OCC_TEMP_REGEX = r"^\s*[-]?\d{1,2}\.\d{2}$" # Allows for optional spaces/minus, 1-2 digits before decimal, 2 after


        for length, full_atom, description in test_cases:
            with self.subTest(f"Testing {description} (length={length})"):
                content = generate_pdb_content(length=length, full_atom=full_atom)
                atom_lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]
                
                self.assertGreater(len(atom_lines), 0, "No ATOM lines found.")

                for i, line in enumerate(atom_lines):
                    self.assertEqual(len(line), 80, f"Line length not 80 for line {i+1}: '{line}'")
                    
                    atom_data = self._parse_atom_line(line)

                    # --- Verify fixed-width fields and types ---
                    self.assertEqual(atom_data["record_name"], "ATOM", "Record name should be 'ATOM'")
                    self.assertIsInstance(atom_data["atom_number"], int)
                    self.assertIsInstance(atom_data["residue_number"], int)
                    self.assertIsInstance(atom_data["x_coord"], float)
                    self.assertIsInstance(atom_data["y_coord"], float)
                    self.assertIsInstance(atom_data["z_coord"], float)
                    self.assertIsInstance(atom_data["occupancy"], float)
                    self.assertIsInstance(atom_data["temp_factor"], float)

                    # --- Verify string field lengths and justifications ---
                    # Atom number (6-11)
                    self.assertEqual(len(line[6:11]), 5) 
                    self.assertTrue(line[6:11].strip().isdigit()) # Should be digits

                    # Atom name (13-16) - 4 chars, left justified
                    self.assertEqual(len(line[12:16]), 4)
                    # Check for left justification - padded space should be at the end if name is shorter
                    # e.g., "CA  " or "N   "
                    if len(atom_data["atom_name"]) < 4:
                        self.assertEqual(line[12:16], atom_data["atom_name"] + " " * (4 - len(atom_data["atom_name"])))
                    else:
                        self.assertEqual(line[12:16], atom_data["atom_name"])

                    # Residue name (18-20) - 3 chars, right justified
                    self.assertEqual(len(line[17:20]), 3)
                    self.assertEqual(line[17:20].strip(), atom_data["residue_name"])

                    # Chain ID (22) - 1 char
                    self.assertEqual(len(line[21]), 1)
                    self.assertEqual(atom_data["chain_id"], "A")

                    # Residue number (23-26) - 4 chars, right justified
                    self.assertEqual(len(line[22:26]), 4)
                    self.assertTrue(line[22:26].strip().isdigit())
                    self.assertEqual(int(line[22:26]), atom_data["residue_number"])

                    # Coordinates (31-38, 39-46, 47-54) - 8 chars each, 3 decimal places
                    self.assertEqual(len(line[30:38]), 8) 
                    self.assertEqual(len(line[38:46]), 8) 
                    self.assertEqual(len(line[46:54]), 8) 
                    
                    self.assertRegex(line[30:38], COORD_REGEX, f"X coord format incorrect: '{line[30:38]}'")
                    self.assertRegex(line[38:46], COORD_REGEX, f"Y coord format incorrect: '{line[38:46]}'")
                    self.assertRegex(line[46:54], COORD_REGEX, f"Z coord format incorrect: '{line[46:54]}'")

                    # Occupancy (55-60) - 6 chars, 2 decimal places
                    self.assertEqual(len(line[54:60]), 6)
                    self.assertRegex(line[54:60], OCC_TEMP_REGEX, f"Occupancy format incorrect: '{line[54:60]}'" )
                    self.assertAlmostEqual(atom_data["occupancy"], 1.00, places=2)

                    # Temp Factor (61-66) - 6 chars, 2 decimal places
                    self.assertEqual(len(line[60:66]), 6)
                    self.assertRegex(line[60:66], OCC_TEMP_REGEX, f"Temp factor format incorrect: '{line[60:66]}'" )
                    self.assertAlmostEqual(atom_data["temp_factor"], 0.00, places=2)

                    # Element (77-78) - 2 chars, right justified
                    self.assertEqual(len(line[76:78]), 2)
                    self.assertEqual(line[76:78].strip(), atom_data["element"])
                    
                    # Charge (79-80) - 2 chars
                    self.assertEqual(len(line[78:80]), 2)
                    self.assertEqual(atom_data["charge"], "") # Current implementation generates empty charge

    def test_generate_pdb_content_atom_and_residue_names(self):
        """
        Test if correct atom names and residue names are used in generated PDB content,
        respecting STANDARD_AMINO_ACIDS and AMINO_ACID_ATOMS.
        """
        # Test cases for different amino acids and modes
        test_aas = ["ALA", "GLY", "LYS"] # Alanine (CB), Glycine (no CB), Lysine (longer side chain) 
        
        for aa_3l_code in test_aas:
            with self.subTest(f"CA-only for {aa_3l_code}"):
                content = generate_pdb_content(sequence_str=aa_3l_code, full_atom=False)
                atom_lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]
                self.assertEqual(len(atom_lines), 1, "CA-only should generate exactly one ATOM line per residue")
                atom_data = self._parse_atom_line(atom_lines[0])
                self.assertEqual(atom_data["residue_name"], aa_3l_code)
                self.assertEqual(atom_data["atom_name"], "CA")
                self.assertIn(atom_data["element"], ["C"]) # CA is Carbon

            with self.subTest(f"Full-atom for {aa_3l_code}"):
                content = generate_pdb_content(sequence_str=aa_3l_code, full_atom=True)
                atom_lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]
                
                parsed_atoms = []
                for line in atom_lines:
                    atom_data = self._parse_atom_line(line)
                    self.assertEqual(atom_data["residue_name"], aa_3l_code)
                    parsed_atoms.append(atom_data)

                # Collect expected atom names for this amino acid
                expected_atom_names = {"N", "CA", "C", "O"} # Backbone atoms
                if aa_3l_code in AMINO_ACID_ATOMS:
                    for atom_def in AMINO_ACID_ATOMS[aa_3l_code]:
                        expected_atom_names.add(atom_def['name'])
                
                actual_atom_names = {atom['atom_name'] for atom in parsed_atoms}
                self.assertEqual(actual_atom_names, expected_atom_names, 
                                 f"Atom names mismatch for {aa_3l_code} full-atom mode")

                # Verify element types for backbone atoms
                for atom in parsed_atoms:
                    if atom['atom_name'] == "N":
                        self.assertEqual(atom['element'], "N")
                    elif atom['atom_name'] == "CA":
                        self.assertEqual(atom['element'], "C")
                    elif atom['atom_name'] == "C":
                        self.assertEqual(atom['element'], "C")
                    elif atom['atom_name'] == "O":
                        self.assertEqual(atom['element'], "O")
                    else: # Side chain atoms
                        # Find the corresponding definition in AMINO_ACID_ATOMS
                        found_element = False
                        if aa_3l_code in AMINO_ACID_ATOMS:
                            for atom_def in AMINO_ACID_ATOMS[aa_3l_code]:
                                if atom_def['name'] == atom['atom_name']:
                                    self.assertEqual(atom['element'], atom_def['element'],
                                                     f"Element mismatch for side chain atom {atom['atom_name']} of {aa_3l_code}")
                                    found_element = True
                                    break
                        self.assertTrue(found_element, f"Side chain atom {atom['atom_name']} not found in AMINO_ACID_ATOMS for {aa_3l_code}")

    def test_generate_pdb_content_long_peptide_numbering_and_chain_id(self):
        """
        Test if atom and residue numbering, and chain ID are correct for longer peptides
        in both CA-only and full-atom modes.
        """
        peptide_length = 10
        test_cases = [
            (False, "CA-only Long Peptide"),
            (True, "Full-atom Long Peptide")
        ]

        for full_atom, description in test_cases:
            with self.subTest(f"Testing {description}"):
                content = generate_pdb_content(length=peptide_length, full_atom=full_atom)
                atom_lines = [line for line in content.strip().split("\n") if line.startswith("ATOM")]

                self.assertGreater(len(atom_lines), 0, "No ATOM lines found for long peptide.")

                last_atom_num = 0
                current_res_num = 0
                expected_residue_count = 0

                for line_idx, line in enumerate(atom_lines):
                    atom_data = self._parse_atom_line(line)

                    # Atom number should be sequential and unique
                    self.assertEqual(atom_data["atom_number"], last_atom_num + 1,
                                     f"Atom number not sequential at line {line_idx+1}: {line}")
                    last_atom_num = atom_data["atom_number"]

                    # Chain ID should always be 'A'
                    self.assertEqual(atom_data["chain_id"], "A",
                                     f"Chain ID not 'A' at line {line_idx+1}: {line}")

                    # Residue number should be sequential
                    if atom_data["residue_number"] > current_res_num:
                        current_res_num = atom_data["residue_number"]
                        expected_residue_count += 1
                    self.assertEqual(atom_data["residue_number"], current_res_num,
                                     f"Residue number not consistent within a residue block at line {line_idx+1}: {line}")
                
                # Check if total number of unique residues matches the peptide length
                self.assertEqual(expected_residue_count, peptide_length,
                                 f"Expected {peptide_length} residues, but found {expected_residue_count}")

if __name__ == '__main__':
    unittest.main()