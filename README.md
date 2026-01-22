# stupid-pdb

A command-line tool to generate Protein Data Bank (PDB) files with full atomic representation for testing, benchmarking, and educational purposes.

> ⚠️ **Important**: The generated structures use idealized geometries and may contain violations of standard structural constraints. These files are intended for **testing computational tools** and **educational demonstrations**, not for simulation or experimental validation.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Validation & Refinement](#validation--refinement)
- [Output PDB Format](#output-pdb-format)
- [Scientific Context](#scientific-context)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

## Features

✨ **Structure Generation**
- Full atomic representation with backbone and side-chain heavy atoms + hydrogens
- Customizable sequence (1-letter or 3-letter amino acid codes)
- Random sequence generation with uniform or biologically plausible frequencies
- **Conformational diversity**: Generate alpha helices, beta sheets, extended chains, or random conformations
- **Rotamer-based side-chain placement** for all 20 standard amino acids (Dunbrack library)

🔬 **Validation Suite**
- Bond length validation
- Bond angle validation
- Ramachandran angle checking (phi/psi dihedral angles)
- Steric clash detection (minimum distance + van der Waals overlap)
- Peptide plane planarity (omega angle)
- Sequence improbability detection (charge clusters, hydrophobic stretches, etc.)

⚙️ **Quality Control**
- `--best-of-N`: Generate multiple structures and select the one with fewest violations
- `--guarantee-valid`: Iteratively generate until a violation-free structure is found
- `--refine-clashes`: Iteratively adjust atoms to reduce steric clashes

📝 **Reproducibility**
- Command-line parameters stored in PDB header (REMARK 3 records)
- Timestamps in generated filenames and headers

## Installation

Install directly from the project directory using pip:

```bash
pip install .
```

This installs the `stupid-pdb` package and makes the `stupid-pdb` command available system-wide.

### Requirements
- Python 3.8+
- NumPy
- Biotite (for residue templates and structure manipulation)

## Quick Start

Generate a simple 10-residue peptide:
```bash
stupid-pdb --length 10
```

Generate and validate a specific sequence:
```bash
stupid-pdb --sequence "ACDEFGHIKLMNPQRSTVWY" --validate --output my_peptide.pdb
```

Generate the best of 10 attempts with clash refinement:
```bash
stupid-pdb --length 20 --best-of-N 10 --refine-clashes 5 --output refined_peptide.pdb
```

## Usage

### Command-Line Arguments

#### **Structure Definition**

- `--length <LENGTH>`: Number of residues in the peptide chain
  - Type: Integer
  - Default: `10`
  - Example: `--length 50`

- `--sequence <SEQUENCE>`: Specify an exact amino acid sequence
  - Formats: 
    - 1-letter codes: `"ACDEFG"`
    - 3-letter codes: `"ALA-CYS-ASP-GLU-PHE-GLY"`
  - Overrides `--length`
  - Example: `--sequence "MVHLTPEEK"`

- `--plausible-frequencies`: Use biologically realistic amino acid frequencies for random generation
  - Based on natural protein composition
  - Ignored if `--sequence` is provided

- `--conformation \u003cCONFORMATION\u003e`: Secondary structure conformation to generate
  - Options: `alpha`, `beta`, `ppii`, `extended`, `random`
  - Default: `alpha` (alpha helix)
  - Choices:
    - `alpha`: Alpha helix (φ=-57°, ψ=-47°)
    - `beta`: Beta sheet (φ=-135°, ψ=135°)
    - `ppii`: Polyproline II helix (φ=-75°, ψ=145°)
    - `extended`: Extended/stretched conformation (φ=-120°, ψ=120°)
    - `random`: Random sampling from allowed Ramachandran regions
  - Example: `--conformation beta`

#### **Validation & Quality Control**

- `--validate`: Run validation checks on the generated structure
  - Checks: bond lengths, bond angles, Ramachandran, steric clashes, peptide planes, sequence improbabilities
  - Reports violations to console

- `--guarantee-valid`: Generate structures until one with zero violations is found
  - Implies `--validate`
  - Use with `--max-attempts` to limit iterations
  - Example: `--guarantee-valid --max-attempts 100`

- `--max-attempts <N>`: Maximum generation attempts for `--guarantee-valid`
  - Default: `100`

- `--best-of-N <N>`: Generate N structures and select the one with fewest violations
  - Implies `--validate`
  - Overrides `--guarantee-valid`
  - Example: `--best-of-N 20`

- `--refine-clashes <ITERATIONS>`: Iteratively adjust atoms to reduce steric clashes
  - Applies after structure selection
  - Iterates until improvements stop or max iterations reached
  - Example: `--refine-clashes 10`

#### **Output Options**

- `--output <FILENAME>`: Custom output filename
  - If omitted, auto-generates: `random_linear_peptide_<length>_<timestamp>.pdb`
  - Example: `--output my_protein.pdb`

- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Logging verbosity
  - Default: `INFO`
  - Use `DEBUG` for detailed validation reports

### Examples

#### Basic Generation

```bash
# Simple 25-residue peptide
stupid-pdb --length 25

# Custom sequence with validation
stupid-pdb --sequence "ELVIS" --validate --output elvis.pdb

# Use biologically realistic frequencies
stupid-pdb --length 100 --plausible-frequencies

# Generate beta sheet conformation
stupid-pdb --length 20 --conformation beta --output beta_sheet.pdb

# Generate extended conformation
stupid-pdb --length 15 --conformation extended

# Generate random conformation (mixed alpha/beta regions)
stupid-pdb --length 30 --conformation random
```

#### Quality Control

```bash
# Generate until valid (may take time!)
stupid-pdb --length 15 --guarantee-valid --max-attempts 200 --output valid.pdb

# Best of 50 attempts
stupid-pdb --length 20 --best-of-N 50 --output best_structure.pdb

# Refine steric clashes (5 iterations)
stupid-pdb --length 30 --refine-clashes 5 --output refined.pdb

# Combined: best of 10 + refinement
stupid-pdb --length 25 --best-of-N 10 --refine-clashes 3 --output optimized.pdb
```

#### Biologically-Inspired Examples

Generate structures that mimic real protein motifs for educational demonstrations:

```bash
# Collagen-like triple helix motif (polyproline II)
# Collagen is rich in proline and glycine with PPII conformation
stupid-pdb --sequence "GPGPPGPPGPPGPPGPPGPP" --conformation ppii --output collagen_like.pdb

# Silk fibroin-like beta sheet
# Silk proteins contain repeating (GAGAGS) motifs forming beta sheets
stupid-pdb --sequence "GAGAGSGAGAGSGAGAGS" --conformation beta --output silk_like.pdb

# Amyloid fibril-like beta structure
# Amyloid fibrils are rich in beta sheets, often with hydrophobic residues
stupid-pdb --sequence "LVEALYLVCGERGFFYTPKA" --conformation beta --best-of-N 10 --output amyloid_like.pdb

# Leucine zipper motif (alpha helix)
# Leucine zippers are alpha-helical with leucine repeats every 7 residues
stupid-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --output leucine_zipper.pdb

# Intrinsically disordered region (random conformation)
# IDRs lack stable structure, rich in charged/polar residues
stupid-pdb --sequence "GGSEGGSEGGSEGGSEGGSE" --conformation random --output disordered_region.pdb

# Transmembrane helix-like structure (extended alpha helix)
# Membrane-spanning regions are often long alpha helices with hydrophobic residues
stupid-pdb --sequence "LVIVLLVIVLLVIVLLVIVL" --conformation alpha --output transmembrane_like.pdb

# Beta-turn rich structure (mixed conformations)
# Proline and glycine favor turns and loops
stupid-pdb --sequence "GPGPGPGPGPGPGPGP" --conformation random --output beta_turn_rich.pdb

# Elastin-like peptide (extended/random)
# Elastin contains repeating VPGVG motifs with flexible structure
stupid-pdb --sequence "VPGVGVPGVGVPGVGVPGVG" --conformation extended --output elastin_like.pdb

# Antimicrobial peptide-like (alpha helix)
# Many AMPs are short amphipathic alpha helices
stupid-pdb --sequence "KWKLFKKIGAVLKVL" --conformation alpha --validate --output amp_like.pdb

# Zinc finger motif-like (mixed structure)
# Zinc fingers have beta sheets and alpha helices
stupid-pdb --sequence "CPHCGKSFSQKSDLVKHQRT" --conformation random --best-of-N 5 --output zinc_finger_like.pdb
```

**Educational Notes:**
- These examples demonstrate **sequence-structure relationships**
- Real proteins would have more complex tertiary structures and post-translational modifications
- Use these for teaching secondary structure concepts, not for actual molecular modeling
- Combine with `--validate` to show how different conformations affect structural quality
- Try `--best-of-N` and `--refine-clashes` to explore quality control strategies

#### For Structural Biologists

```bash
# All natural amino acids with validation report
stupid-pdb --sequence "ACDEFGHIKLMNPQRSTVWY" --validate --log-level DEBUG

# Test structure for MD simulation pipeline
stupid-pdb --length 50 --guarantee-valid --max-attempts 500 --output test_md.pdb

# Benchmark structure with known violations (good for testing validators)
stupid-pdb --length 100 --validate --output benchmark.pdb
```

## Validation & Refinement

### Validation Checks

When `--validate` is enabled, the tool checks for:

1. **Bond Lengths**: Compares N-CA, CA-C, C-N, C-O distances against standard values (±0.05 Å tolerance)

2. **Bond Angles**: Validates N-CA-C, CA-C-N, CA-C-O angles (±5° tolerance)

3. **Ramachandran Angles**: Checks phi/psi dihedral angles against allowed regions
   - Allowed: alpha-helix, beta-sheet, and left-handed alpha-helix regions
   - Glycine and proline have relaxed criteria

4. **Steric Clashes**: Detects atoms that are too close
   - Minimum distance rule: ≥2.0 Å between any atoms
   - van der Waals overlap: atoms closer than sum of vdW radii

5. **Peptide Plane Planarity**: Checks omega (ω) dihedral angles
   - Trans: ~180° (±30° tolerance)
   - Cis: ~0° (±30° tolerance)

6. **Sequence Improbabilities**: Flags unusual sequence patterns
   - Charge clusters (4+ consecutive charged residues)
   - Long hydrophobic stretches (8+ residues)
   - Odd cysteine counts (unpaired cysteines)
   - Poly-proline or poly-glycine runs

### Refinement Strategy

The `--refine-clashes` option uses an iterative approach:
1. Identifies clashing atom pairs
2. Slightly adjusts positions to increase separation
3. Re-validates structure
4. Stops when no improvement or max iterations reached

> **Note**: Refinement focuses on steric clashes and may introduce other violations. Use in combination with `--best-of-N` for better results.

## Output PDB Format

### Structure Representation

- **Full Atomic Model**: All backbone atoms (N, CA, C, O) + side-chain heavy atoms + hydrogens
- **Geometry**: Linear alpha-helix conformation along the X-axis
- **Chain ID**: Always 'A'
- **Residue Numbering**: Sequential from 1
- **Terminal Modifications**: N-terminal and C-terminal hydrogens/oxygens included

### Header Information

Generated PDB files include standard header records:

```
HEADER    PEPTIDE           <DATE>
TITLE     GENERATED LINEAR PEPTIDE OF LENGTH <N>
REMARK 1  This PDB file was generated by the CLI 'stupid-pdb' tool.
REMARK 2  It represents a simplified model of a linear peptide chain.
REMARK 2  Coordinates are idealized and do not reflect real-world physics.
REMARK 3  GENERATION PARAMETERS:
REMARK 3  Command: stupid-pdb --length 10 --validate ...
```

The **REMARK 3** records store the exact command-line arguments used for **reproducibility**.

### Validation Reports

When `--validate` is used, violations are reported:
```
WARNING  --- PDB Validation Report for /path/to/file.pdb ---
WARNING  Final PDB has 5 violations.
WARNING  Bond length violation: N-1-A to CA-1-A. Distance: 1.52Å, Expected: 1.46Å±0.05Å
WARNING  Steric clash (min distance): Atoms CA-3-A and CB-3-A are too close (1.85Å)...
```

## Scientific Context

### Intended Use Cases

✅ **Appropriate Uses:**
- Testing PDB parsers and file I/O
- Benchmarking structure validation tools
- Educational demonstrations of protein structure concepts
- Generating test datasets for bioinformatics pipelines
- Placeholder structures for software development

❌ **Inappropriate Uses:**
- Molecular dynamics simulations
- Homology modeling templates
- Drug docking studies
- Experimental predictions
- Publication-quality structures

### Why "stupid-pdb"?

The name reflects the tool's **intentionally simplistic** approach:
- Uses idealized bond lengths and angles (not energy-minimized)
- Linear backbone geometry (no native-like folding)
- Simplified rotamer placement (limited conformational sampling)
- No solvent, no cofactors, no post-translational modifications

Real protein structures require sophisticated methods like:
- Molecular dynamics with force fields (AMBER, CHARMM)
- Quantum mechanics calculations (DFT)
- Energy minimization and conformational search
- Crystallographic or NMR experimental data

## Limitations

### Structural Limitations

1. **Linear Geometry**: All structures are extended alpha-helixes
   - No beta-sheets, turns, or loops
   - No tertiary or quaternary structure

2. **Idealized Parameters**: Bond lengths/angles from literature averages
   - No force field optimization
   - May deviate from experimental structures

3. **Limited Rotamers**: Only LEU uses rotamer library currently
   - Other residues use template geometries
   - Side chains may have unfavorable conformations

4. **No Environmental Effects**:
   - No solvent (water) molecules
   - No ions or cofactors
   - No pH effects on protonation states

### Validation Limitations

- **Simplified Ramachandran regions**: 3 main regions only (not 100% accurate)
- **VdW radii**: Approximate values, not element/hybridization-specific
- **No electrostatics**: Doesn't check charge-charge interactions
- **No hydrogen bonding**: Doesn't validate H-bond geometry

### Performance Considerations

- `--guarantee-valid` may **never converge** for long sequences (>50 residues)
  - Combinatorial explosion of possible violations
  - Consider using `--best-of-N` instead

- `--refine-clashes` is **iterative and may be slow** for large structures
  - Each iteration requires full re-validation

- Validation runtime scales with sequence length (O(N²) for steric clashes)

## Development

### Running Tests

```bash
# All tests
pytest -v

# With coverage
pytest --cov=stupid_pdb --cov-report=term-missing

# Specific test file
pytest tests/test_generator.py -v
```

**Test Coverage**: 95% overall
- 75 tests covering generation, validation, CLI, and edge cases

### Project Structure

```
stupid-pdb/
├── stupid_pdb/
│   ├── __init__.py
│   ├── main.py          # CLI entry point
│   ├── generator.py     # PDB structure generation
│   ├── validator.py     # Validation checks
│   └── data.py          # Constants and rotamer library
├── tests/
│   ├── test_generator.py
│   ├── test_generator_rotamer.py
│   ├── test_validator.py
│   └── test_main_cli.py
├── setup.py
└── README.md
```

## License

This project is provided as-is for educational and testing purposes.

---

## References & Further Reading

For those interested in proper protein structure modeling:

- **PDB Format Specification**: [https://www.wwpdb.org/documentation/file-format](https://www.wwpdb.org/documentation/file-format)
- **Ramachandran Plot**: Ramachandran, G. N.; Ramakrishnan, C.; Sasisekharan, V. (1963). "Stereochemistry of polypeptide chain configurations"
- **Rotamer Libraries**: Dunbrack, R. L. (2002). "Rotamer libraries in the 21st century"
- **IUPAC Nomenclature**: [https://iupac.qmul.ac.uk/](https://iupac.qmul.ac.uk/)

For production-quality structure generation, consider:
- **MODELLER** (homology modeling)
- **Rosetta** (de novo structure prediction)
- **AlphaFold** (AI-based prediction)
- **PyMOL/Chimera** (structure visualization and manipulation)