# synth-pdb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/main/demo.ipynb)

[![PyPI version](https://badge.fury.io/py/synth-pdb.svg)](https://badge.fury.io/py/synth-pdb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## 📚 Understanding PDB Output - Educational Guide

### Biophysical Realism

**synth-pdb** generates structures with realistic properties that mimic real experimental data:

#### 🌡️ B-factors (Temperature Factors)
**What**: Measure atomic mobility/flexibility (columns 61-66)  
**Formula**: B = 8π²⟨u²⟩ (mean square displacement)  
**Range**: 5-60 Ų  
**Pattern**: Backbone (15-25) < Side chains (20-35) < Termini (30-50)

#### 📊 Occupancy Values  
**What**: Fraction of molecules with atom at position (columns 55-60)  
**Range**: 0.85-1.00  
**Correlation**: High B-factor ↔ Low occupancy  
**Pattern**: Backbone (0.95-1.00) > Side chains (0.85-0.95)

#### 🔄 Rotamer Libraries
**What**: Side chains adopt preferred conformations  
**Source**: Dunbrack library (experimental distributions)  
**Coverage**: All 20 standard amino acids

#### 🧬 Secondary Structures
**What**: Regular backbone patterns (helices, sheets)  
**Control**: Per-region via `--structure` parameter  
**Example**: `--structure "1-10:alpha,11-15:random,16-25:alpha"`

#### 🧪 Residue-Specific Ramachandran Distributions (Experimental/WIP)
**Status**: Experimental (Geometric construction needs Refactoring)
**What**: Realistic backbone geometry based on amino acid type
- **Glycine (GLY)**: Correctly accesses left-handed alpha region (phi > 0) ✅
- **Proline (PRO)**: Correctly restricts phi angles ✅
- **Preset Conformations**: (Alpha/Beta/PPII) Input angles are correct, but final structure geometry may vary due to construction method limitations. *Work in Progress.*

#### 🔗 Disulfide Bonds (SSBOND)
**What**: Covalent bonds between Cysteine residues
**Detection**: Automatic detection of close CYS-CYS pairs (SG-SG distance 2.0-2.2 Å)
**Output**: SSBOND records added to PDB header
**Importance**: Annotates stabilizing post-translational modifications

## Installation

### From PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install synth-pdb
```

This installs the `synth-pdb` package and makes the `synth-pdb` command available system-wide.

### From Source (For Development)

Install directly from the project directory:

```bash
git clone https://github.com/elkins/synth-pdb.git
cd synth-pdb
pip install .
```

### Requirements
- Python 3.8+
- NumPy
- Biotite (for residue templates and structure manipulation)

Dependencies are automatically installed with pip.

## Quick Start

Generate a simple 10-residue peptide:
```bash
synth-pdb --length 10
```

Generate and validate a specific sequence:
```bash
synth-pdb --sequence "ACDEFGHIKLMNPQRSTVWY" --validate --output my_peptide.pdb
```

Generate with mixed secondary structures and visualize:
```bash
synth-pdb --structure "1-10:alpha,11-20:beta" --visualize
```

Generate the best of 10 attempts with clash refinement:
```bash
synth-pdb --length 20 --best-of-N 10 --refine-clashes 5 --output refined_peptide.pdb
```

## 🌟 Feature Spotlight: "Spectroscopically Realistic" Dynamics

Most synthetic PDB generators create static bricks. They might create reasonable geometry, but the "B-factor" column (Column 11) is often just zero or random noise.

**Synth-PDB is different.** It simulates the **physics of protein motion** to generate a unified model of structure AND dynamics.

### The "Structure-Dynamics Link"
We implement the **Lipari-Szabo Model-Free formalism** (Nobel-adjacent physics) directly into the generator:
1.  **Structure Awareness**: The engine analyzes the generated geometry (`alpha-helix` vs `random-coil`).
2.  **Order Parameter ($S^2$) Prediction**: It assigns specific rigidity values:
    *   **Helices**: $S^2 \approx 0.85$ (Rigid H-bond network)
    *   **Loops**: $S^2 \approx 0.65$ (Flexible nanosecond motions)
    *   **Termini**: $S^2 \approx 0.45$ (Disordered fraying)
3.  **Unified Output**:
    *   **PDB B-Factors**: Calculated via $B \propto (1 - S^2)$. When you visualize the PDB in PyMOL, flexible regions *visually* appear thicker/redder, matching real crystal data distributions.
    *   **NMR Relaxation**: $R_1, R_2, NOE$ rates are calculated from the *same* parameters.

**Why this matters**:
> "The correlation between NMR order parameters ($S^2$) and crystallographic B-factors is a bridge between solution-state and solid-state dynamics." — *Fenwick et al., PNAS (2014)*

This feature allows you to test **bioinformatics pipelines** that rely on correlation between sequence, structure, and experimental observables, without needing expensive Molecular Dynamics (MD) simulations.

### 4. Relax (Simulate Dynamics)
Generate relaxation rates ($R_1, R_2, NOE$) with **realistic internal dynamics**:
```bash
python main.py relax --input output/my_peptide.pdb --output output/relaxation_data.nef --field 600 --tm 10.0
```
This module now implements the **Lipari-Szabo Model-Free** formalism with structure-based Order Parameter ($S^2$) prediction:
*   **Helices/Sheets**: $S^2 \approx 0.85$ (Rigid, high $R_1/R_2$)
*   **Loops/Turns**: $S^2 \approx 0.65$ (Flexible, lower $R_1/R_2$)
*   **Termini**: $S^2 \approx 0.45$ (Highly disordered)

This creates realistic "relaxation gradients" along the sequence, perfect for testing dynamics software.

## 🚀 Quick Visual Demo

Want to see the **Physics + Visualization** capabilities in action?

Run this command to generate a **Leucine Zipper** (classic alpha helix), **minimize** its energy using OpenMM, and immediately **visualize** it in your browser:

```bash
synth-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --minimize --visualize
```

This effectively demonstrates:
1.  **Generation**: Creating the alpha-helical backbone.
2.  **Minimization**: "Relaxing" the structure (geometry regularization).
3.  **Visualization**: Launching the interactive 3D viewer.

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

#### **Physics & Advanced Refinement **

- `--minimize`: Run physics-based energy minimization (OpenMM).
  - Uses implicit solvent (OBC2) and AMBER forcefield.
  - Highly recommended for "realistic" geometry.
  - Example: `--minimize`

- `--optimize`: Run Monte Carlo side-chain optimization.
  - Reduces steric clashes by rotating side chains.
  - Example: `--optimize`

- `--forcefield <NAME>`: Specify OpenMM forcefield.
  - Default: `amber14-all.xml`
  - Example: `--forcefield amber14-all.xml`

  - Example: `--forcefield amber14-all.xml`

#### **Synthetic NMR Data **

- `--gen-nef`: Generate synthetic NOE restraints in NEF format.
  - Scans structure for H-H pairs < cutoff.
  - Outputs `.nef` file.
  - Note: Requires hydrogens (use with `--minimize` or internal default).

- `--noe-cutoff <DIST>`: Cutoff distance for NOEs in Angstroms.
  - Default: `5.0`
  - Example: `--noe-cutoff 6.0`

- `--nef-output <FILE>`: Custom output filename for NEF.

#### **Synthetic Relaxation Data **

- `--gen-relax`: Generate synthetic NMR relaxation data ($R_1, R_2, \{^1H\}-^{15}N\ NOE$) in NEF format.
  - Calculates Model-Free parameters ($S^2 \approx 0.85$ for core, $0.5$ for flexible termini).
  - Outputs `_relax.nef` file.
  - **Physics Note**: $NOE$ values depend on tumbling time, not just internal flexibility.

- `--field <MHZ>`: Proton Larmor frequency in MHz.
  - Default: `600.0`
  - Calculates proper spectral density frequencies for this field.

- `--tumbling-time <NS>`: Global rotational correlation time ($\tau_m$) in nanoseconds.
  - Default: `10.0`
  - Controls the overall magnitude of relaxation rates. Larger proteins have larger $\tau_m$.

#### **Output Options**

- `--output <FILENAME>`: Custom output filename
  - If omitted, auto-generates: `random_linear_peptide_<length>_<timestamp>.pdb`
  - Example: `--output my_protein.pdb`

- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Logging verbosity
  - Default: `INFO`
  - Use `DEBUG` for detailed validation reports

- `--seed <INT>`: Random seed for reproducible generation
  - Default: `None` (Random)
  - Example: `--seed 42`
  - Guarantees identical output for the same command.

- `--help`: Show the help message and exit.

### Examples

#### Basic Generation

```bash
# Simple 25-residue peptide
synth-pdb --length 25

# Custom sequence with validation
synth-pdb --sequence "ELVIS" --validate --output elvis.pdb

# Use biologically realistic frequencies
synth-pdb --length 100 --plausible-frequencies

# Generate beta sheet conformation
synth-pdb --length 20 --conformation beta --output beta_sheet.pdb

# Generate extended conformation
synth-pdb --length 15 --conformation extended

# Generate random conformation (mixed alpha/beta regions)
synth-pdb --length 30 --conformation random
```

#### Quality Control

```bash
# Generate until valid (may take time!)
synth-pdb --length 15 --guarantee-valid --max-attempts 200 --output valid.pdb

# Best of 50 attempts
synth-pdb --length 20 --best-of-N 50 --output best_structure.pdb

# Refine steric clashes (5 iterations)
synth-pdb --length 30 --refine-clashes 5 --output refined.pdb

# Combined: best of 10 + refinement
synth-pdb --length 25 --best-of-N 10 --refine-clashes 3 --output optimized.pdb
```

#### Biologically-Inspired Examples

Generate structures that mimic real protein motifs for educational demonstrations:

```bash
# Collagen-like triple helix motif (polyproline II)
# Collagen is rich in proline and glycine with PPII conformation
synth-pdb --sequence "GPGPPGPPGPPGPPGPPGPP" --conformation ppii --output collagen_like.pdb

# Silk fibroin-like beta sheet
# Silk proteins contain repeating (GAGAGS) motifs forming beta sheets
synth-pdb --sequence "GAGAGSGAGAGSGAGAGS" --conformation beta --output silk_like.pdb

# Amyloid fibril-like beta structure
# Amyloid fibrils are rich in beta sheets, often with hydrophobic residues
synth-pdb --sequence "LVEALYLVCGERGFFYTPKA" --conformation beta --best-of-N 10 --output amyloid_like.pdb

# Leucine zipper motif (alpha helix)
# Leucine zippers are alpha-helical with leucine repeats every 7 residues
synth-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --output leucine_zipper.pdb

# Intrinsically disordered region (random conformation)
# IDRs lack stable structure, rich in charged/polar residues
synth-pdb --sequence "GGSEGGSEGGSEGGSEGGSE" --conformation random --output disordered_region.pdb

# Transmembrane helix-like structure (extended alpha helix)
# Membrane-spanning regions are often long alpha helices with hydrophobic residues
synth-pdb --sequence "LVIVLLVIVLLVIVLLVIVL" --conformation alpha --output transmembrane_like.pdb

# Beta-turn rich structure (mixed conformations)
# Proline and glycine favor turns and loops
synth-pdb --sequence "GPGPGPGPGPGPGPGP" --conformation random --output beta_turn_rich.pdb

# Elastin-like peptide (extended/random)
# Elastin contains repeating VPGVG motifs with flexible structure
synth-pdb --sequence "VPGVGVPGVGVPGVGVPGVG" --conformation extended --output elastin_like.pdb

# Antimicrobial peptide-like (alpha helix)
# Many AMPs are short amphipathic alpha helices
synth-pdb --sequence "KWKLFKKIGAVLKVL" --conformation alpha --validate --output amp_like.pdb

# Zinc finger motif-like (mixed structure)
# Zinc fingers have beta sheets and alpha helices
synth-pdb --sequence "CPHCGKSFSQKSDLVKHQRT" --conformation random --best-of-N 5 --output zinc_finger_like.pdb
```

**Educational Notes:**
- These examples demonstrate **sequence-structure relationships**
- Real proteins would have more complex tertiary structures and post-translational modifications
- Use these for teaching secondary structure concepts, not for actual molecular modeling
- Combine with `--validate` to show how different conformations affect structural quality
- Try `--best-of-N` and `--refine-clashes` to explore quality control strategies

#### Visualization-Optimized Examples

These examples are specifically designed to look great in the 3D viewer with `--visualize`:

```bash
# 🧬 Compact Alpha Helix (BEST for visualization)
# Short, tight helix - perfect for interactive viewing
synth-pdb --length 15 --conformation alpha --visualize

# 🔗 Helix-Turn-Helix DNA-Binding Motif
# Classic protein architecture with two helices and a turn
synth-pdb --sequence "AAAAAAGGGAAAAA" --structure "1-6:alpha,7-9:random,10-14:alpha" --visualize

# 🎀 Beta Hairpin
# Two antiparallel beta strands connected by a turn
synth-pdb --sequence "VVVVVGGVVVVV" --structure "1-5:beta,6-8:random,9-12:beta" --visualize

# 🌀 Coiled-Coil Motif
# Two alpha helices - common in structural proteins
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --visualize

# 🧲 Leucine Zipper
# Alpha helix with leucine repeats every 7 residues
synth-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --visualize

# 🦠 Antimicrobial Peptide-like
# Short amphipathic alpha helix
synth-pdb --sequence "KWKLFKKIGAVLKVL" --conformation alpha --visualize

# 🧪 Polyproline II Helix (Collagen-like)
# Left-handed helix, compact and visually distinct
synth-pdb --sequence "GPGPPGPPGPPGPP" --conformation ppii --visualize
```

**Visualization Tips:**
- **Best conformations for viewing**: `alpha` (most compact), `ppii` (distinctive shape)
- **Optimal length**: 10-20 residues for clear visualization
- **In the viewer**: Use "Cartoon" style and "Spectrum" color for best results
- **Interactive**: Rotate with left-click, zoom with scroll, pan with right-click

#### Mixed Secondary Structures

The `--structure` parameter enables creation of realistic protein-like structures with different conformations in different regions:

```bash
# Helix-turn-helix DNA-binding motif
# Two alpha helices connected by a flexible turn region
synth-pdb --length 25 --structure "1-10:alpha,11-15:random,16-25:alpha" --output helix_turn_helix.pdb

# Beta-alpha-beta fold unit
# Common protein architecture with sheet-helix-sheet
synth-pdb --length 30 --structure "1-10:beta,11-15:random,16-25:alpha,26-30:beta" --output bab_fold.pdb

# Zinc finger with realistic structure
# Beta sheet + alpha helix (actual zinc finger architecture)
synth-pdb --sequence "CPHCGKSFSQKSDLVKHQRT" --structure "1-5:beta,6-10:random,11-20:alpha" --output zinc_finger_realistic.pdb

# Immunoglobulin domain
# Multiple beta sheets connected by loops (antibody-like)
synth-pdb --length 40 --structure "1-8:beta,9-12:random,13-20:beta,21-24:random,25-32:beta,33-40:random" --output ig_domain.pdb

# Coiled-coil with flexible linker
# Two helical regions connected by disordered linker
synth-pdb --length 50 --structure "1-20:alpha,21-30:random,31-50:alpha" --output coiled_coil.pdb

# Intrinsically disordered region with structured domain
# Disordered N-terminus, structured C-terminus (common in signaling proteins)
synth-pdb --length 40 --structure "1-15:random,16-40:alpha" --output idr_with_domain.pdb

# Collagen-like with flexibility
# PPII helix with occasional flexible regions (more realistic than uniform)
synth-pdb --sequence "GPGPPGPPGPPGPPGPPGPP" --structure "1-6:ppii,7-9:random,10-20:ppii" --output collagen_flexible.pdb

# Beta-hairpin motif
# Two antiparallel beta strands connected by a turn
synth-pdb --length 20 --structure "1-7:beta,8-12:random,13-20:beta" --output beta_hairpin.pdb
```

**Why This Matters:**
- Real proteins have **mixed secondary structures**, not uniform conformations
- These examples are much more realistic than single-conformation structures
- Useful for teaching protein architecture and domain organization
- Great for testing structure analysis tools with realistic inputs
- Demonstrates how sequence and structure work together

#### Detailed Educational Case Studies

These comprehensive examples demonstrate how to use `synth-pdb` to model specific biological features found in well-known proteins.

**1. Glucagon (Alpha Helix Hormone)**
*29 residues | PDB: 1GCN*
Glucagon is a peptide hormone that raises glucose levels. It folds into a characteristic alpha helix.
```bash
synth-pdb --sequence HSQGTFTSDYSKYLDSRRAQDFVQWLMNT --conformation alpha --refine-clashes 0 --output glucagon.pdb
```
*Educational Concept*: Studying alpha-helical packing and amphipathicity.

**2. Melittin (Bent Helix / Hinge)**
*26 residues | PDB: 2MLT*
The principal toxin in bee venom. It forms two alpha helices separated by a "hinge" region, allowing it to puncture membranes.
```bash
synth-pdb --sequence GIGAVLKVLTTGLPALISWIKRKRQQ --structure "1-11:alpha,12-14:random,15-26:alpha" --refine-clashes 50 --output melittin.pdb
```
*Educational Concept*: Modeling non-linear secondary structures and flexible linkers (hinges).

**3. Bovine Pancreatic Trypsin Inhibitor (BPTI) (Disulfide Bonds)**
*58 residues | PDB: 1BPI*
A classic model for protein folding studies ("The Hydrogen Atom of Protein Folding"). It is stabilized by three disulfide bonds.
```bash
synth-pdb --sequence RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA --conformation random --refine-clashes 100 --output bpti.pdb
```
*Educational Concept*: Automatic detection of disulfide bonds (`SSBOND` records) in a randomly folded state. Run with `--validate` to see if the generator placed cysteines close enough to bond!

**4. Ubiquitin (Complex Mixed Fold)**
*76 residues | PDB: 1UBQ*
A highly conserved regulatory protein with a complex mixed alpha/beta fold (beta grasp fold).
```bash
synth-pdb --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG --structure "1-7:beta,12-16:beta,23-34:alpha,41-45:beta,48-49:beta,56-59:alpha,66-70:beta" --refine-clashes 50 --best-of-N 5 --output ubiquitin.pdb
```
*Educational Concept*: Generating complex, multi-domain topologies and managing steric clashes in larger dense structures.

#### For Structural Biologists

```bash
# All natural amino acids with validation report
synth-pdb --sequence "ACDEFGHIKLMNPQRSTVWY" --validate --log-level DEBUG

# Test structure for MD simulation pipeline
synth-pdb --length 50 --guarantee-valid --max-attempts 500 --output test_md.pdb

# Benchmark structure with known violations (good for testing validators)
synth-pdb --length 100 --validate --output benchmark.pdb
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

7. **Chirality**: Validates L-amino acid stereochemistry
   - Checks improper dihedral angle N-CA-C-CB
   - L-amino acids should have proper chirality (improper dihedral ±60° to ±120°)
   - Glycine is automatically exempt (no CB atom)
   - Detects incorrect stereochemistry (D-amino acids)

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
REMARK 1  This PDB file was generated by the CLI 'synth-pdb' tool.
REMARK 2  It represents a simplified model of a linear peptide chain.
REMARK 2  Coordinates are idealized and do not reflect real-world physics.
REMARK 3  GENERATION PARAMETERS:
REMARK 3  Command: synth-pdb --length 10 --validate ...
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

### Why "synth-pdb"?

The name reflects the tool's **intentionally simplistic** approach:
- Uses idealized bond lengths and angles (not energy-minimized)
- Linear backbone geometry (no native-like folding)
- Rotamer sampling uses Dunbrack library (most common rotamers, not exhaustive sampling)
- No solvent, no cofactors, no post-translational modifications

Real protein structures require sophisticated methods like:
- Molecular dynamics with force fields (AMBER, CHARMM)
- Quantum mechanics calculations (DFT)
- Energy minimization and conformational search
- Crystallographic or NMR experimental data

## Limitations

### Structural Limitations

1. **Linear Peptides Only**: 
   - No cyclic peptides
   - Single chain only (no multi-chain complexes)
   - No tertiary structure (folding) - structures are extended/linear

2. **Idealized Geometry**:
   - Bond lengths and angles use standard values
   - No thermal fluctuations (except small omega angle variation)
   - Structures may not match experimental geometries exactly

3. **Secondary Structure Simplification**:
   - Uses fixed phi/psi angles for each conformation type
   - Real proteins have more variation within secondary structures
   - No complex tertiary interactions

4. **No Environmental Effects**:
   - No solvent (water) molecules
   - No ions or cofactors
   - No pH effects on protonation states
   - No membrane environment for transmembrane peptides

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
pytest --cov=synth_pdb --cov-report=term-missing

# Specific test file
pytest tests/test_generator.py -v
```

**Test Coverage**: 95% overall
- 75 tests covering generation, validation, CLI, and edge cases

### Project Structure

```
synth-pdb/
├── synth_pdb/
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

## 📚 Biophysical References & Further Reading

For students and researchers interested in the physics behind the code, here are key seminal papers:

*   **Cis-Proline (~5% Frequency):**
    *   MacArthur, M. W., & Thornton, J. M. (1991). Influence of proline residues on protein conformation. *J Mol Biol*, 218(2), 397-412.
    *   Weiss, M. S., et al. (1998). Cis-proline. *Acta Cryst D*, 54, 323-329.

*   **NMR Structure Validation & Chirality:**
    *   Montelione, G. T., et al. (2013). Recommendations of the wwPDB NMR Validation Task Force. *Structure*, 21(9), 1563-1570. (Defines standards for geometric validation).
    *   Bhattacharya, A., & Montelione, G. T. (2011). PDBStat: a server for validation of protein NMR structures.

*   **Nuclear Overhauser Effect (NOE) & $r^{-6}$:**
    *   Wüthrich, K. (1986). *NMR of Proteins and Nucleic Acids*. Wiley-Interscience. (The definitive text).
    *   Wüthrich, K. (2003). Nobel Lecture: NMR Studies of Protein Structure and Dynamics.

*   **Chemical Shift Prediction (SPARTA) & Referencing (DSS):**
    *   Shen, Y., & Bax, A. (2010). SPARTA+: a modest improvement in empirical NMR chemical shift prediction... *J Biomol NMR*, 48, 13-22.
    *   Markley, J. L., et al. (1998). Recommendations for the presentation of NMR structures... (IUPAC). *Pure Appl Chem*, 70(1), 117-142. (Defined DSS as the standard).

*   **Internal Dynamics & Model-Free Formalism:**
    *   Lipari, G., & Szabo, A. (1982). Model-free approach to the interpretation of nuclear magnetic resonance relaxation in macromolecules. 1. Theory and range of validity. *J. Am. Chem. Soc.*, 104(17), 4546–4559. (The foundational theory).
    *   Kay, L. E., Torchia, D. A., & Bax, A. (1989). Backbone dynamics of proteins as studied by 15N inverse detected heteronuclear NMR spectroscopy... *Biochemistry*, 28(23), 8972-8979. (The seminal application to proteins).

## License

This project is provided as-is for educational and testing purposes.

---

## Citation

If you use `synth-pdb` in your research, please cite it using the metadata in the `CITATION.cff` file in this repository.

## 🛠️ Software & Libraries

This project relies on the following open-source scientific software:

- **[OpenMM](https://openmm.org/)**: High-performance molecular dynamics toolkit used for physics-based energy minimization (Implicit Solvent/OBC2).
- **[Biotite](https://www.biotite-python.org/)**: Comprehensive library for structural biology involved in PDB IO, atom manipulation, and geometric analysis.
- **[3Dmol.js](https://3dmol.csb.pitt.edu/)**: JavaScript library for molecular visualization used in the `--visualize` browser-based viewer.
- **[NumPy](https://numpy.org/)**: Fundamental package for scientific computing and matrix operations.

### Tools with NEF Support
These external tools can import the data generated by `synth-pdb`:
- **[CCPNMR Analysis](https://ccpn.ac.uk/)**: Premier software for NMR data analysis, assignment, and structure calculation (Native NEF support).
- **[CYANA](http://www.cyana.org/)**: Automated NMR structure calculation.
- **[XPLOR-NIH](https://nmr.cit.nih.gov/xplor-nih/)**: Biomolecular structure determination.

## 📚 References & Scientific Publications

### Key Publications in NMR Structure Validation

1.  **Protein Structure Validation Suite (PSVS)**
    *   Bhattacharya, A., Tejero, R., & Montelione, G. T. (2007). "Evaluating protein structures determined by structural genomics consortia." *Proteins: Structure, Function, and Bioinformatics*, 66(4), 778-795.
    *   [Link to Publisher](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.21165)

2.  **RPF Scores (Recall, Precision, F-measure)**
    *   Huang, Y. J., Powers, R., & Montelione, G. T. (2005). "Protein NMR recall, precision, and F-measure scores (RPF scores): structure quality assessment measures based on information retrieval statistics." *Journal of the American Chemical Society*, 127(6), 1665-1674.
    *   [Link to Publisher](https://pubs.acs.org/doi/10.1021/ja0471963)

3.  **DP Score (Discriminant Power)**
    *   Huang, Y. J., Tejero, R., Powers, R., & Montelione, G. T. (2006). "A topology-constrained distance network algorithm for protein structure determination from NOESY data." *Proteins: Structure, Function, and Bioinformatics*, 62(3), 587-603.
    *   [Link to Publisher](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.20784)

### Data Standards

- **NMR Exchange Format (NEF)**
    *   Gutmanas, A., et al. (2015). "NMR Exchange Format: a unified and open standard for representation of NMR restraint data." *Nature Structural & Molecular Biology*, 22, 433–434.
    *   [Link to Publisher](https://www.nature.com/articles/nsmb.3041)
    *   **Extension Proposal:** "Proposal For Incorporating NMR Relaxation Data In NEF" (GitHub PDF)
        *   [Link to Proposal](https://github.com/NMRExchangeFormat/NEF/blob/master/specification/Proposal%20For%20Incorporating%20NMR%20Relaxation%20Data%20In%20NEF.pdf)

### General Protein Structure References

- **Ramachandran Plot**: Ramachandran, G. N., Ramakrishnan, C., & Sasisekharan, V. (1963). "Stereochemistry of polypeptide chain configurations." *Journal of Molecular Biology*.
- **Rotamer Libraries**: Dunbrack, R. L. (2002). "Rotamer libraries in the 21st century." *Current Opinion in Structural Biology*.
- **PDB Format Specification**: [wwPDB Documentation](https://www.wwpdb.org/documentation/file-format)
- **IUPAC Nomenclature**: [IUPAC-IUBMB Joint Commission on Biochemical Nomenclature](https://iupac.qmul.ac.uk/)

## Biophysics 101: Understanding Energy Minimization

This section explains the science behind the `--minimize` feature.

### 🏔️ The Energy Landscape

Imagine a ball rolling on a hilly landscape.
- **Height** = Potential Energy (unstable)
- **Valleys** = Stable conformations (low energy)
- **Gravity** = Interactions between atoms (forces)

**Energy Minimization** is the process of moving atoms "downhill" to find the nearest stable shape.

```text
      High Energy
      (Unstable)
          |
         / \       Forces push atoms "downhill"
        /   \     (Gradient Descent)
       /     \
      /       \___
     /            \
    /              \__ Low Energy
   /                  (Stable / Minimized)
```

### 💧 Implicit vs. Explicit Solvent

Proteins exist in water. Simulating every water molecule is expensive.
- **Explicit Solvent**: Simulating thousands of H2O molecules. (Accurate but Slow)
- **Implicit Solvent**: Treating water as a mathematical continuous field that shields charges. (Fast and Good Approximation)

`synth-pdb` uses **Implicit Solvent (OBC2)** to get realistic results quickly.

### 🏗️ The Generation Pipeline

How `synth-pdb` builds a protein from scratch:

```text
[User] -> [Generator] -> [Geometry Builder] -> [Sidechain Packer] -> [Energy Minimizer] -> [PDB File]
             ^                  |                    |                      |
             |              (N-CA-C-O)           (Rotamers)             (OpenMM)
             |                                       |                      |
             +---------------------------------------+----------------------+
```

#### NMR: Relaxation & Order Parameters
The "Order Parameter" ($S^2$) derived here is a direct map of the calculated B-factor.
- $S^2 = 1.0$: Perfectly rigid (Low B-factor).
- $S^2 < 0.8$: Flexible (High B-factor).
Use this to validate Model-Free analysis software.

### Synthetic Chemical Shifts (`--gen-shifts`)
Generates predicted chemical shifts ($\delta$) for backbone atoms ($H^N, N, C_\alpha, C_\beta, C', H^\alpha$).

#### Methodology: SPARTA-lite
We implement a simplified version of the SPARTA algorithm:
1.  **Base Value**: Random Coil shift (Wishart et al.).
2.  **Offset**: Based on the exact $\phi, \psi$ angles of the generated structure.
    - **Helix**: $C_\alpha$ +3.1 ppm, $N$ -1.5 ppm.
    - **Sheet**: $C_\alpha$ -1.5 ppm, $N$ +1.2 ppm.

This allows you to verify that your assignment software (e.g. CCPNMR) correctly identifies secondary structure from shifts ("CSI" method).

#### Output Format: NEF
The output `_shifts.nef` file mimics a deposited BMRB entry, enabling direct import into:
-   **CCPNMR Analysis V3** (Drag and drop!)
-   **CYANA** (for automated assignment)
-   **CS-Rosetta** (for structure calculation)

#### CLI Options
- `--gen-shifts`: Enable chemical shift prediction.
- `--shift-output <filename>`: Optional. Specify a custom output filename (default: `[name]_shifts.nef`).

**Example:**
```bash
synth-pdb --sequence "ACDEF" --gen-shifts --shift-output custom_shifts.nef
```



### 🧲 Note for NMR Spectroscopists

If you are coming from an NMR background (XPLOR-NIH, CYANA, CNS):

- **Structure Calculation vs. Generation**: `synth-pdb` mimics the *final stage* of an NMR structure calculation: Geometry Regularization (minimization in implicit solvent).
- **Proton Detection**: Unlike X-ray, NMR relies on 1H spins. That's why we explicitly add hydrogens before minimization—they are the "eyes" of the forcefield, just as they are for NOEs.
- **Ensembles**: Use `--mode decoys` to generate an ensemble of structures. This is analogous to the "bundle" of low-energy structures you calculate to satisfy NOE restraints.
- **Order Parameters**: The generated B-factors in `synth-pdb` follow the same pattern as Order Parameters ($S^2$) or atomic RMSD across an ensemble (low in core, high in tails).

For production-quality structure generation, consider:
- **MODELLER** (homology modeling)
- **Rosetta** (de novo structure prediction)
- **AlphaFold** (AI-based prediction)
- **PyMOL/Chimera** (structure visualization and manipulation)
