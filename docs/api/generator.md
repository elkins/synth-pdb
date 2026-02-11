# generator Module

The `generator` module is the core of synth-pdb, responsible for creating protein structures from amino acid sequences.

## Overview

The generator uses the **NeRF (Natural Extension Reference Frame)** algorithm to build 3D protein structures from internal coordinates (bond lengths, angles, and dihedrals).

## Main Classes

::: synth_pdb.generator.PeptideGenerator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - generate
      group_by_category: true

## Main Functions

::: synth_pdb.generator.generate_pdb_content
    options:
      show_root_heading: true
      show_source: false
      show_signature_annotations: true

## Usage Examples

### Basic Generation

```python
from synth_pdb.generator import PeptideGenerator

# Create generator
gen = PeptideGenerator("ALA-GLY-SER-LEU-VAL")

# Generate structure
peptide = gen.generate(conformation="alpha")

# Get PDB content
pdb_content = peptide.to_pdb()

# Save to file
with open("output.pdb", "w") as f:
    f.write(pdb_content)
```

### Mixed Secondary Structures

```python
# Helix-turn-helix motif
gen = PeptideGenerator("ACDEFGHIKLMNPQRSTVWY")
peptide = gen.generate(
    structure_regions="1-5:alpha,6-10:random,11-15:alpha"
)
```

### Random Sequence Generation

```python
from synth_pdb.generator import generate_pdb_content

# Generate random 20-residue peptide
pdb_content = generate_pdb_content(
    length=20,
    conformation="random",
    use_plausible_frequencies=True  # Use biologically realistic frequencies
)
```

### With Energy Minimization

```python
pdb_content = generate_pdb_content(
    sequence_str="LKELEKELEKELEKEL",  # Leucine zipper
    conformation="alpha",
    minimize_energy=True,
    cap_termini=True
)
```

## Helper Functions

::: synth_pdb.generator._resolve_sequence
    options:
      show_root_heading: true
      show_source: false

::: synth_pdb.generator._sample_ramachandran_angles
    options:
      show_root_heading: true
      show_source: false

::: synth_pdb.generator._detect_disulfide_bonds
    options:
      show_root_heading: true
      show_source: false

## Educational Notes

### NeRF Algorithm

The NeRF (Natural Extension Reference Frame) algorithm builds 3D structures from internal coordinates:

1. **Bond Length**: Distance between consecutive atoms (e.g., N-CA = 1.46 Å)
2. **Bond Angle**: Angle formed by three consecutive atoms (e.g., N-CA-C = 111°)
3. **Dihedral Angle**: Torsion angle formed by four consecutive atoms (e.g., phi, psi)

**Mathematical Foundation**:

Given three atoms (A, B, C) and internal coordinates (bond_length, bond_angle, dihedral), the position of a new atom D is calculated by:

1. Creating a local coordinate system at C
2. Rotating by the dihedral angle
3. Placing D at the specified bond length and angle

This allows building complex 3D structures from simple 1D sequences.

### B-factor Calculation

B-factors (temperature factors) represent atomic mobility:

$$B = 8\pi^2 \langle u^2 \rangle$$

Where $\langle u^2 \rangle$ is the mean square displacement.

synth-pdb calculates B-factors from Order Parameters ($S^2$) using the Lipari-Szabo formalism:

$$B \propto (1 - S^2)$$

**Realistic Ranges**:
- Backbone atoms: 15-25 Ų
- Side-chain atoms: 20-35 Ų
- Terminal residues: 30-50 Ų

See [`_calculate_bfactor`](../api/generator.md#synth_pdb.generator._calculate_bfactor) for implementation details.

## See Also

- [geometry Module](geometry.md) - 3D coordinate calculations
- [physics Module](physics.md) - Energy minimization
- [validator Module](validator.md) - Structure validation
- [Scientific Background: NeRF Geometry](../science/nerf-geometry.md)
