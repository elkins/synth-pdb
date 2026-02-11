# API Reference

Welcome to the synth-pdb API reference. This section provides detailed documentation for all public modules, classes, and functions.

## Quick Links

### Core Modules

<div class="grid cards" markdown>

-   :material-dna:{ .lg .middle } __generator__

    ---

    Main module for generating protein structures from sequences

    [:octicons-arrow-right-24: View docs](generator.md)

-   :material-atom:{ .lg .middle } __physics__

    ---

    Energy minimization and molecular dynamics using OpenMM

    [:octicons-arrow-right-24: View docs](physics.md)

-   :material-check-circle:{ .lg .middle } __validator__

    ---

    Structure validation (bonds, angles, Ramachandran, clashes)

    [:octicons-arrow-right-24: View docs](validator.md)

-   :material-cube-outline:{ .lg .middle } __geometry__

    ---

    3D coordinate calculations using NeRF algorithm

    [:octicons-arrow-right-24: View docs](geometry.md)

</div>

### Scientific Features

<div class="grid cards" markdown>

-   :material-chart-bell-curve:{ .lg .middle } __chemical_shifts__

    ---

    NMR chemical shift prediction (¹H, ¹³C, ¹⁵N)

    [:octicons-arrow-right-24: View docs](chemical_shifts.md)

-   :material-sine-wave:{ .lg .middle } __relaxation__

    ---

    NMR relaxation rates (R₁, R₂, NOE) via Lipari-Szabo

    [:octicons-arrow-right-24: View docs](relaxation.md)

-   :material-molecule:{ .lg .middle } __nmr__

    ---

    NOE restraint generation for structure calculation

    [:octicons-arrow-right-24: View docs](nmr.md)

-   :material-flask:{ .lg .middle } __biophysics__

    ---

    pH titration, salt bridges, PTMs, metal coordination

    [:octicons-arrow-right-24: View docs](biophysics.md)

</div>

### Utilities

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __batch_generator__

    ---

    Vectorized batch generation for AI/ML workflows

    [:octicons-arrow-right-24: View docs](batch_generator.md)

-   :material-eye:{ .lg .middle } __viewer__

    ---

    3D visualization using py3Dmol

    [:octicons-arrow-right-24: View docs](viewer.md)

-   :material-database:{ .lg .middle } __dataset__

    ---

    Bulk dataset generation and export (NPZ, HDF5)

    [:octicons-arrow-right-24: View docs](dataset.md)

</div>

## Usage Patterns

### As a Library

```python
from synth_pdb.generator import PeptideGenerator
from synth_pdb.physics import EnergyMinimizer
from synth_pdb.validator import PDBValidator

# Generate structure
gen = PeptideGenerator("ALA-GLY-SER-LEU-VAL")
peptide = gen.generate(conformation="alpha")

# Minimize energy
minimizer = EnergyMinimizer()
minimized_pdb = minimizer.minimize(
    pdb_file_path="input.pdb",
    output_path="output.pdb"
)

# Validate structure
validator = PDBValidator(pdb_content=minimized_pdb)
validation_report = validator.validate_all()
```

### As a Command-Line Tool

```bash
# Basic usage
synth-pdb --length 20 --conformation alpha --output structure.pdb

# With minimization
synth-pdb --sequence "ACDEFGHIKLMNPQRSTVWY" --minimize --output minimized.pdb

# Batch generation
synth-pdb --mode dataset --num-samples 1000 --output ./dataset
```

## Module Organization

```
synth_pdb/
├── generator.py          # Main structure generation
├── physics.py            # Energy minimization (OpenMM)
├── validator.py          # Structure validation
├── geometry.py           # NeRF algorithm, 3D coordinates
├── chemical_shifts.py    # NMR chemical shift prediction
├── relaxation.py         # NMR relaxation rates
├── nmr.py                # NOE restraints
├── biophysics.py         # pH, salt bridges, PTMs
├── batch_generator.py    # Vectorized batch generation
├── viewer.py             # 3D visualization
├── dataset.py            # Bulk dataset generation
├── data.py               # Rotamer libraries, Ramachandran data
├── main.py               # CLI entry point
└── ...                   # Additional utilities
```

## Design Philosophy

synth-pdb follows these design principles:

1. **Code as Textbook**: Extensive educational comments explaining the biophysical reasoning
2. **Modular Architecture**: Clear separation of concerns (generation, physics, validation)
3. **Scientific Rigor**: Proper implementation of established methods with citations
4. **Performance**: Vectorized operations, optional Numba JIT compilation
5. **Flexibility**: Works as both library and command-line tool

## Next Steps

- [generator Module](generator.md) - Start with the core generation module
- [User Guides](../guides/beginners.md) - Learn how to use synth-pdb effectively
- [Examples Gallery](../examples/gallery.md) - Browse copy-paste examples
