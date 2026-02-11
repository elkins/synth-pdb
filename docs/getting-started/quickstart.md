# Quick Start

Get up and running with synth-pdb in 5 minutes.

## Installation

=== "PyPI (Recommended)"

    ```bash
    pip install synth-pdb
    ```

=== "From Source"

    ```bash
    git clone https://github.com/elkins/synth-pdb.git
    cd synth-pdb
    pip install .
    ```

## Your First Structure

Generate a simple 10-residue alpha helix:

```bash
synth-pdb --length 10 --conformation alpha --output my_first_helix.pdb
```

This creates a PDB file with:
- ✅ Full atomic representation (backbone + side-chains + hydrogens)
- ✅ Realistic B-factors and occupancy values
- ✅ Proper bond geometry and angles

## Visualize It

View your structure interactively in the browser:

```bash
synth-pdb --length 10 --conformation alpha --visualize
```

![Visualization Example](../images/quickstart_viz.png)

## Add Physics

Generate a more realistic structure with energy minimization:

```bash
synth-pdb --length 20 --conformation alpha --minimize --output minimized.pdb
```

This uses OpenMM to:
- Regularize bond lengths and angles
- Resolve steric clashes
- Apply implicit solvent effects

## Next Steps

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } __First Structure Tutorial__

    ---

    Detailed walkthrough of generating your first protein structure

    [:octicons-arrow-right-24: Start tutorial](first-structure.md)

-   :material-console:{ .lg .middle } __Command-Line Reference__

    ---

    Complete reference of all available options and flags

    [:octicons-arrow-right-24: View reference](../guides/cli-reference.md)

-   :material-image-multiple:{ .lg .middle } __Examples Gallery__

    ---

    Browse inspiring examples and copy-paste commands

    [:octicons-arrow-right-24: Explore gallery](../examples/gallery.md)

-   :material-robot:{ .lg .middle } __AI/ML Integration__

    ---

    Learn how to use synth-pdb for machine learning workflows

    [:octicons-arrow-right-24: ML guide](../guides/ml-researchers.md)

</div>

## Common Use Cases

### Testing Bioinformatics Tools

```bash
# Generate a test dataset
synth-pdb --mode dataset --num-samples 100 --output ./test_data
```

### Educational Demonstrations

```bash
# Show different secondary structures
synth-pdb --structure "1-10:alpha,11-20:beta" --visualize
```

### NMR Data Generation

```bash
# Generate structure with synthetic NMR observables
synth-pdb --length 30 --gen-nef --gen-relax --output nmr_test.pdb
```

## Getting Help

!!! question "Need help?"
    
    - 📖 Check the [User Guides](../guides/beginners.md)
    - 💬 Ask questions on [GitHub Discussions](https://github.com/elkins/synth-pdb/discussions)
    - 🐛 Report bugs on [GitHub Issues](https://github.com/elkins/synth-pdb/issues)
