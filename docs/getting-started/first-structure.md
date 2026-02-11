# Your First Structure

This tutorial walks you through generating your first protein structure with synth-pdb.

## Step 1: Choose Your Sequence

You can specify a sequence in two ways:

1. **Random length**: `--length 20`
2. **Specific sequence**: `--sequence "ALA-GLY-SER-LEU-VAL"`

## Step 2: Choose Conformation

- `alpha` - Alpha helix
- `beta` - Beta sheet
- `random` - Random coil
- `polyproline` - Polyproline II helix

## Step 3: Generate

```bash
synth-pdb --length 20 --conformation alpha --output my_structure.pdb
```

## Step 4: Visualize

```bash
synth-pdb --length 20 --conformation alpha --visualize
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Examples Gallery](../examples/gallery.md)
