# Interactive Visualization

`synth-pdb` includes a powerful built-in 3D viewer powered by [3Dmol.js](https://3dmol.csb.pitt.edu/). This allows you to instantly visualize your generated structures, energy minimization results, and NMR restraints directly in your web browser.

## The `--visualize` Flag

Simply add `--visualize` to any generation command to launch the viewer.

```bash
synth-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --visualize
```

This will:
1.  Generate the PDB file.
2.  Create a temporary HTML page.
3.  Open it in your default web browser.

## Viewer Controls

The viewer is fully interactive:

*   **Rotate**: Left-click and drag.
*   **Zoom**: Scroll wheel (or pinch zoom).
*   **Pan**: Right-click (or Ctrl+Left-click) and drag.

### Feature Toggles
*   **👻 Ghost Mode**: Makes the protein transparent (0.4 opacity). Useful for seeing internal details like NOE restraints inside a Space-Filling (Sphere) model.
*   **🔴 Restraints**: Toggles the visibility of NOE cylinders on/off to declutter the view.

## Features

### Visualization Styles
Use the dropdown menu to change the representation:
*   **Cartoon**: (Default) Shows the secondary structure (helices as ribbons, sheets as arrows). Best for seeing the overall fold.
*   **Stick**: Shows all bonds as sticks. Best for seeing chemical details and side-chain packing.
*   **Sphere**: Space-filling model (Van der Waals radii). Best for seeing packing density and surface.
*   **Line**: Wireframe. Lightweight, good for seeing through the structure.

### Color Schemes
*   **Spectrum**: (Default) Rainbow gradient from Blue (N-terminus) to Red (C-terminus).
*   **Chain**: Colors by chain ID (all 'A' for synth-pdb).
*   **Element**: Standard CPK coloring (Carbon=Green/Grey, Oxygen=Red, Nitrogen=Blue, Sulfur=Yellow).
*   **SS**: Secondary Structure (Helices=Magenta, Sheets=Yellow, Loops=White).

### NMR Restraints (NOEs)
If you generate NEF restraints using `--gen-nef`, they will validly appear as **translucent red cylinders** connecting the protons.

```bash
# Generate structure, Minimize, Create NEF restraints, and Visualize
synth-pdb --length 20 --minimize --gen-nef --visualize
```

*   **Cylinders**: Represent the NOE distance restraint.
*   **Visualization**: Allows you to verify that the restraints match the geometry (i.e., the connected atoms are indeed close).

## Visual Learning Examples

Here are curated examples designed to demonstrate specific structural biology concepts visually.

### 1. The Alpha Helix (Leucine Zipper)
Observe the tight packing and the protruding side chains.
```bash
synth-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --visualize
```

### 2. The Beta Hairpin
See two antiparallel beta strands connected by a turn. Note the hydrogen bonds (implied) between the strands.
```bash
synth-pdb --sequence "VVVVVGGVVVVV" --structure "1-5:beta,6-8:random,9-12:beta" --visualize
```

### 3. Polyproline II Helix (Collagen-like)
Compare this left-handed, extended helix to the standard right-handed alpha helix.
```bash
synth-pdb --sequence "GPGPPGPPGPPGPP" --conformation ppii --visualize
```

### 4. Minimization in Action
Generate a structure with clashes, then minimize it. Compare the "before" and "after" (by running two commands) to see how physics relaxes the geometry.
```bash
# Before (Clashed)
synth-pdb --length 30 --conformation random --visualize

# After (Relaxed)
synth-pdb --length 30 --conformation random --minimize --visualize
```
