# Technical Implementation Notes

This document captures key engineering decisions and "gotchas" encountered during the development of `synth-pdb`.

## Physics Engine (OpenMM)

### Implicit Solvent Configuration
When using OpenMM's `app.OBC2` (Onufriev-Bashford-Case) implicit solvent model with the Amber14 forcefield, the standard `amber14-all.xml` file is **not sufficient**.

**The Issue:**
Attempting to create a system with `implicitSolvent=app.OBC2` without loading the specific solvent parameters results in a `ValueError` or a silent fallback to Vacuum electrostatics (if error handling isn't strict).

**The Solution:**
You must explicitly load the implicit solvent XML file alongside the main forcefield.

```python
# Correct initialization for Implicit Solvent
forcefield = app.ForceField(
    'amber14-all.xml',      # Main atom types
    'amber14/tip3pfb.xml',  # Water model (required reference)
    'implicit/obc2.xml'     # <--- CRITICAL: Defines OBC2 parameters
)

system = forcefield.createSystem(
    topology,
    implicitSolvent=app.OBC2, 
    nonbondedMethod=app.NoCutoff # Implicit/Vacuum requires NoCutoff
)
```

## Visualization (3Dmol.js)

### NMR Restraints & Hydrogens
Standard PDB visualization often strips hydrogen atoms to improve rendering performance. However, **NMR NOE restraints are defined between protons**.

**The Issue:**
If you load a model into `3Dmol.js` with default settings, it may remove hydrogens. Consequently, attempts to draw cylinders between protons (e.g., `HA` to `HB`) fail because the target atoms simply "don't exist" in the viewer's object model.

**The Solution:**
You must enable the `keepH` flag when adding the model.

```javascript
viewer.addModel(pdbData, "pdb", {keepH: true}); // <--- Essential for NMR
```

### Fuzzy Atom Selection
Discrepancies between generated PDB chain IDs (often defaults to 'A') and restraint lists (which might imply 'A' or be empty) are common.

**The Strategy:**
To ensure restraints are drawn even if metadata is slightly mismatched:
1.  **Strict Match**: Try selecting atom by `Chain + Residue + Name`.
2.  **Fallback**: If strict match returns 0 atoms, retry with `Residue + Name` (ignoring chain).
3.  **Debug**: If both fail, log a detailed error listing available atoms in that residue.
