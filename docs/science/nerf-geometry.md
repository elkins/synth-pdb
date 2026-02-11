# NeRF Geometry

The **NeRF (Natural Extension Reference Frame)** algorithm is a fundamental technique for building 3D molecular structures from internal coordinates.

## Overview

NeRF converts internal coordinates (bond lengths, angles, and dihedrals) into 3D Cartesian coordinates. This is essential for protein structure generation because:

- **Compact representation**: Internal coordinates are more compact than Cartesian coordinates
- **Chemical validity**: Bond lengths and angles follow known chemical constraints
- **Efficient sampling**: Sampling in dihedral space is more efficient than Cartesian space

## Mathematical Foundation

### Internal Coordinates

For a chain of atoms, we define:

1. **Bond Length** ($r$): Distance between consecutive atoms
   - Example: N-CA bond = 1.46 Å
   
2. **Bond Angle** ($\theta$): Angle formed by three consecutive atoms
   - Example: N-CA-C angle = 111°
   
3. **Dihedral Angle** ($\phi$): Torsion angle formed by four consecutive atoms
   - Example: Phi (φ) and Psi (ψ) backbone dihedrals

### The NeRF Algorithm

Given three atoms **A**, **B**, **C** with known positions, and internal coordinates for a new atom **D**:

- Bond length: $r_{CD}$
- Bond angle: $\theta_{BCD}$
- Dihedral angle: $\phi_{ABCD}$

The algorithm computes the position of **D** as follows:

1. **Create local coordinate system** at C:
   - **z-axis**: Along the C→B direction
   - **x-axis**: In the plane of A-B-C, perpendicular to z
   - **y-axis**: Perpendicular to both x and z

2. **Place D in local coordinates**:
   ```
   D_local = [
       r * sin(θ) * cos(φ),
       r * sin(θ) * sin(φ),
       r * cos(θ)
   ]
   ```

3. **Transform to global coordinates**:
   - Apply rotation matrix to align local axes with global axes
   - Translate by position of C

## Application to Proteins

### Backbone Construction

The protein backbone is built iteratively:

```
N₁ → CA₁ → C₁ → N₂ → CA₂ → C₂ → ...
```

Each atom is placed using NeRF with:
- **Fixed bond lengths**: From crystallographic data
- **Fixed bond angles**: From chemical constraints
- **Variable dihedrals**: Sampled from Ramachandran distributions

### Side-Chain Placement

Side-chains are added using:
- **Rotamer libraries**: Pre-computed favorable conformations
- **NeRF algorithm**: To place each side-chain atom
- **Steric constraints**: To avoid clashes

## Implementation in synth-pdb

The `geometry` module implements NeRF for protein construction:

```python
from synth_pdb.geometry import place_atom

# Place a new atom given three reference atoms
new_position = place_atom(
    atom_a=pos_a,  # Position of atom A
    atom_b=pos_b,  # Position of atom B
    atom_c=pos_c,  # Position of atom C
    bond_length=1.52,  # C-C bond
    bond_angle=111.0,  # degrees
    dihedral=180.0     # degrees
)
```

## Advantages

1. **Chemical validity**: Structures automatically satisfy bond constraints
2. **Efficiency**: O(n) complexity for n atoms
3. **Interpretability**: Dihedrals directly correspond to conformational freedom
4. **Sampling**: Easy to sample conformational space

## Limitations

1. **Rigid geometry**: Bond lengths and angles are typically fixed
2. **Sequential construction**: Requires a defined atom ordering
3. **Numerical precision**: Small errors can accumulate in long chains

## See Also

- [Ramachandran Plots](ramachandran.md) - Dihedral angle distributions
- [Rotamer Libraries](rotamers.md) - Side-chain conformations
- [geometry Module](../api/geometry.md) - Implementation details

## References

1. Parsons, J., et al. (2005). "Practical conversion from torsion space to Cartesian space for in silico protein synthesis." *Journal of Computational Chemistry*, 26(10), 1063-1068.
2. Coutsias, E. A., et al. (2004). "Using quaternions to calculate RMSD." *Journal of Computational Chemistry*, 25(15), 1849-1857.
