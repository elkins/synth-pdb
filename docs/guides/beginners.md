# Beginners

Welcome to `synth-pdb`, a powerful tool designed to generate protein structures for various applications, especially in AI/ML research and educational settings. If you're new to protein modeling, computational structural biology, or simply looking for an efficient way to create diverse protein structures, you're in the right place!

## What is `synth-pdb`?

`synth-pdb` is a high-performance Python library that allows you to synthesize Protein Data Bank (PDB) files programmatically. It focuses on generating realistic and diverse protein structures, ranging from simple peptides to complex globular proteins, with control over various biophysical properties.

## Why use `synth-pdb`?

*   **AI/ML Data Generation:** Create large, diverse datasets of protein structures for training machine learning models in areas like protein folding, design, and function prediction.
*   **Education & Visualization:** Generate custom protein examples to illustrate structural biology concepts in classrooms or interactive tutorials.
*   **Hypothesis Testing:** Quickly prototype and test ideas about protein structure-function relationships.
*   **Flexibility:** Control parameters such as amino acid sequence, secondary structure elements, and global folds.

## Getting Started: Your First Protein

Before you begin, ensure you have `synth-pdb` installed. If not, please refer to the [Installation Guide](../getting-started/installation.md).

Once installed, generating a simple protein is straightforward. Let's create a small alanine peptide:

```python
from synth_pdb.generator import generate_protein

# Generate a simple 5-residue poly-alanine peptide
# This will create a PDB file named 'poly_ala_5.pdb' in your current directory
protein_model = generate_protein(sequence="AAAAA", output_filepath="poly_ala_5.pdb")
print(f"Generated protein: {protein_model.filepath}")
```

This simple command creates a PDB file that you can then visualize using tools like PyMOL, VMD, or NGL Viewer.

## Key Concepts

As you delve deeper into `synth-pdb`, you'll encounter concepts like:

*   **PDB Files:** The standard file format for recording protein and nucleic acid structures.
*   **Amino Acids:** The building blocks of proteins, linked together in a specific sequence.
*   **Residues:** Individual amino acid units within a protein chain.
*   **Torsional Angles (Phi/Psi):** Key angles that define the backbone conformation of a protein.

## Where to go next?

*   **Quick Start Guide:** Learn more about basic usage and features in the [Quick Start Guide](../getting-started/quickstart.md).
*   **Interactive Tutorials:** Explore practical examples and advanced functionalities in the [Tutorials](../tutorials/gfp_molecular_forge.ipynb) section.
*   **API Reference:** Dive into the detailed documentation of specific modules and functions in the [API Reference](../api/overview.md).

Happy generating!
