---
title: 'synth-pdb: A Python tool for generating realistic peptide structures and synthetic NMR data'
tags:
  - Python
  - biology
  - bioinformatics
  - structural biology
  - NMR
  - protein structure
  - molecular dynamics
authors:
  - name: George Elkins
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 23 January 2026
bibliography: paper.bib
---

# Summary

``synth-pdb`` is a command-line tool and Python library designed to generate biologically realistic protein structures and associated synthetic Nuclear Magnetic Resonance (NMR) data. Unlike simple random walk algorithms, ``synth-pdb`` employs biophysical constraints derived from the Ramachandran plot [@Ramachandran1963] and rotamer libraries to construct valid backbone and side-chain geometries. It includes an energy minimization pipeline using OpenMM [@Eastman2017] to resolve steric clashes and ensure physical plausibility. Additionally, the tool generates synthetic NMR observables—including Nuclear Overhauser Effect (NOE) restraints, chemical shifts, and relaxation parameters—exported in the NMR Exchange Format (NEF) [@Gutmanas2015], facilitating the benchmarking of bioinformatics tools and NMR assignment software.

# Statement of need

In the fields of structural biology and bioinformatics, researchers frequently require datasets of protein structures to test algorithms, train machine learning models, or validatate analytical pipelines. While the Protein Data Bank (PDB) [@Berman2000] contains over 200,000 experimental structures, relying solely on experimental data has limitations:
1.  **Bias**: PDB data is biased toward crystallizable or stable proteins.
2.  **Complexity**: Experimental files often contain artifacts, missing atoms, or non-standard residues that complicate initial testing.
3.  **Lack of Ground Truth**: When developing algorithms for NMR assignment or structure calculation, "perfect" synthetic data is essential for unit testing and validation before applying methods to noisy experimental data.

Existing tools often focus on *ab initio* folding (e.g., Rosetta, AlphaFold), which are computationally expensive and intended for prediction rather than rapid test-data generation. ``synth-pdb`` fills this gap by providing a lightweight, deterministic generator that produces chemically valid, full-atom PDB files with user-defined secondary structures (helices, sheets) in seconds. It serves educators teaching protein geometry, developers testing PDB parsers, and spectroscopists validating NMR software.

# Functionality

``synth-pdb`` enables users to:
1.  **Generate Structures**: Create linear peptides with specific sequences and conformations (alpha-helix, beta-sheet, polyproline-II).
2.  **Control Geometry**: Define mixed secondary structures (e.g., residues 1-10 alpha, 11-20 beta) to simulate variable protein domains.
3.  **Ensure Quality**: Run geometric validation checks (bond lengths, angles, chirality) and optimization routines (side-chain packing, energy minimization).
4.  **Simulate NMR Data**:
    *   **NOEs**: Calculate inter-atomic distances to simulate proximity restraints.
    *   **Relaxation**: Generate $R_1$, $R_2$, and Heteronuclear NOE values based on Model-Free formalism.
    *   **Chemical Shifts**: Predict backbone shifts using a SPARTA-lite approach [@Shen2010].

# Implementation

The core generation logic builds the polypeptide chain atom-by-atom in internal coordinates (bond lengths, angles, and torsions) before converting to Cartesian coordinates using the NeRF (Natural Extension Reference Frame) algorithm. The software relies on ``biotite`` [@Kunzmann2018] for PDB file handling and basic structural analysis.

Energy minimization is performed via ``OpenMM`` [@Eastman2017], utilizing the AMBER14 forcefield and OBC2 implicit solvent model to relax the generated structures into local energy minima. This step is critical for resolving steric clashes that inevitably arise during random chain generation.

# References
