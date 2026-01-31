# Project Roadmap & Future Horizons
**Project**: `synth-pdb`  
**Focus**: Biophysical Realism, Computational Efficiency, AI Research Support  
**Strategy**: Weighted by Highest Reward / Lowest Risk

## ✅ Recently Completed
The following features have been successfully implemented and verified:
- **Batched Generation (GPU-First)**: Vectorized SIMD/GPU-ready data factory.
- **D-Amino Acid Support**: Mirror-image peptide generation and validation.
- **Hard Decoy Generation**: Gaussian drift and sequence threading for AI training.
- **Numba JIT Support**: Core geometry kernels optimized for high-performance.
- **Cyclic Peptide Support**: Head-to-tail cyclization for therapeutic modeling.

---

## 🚀 Next Horizons (Prioritized)
Sorted by **High-Value / Low-Risk** first.

### 1. ML Integration Examples 🤖 (High Value / Low Risk)
*   **Concept**: Provide Jupyter notebooks/scripts demonstrating zero-copy handover from `BatchedPeptide` to PyTorch/JAX/MLX.
*   **Why**: Closes the loop for AI researchers. Shows how to use the "contiguous tensor" output in real training pipelines.
*   **Implementation**: Create an `examples/ml_loading/` directory with clear demonstrations.

### 2. Hardware Benchmarking Suite ⏱️ (Medium-High Value / Low Risk)
*   **Concept**: A formal tool to compare Serial vs. Batched performance across hardware (CPU vs. Apple Silicon M-series).
*   **Why**: Quantifies the efficiency gains of the "GPU-First" architecture and justifies its use for large-scale data production.
*   **Implementation**: New `benchmarks/` module with automated performance plotting.

### 3. "Chromophore" Educational Case (GFP) 🌈 (Medium-High Value / Medium Risk)
*   **Concept**: Specialized test case for a Green Fluorescent Protein (GFP) fragment, focusing on the Ser-Tyr-Gly cyclization.
*   **Why**: Adds coordination chemistry and post-translational modification (PTM) education.
*   **Implementation**: Extend `cofactors.py` or create `special_chemistry.py` to handle the unique SYG ring closure.

---

## 🔬 Section 1: Biophysical Realism
**Goal**: Generate structures that are indistinguishable from experimental data.

### 1. Explicit Solvent "Water Box" 💧
*   **Concept**: Simulate the protein in a box of explicit water molecules (TIP3P model) rather than implicit solvent.
*   **Use Case**: Capturing "water-bridging" interactions that stabilize specific conformations.
*   **Risk**: Medium. Increases minimization time significantly.

### 2. 💍 Macrocycle Design Lab (Cyclic Peptides)
*   **Objective**: Explore the generation of random cyclic peptides and visualize the "closure" of the macrocycle via physics-based minimization.
*   **Key Features**: `--cyclic`, `--minimize`, and `--refine-clashes`.
*   **Value**: Macrocycles are at the forefront of drug discovery. An AI model that can design cyclic peptides needs a training set of realistic closed loops, which standard PDBs lack in sufficient quantity.

---

## 🤖 Section 2: AI Research Support
**Goal**: Provide data that trains better models (AlphaFold, RFDiffusion, etc.).

### 1. Co-Evolutionary Constraints (MSA) 🧬
*   **Concept**: Generate a *family* of sequences that imply a specific fold, simulating evolution.
*   **Implementation**: Enhance `dataset.py` to generate correlated mutations based on the contact map.

### 2. Full Orientogram Export ($\omega, \theta, \phi$ angles) 📐
*   **Concept**: Export the full 6D inter-residue orientations used by models like trRosetta/RoseTTAFold.
*   **Changes**: `synth_pdb/processing.py`.

### 3. 📁 Bulk Dataset Factory (NPZ Pipeline)
*   **Objective**: Show the end-to-end workflow of generating a 10,000-sample dataset and loading it directly into a PyTorch DataLoader.
*   **Key Features**: `--mode dataset`, `--dataset-format npz`, and `dataset_manifest.csv`.
*   **Value**: Highlights the scalability. Instead of parsing thousands of individual PDB files (which is slow), it treats the generator as a streaming tensor source.

### 4. 🧠 Multi-Modal Transformer Training (Structure + NMR)
*   **Objective**: Generate synchronized structural coordinates (PDB) and synthetic experimental data (NEF Chemical Shifts) for multi-modal AI training.
*   **Key Features**: `--gen-nef`, `predict_chemical_shifts`, and `predict_order_parameters`.
*   **Value**: Demonstrates how synth-pdb can train models to predict experimental observables (like NMR shifts) directly from sequence or 3D geometry.

### 5. 📉 Torsion Angle Drift & Distribution Analysis
*   **Objective**: Statistically analyze how adding "Drift" to Ramachandran angles affects the overall fold and the radius of gyration.
*   **Key Features**: `--drift`, `dihedral_backbone`, and `BatchedGenerator`.
*   **Value**: Useful for researchers testing the robustness of their GNNs (Graph Neural Networks) to small backbone perturbations.

---

## 📉 Long-Term / High Risk
*   **Protein-Protein Docking**: Complex multi-chain physics.
*   **Metadynamics Simulation**: Better suited for dedicated MD engines.
*   **Fragment Assembly (Rosetta-style)**: Massive database dependencies.
