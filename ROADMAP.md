# Potential Enhancements Analysis
**Project**: `synth-pdb`  
**Focus**: Biophysical Realism, Computational Efficiency, AI Research Support  
**Strategy**: Weighted by Highest Reward / Lowest Risk

## 🏆 Top Recommendation: The "Golden Zone"
These features offer the highest value for the least effort/risk.

| Feature | Category | Reward | Risk | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **1. Cyclic Peptide Support** | Realism/AI | ⭐⭐⭐⭐⭐ | 📉 Low | Essential for therapeutic peptide modeling (hot area in AI drug discovery). |
| **2. D-Amino Acid Support** | Realism/AI | ⭐⭐⭐⭐⭐ | 📉 Low | Critical for modeling non-ribosomal peptides and peptidomimetics. |
| **3. Hard Decoy Generation** | AI Support | ⭐⭐⭐⭐ | 📉 Low | AI models need *negative* examples (near-native but incorrect) to learn robustly. |
| **4. Numba JIT Compilation** | Efficiency | ⭐⭐⭐⭐ | 📉 Low | Pure Python geometry kernels are slow. `@jit` decorators provide "free" 10-50x speedups. |

---

## 🔬 Section 1: Biophysical Realism
**Goal**: Generate structures that are indistinguishable from experimental data.

### 1. Head-to-Tail Cyclization 🔄 (Highest Priority)
*   **Concept**: Bond the N-terminus to the C-terminus.
*   **Use Case**: Cyclic peptides (e.g., Cyclosporin) have high metabolic stability and are prime targets for AI generation.
*   **Risk**: Low. Requires a final bond closure step and energy minimization to resolve strain.
*   **Changes**: Update `generator.py` to add bond, `physics.py` to relax loop.

### 2. D-Amino Acids & Non-Standard Chirality 💊
*   **Concept**: Support D-Stereoisomers (mirror image residues).
*   **Use Case**: Most therapeutic peptides use D-amino acids to resist enzymatic degradation.
*   **Risk**: Low. Just requires inverting Chitrality checks and accessing D-AA templates in OpenMM.
*   **Changes**: Update `data.py` (Residue definitions) and `generator.py` (Phi/Psi inversion).

### 3. Explicit Solvent "Water Box" 💧
*   **Concept**: Simulate the protein in a box of explicit water molecules (TIP3P model) rather than implicit solvent.
*   **Use Case**: Capturing "water-bridging" interactions that stabilize specific conformations. High fidelity.
*   **Risk**: Medium. Increases minimization time significantly (seconds -> minutes).
*   **Changes**: `physics.py` (Add `Modeller.addSolvent`).

---

## ⚡ Section 2: Computational Efficiency
**Goal**: Generate millions of structures for Deep Learning training sets.

### 1. Numba JIT Optimization 🚀
*   **Concept**: Use Just-In-Time compilation for the heavy math kernels.
*   **Reward**: The NeRF algorithm (internal -> cartesian coordinates) is a tight loop. Numba can make this C++ fast.
*   **Risk**: Low. Non-invasive decorators.
*   **Changes**: `synth_pdb/geometry.py`.

### 2. Batched Generation (GPU-First) 🏎️
*   **Concept**: Rewrite the generator to produce `N` structures at once using Tensor operations (PyTorch/JAX).
*   **Reward**: Massive throughput improvement for dataset generation.
*   **Risk**: **High**. Requires rewriting the core logic from imperative loops to vectorized tensor math.
*   **Changes**: New module `generator_batch.py`.

---

## 🤖 Section 3: AI Research Support
**Goal**: Provide data that trains better models (AlphaFold, RFDiffusion, etc.).

### 1. "Hard" Decoy Generation 🎯
*   **Concept**: Generate structures that *look* physically plausible (good bond lengths/angles) but have subtle, high-energy flaws (e.g., buried charges, wrong rotamer packing).
*   **Use Case**: Training "Structure Quality Assessment" (QA) models. They need hard negatives to learn physics.
*   **Implementation**: Run optimization but *stop early*, or intentionally introduce singular "mutations" into a minimized structure without relaxing it.

### 2. Co-Evolutionary Constraints (MSA) 🧬
*   **Concept**: When generating a dataset, generate a *family* of sequences that imply a specific fold, simulating evolution.
*   **Use Case**: Testing "MSA-based" folding models like AlphaFold2.
*   **Changes**: Enhance `dataset.py` to generate correlated mutations based on the contact map.

### 3. Distogram & Orientogram Export 📐
*   **Concept**: Export the "Ground Truth" labels used by modles like trRosetta/AlphaFold (Distance Maps + Orientation Angles).
*   **Status**: Partially implemented (Contact Map), but full Orientogram ($\omega, \theta$ angles) is missing.
*   **Changes**: `synth_pdb/processing.py`.

---

## 📉 Low Priority / High Risk (Avoid for now)
*   **Protein-Protein Docking**: Too complex for this tool's scope.
*   **Metadynamics Simulation**: Better suited for dedicated MD engines (GROMACS/OpenMM scripts).
*   **Fragment Assembly (Rosetta-style)**: Requires massive database dependencies.
