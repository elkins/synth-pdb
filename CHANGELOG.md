# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.20.1] - 2026-02-21

### Changed
- **Numpy Compatibility**: Relaxed numpy dependency pin to `<3.0.0` to resolve binary incompatibility errors (`numpy.dtype size changed`) in Google Colab and other environments using `numpy 2.x`.
- **Dependency Update**: Bumped `synth-nmr` dependency to `>=0.6.1` to incorporate upstream numpy compatibility fixes.

---

## [1.19.1] - 2026-02-19

### Fixed
- **PLM Tutorial Bug**: Fixed a `TypeError` in `docs/tutorials/plm_embeddings.ipynb` where an obsolete `ss_type` argument was used instead of `conformation` for `generate_pdb_content()`.

### Changed
- **Educational Enhancements**: Significantly expanded the PLM tutorial notebook (`plm_embeddings.ipynb`) with plain-language explanations of Protein Language Models, embeddings, similarity metrics, and UMAP clustering for chemists and biologists without machine learning backgrounds.

---

## [1.19.0] - 2026-02-19

### Added
- **PLM Embeddings**: Integrated ESM-2 protein language model support via `synth_pdb.quality.plm`. Generates per-residue and pooled embeddings from generated structures, enabling zero-shot quality scoring and downstream ML tasks.
- **PLM Tutorial**: Added `docs/tutorials/plm_embeddings.ipynb` Colab-compatible notebook demonstrating ESM-2 embedding extraction and visualization.
- New optional dependency group `[plm]` (`torch>=2.0.0`, `transformers>=4.30.0`).

---

## [1.18.0] - 2026-02-19

### Added
- **GNN Quality Scorer**: New `synth_pdb.quality.gnn` module with a Graph Neural Network model for protein structure quality assessment. Nodes represent residues; edges encode sequence proximity and spatial contacts.
- **GNN Training Script**: `scripts/train_gnn_quality.py` for training the GNN on labelled structure datasets.
- **Random Forest Baseline**: Included alongside GNN as an interpretable quality-filter baseline (`synth_pdb.quality.rf_model`).
- New optional dependency group `[gnn]` (`torch>=2.0.0`, `torch_geometric>=2.4.0`, `scikit-learn`, `joblib`).

### Changed
- `[ai]` optional-dependency group now covers the Random Forest model; `[gnn]` covers the full GNN stack.

---

## [1.17.0] - 2026-02-11

### Changed
- **NMR Package Integration**: NMR functionality now provided by the [`synth-nmr`](https://github.com/elkins/synth-nmr) package
- Maintained 100% backward compatibility via compatibility shims
- All existing code continues to work without changes

### Added
- Dependency on `synth-nmr>=0.1.0`

### Removed
- ~1,200 lines of duplicate NMR code (now imported from synth-nmr)

## [1.15.0] - 2026-02-02
### Added
- **ML Handover Notebooks**: Added zero-copy handover examples for **JAX**, **MLX**, and **PyTorch** in `examples/ml_loading/`.
- **Vectorized Batch Generation**: Exposed `BatchedGenerator` and `BatchedPeptide` via `synth_pdb.generator` for high-performance AI training pipelines.
- **Salt Bridge Consolidation**: Unified salt bridge force parameters to prevent global parameter conflicts in complex structures.

### Fixed
- **Cyclic Peptide Physics**: Refined covalent ring closure using a surgical linear-to-cyclic conversion strategy, bypassing OpenMM template matching limitations.
- **Physics Preprocessing**: Resolved an `UnboundLocalError` in the simulation engine that caused crashes in specific edge-case topologies.
- **Notebook Robustness**: Added graceful dependency checks and precision-safe assertions (`assert_allclose`) to ML handover notebooks.
- **Test Stability**: Suppressed verbose Numba debug logging and fixed mock assertion failures in the physics test suite.

## [1.14.0] - 2026-01-31
### Added
- **D-Amino Acid Support**: Support for generating and validating peptides with D-amino acids using the `D-` prefix in sequences.
- **PDB Compatibility**: Automatic conversion of D-amino acids to standard 3-letter codes (e.g., `DAL`, `DPH`).
- **Educational Enhancements**: Detailed comments explaining chiral mirroring and stereochemistry.
- **New Tests**: Comprehensive TDD suite for D-amino acid generation and validation.


## [1.13.1] - 2026-01-30

### Added
- **EGF Generation Example**: Added a new example script `examples/generate_egf.py` demonstrating the generation of a complex 53-residue protein with disarmament minimization and synthetic NMR data.

### Fixed
- **Validator Stability**: Fixed a critical `TypeError` in `validator.py` where terminal caps or incomplete backbone atoms could cause a crash during bond angle validation.
- **Regression Testing**: Added automated regression tests for the validator crash to prevent future regressions.

## [1.13.0] - 2026-01-30

### Added
- **Cyclic Peptide Support**: Implemented head-to-tail macrocyclization with automated terminal atom removal (OXT/H1-3) and physics-based bond closure.
- **Numba JIT Acceleration**: Integrated `@njit` compilation for NeRF geometry engines, Lipari-Szabo spectral density, and Ring Current calculations, achieving **50-100x speedups**.
- **Visual Connectivity**: Automated `CONECT` record generation for cyclic bonds and disulfide bridges to ensure seamless representation in the 3D viewer.
- **Educational References**: Added seminal scientific citations for macrocyclization (Horton, Craik) and deep-dive biophysical commentary to the codebase and README.

### Fixed
- **Proline Minimization**: Resolved a bug where Proline residues in cyclic peptides caused OpenMM template errors by stripping illegal amide hydrogens.
- **Metadata Persistence**: Fixed an issue where PTM residue names (SEP, TPO, PTR) were lost during the minimization-to-assembly pipeline.

## [1.12.0] - 2026-01-29

### Added
- **Beta-Turn Geometries**: Implemented physics-based construction for Type I, II, I', II', and VIII beta-turns. Added `--structure` CLI argument (e.g., `'3-6:typeII'`) for precise loop modeling.
- **J-Coupling Prediction**: Added generation of $^3J_{H_NH_\alpha}$ scalar couplings using the Karplus equation ($A \cos^2\phi + B \cos\phi + C$). Output available via `main.py` (CSV export).
- **Cis-Proline Isomerization**: Added `--cis-proline-frequency` to simulate biologically realistic non-canonical conformations (~5% frequency).
- **Post-Translational Modifications (PTMs)**: Added `--phosphorylation-rate` to simulate Ser/Thr/Tyr phosphorylation, converting residues to SEP/TPO/PTR for downstream MD/NMR analysis.
- **Performance**: Vectorized geometry kernels and improved OpenMM platform selection (CUDA/Metal preference with CPU fallback).

### Fixed
- **CLI Regressions**: Fixed `AttributeError` caused by missing CLI arguments for new biophysics features.
- **Variable Scoping**: Resolved `NameError` in `generator.py` related to rotamer selection aliases.
## [1.11.0] - 2026-01-29

### Added
- **Pre-Proline Backbone Realism**: Implemented specific conformational sampling for residues preceding Proline (favoring Extended/Beta, restricting Alpha). This significantly reduces steric clashes.
- **Biophysical Efficiency**: Validated that improving backbone realism reduces energy minimization time by **>60%** (2.42s -> 0.91s) by providing physically sound starting structures.
- **Advanced Chemical Shifts**: Added **Ring Current Effects** (Haigh-Mallion point-dipole model) to chemical shift prediction. Protons above aromatic rings are now correctly shielded, and in-plane protons deshielded.
- **SASA-Modulated Relaxation**: Implemented Solvent Accessible Surface Area (SASA) calculation to modulate Order Parameters ($S^2$). Buried residues are now modeled as more rigid than exposed ones.
- **SSBOND Robustness**: Enhanced disulfide bond detection with strict 1-to-1 pairing logic and a defined capture radius (8.0 Å) to prevent multi-bond artifacts in dense structures.

### Fixed
- **SSBOND Regression**: Fixed an issue where single Cysteines could form multiple disulfide bonds.
- **SASA Calculations**: Added robust handling for `NaN` values in SASA calculation for small/mock structures.

## [1.10.0] - 2026-01-28

### Added
- **Full Rotamer Library**: Expanded backbone-dependent rotamer library to support **All 20 Standard Amino Acids** (previously limited). Includes charged (ARG, LYS, GLU, ASP) and aromatic residues with biophysically accurate probabilities.
- **Side-Chain Validation**: Implemented `validate_side_chain_rotamers()` in `PDBValidator`. It now checks if generated Chi1/Chi2 angles conform to the library distributions (with configurable tolerance).
- **Chirality Validation**: Added `validate_chirality()` to ensure L-amino acid stereochemistry (checking improper dihedrals).
- **Validation Integration**: Updated CLI (`main.py`) to run the full suite of validation checks (including rotamers and chirality) whenever `--validate`, `--best-of-N`, or `--guarantee-valid` is used.
- **Educational Notes**: Added extensive comments explaining Rotamer libraries, Staggered conformations, and Validation logic.

### Changed
- **CLI Robustness**: Refactored `main.py` validation calls to use `validator.validate_all()`, ensuring no checks are silently skipped in the future.
- **Tests**: Replaced incomplete mocks with robust TDD cases for rotamer violations.
- **Project Config**: Updated `pyproject.toml` to fix `setuptools` deprecation warnings (retaining backward compatibility).

### Fixed
- **Missing Validation**: Fixed an issue where `main.py` was selectively running only some validation checks, ignoring newly added ones.

## [1.9.0] - 2026-01-27

### Added
- **Feature**: Metal Ion Coordination (Zinc detection and injection).
- **Feature**: Disulfide Bond detection (SSBOND records).
- **Feature**: Salt Bridge stabilization in Energy Minimization.

## [1.8.0] - 2026-01-20

### Added
- **Feature**: NEF (NMR Exchange Format) IO support.
- **Feature**: Chemical Shift Prediction (SPARTA+ style logic).
