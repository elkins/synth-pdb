# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
