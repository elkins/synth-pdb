# Installation Guide

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Dependencies**: NumPy, Biotite, OpenMM (optional), Numba (optional)

## Installation Methods

### PyPI (Recommended)

The easiest way to install synth-pdb is via PyPI:

```bash
pip install synth-pdb
```

This installs:
- The `synth-pdb` command-line tool
- The `synth_pdb` Python library
- All required dependencies

### From Source

For development or to get the latest features:

```bash
git clone https://github.com/elkins/synth-pdb.git
cd synth-pdb
pip install -e .
```

The `-e` flag installs in "editable" mode, so changes to the source code are immediately reflected.

### Development Installation

To install with development dependencies (testing, linting, etc.):

```bash
git clone https://github.com/elkins/synth-pdb.git
cd synth-pdb
pip install -e ".[dev]"
```

## Optional Dependencies

### OpenMM (Physics Engine)

For energy minimization and MD equilibration:

```bash
pip install openmm
```

Or via conda (recommended for Apple Silicon):

```bash
conda install -c conda-forge openmm
```

!!! tip "Apple Silicon Users"
    OpenMM on M1/M2/M3/M4 Macs supports GPU acceleration via OpenCL/Metal, providing 5x speedup over CPU.

### Numba (JIT Compilation)

For 50-100x speedup on geometry and NMR calculations:

```bash
pip install numba
```

## Verification

Verify your installation:

```bash
synth-pdb --version
```

Expected output:
```
synth-pdb version 1.19.0
```

Test basic functionality:

```bash
synth-pdb --length 5 --output test.pdb
```

This should create a `test.pdb` file in the current directory.

## Troubleshooting

### ImportError: No module named 'openmm'

If you see this error when using `--minimize`:

```bash
pip install openmm
```

Or skip minimization:

```bash
synth-pdb --length 10  # Works without OpenMM
```

### Apple Silicon: OpenMM Not Found

Use conda instead of pip:

```bash
conda install -c conda-forge openmm
```

### Windows: Long Path Issues

Enable long paths in Windows:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Generate your first structure
- [First Structure Tutorial](first-structure.md) - Detailed walkthrough
