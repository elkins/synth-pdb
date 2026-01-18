# stupid-pdb

A simple tool to naively generate Protein Data Bank (PDB) files.
The resulting linear peptide may not necessarily be biophysically realistic or useful for any purpose.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Output PDB Format](#output-pdb-format)
- [Logging](#logging)
- [Development and Extensibility](#development-and-extensibility)

## Installation

You can install `stupid-pdb` directly from the project directory using pip:

```bash
pip install .
```

This will install the `stupid-pdb` package and make the `stupid-pdb` command available in your terminal.

## Usage

The `stupid-pdb` tool is a command-line interface (CLI) application.

### Running after `pip install .`

If you have installed the package using `pip install .`, you can use the `stupid-pdb` command from any directory:

```bash
stupid-pdb --length 50
```

### Running directly with Python

You can also run the tool directly from the project root directory without installing it via pip:

```bash
python -m stupid_pdb.main --length 50
```

This will create a file named `random_linear_peptide_50_YYYYMMDD_HHMMSS.pdb` in your current directory.

### Command-line Arguments

-   `--length <LENGTH>`: Specifies the length of the amino acid sequence (number of residues).
    -   Type: Integer
    -   Default: `10`
    -   Example: `--length 100`

-   `--output <FILENAME>`: Optional. Provides a custom filename for the output PDB file.
    -   Type: String
    -   Example: `--output my_custom_protein.pdb`

-   `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Sets the verbosity of the logging output.
    -   Type: String
    -   Default: `INFO`
    -   Example: `--log-level DEBUG` (for verbose debugging information)

-   `--full-atom`: A flag that, when present, instructs the tool to generate a full atomic representation (N, CA, C, O, and side chain atoms) for each residue, rather than just the C-alpha atom.
    -   Type: Boolean flag (no value needed)
    -   Example: `--full-atom`

-   `--sequence <SEQUENCE_STRING>`: Specify an amino acid sequence (e.g., 'AGV' or 'ALA-GLY-VAL'). When provided, this sequence will be used instead of random generation.
    -   Type: String
    -   Example: `--sequence "AGV"` or `--sequence "ALA-GLY-VAL"`

-   `--plausible-frequencies`: A flag that, when present, instructs the tool to use biologically plausible amino acid frequencies for random sequence generation. This option is ignored if `--sequence` is provided.
    -   Type: Boolean flag (no value needed)
    -   Example: `--plausible-frequencies`

### Examples:

1.  **Generate a 25-residue protein with default filename (uniform random sequence):**
    ```bash
    stupid-pdb --length 25
    ```

2.  **Generate a 10-residue protein with a custom filename (CA-only, uniform random sequence):**
    ```bash
    stupid-pdb --length 10 --output short_peptide.pdb
    ```

3.  **Generate a 50-residue protein with debug logging (CA-only, uniform random sequence):**
    ```bash
    stupid-pdb --length 50 --log-level DEBUG
    ```

4.  **Generate a 5-residue protein with full atomic detail (uniform random sequence):**
    ```bash
    stupid-pdb --length 5 --full-atom --output full_atom_peptide.pdb
    ```

5.  **Generate a 100-residue protein using biologically plausible amino acid frequencies:**
    ```bash
    stupid-pdb --length 100 --plausible-frequencies --output plausible_peptide.pdb
    ```

6.  **Generate a protein with a specific sequence:**
    ```bash
    stupid-pdb --sequence "AGV" --output my_specific_peptide.pdb
    # or using 3-letter codes
    stupid-pdb --sequence "ALA-GLY-VAL" --output another_specific_peptide.pdb
    ```


## Output PDB Format

The generated PDB files can now contain either:
-   **C-alpha only**: This is the default. Each residue is represented by a single `ATOM` record for its alpha-carbon (CA).
-   **Full atomic representation**: When the `--full-atom` flag is used, each residue will include `ATOM` records for its backbone atoms (Nitrogen, Alpha-Carbon, Carbonyl Carbon, and Carbonyl Oxygen) and all heavy atoms of its side chain (e.g., C-beta, C-gamma, etc.). Hydrogen atoms are not included in this rudimentary full atom model.

In both cases, the protein is represented as a linear chain along the X-axis, with a fixed distance of 3.8 Angstroms between consecutive C-alpha atoms. The full atomic representation positions atoms relative to this backbone geometry using simplified bond lengths and angles.

Each `ATOM` record includes:
-   Atom number
-   Atom name (e.g., 'N', 'CA', 'C', 'O', 'CB', 'CG', etc.)
-   Residue name (3-letter code, randomly chosen)
-   Chain ID (always 'A')
-   Residue number
-   X, Y, Z coordinates (Y and Z will vary for full atom models)
-   Occupancy and Temperature Factor (fixed values)
-   Element (e.g., 'C', 'N', 'O', 'S')

## Logging

The tool uses Python's standard `logging` module.
-   `INFO` level messages provide general updates on the process.
-   `DEBUG` level messages offer detailed insights into internal operations.
-   `WARNING` messages indicate unusual but non-critical conditions.
-   `ERROR` and `CRITICAL` messages are used for exceptions and severe failures.

You can control the logging verbosity using the `--log-level` argument.

## Development and Extensibility

The project is structured with modularity in mind:
-   `stupid_pdb/data.py`: Defines static data like standard amino acids.
-   `stupid_pdb/generator.py`: Contains the core logic for sequence generation and PDB content formatting.
-   `stupid_pdb/main.py`: Handles command-line argument parsing and orchestrates the generation process.
-   `tests/`: Contains unit tests for the core logic.

This separation of concerns allows for easy extension, such as adding new conformation types (e.g., helices, sheets) or supporting different atom types in the PDB output.
