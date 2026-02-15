"""
NEF (NMR Exchange Format) I/O for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import write_nef_file, read_nef_restraints

See: https://github.com/elkins/synth-nmr
"""

# Re-export from synth-nmr for backward compatibility
from synth_nmr.nef_io import (
    read_nef_restraints,
    write_nef_file,
    write_nef_relaxation,
    write_nef_chemical_shifts,
)

__all__ = [
    "read_nef_restraints",
    "write_nef_file",
    "write_nef_relaxation",
    "write_nef_chemical_shifts",
]
