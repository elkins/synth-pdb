"""
Chemical Shift prediction for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import predict_chemical_shifts, calculate_csi

See: https://github.com/elkins/synth-nmr
"""

# Re-export from synth-nmr for backward compatibility
from synth_nmr.chemical_shifts import (
    calculate_csi,
    predict_chemical_shifts,
    get_secondary_structure,
    RANDOM_COIL_SHIFTS,
    SECONDARY_SHIFTS,
    # Private functions used in tests
    _calculate_ring_current_shift,
    _get_aromatic_rings,
)

__all__ = [
    "predict_chemical_shifts",
    "calculate_csi",
    "get_secondary_structure",
    "RANDOM_COIL_SHIFTS",
    "SECONDARY_SHIFTS",
    "_calculate_ring_current_shift",
    "_get_aromatic_rings",
]
