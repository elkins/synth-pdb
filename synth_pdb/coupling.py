"""
Coupling utilities for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_hn_ha_coupling, predict_couplings_from_structure

See: https://github.com/elkins/synth-nmr
"""

# Re-export from synth-nmr for backward compatibility
from synth_nmr.coupling import (
    calculate_hn_ha_coupling,
    predict_couplings_from_structure,
)

__all__ = [
    "calculate_hn_ha_coupling",
    "predict_couplings_from_structure",
]
