"""
NMR Relaxation calculations for synth-pdb.

This module now provides compatibility shims that re-export from the synth-nmr package.
For direct usage of NMR functionality, consider using synth-nmr directly:
    pip install synth-nmr
    from synth_nmr import calculate_relaxation_rates, predict_order_parameters

See: https://github.com/elkins/synth-nmr
"""

# Re-export from synth-nmr for backward compatibility
from synth_nmr import (
    calculate_relaxation_rates,
    predict_order_parameters,
)
from synth_nmr.relaxation import spectral_density, njit

__all__ = [
    "calculate_relaxation_rates",
    "predict_order_parameters",
    "spectral_density",
    "njit",
]
