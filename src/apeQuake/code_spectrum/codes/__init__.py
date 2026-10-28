"""Code editions. Importing this package registers every model."""
from .asce7_10 import ASCE7_10Spectrum
from .asce7_16 import ASCE7_16Spectrum, ASCE7_16VerticalSpectrum
from .asce7_22 import ASCE7_22Spectrum
from .asce7_22_mprs import MPRS_PERIODS, ASCE7_22MultiPeriodSpectrum
from .nec import NECSpectrum

__all__ = [
    "MPRS_PERIODS",
    "ASCE7_10Spectrum",
    "ASCE7_16Spectrum",
    "ASCE7_16VerticalSpectrum",
    "ASCE7_22MultiPeriodSpectrum",
    "ASCE7_22Spectrum",
    "NECSpectrum",
]
