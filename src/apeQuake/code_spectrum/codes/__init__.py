"""Code editions. Importing this package registers every model."""
from .asce7_10 import ASCE7_10Spectrum
from .asce7_16 import ASCE7_16Spectrum
from .asce7_22 import ASCE7_22Spectrum
from .nec import NECSpectrum

__all__ = ["ASCE7_10Spectrum", "ASCE7_16Spectrum", "ASCE7_22Spectrum", "NECSpectrum"]
