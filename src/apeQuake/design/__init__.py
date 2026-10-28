"""Design-level parameters built on the code spectra: ASCE 7 and NEC-SE-DS.

* :mod:`apeQuake.design.asce7` — importance factor ``Ie``, seismic design
  category, equivalent lateral force (ASCE 7-10 / 7-16 / 7-22, including the
  7-22 Method 1 from the multi-period spectrum).
* :mod:`apeQuake.design.nec` — importance ``I``, period, base shear and force
  distribution of NEC-SE-DS 6.3.
"""
from . import asce7, nec

__all__ = ["asce7", "nec"]
