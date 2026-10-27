# Code spectrum

Building-code elastic design spectra (NEC-15, ASCE 7-10, 7-16, 7-22 two-period
and multi-period), their comparison with a record's response spectrum via
`rec.code_spectrum`, and their export to ETABS, SAP2000 and OpenSees.

::: apeQuake.code_spectrum.composite.CodeSpectrum

::: apeQuake.code_spectrum.base.CodeSpectrumModel

::: apeQuake.code_spectrum.base.AsceTwoPeriodSpectrum

::: apeQuake.code_spectrum.base.register_code

## ASCE 7-22 multi-period

::: apeQuake.code_spectrum.codes.asce7_22_mprs
    options:
      members:
        - ASCE7_22MultiPeriodSpectrum
        - MPRS_PERIODS
        - infer_sms_sm1

## Export

::: apeQuake.code_spectrum.export
    options:
      members:
        - to_etabs
        - to_sap2000
        - to_opensees
        - to_csv
        - sampled
