# ASCE 7-22 multi-period design spectrum

`ASCE7_22MultiPeriodSpectrum` implements the **multi-period** design response
spectrum of ASCE 7-22 Section 11.4.5.1 (5 % damping, `Sa` in g). Section 11.4.5
makes it the default design spectrum of 7-22. The
[two-period spectrum](code-spectrum-asce7-22.md) is permitted only where the
multi-period values are not available from the USGS Seismic Design Geodatabase
(Exception 2).

## Usage

The input is the multi-period 5 %-damped **MCE_R** spectrum the geodatabase
gives for the site class, normally at the 22 periods of item 1
(`MPRS_PERIODS`: 0, 0.01, 0.02 … 7.5, 10 s), plus `SMS` / `SM1` from the same
response:

```python
from apeQuake.code_spectrum import MPRS_PERIODS

rec.code_spectrum.add(
    "ASCE 7-22 multi-period",
    periods=MPRS_PERIODS, sa_mcer=sa_mcer,     # MCE_R ordinates [g]
    sms=1.53, sm1=1.21,                        # geodatabase, Section 11.4.3
    tl=8.0, site_class="CD",
)
rec.code_spectrum.compute(t_max=6.0)
```

Standalone:

```python
from apeQuake.code_spectrum import ASCE7_22MultiPeriodSpectrum

m = ASCE7_22MultiPeriodSpectrum(MPRS_PERIODS, sa_mcer, tl=8.0, site_class="CD",
                                sms=1.53, sm1=1.21)
m.sa([0.1, 0.5, 1.0, 12.0])
m.parameters()     # SMS, SM1, SDS, SD1, SMS_SM1_source, TL, site_class, level, ...
m.two_period()     # the 11.4.5.2 spectrum from the same SDS / SD1, for comparison
```

## Inputs

| Input | Meaning |
|---|---|
| `periods`, `sa_mcer` | MCE_R ordinates in g for `site_class`. Periods strictly increasing and starting at 0; `parameters()["standard_periods"]` tells whether they are exactly the 22 of item 1 |
| `tl` | Long-period transition period in s (Figures 22-14 to 22-17 or the geodatabase) |
| `site_class` | `A`, `B`, `BC`, `C`, `CD`, `D`, `DE`, `E`. `F` raises (Sections 11.4.7, 21.1) |
| `sms`, `sm1` | Site-adjusted MCE_R parameters from the geodatabase. Optional, but give both or neither |
| `level` | `"design"` (default, 2/3 of the ordinates) or `"mcer"` (the ordinates themselves, Section 11.4.6) |

## Equations (Section 11.4.5.1)

1. At the tabulated periods, `Sa = 2/3 × MCE_R ordinate`.
2. Between them (`T < 10 s`), linear interpolation.
3. Beyond 10 s: `Sa = Sa(10) · 10 / T` for `T ≤ TL`, and
   `Sa = Sa(10) · 10 TL / T²` for `T > TL`. These are applied as written, so
   with `TL < 10 s` the second branch starts right after 10 s.

`SDS = 2/3 SMS` and `SD1 = 2/3 SM1` (Eqs. 11.4-1 and 11.4-2) are what the SDC,
the ELF procedure and `two_period()` use.

## When `SMS` / `SM1` are not given

They are inferred from the ordinates with the Section 21.4 rules, and
`parameters()["SMS_SM1_source"]` says so:

- `SMS = 0.9 × max Sa` over `0.2 ≤ T ≤ 5 s`;
- `SM1 = 0.9 × max (T · Sa)` over 1–2 s where `v̄s > 442 m/s` (Site Classes A,
  B, BC and C) or over 1–5 s otherwise (CD, D, DE, E), but not less than
  `Sa(1 s)`.

This is only a fallback. Section 21.4 is written for site-specific spectra, and
the geodatabase publishes `SMS` and `SM1` directly.

!!! note "Difference from codeSpectra"
    codeSpectra's `ASCE7_22.from_mprs` applies the 1–5 s window to Site Class C.
    Table 20.2-1 puts C above 442 m/s, so apeQuake uses 1–2 s. Otherwise the two
    agree exactly on the spectrum from 0 to 16 s.

## Exporting

Every code spectrum can be written for ETABS, SAP2000 or OpenSees; see
[Exporting code spectra](code-spectrum-export.md). For this model, pass
`t_max >= 10` to export the whole tabulated range.
