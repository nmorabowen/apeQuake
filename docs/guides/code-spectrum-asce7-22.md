# ASCE 7-22 two-period design spectrum

`ASCE7_22Spectrum` implements the **two-period** design response spectrum of
ASCE 7-22 Section 11.4.5.2 (5 % damping, `Sa` in g).

!!! warning "7-22 has no Fa / Fv tables"
    ASCE 7-10 and 7-16 obtain `SMS = Fa Ss` and `SM1 = Fv S1` from Tables
    11.4-1 and 11.4-2. ASCE 7-22 removed those tables: the site-class-adjusted
    `SMS` and `SM1` (Section 11.4.3) come **directly from the USGS Seismic Design
    Geodatabase** for the site class. This model therefore takes `sms` / `sm1`
    as inputs and does no site-coefficient interpolation. Use the 7-10 or 7-16
    models if you start from mapped `Ss` / `S1`.

## Usage

```python
rec.code_spectrum.add("ASCE7-22", sms=1.5, sm1=0.9, site_class="D", tl=8.0)
rec.code_spectrum.compute()
```

Standalone:

```python
from apeQuake.code_spectrum.codes.asce7_22 import ASCE7_22Spectrum

m = ASCE7_22Spectrum(sms=1.5, sm1=0.9, tl=8.0, site_class="D")
m.parameters()      # SMS, SM1, SDS, SD1, TL, site_class, level, T0, Ts
m.sa([0.1, 0.5, 1.0])

# Site-specific values (Chapter 21, or any external SDS/SD1 pair):
ASCE7_22Spectrum.from_sds_sd1(sds=0.9, sd1=0.5, tl=6.0)
```

## Inputs

| Input | Meaning |
|---|---|
| `sms`, `sm1` | Site-class-adjusted MCE_R accelerations in g (Section 11.4.3), from the USGS geodatabase for `site_class`. Both > 0 |
| `tl` | Long-period transition period in s (Figures 22-14 to 22-17 or the geodatabase); must exceed `Ts` |
| `site_class` | `A`, `B`, `BC`, `C`, `CD`, `D`, `DE`, `E` (Table 20.2-1). Reported in `parameters()`; the accelerations must already correspond to it. `F` raises `ValueError` (site response analysis, Sections 11.4.7 and 21.1) |
| `level` | `"design"` (default) or `"mcer"` |

`from_sds_sd1(sds, sd1, tl, site_class="site-specific", level="design")` bypasses
`sms` / `sm1`; it sets `SMS = 1.5 SDS`, `SM1 = 1.5 SD1`, and does not validate the
site-class label, so Site Class F studies (Section 21.4 values) can be entered.

## Equations

- `SDS = 2/3 SMS` (Eq. 11.4-1), `SD1 = 2/3 SM1` (Eq. 11.4-2).
- `T0 = 0.2 Ts`, `Ts = SD1 / SDS`.
- `T < T0`: `Sa = SDS (0.4 + 0.6 T / T0)` (Eq. 11.4-3).
- `T0 <= T <= Ts`: `Sa = SDS`.
- `Ts < T <= TL`: `Sa = SD1 / T` (Eq. 11.4-4).
- `T > TL`: `Sa = SD1 TL / T^2` (Eq. 11.4-5).
- `level="mcer"` (Section 11.4.6): the MCE_R spectrum is 1.5 times the design
  spectrum, i.e. the same shape built from `SMS` / `SM1`.

## Difference from the multi-period procedure

ASCE 7-22 makes the 22-point **multi-period** design spectrum (Section 11.4.5.1,
USGS geodatabase ordinates, linear interpolation) the default; the two-period
spectrum is the fallback when multi-period values are unavailable (Section
11.4.5, Exception 2). The two can differ materially, especially at long periods.
This class does not implement the multi-period spectrum.

## Limitations

- The `SMS` / `SM1` values are not computed here; obtain them from the USGS
  geodatabase for the exact site class and coordinates.
- Sections 11.4.2 to 11.4.7 and Eqs. 11.4-1 to 11.4-5 were checked against the
  printed ASCE/SEI 7-22: the nine site classes (A, B, BC, C, CD, D, DE, E, F),
  `SS`, `S1`, `SMS` and `SM1` taken from the USGS Seismic Design Geodatabase
  (11.4.3, no `Fa` / `Fv` tables), `SDS = 2/3 SMS`, `SD1 = 2/3 SM1`, the
  two-period spectrum of 11.4.5.2 and `MCER = 1.5 x design` (11.4.6).
- Site Class F, seismic isolation and damped structures need site-specific
  analysis (Chapter 21).
