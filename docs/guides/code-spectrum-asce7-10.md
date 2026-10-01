# ASCE 7-10 design spectrum

`ASCE7_10Spectrum` implements the two-period design response spectrum of
**ASCE/SEI 7-10**, Chapter 11, Section 11.4 (5 % damping, `Sa` in g). Section,
table and equation numbers below are those printed in the standard.

## Usage

```python
rec.code_spectrum.add("ASCE7-10", ss=1.5, s1=0.6, site_class="D", tl=8.0)
rec.code_spectrum.compute()
rec.code_spectrum.plot()
```

Or standalone:

```python
from apeQuake.code_spectrum.codes.asce7_10 import ASCE7_10Spectrum

m = ASCE7_10Spectrum(ss=1.5, s1=0.6, site_class="D", tl=8.0)
m.parameters()          # Ss, S1, site_class, Fa, Fv, SMS, SM1, SDS, SD1, T0, Ts, TL, level
m.sa([0.1, 0.5, 1.0])   # g
m.sd(1.0)               # m

mcer = ASCE7_10Spectrum(ss=1.5, s1=0.6, site_class="D", tl=8.0, level="mcer")

# Site-specific or externally computed SDS / SD1 (bypasses Tables 11.4-1 and 11.4-2)
m2 = ASCE7_10Spectrum.from_sds_sd1(sds=1.0, sd1=0.6, tl=8.0)
```

## Inputs

| Input | Meaning |
|---|---|
| `ss`, `s1` | Mapped MCE_R spectral accelerations in g at 0.2 s and 1 s (11.4.1, Figs. 22-1 to 22-6); both > 0 |
| `site_class` | `"A"` to `"E"`. `"F"` raises `ValueError`: a site response analysis is required (11.4.7, Section 21.1) |
| `tl` | Long-period transition period in s (Figs. 22-12 to 22-16); must exceed `Ts = SD1/SDS` |
| `level` | `"design"` (default) or `"mcer"` (design spectrum times 1.5, 11.4.6) |

`from_sds_sd1(sds, sd1, tl, level)` takes the *design* `SDS`, `SD1` directly (for
example from a Chapter 21 site-specific study). `ss`, `s1`, `site_class`, `Fa` and
`Fv` are then `None` (and absent from `parameters()`), and `SMS = 1.5 SDS`,
`SM1 = 1.5 SD1`.

## Equations and tables

Site coefficients (11.4.3, Tables 11.4-1 and 11.4-2, p. 55). Values are
interpolated linearly in `Ss` / `S1` between columns and clamped at the end
columns (the tables read `Ss <= 0.25` and `Ss >= 1.25`, `S1 <= 0.1` and `S1 >= 0.5`).

Table 11.4-1, `Fa`:

| Site class | Ss <= 0.25 | 0.5 | 0.75 | 1.0 | >= 1.25 |
|---|---|---|---|---|---|
| A | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| C | 1.2 | 1.2 | 1.1 | 1.0 | 1.0 |
| D | 1.6 | 1.4 | 1.2 | 1.1 | 1.0 |
| E | 2.5 | 1.7 | 1.2 | 0.9 | 0.9 |

Table 11.4-2, `Fv`:

| Site class | S1 <= 0.1 | 0.2 | 0.3 | 0.4 | >= 0.5 |
|---|---|---|---|---|---|
| A | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| C | 1.7 | 1.6 | 1.5 | 1.4 | 1.3 |
| D | 2.4 | 2.0 | 1.8 | 1.6 | 1.5 |
| E | 3.5 | 3.2 | 2.8 | 2.4 | 2.4 |

```
SMS = Fa Ss          (11.4-1)
SM1 = Fv S1          (11.4-2)
SDS = 2/3 SMS        (11.4-3)
SD1 = 2/3 SM1        (11.4-4)
```

Spectrum (11.4.5, Fig. 11.4-1):

```
Sa = SDS (0.4 + 0.6 T/T0)     T < T0            (11.4-5)
Sa = SDS                      T0 <= T <= Ts
Sa = SD1 / T                  Ts < T <= TL      (11.4-6)
Sa = SD1 TL / T^2             T > TL            (11.4-7)
T0 = 0.2 SD1/SDS,  Ts = SD1/SDS
```

MCE_R spectrum (11.4.6): the design spectrum multiplied by 1.5, with the same
`T0`, `Ts` and `TL`.

## Worked example

`Ss = 1.5`, `S1 = 0.6`, Site Class D, `TL = 8 s`: `Fa = 1.0`, `Fv = 1.5` (both end
columns), `SMS = 1.5`, `SM1 = 0.9`, `SDS = 1.0`, `SD1 = 0.6`, `Ts = 0.6 s`,
`T0 = 0.12 s`; `Sa(2 s) = 0.6/2 = 0.30 g`.

## Limitations

- Site Class F is not supported (11.4.7); use `from_sds_sd1` with site-specific values.
- Only the two-period spectrum of 11.4.5 is built; the vertical spectrum, the
  site-specific procedures of Chapter 21 and the seismic design category tables
  (11.6) are not implemented.
- `Ss`, `S1` and `TL` are inputs: the mapped values must come from the Chapter 22
  maps or the USGS tool; apeQuake does not look them up.
