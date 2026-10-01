# ASCE 7-16 design spectrum

`ASCE7_16Spectrum` implements the two-period design response spectrum of
**ASCE/SEI 7-16, Chapter 11 (11.4)**, 5 % damping, `Sa` in units of g.
Section, table and equation numbers are those printed in ASCE 7-16.

## Usage

```python
rec.code_spectrum.add("ASCE7-16", ss=1.0, s1=0.4, site_class="C", tl=8.0)
rec.code_spectrum.compute()
rec.code_spectrum.plot()
```

Or standalone:

```python
from apeQuake.code_spectrum.codes.asce7_16 import ASCE7_16Spectrum

m = ASCE7_16Spectrum(ss=1.0, s1=0.4, site_class="C", tl=8.0)
m.parameters()        # ss, s1, site_class, level, Fa, Fv, SMS, SM1, SDS, SD1, T0, Ts, TL, exception_applied
m.sa([0.1, 0.5, 1.0]) # g
m.sd(1.0)             # m

mce = ASCE7_16Spectrum(ss=1.0, s1=0.4, site_class="C", tl=8.0, level="mcer")  # 11.4.7
site = ASCE7_16Spectrum.from_sds_sd1(sds=1.1, sd1=0.7, tl=6.0)  # site-specific values
```

## Inputs

| Input | Meaning |
|---|---|
| `ss`, `s1` | Mapped MCE_R spectral accelerations (g), both `> 0` (Figs. 22-1 to 22-8) |
| `site_class` | `"A"` to `"E"`. `"F"` raises `ValueError` (site response analysis, 11.4.8 / 21.1) |
| `tl` | Long-period transition period (s), Figs. 22-14 to 22-17; must exceed `Ts` |
| `level` | `"design"` (default, `SDS`/`SD1`) or `"mcer"` (11.4.7: 1.5 x the design spectrum) |
| `allow_exception` | Apply the 11.4.8 exceptions instead of raising (see below) |
| `vs_measured` | Site Class B only: `False` (no on-site Vs measurement) gives `Fa = Fv = 1.0` (11.4.3) |
| `default_class` | Site Class D chosen as the 11.4.3 default: `Fa >= 1.2` (11.4.4) |

## Equations and tables

- Site coefficients (11.4.4): `Fa` from Table 11.4-1 and `Fv` from Table 11.4-2,
  linear in `Ss` / `S1` between columns, clamped at the end columns.
- Eq. 11.4-1 `SMS = Fa Ss`; Eq. 11.4-2 `SM1 = Fv S1`.
- Eq. 11.4-3 `SDS = 2/3 SMS`; Eq. 11.4-4 `SD1 = 2/3 SM1`.
- Design spectrum (11.4.6, Fig. 11.4-1), `T0 = 0.2 SD1/SDS`, `Ts = SD1/SDS`:
  Eq. 11.4-5 `Sa = SDS (0.4 + 0.6 T/T0)` for `T < T0`; `Sa = SDS` for
  `T0 <= T <= Ts`; Eq. 11.4-6 `Sa = SD1/T` for `Ts < T <= TL`; Eq. 11.4-7
  `Sa = SD1 TL / T^2` for `T > TL`.
- MCE_R spectrum (11.4.7): 1.5 x the design spectrum.

Table 11.4-1, `Fa`:

| Site class | Ss <= 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | >= 1.50 |
|---|---|---|---|---|---|---|
| A | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| B | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| C | 1.3 | 1.3 | 1.2 | 1.2 | 1.2 | 1.2 |
| D | 1.6 | 1.4 | 1.2 | 1.1 | 1.0 | 1.0 |
| E | 2.4 | 1.7 | 1.3 | see 11.4.8 | see 11.4.8 | see 11.4.8 |

Table 11.4-2, `Fv` (D and E cells for `S1 >= 0.2` are flagged "see 11.4.8"):

| Site class | S1 <= 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | >= 0.6 |
|---|---|---|---|---|---|---|
| A | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| B | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| C | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.4 |
| D | 2.4 | 2.2 | 2.0 | 1.9 | 1.8 | 1.7 |
| E | 4.2 | 3.3 | 2.8 | 2.4 | 2.2 | 2.0 |

These differ from ASCE 7-10 (e.g. Site Class B values below 1.0, Site Class D
and E rows, and the "see 11.4.8" cells).

## Site-specific study rules (11.4.8)

ASCE 7-16 requires a ground motion hazard analysis (Chapter 21.2) for:

1. seismically isolated structures and structures with damping systems with
   `S1 >= 0.6` (depends on the structure; **not checked** by this class);
2. Site Class E with `Ss >= 1.0`;
3. Site Class D or E with `S1 >= 0.2`;

and a site response analysis (21.1) for Site Class F.

By default the constructor raises `ValueError` naming the clause. With
`allow_exception=True` the matching exception is applied:

| Case | Exception | What the class does |
|---|---|---|
| E, `Ss >= 1.0` | 1 | `Fa` taken as that of Site Class C (Table 11.4-1 C row) |
| D, `S1 >= 0.2` | 2 | `Cs` from Eq. 12.8-2 for `T <= 1.5 Ts`; 1.5 x Eq. 12.8-3 for `TL >= T > 1.5 Ts`, 1.5 x Eq. 12.8-4 for `T > TL`. Implemented as a spectrum **shape override** (`Cs` is proportional to `Sa`): `Sa = SDS` up to `1.5 Ts`, then `1.5 SD1/T`, then `1.5 SD1 TL/T^2`; continuous at `1.5 Ts`. Needs `TL > 1.5 Ts`. The tabulated `Fv` is used. |
| E, `S1 >= 0.2` | 3 | Only valid for `T <= Ts` with the equivalent lateral force procedure. This is a condition on the design procedure: spectrum numbers are the tabulated ones and **you must verify the condition**. |

`parameters()["exception_applied"]` records which exceptions were used.
None of the exceptions may be used for seismically isolated structures or
structures with damping systems.

`ASCE7_16Spectrum.from_sds_sd1(sds, sd1, tl)` builds the standard shape from
externally computed values (site-specific analysis, Chapter 21.4). It bypasses
the tables and all 11.4.8 checks; you are responsible for the Chapter 21
minimums (e.g. the 80 % floor of 21.3).

## Limitations

- Two-period spectrum only; no simplified-procedure `Fa` (12.14.8.1) and no
  `FPGA` / `PGAM`.
- The wording of the 11.4.8 exceptions follows the ASCE 7-16 text as quoted in
  secondary sources; later supplements may refine it (check the edition you
  design to).
- `TL` and the mapped values are inputs; nothing is looked up from the maps.
- For `E` with `0.75 < Ss < 1.0` the open Table 11.4-1 cell at `Ss = 1.0` is
  closed with the Site Class C value (the exception-1 value) for interpolation.
