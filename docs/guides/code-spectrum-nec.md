# NEC-15 design spectrum (Ecuador)

`NECSpectrum` implements the elastic design spectrum of **NEC-SE-DS**
(*Peligro sismico y requisitos de diseno sismo resistente*, NEC-15), 5 % damping,
with `Sa` in units of g. Section, table and page numbers below are those printed
in the standard.

## Usage

```python
rec.code_spectrum.add("NEC", z=0.40, site_class="D", region="sierra")
rec.code_spectrum.compute()
rec.code_spectrum.plot()
```

Or standalone:

```python
from apeQuake.code_spectrum.codes.nec import NECSpectrum

m = NECSpectrum(z=0.40, site_class="D", region="sierra")
m.parameters()          # Z, eta, Fa, Fd, Fs, r, T0, Tc, TL, site_class, region
m.sa([0.1, 0.5, 1.0])   # g
m.sd(1.0)               # m   (3.3.2)
m.reduced_sa(0.8, r_factor=6, phi_p=0.9, phi_e=1.0, importance=1.3)  # V/W (6.3.2)
```

## Inputs

| Input | Meaning |
|---|---|
| `z` | Zone factor Z (fraction of g, > 0) |
| `site_class` | `"A"` to `"E"`. `"F"` raises `ValueError`: a site-specific response study is required (3.2 and 10.5.4) |
| `region` | `"costa"` (1.80), `"sierra"` (2.48), `"oriente"` (2.60), `"galapagos"` (2.48), `"esmeraldas"` (2.48) |
| `eta` | optional override of the regional value |
| `fa`, `fd`, `fs` | optional overrides (e.g. from a microzonation study, 3.3.1); each one replaces its table value |
| `component` | `"horizontal"` (default) or `"vertical"` (2/3 of horizontal, 3.4.2) |
| `ascending_branch` | `True` (default): full Fig. 3 curve, left branch for T < T0 (`Sa(0) = Z Fa`). `False`: plateau down to T = 0 (static analysis / fundamental mode) |

## Equations

Site coefficients (3.2.2, Tables 3, 4, 5, pp. 31-32): `Fa`, `Fd`, `Fs` are
interpolated linearly in Z between the columns 0.15, 0.25, 0.30, 0.35, 0.40 and
>= 0.50, and clamped to the end columns outside that range.

Spectrum (3.3.1, pp. 33-35):

```
Sa = eta Z Fa                     0 <= T <= Tc
Sa = eta Z Fa (Tc / T)^r          T > Tc
T0 = 0.10 Fs Fd / Fa
Tc = 0.55 Fs Fd / Fa
TL = 2.4 Fd                       (at most 4 s for site classes D and E)
r  = 1.0 (classes A-D),  1.5 (class E)
```

`eta` (3.3.1, p. 34): 1.80 for the Costa provinces **except Esmeraldas**; 2.48 for
the Sierra, Esmeraldas and Galapagos; 2.60 for the Oriente.

Left branch (3.3.1, p. 35, Fig. 3): `Sa = Z Fa [1 + (eta - 1) T / T0]` for
`T <= T0`. NEC states it is for dynamic analysis of modes **other than the
fundamental**; for static analysis and the fundamental mode the plateau reaches
T = 0 (appendix 10.1.2). The default draws the full Fig. 3 curve
(`Sa(0) = Z Fa`), which is what you want when comparing with a record;
pass `ascending_branch=False` for the plateau (`Sa(0) = eta Z Fa`).
`reduced_sa`, used for force-based design, always takes the plateau.

Displacement spectrum (3.3.2, p. 36): `Sd = Sa g (T / 2 pi)^2` for `T <= TL`; it is
held constant beyond `TL` (Fig. 4), with `TL` capped at 4 s for classes D and E.

Vertical component (3.4.2, p. 37): `Ev >= 2/3 Eh`; the model uses exactly 2/3 of
the horizontal spectrum. The code gives no period range for it.

Design ordinate (6.3.2, p. 61): the base shear is
`V = I Sa(Ta) W / (R phiP phiE)`, so `reduced_sa` returns
`I Sa(T) / (R phiP phiE)`. It always uses the plateau-from-zero horizontal
spectrum. (The text of 6.3.2 points to 3.3.2 for `Sa`; that is an erratum,
it is 3.3.1.)

## Limitations

- Site class F, and essential / special structures that require hazard-curve
  based ordinates (6.3.2), are outside the tabulated procedure.
- Z below 0.15 or above 0.50 is clamped to the table end columns; the standard
  tabulates nothing outside that range.
- The vertical component is the general-case 2/3 factor only; near-field
  essential structures (3.4.3) need a site response study.
- The 3.3.2 formula for `T > TL` is ambiguous in the printed text (the
  formula uses `TL` in the period factor, the figure shows a constant). The
  constant branch is implemented.
- No Ta estimation, R tables, irregularity factors or drift checks: this
  module only builds spectra.
