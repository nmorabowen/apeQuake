# Design: Ie, SDC and ELF

`apeQuake.design` turns code-spectrum parameters into design quantities:
`apeQuake.design.asce7` for ASCE/SEI 7-10, 7-16 and 7-22, and
`apeQuake.design.nec` for NEC-SE-DS 2015. All tables and equations were
checked against the standards' PDFs.

## ASCE 7: importance factor and seismic design category

```python
from apeQuake.code_spectrum.codes import ASCE7_16Spectrum
from apeQuake.design import asce7

m = ASCE7_16Spectrum(ss=1.5, s1=0.6, site_class="C", tl=8.0)

asce7.importance_factor("III")                       # 1.25 (Table 1.5-2)
asce7.seismic_design_category(m.sds, m.sd1, s1=m.s1, risk_category="IV")
# SDCResult(sdc='D', from_sds='D', from_sd1='D', risk_category='IV', s1_override=False)
```

The SDC follows Section 11.6. If `S1 ≥ 0.75` it is E (Risk Category I–III) or F
(IV). Otherwise it is the more severe of Table 11.6-1 (`SDS`) and Table 11.6-2
(`SD1`). `s1` is required. Pass `None` only if it is unknown, and the result
then reports `s1_override=None` (the override was not checked).

## ASCE 7: equivalent lateral force (Section 12.8)

```python
r = asce7.base_shear(
    W=50_000, R=8.0, sds=m.sds, sd1=m.sd1, s1=m.s1, tl=m.tl, risk_category="II",
    T=2.1, hn=40.0, structure_type="steel_moment_frame",     # T capped at Cu Ta
    weights=w, heights=h,                                     # optional Fx, Vx
    edition="7-16",
)
r.V, r.Cs, r.T, r.Ta, r.Cu, r.k, r.governing    # e.g. governing 'cap eq 12.8-3'
r.Fx, r.Vx                                      # story forces and shears
r.breakdown                                     # every Cs equation and limit
```

- `Ta = Ct hn^x` (Table 12.8-2), with `hn` in m (`metric=True`) or ft.
- With `hn` given, an analysis `T` is capped at `Cu Ta` (Table 12.8-1), unless
  you pass `apply_upper_limit=False`. Without `T`, `Ta` is used.
- `Cs = SDS / (R/Ie)`, capped by `SD1 / (T R/Ie)` (or `SD1 TL / (T² R/Ie)`) and
  floored by `0.044 SDS Ie ≥ 0.01` and, where `S1 ≥ 0.6`, by `0.5 S1 / (R/Ie)`.
- `k` is 1 below 0.5 s, 2 above 2.5 s, and linear between. Pass
  `k_interpolate=False` to use 2, which the code also allows.

### ASCE 7-22 Method 1

ASCE 7-22 Section 12.8.1.1 adds **Method 1**: `Cs = Sa(T) / (R/Ie)`, read from
the multi-period design spectrum. Where `T` is below the period of the maximum
`Sa`, the maximum is used. Pass the spectrum to select it:

```python
from apeQuake.code_spectrum import ASCE7_22MultiPeriodSpectrum, MPRS_PERIODS

sp = ASCE7_22MultiPeriodSpectrum(MPRS_PERIODS, sa_mcer, tl=8.0, site_class="CD",
                                 sms=sms, sm1=sm1)
r = asce7.base_shear(W=..., R=8.0, sds=sp.sds, sd1=sp.sd1, s1=s1, tl=sp.tl,
                     T=1.2, edition="7-22", spectrum=sp)
r.breakdown["method"]      # 1
```

Without `spectrum`, 7-22 uses **Method 2** (the classical form). Method 1 is
not allowed where Exception 2 of Section 11.4.5 applies, i.e. with the
two-period spectrum. Equation numbers in `breakdown` follow the edition: 7-22
renumbered them to 12.8-2 … 12.8-8.

Not implemented: `Ta = 0.1 N`, the shear-wall `Cq/√Cw` period, the Section
11.6 exception that assigns the SDC from Table 11.6-1 alone, and the optional
`SDS = 1.0` of Section 12.8.1.3.

## NEC-SE-DS: base shear (6.3)

```python
from apeQuake.code_spectrum.codes import NECSpectrum
from apeQuake.design import nec

sp = NECSpectrum(z=0.4, site_class="D", region="costa")
r = nec.base_shear(sp, W=50_000, R=8.0, I=nec.importance_factor("especial"),
                   phi_p=0.9, phi_e=0.9, hn=24.0, structure_type="hormigon_porticos",
                   weights=w, heights=h)
r.V, r.Sa_Ta, r.Ta, r.shear_coefficient, r.Fx, r.Vx
```

- `I` comes from Tabla 6: esencial 1.5, especial 1.3, otra 1.0.
- Método 1: `Ta = Ct hn^α` (6.3.3 a, `hn` in m). A Método 2 period passed as
  `T` is capped at 1.30 times Método 1 (6.3.3 b).
- `V = I Sa(Ta) W / (R φP φE)` (6.3.2). The plateau runs down to T = 0, because
  the ascending branch is not used for static design.
- `k` is 1, `0.75 + 0.5 T`, or 2 (6.3.5).

Pass the **elastic horizontal** spectrum: the reduction is applied inside
`base_shear`.

Not implemented: the alternative `Ct = 0.0055/√Cw` for wall structures.
