# Spectrum operations

Every code spectrum model has operations that return a new model, a
`DerivedSpectrum`. The result plots, compares, exports and combines like any
code spectrum, and keeps every corner period of the spectra it came from.

```python
from apeQuake.code_spectrum.codes import ASCE7_16Spectrum, NECSpectrum

a = ASCE7_16Spectrum(ss=1.5, s1=0.6, site_class="C", tl=8.0)
n = NECSpectrum(z=0.4, site_class="D", region="costa")

a.scaled(1.5)                                   # every ordinate x 1.5
a.reduced(8.0, Ie=1.25)                         # Sa Ie / R
n.reduced(6.0, Ie=1.3, phi_p=0.9, phi_e=0.9)    # NEC: I Sa / (R phiP phiE)
a.envelope(n)                                   # point-wise maximum
site.floored_by(a, 0.80)                        # ASCE 7 21.3: >= 80 % of 11.4
n.capped_by(a)                                  # point-wise minimum
a.peak()                                        # (T, Sa) of the maximum

rec.code_spectrum.add(a.envelope(n, label="NEC / ASCE envelope"))
```

| Method | Result |
|---|---|
| `scaled(f)` | `f · Sa` |
| `reduced(R, Ie=1, phi_p=1, phi_e=1)` | `Sa · Ie / (R φP φE)` |
| `envelope(*others)` | `max(Sa, other.Sa, …)` |
| `floored_by(other, ratio=1)` | `max(Sa, ratio · other.Sa)` |
| `capped_by(other, ratio=1)` | `min(Sa, ratio · other.Sa)` |
| `peak(t_max=4)` | `(T, Sa)` at the maximum ordinate |

`parameters()` of the result lists the factors and the codes it was built
from. `sd()` is the generic `Sa g (T/2π)²`. For NEC static design, use
`NECSpectrum.reduced_sa`, which also drops the ascending branch.
