# Response spectra

`rec.response_spectra` solves the single-degree-of-freedom equation of motion
for each oscillator period and component, returning displacement (Sd),
velocity (Sv), and acceleration (Sa) spectra.

The equation of motion (unit mass) is

$$ \ddot{u} + 2\,\xi\,\omega\,\dot{u} + \omega^2 u = -a_g(t), \qquad \omega = \tfrac{2\pi}{T} $$

integrated with **Newmark's average-acceleration** scheme (γ = 0.5, β = 0.25),
which is unconditionally stable. The per-period solve runs on a Numba-parallel
kernel when Numba is available, with an identical pure-Python fallback.

## Multi-component compute & plot

```python
import numpy as np

rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)

T = np.linspace(0.05, 5.0, 200)
rec.response_spectra.compute(
    periods=T,
    damping=0.05,        # ξ
    sa_mode="pseudo",    # 'pseudo' (ω²·Sd) or 'absolute' (max|ü + a_g|)
)

rec.response_spectra.plot(quantity="Sa", representation="loglog")
rec.response_spectra.plot(quantity="Sd", combined=True)
```

After `compute()` the results live on the object:

```python
rec.response_spectra.periods   # T grid
rec.response_spectra.Sa["X"]   # per-component dicts
rec.response_spectra.Sv["X"]
rec.response_spectra.Sd["X"]
```

## Sa: pseudo vs. absolute

| `sa_mode` | Definition | Use when |
|-----------|------------|----------|
| `"pseudo"` | Sa = ω²·Sd | Standard design spectra; Sa, Sv, Sd are consistent by construction. |
| `"absolute"` | Sa = max\|ü + a_g\| | You need true total acceleration (e.g. floor demands at high damping). |

## Single-component convenience API

For one component, `compute_response_spectrum` returns the arrays directly:

```python
rs = rec.response_spectra.compute_response_spectrum(
    periods=T, component="X", sa_mode="absolute"
)
rs["T"], rs["Sa"], rs["Sv"], rs["Sd"]

rec.response_spectra.plot_response_spectrum(rs)   # stacked Sa/Sv/Sd
```

The registered filter pipeline is applied here on the same terms as
`compute()`: when reading from the record it is on by default; pass
`use_filters=False` to skip it.

## A single SDOF history

To inspect one oscillator's full response (not just the peaks):

```python
out = rec.response_spectra.newmark_sdof(T=1.0, component="X", plot=True)
out["displacement"], out["velocity"], out["total_acceleration"]
```

See the [`ResponseSpectra` API](../api/response_spectra.md) for every option.

## RotD00, RotD50, RotD100

Orientation-independent spectra of two horizontal components (Boore, 2010):
the oscillator responses are rotated through 0–180° and, per period, the
pseudo-acceleration `ω² max|u_θ|` is reduced to percentiles over the angles.

```python
r = rec.response_spectra.rotd(T, components=("X", "Y"))   # angle_step=1 deg
r["RotD50"], r["RotD100"], r["RotD00"]
r["angle_RotD100"]       # orientation of the maximum per period
r["Sa_angles"]           # (nT, n_angles) PSA at every angle

rec.response_spectra.rotd(T, percentiles=(50, 84, 100))   # any percentiles
rec.response_spectra.plot_rotd()                          # from the cached result
```

`rotd` uses the same input and filter-pipeline convention as `compute`
(`df=`, `use_filters=`). It is exact: the oscillator is linear, so rotating its
response equals rotating the input record.
