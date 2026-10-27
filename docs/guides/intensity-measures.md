# Intensity measures

`rec.intensity_measures` provides peak, energy and duration measures on a
stateful working DataFrame (`df_im`), with the same `apply_*` filter wrappers
as the other processors.

## Summary table

```python
rec.intensity_measures.summary()
#              PGA     PGV     PGD  D5-75  D5-95      Ia
# component
# X          0.161  0.0060  0.0014   1.20   2.41  0.0696
```

One row per component. `Ia` (m/s) is included when the units are known
(`units=` or `Record.units`).

## Velocity, displacement and peaks

Velocity and displacement are integrated from the working acceleration
(exact for acceleration linear between samples, zero initial conditions):

```python
t, v = rec.intensity_measures.velocity("X")
t, d = rec.intensity_measures.displacement("X")
rec.intensity_measures.pga("X")   # max|a|, record units
rec.intensity_measures.pgv("X")   # max|v|, record units x s   (cm/s^2 -> cm/s)
rec.intensity_measures.pgd("X")   # max|d|, record units x s^2 (cm/s^2 -> cm)
```

!!! warning "Correct the baseline before integrating"
    A small acceleration offset grows linearly in velocity and quadratically
    in displacement. Apply `apply_baseline_correction()` (or a high-pass) to
    unprocessed records before reading PGV / PGD. PEER NGA records are
    already processed.

## Baseline correction

`apply_baseline_correction(order=2)` fits the polynomial
`c0 t + c1 t² + c2 t³` to the integrated velocity by least squares and
removes its derivative from the acceleration (the correction of
EarthquakeSignal's `BaselineCorrection`):

```python
im = rec.intensity_measures
im.apply_baseline_correction()     # order=2 by default
im.pgd("X")
im.reset_df()
```

It is also a plain filter, `rec.filter.baseline_correction(order=2, df=...)`,
usable in a response-spectra pipeline, and a wrapper on `Spectrum` and
`Spectrogram` (`apply_baseline_correction`).

## Arias intensity

$I_a = \dfrac{\pi}{2g}\int_0^{t_{\text{end}}} a(t)^2\,dt$, in m/s. It is not
unit-agnostic, so the acceleration unit is required:

```python
rec = Record.from_files(x="ACH1_201604162359_E_100.txt")   # units from header
rec.intensity_measures.arias_intensity("X")                 # m/s

Record(x=acc_g, dt=0.01).intensity_measures.arias_intensity("X", units="g")
```

## Husid curve

The Husid curve is the normalized cumulative energy
$\int_0^t a^2\,d\tau \big/ \int_0^{t_{\text{end}}} a^2\,d\tau$, a monotone curve
from 0 to 1 (proportional to the build-up of Arias intensity):

```python
t_h, husid = rec.intensity_measures.husid("X")
```

## Significant duration

The significant duration D(p₁–p₂) is the time between the instants at which the
Husid curve reaches fractions `p1` and `p2`:

```python
d575 = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.75)
d595 = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
```

The common metrics are **D5–75** (`p1=0.05, p2=0.75`) and **D5–95**
(`p1=0.05, p2=0.95`); any `0 ≤ p1 < p2 ≤ 1` is allowed.

For every component at once:

```python
durations = rec.intensity_measures.significant_all(p1=0.05, p2=0.95)
# {'X': 12.3, 'Y': 11.8, 'Z': 9.4}
```

## Preprocessing

Duration measures are sensitive to baseline drift, so detrend/taper first:

```python
rec.intensity_measures.apply_base_filters(detrend=True, demean=True, taper=0.05)
rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
rec.intensity_measures.reset_df()
```

You can also pass a one-off DataFrame via `df=` to any method without touching
the stateful working copy.

See the [`IntensityMeasures` API](../api/intensity_measures.md) for details.
