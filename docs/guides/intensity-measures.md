# Intensity measures

`rec.intensity_measures` provides duration and energy-accumulation measures on
a stateful working DataFrame (`df_im`), with the same `apply_*` filter wrappers
as the other processors.

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
