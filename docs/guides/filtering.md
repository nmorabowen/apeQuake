# Filtering

The `Filter` composite (`rec.filter`) wraps ObsPy's Butterworth filters,
detrending, and tapering. Two rules hold everywhere:

1. **Filters are specified by period** (seconds), not frequency. Internally
   `freq = 1 / Tc`.
2. **Nothing mutates `Record.df`.** Every method returns a new DataFrame.

## Direct use

```python
df = rec.filter.detrend(type="demean")               # 'demean' or 'linear'
df = rec.filter.taper(max_percentage=0.05)
df = rec.filter.band_pass(Tc_low=0.1, Tc_high=5.0)    # keep 0.1 s … 5 s
df = rec.filter.low_pass(Tc=0.1)
df = rec.filter.high_pass(Tc=5.0)
df = rec.filter.band_stop(Tc_low=0.1, Tc_high=5.0)
```

Band-pass period bounds map to frequencies as `freqmin = 1/Tc_high`,
`freqmax = 1/Tc_low`. Butterworth `corners` (default 4) and `zerophase`
(default `True`) are exposed on every filter.

## Chaining

Pass the previous result via `df=` to chain transparently:

```python
df = rec.filter.detrend(type="demean")
df = rec.filter.taper(df=df, max_percentage=0.05)
df = rec.filter.band_pass(df=df, Tc_low=0.1, Tc_high=5.0)
```

## Pipelines on a processor

`ResponseSpectra` accepts a **registered pipeline** that is applied at compute
time:

```python
rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)
rec.response_spectra.add_filter(rec.filter.band_pass, Tc_low=0.1, Tc_high=5.0)

rec.response_spectra.describe_filters()
# "detrend(type='demean') → taper(max_percentage=0.05) → band_pass(...)"

rec.response_spectra.compute(periods=T)   # pipeline applied to record.df
rec.response_spectra.clear_filters()
```

When the input comes from the record (`df=None`), the pipeline is applied by
default. When you pass your own `df`, the pipeline is applied only with
`use_filters=True`.

## Stateful processors

`Spectrum`, `Spectrogram`, and `IntensityMeasures` keep a mutable working
DataFrame and offer `apply_*` wrappers plus a one-shot preset:

```python
rec.spectrum.apply_detrend(type="linear")
rec.spectrum.apply_band_pass(Tc_low=0.1, Tc_high=5.0)

# or the standard preprocessing in one call
rec.spectrum.apply_base_filters(detrend=True, demean=True, taper=0.05,
                                band=(0.1, 5.0))

rec.spectrum.reset_df()   # back to the pristine record
```

Each `apply_*` invalidates cached outputs and logs to `applied_filters`.
