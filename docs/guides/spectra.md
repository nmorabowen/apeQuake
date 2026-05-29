# Spectra & spectrograms

## Amplitude spectrum

`rec.spectrum` computes a single-sided FFT amplitude spectrum per component on
its stateful working DataFrame.

```python
rec.spectrum.apply_base_filters(detrend=True, demean=True, taper=0.05)
rec.spectrum.compute(scale="amplitude")   # 'amplitude' or 'none'
rec.spectrum.plot(representation="loglog")
```

- `compute()` does **not** filter — it uses `df_spectrum` as-is. Preprocess
  with the `apply_*` wrappers first.
- `scale="amplitude"` divides by `n` and doubles interior bins (single-sided
  convention), so a pure sinusoid recovers its physical amplitude.
- Plot representations: `linear`, `semilogx`, `semilogy`, `loglog`. Combined
  (`combined=True`) overlays components; otherwise one subplot each.

Pull the arrays directly:

```python
freqs, amps, comps = rec.spectrum.get_arrays(fmin=0.1, fmax=25.0)
```

### Comparing records

```python
base = rec.spectrum
base.compute()
base.compare(other_records=[rec2, rec3],
             labels=["M1", "M2", "M3"],
             align="interp")   # 'strict' requires identical grids
```

## Spectrogram

`rec.spectrogram` produces time–frequency power (dB) via
`matplotlib.mlab.specgram`, with a secondary period axis. It follows the same
**preprocess → compute → plot** pipeline as the other processors: filter the
stateful working DataFrame with the `apply_*` wrappers, then plot.

```python
# 1. Preprocess (mutates df_spectrogram, logged in applied_filters)
rec.spectrogram.apply_base_filters(detrend=True, taper=0.05, band=(0.1, 10.0))

# 2. Set plot ranges
rec.spectrogram.set_plot_limits(fmin=0.1, fmax=25.0, tmin=0.0, tmax=40.0)

# 3. Plot — computes once and caches; no re-filtering at plot time
rec.spectrogram.plot_spectrogram(NFFT=256, noverlap=128, cmap="seismic")
```

`plot_spectrogram()` consumes the cached `compute()` result, recomputing only
if it is missing or was produced with a different `NFFT`/`noverlap`. Because
preprocessing lives entirely in the `apply_*` path, there is **no double
filtering** — the spectrogram reflects exactly the current `df_spectrogram`.
Any `apply_*` call invalidates the cache so the next plot recomputes.

`set_plot_limits()` controls the time window (`tmin/tmax`), frequency window
(`fmin/fmax`), and color scale (`pmin/pmax`); `reset_plot_limits()` clears
them. A frequency floor `> 0` is required for the period secondary axis (the
`fmin` argument is just the fallback when no floor was set).

You can also compute without plotting and read the arrays:

```python
rec.spectrogram.compute(NFFT=256, noverlap=128)
rec.spectrogram.freqs    # (n_freqs,)
rec.spectrogram.bins     # (n_time,)
rec.spectrogram.power    # {comp: (n_freqs, n_time) dB array}
```

See the [`Spectrum`](../api/spectrum.md) and
[`Spectrogram`](../api/spectrogram.md) API pages for full signatures.
