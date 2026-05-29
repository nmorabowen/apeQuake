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
`matplotlib.mlab.specgram`, with a secondary period axis.

```python
rec.spectrogram.set_plot_limits(fmin=0.1, fmax=25.0, tmin=0.0, tmax=40.0)
rec.spectrogram.plot_spectrogram(
    NFFT=256,
    noverlap=128,
    detrend=True,         # plot-time preprocessing
    taper=0.05,
    band=(0.1, 10.0),     # optional band-pass, in periods (s)
    cmap="seismic",
)
```

`set_plot_limits()` controls the time window (`tmin/tmax`), frequency window
(`fmin/fmax`), and color scale (`pmin/pmax`); `reset_plot_limits()` clears
them. A frequency `fmin > 0` is required for the period secondary axis.

!!! tip "Preprocessing applies twice if you're not careful"
    `plot_spectrogram()` applies its own `detrend`/`taper`/`band` at plot time
    on top of whatever is already in `df_spectrogram`. If you pre-filtered with
    `apply_*`, pass `detrend=False, taper=0.0` to avoid double processing.

See the [`Spectrum`](../api/spectrum.md) and
[`Spectrogram`](../api/spectrogram.md) API pages for full signatures.
