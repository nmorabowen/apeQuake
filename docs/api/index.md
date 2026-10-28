# API Reference

apeQuake is organized around a single [`Record`](record.md) that owns a set of
composite processors. Each processor is documented on its own page; signatures
and docstrings below are generated directly from the source.

| Composite | Accessor | Purpose |
|-----------|----------|---------|
| [`Record`](record.md) | — | Multi-component ground-motion container |
| [Readers](readers.md) | `Record.from_files` | PEER `.AT2`, RENAC and text record files |
| [`Filter`](filters.md) | `rec.filter` | Detrend, taper, baseline correction, band-pass / -stop, low/high-pass |
| [`Spectrum`](spectrum.md) | `rec.spectrum` | FFT amplitude spectra |
| [`Spectrogram`](spectrogram.md) | `rec.spectrogram` | Time–frequency spectrograms |
| [`ResponseSpectra`](response_spectra.md) | `rec.response_spectra` | Newmark SDOF Sd / Sv / Sa, RotD00/50/100 |
| [`IntensityMeasures`](intensity_measures.md) | `rec.intensity_measures` | PGA / PGV / PGD, Arias intensity, significant duration, Husid |
| [`PlotRecord`](plot_record.md) | `rec.plot_record` | Time-history plots & band-pass comparisons |
| [`EcuadorHazard`](hazard.md) | `apeQuake.hazard` | IG-EPN seismic hazard of Ecuador: UHS, hazard curves, sources, catalogs |

```python
from apeQuake import Record

rec = Record(x=acc, dt=0.01)
rec.filter             # Filter
rec.spectrum           # Spectrum
rec.spectrogram        # Spectrogram
rec.response_spectra   # ResponseSpectra
rec.intensity_measures # IntensityMeasures
rec.plot_record        # PlotRecord
```
