<div align="center">

<img src="docs/assets/logo.svg" alt="apeQuake logo" width="120" />

# apeQuake

**A modular Python toolkit for seismic ground-motion processing** — filtering, amplitude spectra, time–frequency spectrograms, Newmark response spectra, and engineering intensity measures.

[![Docs](https://img.shields.io/badge/docs-nmorabowen.github.io%2FapeQuake-c0392b)](https://nmorabowen.github.io/apeQuake/)
[![Python](https://img.shields.io/badge/python-3.10%2B-0b2540)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-0b2540)](#-license)

### 📖 [**Read the documentation →**](https://nmorabowen.github.io/apeQuake/)

</div>

---

## Overview

apeQuake provides a clean, extensible framework for working with earthquake
acceleration time histories in engineering workflows. Everything hangs off a
single, transparent **`Record`** object and a set of composable processors:

| Composite | Accessor | Purpose |
|-----------|----------|---------|
| Filtering | `rec.filter` | Detrend, taper, band-pass / -stop, low/high-pass |
| Amplitude spectrum | `rec.spectrum` | Single-sided FFT amplitude spectra |
| Spectrogram | `rec.spectrogram` | Time–frequency power (dB) |
| Response spectra | `rec.response_spectra` | Newmark SDOF Sd / Sv / Sa |
| Intensity measures | `rec.intensity_measures` | Significant duration, Husid curve |
| Plotting | `rec.plot_record` | Time histories & band-pass comparisons |

All operations use DataFrames, support multi-component (X/Y/Z) records, and
follow a transparent, reproducible design with **no hidden state** — the
baseline time history is never mutated unless you ask.

## Features

- **1–3 components** (X, Y, Z) on a uniform time grid.
- **Automatic `dt` detection & interpolation** for irregular time steps.
- **Composable filter pipelines** — declare detrend → taper → band-pass once,
  reuse across spectra, spectrograms, and response spectra.
- **Newmark response spectra** (average acceleration, γ=0.5, β=0.25),
  Numba-accelerated with a pure-Python fallback; pseudo (ω²·Sd) or absolute
  (max|ü + a_g|) Sa.
- **Spectra & spectrograms** with dual frequency↔period axes.
- **Intensity measures** — significant duration (D5–75, D5–95, custom) and the
  normalized Husid curve.

## Installation

```bash
git clone https://github.com/nmorabowen/apeQuake.git
cd apeQuake
pip install -e .
```

Depends on `numpy`, `pandas`, `scipy`, `matplotlib`, `numba`, and `obspy`
(installed automatically). Optional extras: `pip install -e ".[dev]"` for
tests/linting, `pip install -e ".[docs]"` to build the docs site.

## Quick start

```python
import numpy as np
from apeQuake import Record

# Synthetic acceleration
dt = 0.01
t = np.arange(0, 20, dt)
x = 0.5 * np.sin(2 * np.pi * 1.5 * t)

rec = Record(x=x, dt=dt, name="Demo")

# --- Filtering pipeline for the response spectrum ---
rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)

T = np.linspace(0.05, 5.0, 200)
rec.response_spectra.compute(periods=T)                 # pipeline applied
rec.response_spectra.plot(quantity="Sa", representation="loglog")

# Absolute-acceleration Sa (max |ü + a_g|)
rec.response_spectra.compute(periods=T, sa_mode="absolute")

# Single-component convenience API
rs = rec.response_spectra.compute_response_spectrum(periods=T, component="X")
rec.response_spectra.plot_response_spectrum(rs)

# Intensity measures
d595 = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
```

See the [**documentation**](https://nmorabowen.github.io/apeQuake/) for the full
guides and API reference.

## Composite architecture

Each `Record` exposes a set of modular processors:

```python
rec.filter             # low-level signal filtering
rec.spectrum           # amplitude spectra
rec.spectrogram        # time–frequency analysis
rec.response_spectra   # engineering RS (Sd / Sv / Sa)
rec.intensity_measures # intensity measures (duration, Husid, …)
rec.plot_record        # time-history plotting
```

Each module can apply its own filter pipeline and operate on either the
baseline record (`rec.df`) or a user-provided DataFrame (`df=`), and never
modifies the original time history unless explicitly requested.

## Project structure

```
apeQuake/
├── src/apeQuake/
│   ├── core/                 # Record + types
│   ├── filters/              # ObsPy-backed Filter toolbox
│   ├── spectrum/             # amplitude spectra
│   ├── spectrogram/          # time–frequency spectrograms
│   ├── response_spectra/     # Newmark Sd / Sv / Sa
│   ├── intensity_measures/   # duration, Husid
│   └── plot_record/          # time-history plotting
├── tests/                    # pytest suite (Newmark validated vs scipy)
├── docs/                     # MkDocs site
└── mkdocs.yml
```

## Roadmap

- [x] Significant duration
- [x] Newmark response spectra
- [ ] Arias Intensity
- [ ] Cumulative Absolute Velocity (CAV)
- [ ] RotD50 response spectra
- [ ] Batch processing utilities
- [ ] PEER/NGA flatfile importer
- [ ] PyPI package

## Contributing

Pull requests are welcome! Please open an issue to discuss improvements or new
features. Run the test suite with:

```bash
pip install -e ".[dev]"
pytest -q
```

## 📜 License

MIT License © 2025 Nicolás Mora Bowen

## Citation

If you use apeQuake in research, please cite this repository:

> Nicolás Mora Bowen (2025). *apeQuake: A modular toolkit for seismic
> ground-motion processing and engineering response analysis.* GitHub
> Repository. https://github.com/nmorabowen/apeQuake

---

<div align="center">
<sub>Part of José Abell's <em>El Ladruño Research Group</em>.</sub>
</div>
