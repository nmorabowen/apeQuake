🐒 apeQuake

A modular Python toolkit for seismic ground-motion processing, filtering, spectra, spectrograms, and engineering intensity measures.

📌 Overview

apeQuake provides a clean, extensible framework for working with earthquake acceleration time histories in engineering workflows.
It is built around a central Record object and a set of powerful composite modules:

Record.filter — preprocessing pipeline (detrend, taper, bandpass…)

Record.spectrum — amplitude spectral analysis

Record.spectrogram — high-quality time–frequency spectrograms

Record.response_spectra — Newmark-based Sd, Sv, Saₚₛ spectra

Record.IM — intensity measures (significant duration, Husid, AI, CAV…)

All operations use DataFrames, support multi-component (X/Y/Z) records, and follow a transparent, reproducible design with no hidden state.

✨ Key Features
Ground-motion Record Handling

1–3 components (X, Y, Z)

Automatic dt detection & interpolation for irregular time steps

Unified DataFrame structure: time, components, metadata

Composable Filtering System

Filters can be applied directly or chained as pipelines:

rec.filter.detrend(type="demean")
rec.filter.taper(max_percentage=0.05)
rec.filter.band_pass(Tc_low=0.1, Tc_high=5.0)


Or attach them to another composite:

rec.spectrogram.add_filter(rec.filter.detrend, type="demean")
rec.spectrogram.add_filter(rec.filter.band_pass, Tc_low=0.1, Tc_high=5.0)

Spectral Tools
📡 Amplitude Spectrum

Linear, semi-log, log-log representations

Combined or per-component plots

Frequency ↔ Period dual axes

🎨 Spectrograms

Smooth contour spectrograms

Period (right axis) & frequency (left axis)

Optional filtering pipelines

📈 Response Spectra (Sd, Sv, Saₚₛ)

Newmark Average Acceleration (γ=0.5, β=0.25)

Configurable damping

Sa definition mode: pseudo (ω²Sd) or absolute max(|u¨ + a_g|)

Single-component helpers: `newmark_sdof(...)`, `compute_response_spectrum(...)`

Per-component spectra

Combined and individual plotting modes

📊 Intensity Measures

Available under Record.IM:

Significant duration:

D5–75, D5–95, or custom (p1, p2)

Husid curve (normalized Arias intensity)

Extensible structure for:

Arias Intensity

CAV

PGA, PGV, PGD

Energy-based metrics

🛠 Installation
Local install (development mode)
git clone https://github.com/<your-user>/apeQuake.git
cd apeQuake
pip install -e .

PyPI (future)
pip install apeQuake

🚀 Quick Start
from apeQuake.core import Record
import numpy as np

# Example synthetic acceleration
dt = 0.01
t = np.arange(0, 20, dt)
x = 0.5 * np.sin(2*np.pi*1.5 * t)

rec = Record(x=x, dt=dt)

# --- Filtering pipeline for response spectra ---
rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)

T = np.linspace(0.05, 5.0, 100)

rec.response_spectra.compute(periods=T)
rec.response_spectra.plot(quantity="Sa", representation="loglog")

# absolute-acceleration Sa (max |u¨ + a_g|)
rec.response_spectra.compute(periods=T, sa_mode="absolute")

# single-component convenience API
rs = rec.response_spectra.compute_response_spectrum(periods=T, component="X", sa_mode="absolute")
rec.response_spectra.plot_response_spectrum(rs)

🎛 Composite Architecture

Each Record exposes a set of modular processors:

rec.filter            # low-level signal filtering
rec.spectrum          # amplitude spectra
rec.spectrogram       # time-frequency analysis
rec.response_spectra  # engineering RS (Sd/Sv/Sa)
rec.IM                # intensity measures (duration, AI, CAV…)


Each module:

can apply its own filter pipeline

can operate on:

the baseline record (rec.df)

or a user-provided custom DataFrame (df= argument)

never modifies the original time history unless explicitly requested

This ensures full transparency and reproducibility.

📂 Project Structure
apeQuake/
    core/
        record.py
        types.py
    filters/
        filters.py
    spectrum/
        spectrum.py
    spectrogram/
        spectrogram.py
    response_spectra/
        response_spectra.py
    intensity_measures/
        intensity_measures.py
    examples/
        basic_usage.ipynb

📘 Documentation

(coming soon)

API Reference

Example notebooks

Seismological vs Engineering conventions

Validation against OpenSees SDOF and PEER tools

🗺 Roadmap

✔ Significant duration

✔ Newmark response spectra

Arias Intensity

Cumulative Absolute Velocity (CAV)

Husid plotting utilities

RotD50 response spectra

Batch processing utilities

PEER/NGA flatfile importer

PyPI package + docs website

If you want, I can generate strike-through completion tracking in the README.

🤝 Contributing

Pull requests are welcome!
Please open an issue to discuss improvements or new features.

📜 License

MIT License © 2025 Nicolás Mora Bowen

📣 Citation

If you use apeQuake in research, please cite this repository:

Nicolas Mora Bowen (2025). apeQuake: A modular toolkit for seismic ground-motion processing and engineering response analysis. GitHub Repository.