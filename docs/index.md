---
hide:
  - navigation
---

<div class="ape-hero" markdown>
<img class="ape-hero__mark" src="assets/logo.svg" alt="apeQuake mark" />
<div>
  <div class="ape-hero__word">apeQuake</div>
  <div class="ape-hero__sub">LADRUÑO</div>
</div>
</div>

A modular Python toolkit for **seismic ground-motion processing** — filtering,
amplitude spectra, time–frequency spectrograms, Newmark response spectra, and
engineering intensity measures. Everything hangs off a single, transparent
[`Record`](guides/records.md) object built on pandas DataFrames, with no hidden
state and full multi-component (X/Y/Z) support.

```python
import numpy as np
from apeQuake import Record

dt = 0.01
t = np.arange(0, 30, dt)
acc = np.loadtxt("my_motion.txt")           # ground acceleration

rec = Record(x=acc, dt=dt, name="My Motion")

# Filter pipeline → response spectra
rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)

T = np.linspace(0.05, 5.0, 200)
rec.response_spectra.compute(periods=T)
rec.response_spectra.plot(quantity="Sa", representation="loglog")
```

## Where do you want to start?

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } &nbsp; __[Quickstart](guides/quickstart.md)__

    ---

    *Show me the shortest path.*

    Build a `Record`, run a response spectrum, read off significant
    duration — end to end in a dozen lines.

-   :material-pulse:{ .lg .middle } &nbsp; __[The Record object](guides/records.md)__

    ---

    *What's the central abstraction?*

    Construction from arrays, automatic `dt` detection and interpolation,
    the canonical DataFrame, and the composite processors.

-   :material-filter-variant:{ .lg .middle } &nbsp; __[Filtering](guides/filtering.md)__

    ---

    *Detrend, taper, band-pass.*

    The ObsPy-backed filter toolbox and the composable, never-mutate
    pipeline model shared across every processor.

-   :material-sine-wave:{ .lg .middle } &nbsp; __[Spectra & spectrograms](guides/spectra.md)__

    ---

    *Frequency content over time.*

    Single-sided amplitude spectra, dual frequency↔period axes, and
    smooth contour spectrograms.

-   :material-chart-bell-curve:{ .lg .middle } &nbsp; __[Response spectra](guides/response-spectra.md)__

    ---

    *Sd, Sv, Sa via Newmark.*

    Average-acceleration SDOF integration (Numba-accelerated),
    pseudo vs. absolute Sa, per-component plots.

-   :material-book-open-variant:{ .lg .middle } &nbsp; __[API reference](api/index.md)__

    ---

    *Look up a method.*

    The complete surface: `Record`, `Filter`, `Spectrum`,
    `Spectrogram`, `ResponseSpectra`, `IntensityMeasures`.

</div>

## Design principles

- **One source of truth.** Every processor reads from `Record.df` and returns
  new DataFrames — the baseline time history is never mutated unless you ask.
- **Composable filters.** A detrend → taper → band-pass pipeline is declared
  once and reused across spectra, spectrograms, and response spectra.
- **Reproducible.** No global state, explicit units, transparent conventions
  (pseudo vs. absolute Sa, Newmark γ/β, seismological vs. engineering sign).

---

## Credits

**Developed by:** Nicolás Mora Bowen

Part of José Abell's *El Ladruño Research Group*.

Released under the MIT License.
