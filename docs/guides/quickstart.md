# Quickstart

## Install

```bash
git clone https://github.com/nmorabowen/apeQuake.git
cd apeQuake
pip install -e .
```

apeQuake depends on `numpy`, `pandas`, `scipy`, `matplotlib`, `numba`, and
`obspy` (all installed automatically).

## A complete pass

```python
import numpy as np
from apeQuake import Record

# 1. Build a record from a single acceleration component
dt = 0.01
t = np.arange(0, 30, dt)
acc = np.loadtxt("my_motion.txt")        # ground acceleration, len == len(t)
rec = Record(x=acc, dt=dt, name="My Motion")

print(rec)   # <Record name='My Motion' components=[X] npts=3000 dt=0.01 (as-provided)>
```

### Look at the time history

```python
rec.plot_record.plot()
```

### Amplitude spectrum

```python
rec.spectrum.apply_base_filters(detrend=True, demean=True, taper=0.05)
rec.spectrum.compute()
rec.spectrum.plot(representation="loglog")
```

### Response spectrum (Sd / Sv / Sa)

```python
# Declare a preprocessing pipeline once …
rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)

# … then compute over a period grid
T = np.linspace(0.05, 5.0, 200)
rec.response_spectra.compute(periods=T)            # pipeline applied by default
rec.response_spectra.plot(quantity="Sa", representation="loglog")
```

### Significant duration

```python
d595 = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
print(f"D5-95 = {d595:.2f} s")

t_h, husid = rec.intensity_measures.husid("X")     # normalized Arias curve
```

## Multi-component records

Pass any of `x`, `y`, `z`. Every processor then operates per component:

```python
rec = Record(x=ax, y=ay, z=az, dt=dt, name="3C station")
rec.response_spectra.compute(periods=T)
rec.response_spectra.plot(quantity="Sa", combined=True)   # all comps, one axes
```

## Where next

- [The Record object](records.md) — construction, `dt` inference, interpolation.
- [Filtering](filtering.md) — the pipeline model in detail.
- [Response spectra](response-spectra.md) — pseudo vs. absolute Sa, Newmark.
