# The Record object

`Record` is the single entry point. It stores 1–3 acceleration components on a
**uniform time grid** as a canonical pandas DataFrame (`time | X | Y | Z`, only
the present components included) and attaches every processor as a composite.

## Construction

`Record` is keyword-only. Provide at least one component:

```python
from apeQuake import Record

# From dt (time generated as arange(npts) * dt)
rec = Record(x=acc, dt=0.01)

# From an explicit time array
rec = Record(x=acc, time_array=t)

# Multi-component, with a name
rec = Record(x=ax, y=ay, z=az, dt=0.005, name="Station 12")
```

| Argument | Meaning |
|----------|---------|
| `x`, `y`, `z` | Component arrays (≥1 required, equal length). |
| `time_array` | Optional time vector (same length as components). |
| `dt` | Desired time step. Required if `time_array` is `None`. |
| `name` | Optional label, used in `repr` and plot titles. |

## dt detection & interpolation

The constructor resolves the effective `dt` and decides whether to resample:

- **`time_array=None`** → `dt` is required; time is `arange(npts) * dt`.
- **`time_array` given, `dt=None`** → if the grid is uniform (within a small
  tolerance), that spacing is used as-is; if it is non-uniform, the record is
  linearly interpolated onto `dt = min(Δt)`.
- **`time_array` given, `dt` given** → if the grid already matches `dt` it is
  used as-is; otherwise the record is interpolated onto the requested `dt`.

Whether interpolation happened is recorded:

```python
rec._interpolation_flag   # True if the components were resampled
rec.init_logs             # human-readable log of the construction decisions
```

!!! note "Uniform grids from `np.arange` / `np.linspace`"
    Floating-point round-off in an otherwise-uniform grid is tolerated — such a
    grid is treated as uniform and **not** resampled.

## Accessing the data

```python
rec.time          # 1D time vector
rec.data          # (npts, ncomp) array of present components
rec.x, rec.y, rec.z   # individual components (AttributeError if absent)
rec.components    # e.g. ('X', 'Y')
rec.dt            # effective time step
rec.df            # the canonical DataFrame (do not mutate)
```

## The composites

Each `Record` exposes six processors. They read from `rec.df` (or a DataFrame
you pass explicitly) and never mutate the baseline:

```python
rec.filter             # signal filtering          → guides/filtering.md
rec.spectrum           # FFT amplitude spectra      → guides/spectra.md
rec.spectrogram        # time–frequency analysis    → guides/spectra.md
rec.response_spectra   # Newmark Sd / Sv / Sa       → guides/response-spectra.md
rec.intensity_measures # duration, Husid            → guides/intensity-measures.md
rec.plot_record        # time-history plotting
```

See the [`Record` API](../api/record.md) for the full signature.
