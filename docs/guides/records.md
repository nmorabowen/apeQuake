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
| `units` | Optional acceleration unit label: `"g"`, `"m/s2"` or `"cm/s2"` (aliases `"m/s^2"`, `"cm/s^2"`, `"gal"`). Never rescales the data; read by unit-dependent measures such as Arias intensity. |

## Reading record files

`Record.from_files` reads one file per component and checks that they share
the time step:

```python
# PEER NGA (.AT2, units of g)
rec = Record.from_files(x="RSN1_HELENA.A_A-HMC180.AT2",
                        y="RSN1_HELENA.A_A-HMC270.AT2",
                        z="RSN1_HELENA.A_A-HMCDWN.AT2")

# RENAC, IG-EPN Ecuador (.txt, cm/s^2): map E/N/Z to X/Y/Z yourself
rec = Record.from_files(x="ACH1_201604162359_E_100.txt",
                        y="ACH1_201604162359_N_100.txt",
                        z="ACH1_201604162359_Z_100.txt")
rec.units          # 'cm/s2', from the file header
rec.source_files   # {'X': ..., 'Y': ..., 'Z': ...}
```

| Format | Detected by | Time step | Units |
|--------|-------------|-----------|-------|
| `"at2"` | `.AT2` extension, `PEER` or `NPTS=` header | `DT=` on header line 4 (new or old layout) | `IN UNITS OF G` line |
| `"renac"` | `RENAC` or `Frecuencia de muestreo` header | `1 / Frecuencia de muestreo (Hz)` | `Unidades:` line |
| `"txt"` | anything else | a header line with `DT` (`DT = 0.005`), or `dt=` | a header line with `Units` / `Unidades`, if any |

- `units=` converts every file to one unit (`Record.from_files(..., units="cm/s2")`);
  without it, files that declare a unit must agree, and mixing files with and
  without a declared unit raises.
- Components of different length are padded with trailing zeros to the
  longest; padding and conversions are listed in `rec.init_logs`.
- A non-numeric line inside the data block raises `ValueError` with its line
  number: gaps are never filled silently.
- The readers are also available on their own, see [Readers](../api/readers.md):

```python
from apeQuake.readers import read_record_file

f = read_record_file("RSN1_HELENA.A_A-HMC180.AT2")
f.data, f.dt, f.units, f.metadata
```

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
rec.response_spectra   # Newmark Sd / Sv / Sa, RotD → guides/response-spectra.md
rec.intensity_measures # peaks, Arias, duration     → guides/intensity-measures.md
rec.plot_record        # time-history plotting
```

See the [`Record` API](../api/record.md) for the full signature.
