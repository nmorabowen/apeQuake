# Exporting code spectra

Any code spectrum (NEC-15, ASCE 7-10, 7-16, 7-22 two-period or multi-period)
can be written in the formats analysis programs read:

```python
from apeQuake.code_spectrum import to_etabs, to_sap2000, to_opensees, to_csv

to_etabs(model, "nec_design.txt", t_max=6.0, units="m/s2")
to_sap2000(model, "nec_design_sap.txt")
to_opensees(model, "rs.tcl", tag=1)                            # Tcl
to_opensees(model, "rs.py", style="python", units="m/s2")      # OpenSeesPy
to_csv(model, "spectrum.csv")

# or through the record composite, by label
rec.code_spectrum.export("NEC-15", "nec.txt", format="etabs", units="cm/s2")
```

| Argument | Meaning |
|---|---|
| `periods` | Periods to write. Default: `model.default_periods(t_max, n)`, a grid that contains every knee period (`T0`, `Ts`, `TL`, the multi-period nodes) |
| `t_max`, `n` | Range and density of the default grid (default 4 s, 200 points) |
| `units` | `"g"` (default), `"m/s2"` or `"cm/s2"`; the aliases of `apeQuake.core.units` (`"m/s^2"`, `"gal"`, ...) also work and are written under their canonical name |
| `tag`, `style` | OpenSees only: time-series tag, `"tcl"` or `"python"` |

Periods are written with 6 decimals, and periods that are equal at that
precision are written once.

## ETABS and SAP2000

The file has two columns, `period<TAB>value`. Import it with
*Define > Functions > Response Spectrum > From File*:

- *Period vs Value*;
- *Header Lines to Skip* = the number on the first line of the file
  (`Header lines to skip: 5`);
- set the function's acceleration units to the `units` you exported.

The header records the code, the site class and level, and every parameter
(`SDS`, `SD1`, `TL` …), so the file documents where its values came from.

## OpenSees

`to_opensees` writes a `Path` time series whose "time" axis is the period,
which is how a design spectrum is handed to an OpenSees response-spectrum
analysis:

```tcl
timeSeries Path 1 -time {0.000000 0.020101 ...} -values {0.480000 0.635065 ...}
```

The OpenSeesPy version writes plain float lists and runs on its own. The
suite executes it against a stand-in module, and it was also checked in real
OpenSees: the series returns the spectral ordinate at all 201 exported
periods.

!!! warning "Not a ground motion"
    The series is indexed by period, not by time. Never use it in a
    `UniformExcitation` pattern.
