# Seismic hazard of Ecuador (IG-EPN)

`apeQuake.hazard` bundles the public products of the IG-EPN probabilistic seismic hazard
model for Ecuador (Beauval et al., 2018) as an offline database: uniform hazard spectra
for every 9 km cell of the country, hazard curves for the cantonal capitals, the seismic
source model and the earthquake catalogs.

!!! warning "Reference values on rock"
    All values are for a generic **rock site (Vs30 = 760 m/s)**. IG-EPN publishes them as
    reference values: they do not replace a site-specific study or the design spectrum
    of NEC-SE-DS, which adds site amplification (Fa, Fd, Fs) and its own zoning.

!!! note "IG-EPN model vs. NEC hazard curves"
    [`apeQuake.nec`](nec-hazard.md) holds the hazard curves printed in NEC-SE-DS (NEC-15)
    for 23 provincial capitals, which belong to the code. `apeQuake.hazard` holds the newer
    IG-EPN model (Beauval et al., 2018) on a 3146-cell grid, with logic-tree percentiles.
    The two models differ. Use the NEC curves for code work, and the IG-EPN model for
    site-specific studies and comparisons.

## What is in the database

```python
from apeQuake.hazard import EcuadorHazard

hz = EcuadorHazard()
hz.inventory()
```

| Dataset | Rows | Accessor |
|---------|-----:|----------|
| Hazard grid (UHS) | 3146 cells × TR 475 / 2475 × mean, q16, q50, q84 × 8 periods | `site()`, `uhs_at()`, `hazard_map()` |
| Hazard curves (digitized) | 365 capital cells × 8 periods | `site(...).hazard_curve()` |
| Cantonal capitals | 366 | `capitals()` |
| Capital population | 223 | `population()` |
| Seismic sources / faults | 22 / 8 | `sources()`, `faults()` |
| Catalogs | 2502 shallow, 2039 deep, 1967 historical | `catalog(kind)` |
| Provinces / cantons / parishes | 25 / 224 / 1040 | `admin(level)` |
| Last 180 days of events | live | `fetch_recent_events()` |

Periods: PGA (0.0), 0.05, 0.07, 0.1, 0.2, 0.5, 1.0 and 2.0 s.

## Hazard at a site

By coordinates (lat, lon in degrees) or by place name; names are matched against
capitals, parishes, cantons and provinces, ignoring accents and case:

```python
quito = hz.site("Quito")
gye   = hz.site(-2.19, -79.89)
hz.find("Baños")          # list the candidates when a name is ambiguous
```

The point is assigned to the IG-EPN cell whose polygon contains it (the cells are not a
pure lattice; some are merged with a neighbour). Use `interp="bilinear"` to interpolate
between the four surrounding cell centroids instead.

## Uniform hazard spectra at any return period

```python
quito.uhs(475)                      # published values
quito.uhs(975)                      # from the hazard curve
quito.uhs(2475, stat="q84")         # logic-tree percentile
quito.uhs_table([72, 225, 475, 975, 2475])
quito.plot_uhs([475, 975, 2475])    # q16-q84 band for the published TRs
```

Probabilities of exceedance convert with `tr_from_poe`:

```python
from apeQuake.hazard import tr_from_poe
tr_from_poe(0.10, 50)    # 474.6 yr
tr_from_poe(0.05, 50)    # 974.8 yr
```

Every output has a `source` column saying where each number comes from:

| `source` | Meaning |
|----------|---------|
| `published` | IG-EPN value at TR = 475 or 2475 yr |
| `digitized` | read from the IG-EPN hazard curve of a capital cell |
| `fit` | power law λ = k₀·Sa⁻ᵏ through the two published points (between 475 and 2475 yr) |
| `fit-extrapolated` | the same power law outside 475–2475 yr; a warning is issued |

## Hazard curves

```python
quito.hazard_curve(0.2)            # sa_g, rate [1/yr], tr [yr], source
quito.return_period(0.0, 0.5)      # TR of exceeding PGA = 0.5 g
quito.plot_hazard_curves()
```

IG-EPN publishes hazard curves only as images, for the 365 cells that contain a cantonal
capital. They were digitized and checked against the published UHS: the curve must give
λ = 1/475 and 1/2475 at the published accelerations. The median error is 0.010 decades
(about 2.5 % in rate), the maximum 0.065. The curves of other cells are the power-law
fit, which is exact at both published points but straight in log-log space. Real hazard
curves bend down, so the fit overestimates the rate away from the two points.

## Maps

`hazard_map()` returns the values behind every map, one row per cell. For return periods
other than 475 / 2475 yr it uses the per-cell power-law fit:

```python
hz.hazard_map(975, period=0.2)               # cell_id, lat, lon, sa
```

### Static maps with overlays

`plot_map()` draws the hazard cells with matplotlib, with optional overlays. The result
is ready for reports and notebooks:

```python
hz.plot_map(
    475, period=0.0,
    faults=True,                             # fault model
    sources=["crustal", "interface"],        # also "inslab", "background", or True
    catalog="historical", min_mw=6,          # or "shallow", "deep", a list, a DataFrame
    capitals=True,
    points=["Quito", (-0.95, -80.73, "Site A")],
)
hz.plot_map(975, period=0.2, points=[(-0.25, -78.45, "Proyecto")], extent="points")
```

`points` accepts place names, `(lat, lon)` or `(lat, lon, label)` tuples, `HazardSite`
objects, or a DataFrame with `lat`, `lon` and an optional `label`. Each point is labelled
with its value from `HazardSite.uhs`, so a capital cell uses its digitized curve.
`extent="points"` zooms the map to them. `catalog` also takes the live events:
`catalog=fetch_recent_events()`.

### Interactive map

`explore()` writes a standalone HTML page that runs in any browser (Leaflet), with the
data embedded:

```python
hz.explore("hazard.html", points=["Quito", (-0.25, -78.45, "Proyecto")],
           recent=True, open_browser=True)
```

- **Base maps:** light gray, OpenStreetMap, terrain and satellite.
- **Layers you can toggle:** the hazard grid, faults, source zones by type, each catalog
  (epicenters sized by Mw; `recent=True` adds the IG-EPN last 180 days), the cantonal
  capitals and your points.
- **Controls:** return period (presets or any value), period and statistic re-color the
  map in the browser. Values at TR other than 475 / 2475 come from the power-law fit,
  the same as `hazard_map()`.
- **Click a cell** for its mean, q16 and q84 values and a UHS chart. **Click a point**
  for its UHS at `point_trs`, computed by apeQuake.
- **"Go to lat, lon"** finds the cell that contains any coordinate, with the same
  point-in-polygon rule as `site()`.

- **Hover a cell** for its value at a glance. **Minimum Mw** thins the earthquake
  layers, and **Download map values (CSV)** exports exactly what the map shows (the
  table view of the map).
- **Theme:** Auto follows the system setting; Light and Dark are remembered. The base
  map follows the theme.

The page is self-contained: the map library (Leaflet, BSD-2-Clause) and all the data are
embedded, so it also opens where external scripts are blocked, such as sandboxed file
previews, mail or chat attachments, or offline. Only the base-map tiles come from the
internet; without them the hazard and overlays still draw on a blank background. The
file is about 2.9 MB. `inline_leaflet=False` loads Leaflet from a CDN (with integrity
checks) instead, for a file about 150 kB smaller.

### How the colors are chosen

Every plot and map uses one set of design tokens, checked with a palette validator
rather than chosen by eye:

| What | Encoding | Why |
|------|----------|-----|
| Hazard (map cells) | one blue hue, light = low, dark = high | magnitude is sequential, and a one-hue ramp leaves room for overlays |
| Return periods, spectral periods (lines) | steps of the same blue, short → long | they are ordered, not categories |
| Source-zone types | orange, aqua, violet (background zones gray) | the only identity colors on a map; a map can carry at most three |
| Faults, earthquakes, sites | ink with a thin surface-colored ring | readable on any hazard color |
| Earthquake catalogs | fill and outline: shallow hollow, deep tinted, historical dashed, live solid | identity without spending a fourth color |

A one-hue ramp separates about five ordered lines. That is why `plot_hazard_curves()`
shows five periods by default (PGA, 0.2, 0.5, 1 and 2 s). Ask for more and the curves
are also labelled at their ends. Every plot takes `theme="light"` or `"dark"`; in dark
mode the hazard ramp flips, so low hazard recedes into the background.

## Sources and catalogs

```python
hz.sources()                  # a, b, rate of Mw >= 4.5, Mmax, geometry (GeoJSON)
hz.faults()                   # slip rates, Mmax, dip
hz.catalog("historical")      # 1587-2021
hz.catalog("shallow")         # homogenized catalog used in the PSHA

from apeQuake.hazard import fetch_recent_events
fetch_recent_events()         # IG-EPN, last 180 days (needs internet)
```

## Citing

The data belongs to IG-EPN. Cite the model and the provider:

```python
print(EcuadorHazard.citation())
```

> Beauval C. et al. (2018). A New Seismic Hazard Model for Ecuador.
> *Bull. Seismol. Soc. Am.* 108(3A), 1443–1464. doi:10.1785/0120170259

> Yepes H. et al. (2016). A new view for the geodynamics of Ecuador: Implication in
> seismogenic source definition and seismic hazard assessment. *Tectonics* 35,
> 1249–1279. doi:10.1002/2015TC003941

Provenance (URLs, retrieval date, record counts) is in `hz.manifest()`. The snapshot is
regenerated with `scripts/igepn/fetch_igepn.py`, and the curves with
`scripts/igepn/digitize_hazard_curves.py`.
