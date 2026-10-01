# IG-EPN seismic-hazard database (snapshot)

Data in this folder is a normalized snapshot of the public products of the
**Mapa Digital Interactivo del Peligro Sísmico Probabilístico para el Ecuador**,
published by the **Instituto Geofísico – Escuela Politécnica Nacional (IG-EPN)**:

- Portal: <https://www.igepn.edu.ec/mapas/peligro-sismico/mapa-peligro-sismico.html>
- StoryMap: <https://storymaps.arcgis.com/stories/f885bc190d6442fa9fef91f8145d063e>
- Feature services: <https://services8.arcgis.com/WFUbp7xZ6MJpsSuI/arcgis/rest/services/>

The data belongs to IG-EPN. apeQuake redistributes it unchanged in value
(only reformatted) for convenience and offline use, with attribution. The MIT license of
apeQuake covers the code, **not** this data. Retrieval date and record counts are in
`manifest.json`. Regenerate with `python scripts/igepn/fetch_igepn.py`.

IG-EPN states that these results are **reference values on rock** and do not replace
site-specific studies or the seismic design code (NEC-SE-DS).

## How to cite

Cite the hazard model and IG-EPN as the data provider:

> Beauval C., Marinière J., Yepes H., Audin L., Nocquet J.M., Alvarado A., Baize S.,
> Aguilar J., Singaucho J.C., Jomard H. (2018). A New Seismic Hazard Model for Ecuador.
> *Bulletin of the Seismological Society of America*, 108(3A), 1443–1464.
> doi:10.1785/0120170259

> Yepes H., Audin L., Alvarado A., Beauval C., Aguilar J., Font Y., Cotton F. (2016).
> A new view for the geodynamics of Ecuador: Implication in seismogenic source definition
> and seismic hazard assessment. *Tectonics*, 35, 1249–1279. doi:10.1002/2015TC003941

> Instituto Geofísico – Escuela Politécnica Nacional (IG-EPN). Mapa Digital Interactivo del
> Peligro Sísmico Probabilístico para el Ecuador. Retrieved <date in manifest.json> from
> https://www.igepn.edu.ec/mapas/peligro-sismico/mapa-peligro-sismico.html

For the historical catalog also cite Beauval et al. (2010), *Geophys. J. Int.* 181(3),
1613–1633, doi:10.1111/j.1365-246X.2010.04569.x.

## Contents

All accelerations in g, rock (Vs30 = 760 m/s), geometric-mean horizontal component.
Coordinates WGS84 (EPSG:4326).

| File | Rows | Content |
|------|-----:|---------|
| `hazard_cells.csv` | 3146 | Grid cells (0.08° ≈ 9 km): `cell_id`, centroid `lat`, `lon`, `area_km2` |
| `hazard_cells.geojson.gz` | 3146 | Cell polygons (clipped to the national border) |
| `hazard_grid.csv.gz` | 25168 | UHS ordinates per `cell_id` × `tr` (475, 2475) × `stat` (mean, q16, q50, q84); columns `T0.00` (PGA) … `T2.00` |
| `capitals.csv` | 366 | Cantonal / provincial capitals: names, canton, province, cell, PGA 475 / 2475 |
| `hazard_curves_capitals.csv.gz` | 116800 | **Digitized** mean hazard curves (`cell_id`, `period`, `sa_g`, `rate` in 1/yr), 40 points per curve |
| `hazard_curves_capitals_qc.csv` | 2920 | Quality control per digitized curve (see below) |
| `cabeceras_population.csv` | 223 | Cantonal capitals: census 2010 population, households, dwellings, 2019 projection, 2022 census, PGA |
| `sources_crust_interface.geojson` | 13 | Area sources (crustal + interface): a, b, λ(Mw≥4.5), Mmax, depth range, dip / rake / strike |
| `sources_beauval2018.geojson` | 22 | Full source set of Beauval et al. (2018), including in-slab sources |
| `faults.geojson` | 8 | Fault model: geodetic / geologic slip rate, Mmax, dip, depth limits, focal mechanism, length |
| `catalog_shallow.csv.gz` | 2502 | Homogenized catalog used in the PSHA, shallow events |
| `catalog_deep.csv.gz` | 2039 | Homogenized catalog, intermediate-depth events |
| `catalog_historical.csv.gz` | 1967 | Historical and instrumental seismicity 1587–2021 |
| `admin_{provinces,cantons,parishes}.csv` | 25 / 224 / 1040 | Administrative units (INEC DPA) with names, codes and polygon centroids |
| `admin_*.geojson.gz` | | Simplified polygons (~200 m tolerance) of the administrative units |

`sources_inslab` is published empty by IG-EPN; the in-slab sources are in
`sources_beauval2018.geojson`. The rolling "last 180 days" event list is not snapshotted;
query it live.

## Digitized hazard curves

IG-EPN publishes hazard curves only as images, for the 366 cantonal-capital cells (365
unique cells). They were digitized with `scripts/igepn/digitize_hazard_curves.py`
(alpha-unmixing color classification, Viterbi line tracing, axis calibration on the
published UHS). Every curve is checked against the published mean UHS:
λ(Sa_TR) must equal 1/TR at TR = 475 and 2475 yr.

- `dlog10_rate_475`, `dlog10_rate_2475`: error at those check points (log10 decades;
  0.043 = 10 % in rate). NaN means the check point falls outside the recovered span
  (for example, hidden behind the legend box).
- `sa_min`, `sa_max`: recovered span. Parts hidden by the legend (upper right) or fully
  covered by another curve without confirmation are not invented.
- `coincident_with`: the curve was drawn exactly on top of this period's curve over part
  of its span. That stretch was taken from the other curve, only where this curve's own
  published UHS values confirm it.

Overall: median error 0.010 decades, p95 0.018 decades, max 0.065 decades.
These are digitized values (pixel resolution ≈ 0.004 g in Sa, ≈ 3 % in rate), not the
original numerical output of the hazard model.
