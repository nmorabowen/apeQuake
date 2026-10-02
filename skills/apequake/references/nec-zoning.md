# NEC-SE-DS zoning: `apeQuake.nec.zone_at`

Merged in apeQuake#10 (2026-10-01). Module `src/apeQuake/nec/zoning.py`.

```python
from apeQuake.nec import zone_at, region_at, table19, ZONE_NAMES, OutsideEcuadorError

z = zone_at(-2.1347, -79.5872)       # Milagro
z.z, z.zone, z.province, z.region, z.eta      # 0.30, 'III', 'GUAYAS', 'costa', 1.80
z.boundary_km, z.z_across_boundary            # 3.4, 0.35   (inf / None if none within ~20 km)
z.nearest_listed                              # ListedTown(town, parish, canton, province, z, distance_km)
z.source                                      # 'figura-1' | 'galapagos'
z.warnings                                    # tuple[str, ...]
z.to_dict()                                   # JSON-ready (inf -> None)
```

- `ZONE_NAMES = {0.15: "I", 0.25: "II", 0.30: "III", 0.35: "IV", 0.40: "V", 0.50: "VI"}`.
- `region_at(lat, lon) -> (PROVINCE, region)`; regions `costa | sierra | oriente | esmeraldas |
  galapagos` (keys of `code_spectrum.codes.nec.ETA_BY_REGION`: 1.80 / 2.48 / 2.60 / 2.48 / 2.48).
  "ZONA NO DELIMITADA" -> `costa` with a warning.
- Outside Ecuador (or > 3 km off the map mask) -> `OutsideEcuadorError` (a `ValueError`).
- Galápagos: Z = 0.30 for the whole province (figure inset), no listed towns within 50 km.
- Warnings: boundary within 5 km; Tabla 19 town within 10 km with a different Z.

## Data and how it was made

- `nec/data/nec_zone_map.npz`: `zone` (uint8 class index, 255 = outside), `lon`, `lat` (pixel
  centres), `z_values`. ~1.5 km/px (73.6 px/deg), 476 × 434 crop.
- `nec/data/nec_table19.csv`: 545 rows (poblacion, parroquia, canton, provincia, z,
  parish_code, lat, lon); 544 joined to the IG-EPN parish layer (lat/lon = parish point).
- Rebuild: `python scripts/digitize_nec_zone_map.py --nec-dir "<folder with
  NEC-SE-DS-Peligro-Sísmico-parte-*.pdf>"` (local copy:
  `C:\Users\nmora\Dropbox\nmb\Clases\Diseño Sismoresistente NRC 2985\Libros\Codigos`).
  Needs PyMuPDF. Writes the npz/csv and `data-raw/nec/figura1.png` + `figura1_qc.png`.
- Method: Figura 1 is extracted from parte-1 (741 × 503 px), georeferenced from the graticule
  ticks just outside the frame (residual < 0.4 px, plate carrée), masked with the province
  polygons, classified by **hue** (the overlay is translucent over hillshade, RGB fails):
  red < 14° (0.50), orange < 37.5° (0.40), yellow < 56° (0.35), 56-99°: pale 0.30 if
  saturation < 0.53 else green 0.25, dark green < 150° (0.15). Water / grey city patches / text
  and a 2-px halo around them are unclassified (colour-mixed edges read as yellow, which once
  put Guayaquil at 0.35), filled from the nearest classified pixel, then a 7 × 7 majority filter ×3.
- Tabla 19 parse: column edges from each page's header row (parte-2 p.48 is offset), the PDF
  font maps Ñ -> Ð (also in the IG-EPN admin layer: normalise both), join on (province, canton,
  parish | town).

## Validation (regression test `test_map_agrees_with_table19` ≥ 0.90)

- Map Z = Tabla 19 Z at the parish point: 91 %; Tabla 19 Z present inside the parish: 98 %.
- Remaining: near-boundary points, and NEC internal contradictions (Quevedo, Vinces, Milagro
  listed 0.35 inside the 0.30 ellipse; Simón Bolívar, Olmedo, Pedro Carbo, Puerto Quito, La
  Concordia, Samborondón, Pungalá, Pilahuín... listed twice with different Z).
- NEC §3.1.1: Tabla 19 for listed towns; if the map is hard to read, the nearest listed town.
  The user chose map-only lookup; the result carries the table check so the engineer decides.
