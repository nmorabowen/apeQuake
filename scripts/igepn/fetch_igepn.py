"""
Snapshot the IG-EPN public seismic-hazard products into apeQuake's own database.

Source: "Mapa Digital Interactivo del Peligro Sísmico Probabilístico para el Ecuador",
Instituto Geofísico - Escuela Politécnica Nacional (IG-EPN),
https://www.igepn.edu.ec/mapas/peligro-sismico/mapa-peligro-sismico.html
The StoryMap is backed by public ArcGIS FeatureServer layers; this script queries them
directly (JSON / GeoJSON) and writes normalized tables to
``src/apeQuake/hazard/data/igepn``. Raw downloads (hazard-curve JPGs, UHS .txt used
for cross-checks) go to ``data-raw/igepn`` (git-ignored).

Usage:
    python scripts/igepn/fetch_igepn.py            # tables + capital hazard-curve images
    python scripts/igepn/fetch_igepn.py --no-images

Stdlib only, so the snapshot can be regenerated without extra dependencies.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "apeQuake" / "hazard" / "data" / "igepn"
RAW = ROOT / "data-raw" / "igepn"

BASE = "https://services8.arcgis.com/WFUbp7xZ6MJpsSuI/arcgis/rest/services/"
STORYMAP = "https://www.igepn.edu.ec/mapas/peligro-sismico/mapa-peligro-sismico.html"
STORYMAP_ITEM = "https://storymaps.arcgis.com/stories/f885bc190d6442fa9fef91f8145d063e"

SVC_475 = "Peligro_sísmico_Tr_475_años_WFL1"
SVC_2475 = "Peligro_Sísmico_Tr_2475_años_l"
SVC_CAP = "cantones_y_PGA_"
SVC_SRC = "Fuentes_sismogenéticas_ec_WFL1"
SVC_HIST = "Sismicidad_histórica_1587___WFL1"
SVC_DPA = "SHP_DPA"

PERIODS = (0.0, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0)
# Field tokens are inconsistent across layers (SA_1 vs SA_10 both mean T = 1.0 s).
_PERIOD_TOKEN = {"005": 0.05, "007": 0.07, "01": 0.1, "02": 0.2, "05": 0.5,
                 "1": 1.0, "10": 1.0, "2": 2.0, "20": 2.0}
# Suffix encodes the exceedance probability in 50 yr: 10 % -> TR 475, 2 % -> TR 2475.
_TR_SUFFIX = {"01": 475, "012": 475, "002": 2475}
_FIELD_RE = re.compile(r"^(?:PGA|SA_(\d+))_(\d+)$")

# (service, layer) -> statistic; quantile layers hold BOTH return periods.
HAZARD_LAYERS = [
    (SVC_475, 0, "mean"),
    (SVC_2475, 0, "mean"),
    (SVC_475, 1, "q16"),
    (SVC_475, 2, "q50"),
    (SVC_475, 3, "q84"),
]


# ----------------------------------------------------------------------------- HTTP

def _get(url: str, params: dict | None = None, *, binary: bool = False, retries: int = 4):
    url = urllib.parse.quote(url, safe=":/")  # service names contain í / ñ
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            return data if binary else data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - network flakiness, retry with backoff
            if k == retries - 1:
                raise
            time.sleep(2.0 * (k + 1))


def _json(url: str, params: dict | None = None) -> dict:
    d = json.loads(_get(url, params))
    if "error" in d:
        raise RuntimeError(f"{url}: {d['error']}")
    return d


def layer_url(svc: str, layer: int) -> str:
    return f"{BASE}{svc}/FeatureServer/{layer}"


def query_all(svc: str, layer: int, *, geometry: bool = False, fmt: str = "json",
              offset_deg: float | None = None) -> list[dict]:
    """Page through a layer (maxRecordCount = 2000) and return all features."""
    oid = _json(layer_url(svc, layer), {"f": "json"}).get("objectIdField", "OBJECTID")
    feats: list[dict] = []
    start = 0
    while True:
        params = {
            "where": "1=1", "outFields": "*", "returnGeometry": str(geometry).lower(),
            "orderByFields": oid, "outSR": 4326,
            "resultOffset": start, "resultRecordCount": 2000, "f": fmt,
        }
        if offset_deg:
            params["maxAllowableOffset"] = offset_deg
            params["geometryPrecision"] = 5
        params = {k: v for k, v in params.items() if v != ""}
        d = _json(layer_url(svc, layer) + "/query", params)
        batch = d["features"]
        feats.extend(batch)
        exceeded = d.get("exceededTransferLimit") or d.get("properties", {}).get(
            "exceededTransferLimit")
        if not batch or not exceeded:
            break
        start += len(batch)
    return feats


def attrs_df(svc: str, layer: int) -> pd.DataFrame:
    return pd.DataFrame([f["attributes"] for f in query_all(svc, layer)])


# ----------------------------------------------------------------------------- hazard grid

def hazard_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: dict[tuple[str, int, str], dict[float, float]] = {}
    cells = None
    for svc, layer, stat in HAZARD_LAYERS:
        df = attrs_df(svc, layer)
        if cells is None or svc == SVC_475 and layer == 0:
            cells = df[["ID", "latitud", "longitud"]].copy()
        for col in df.columns:
            m = _FIELD_RE.match(col)
            if not m:
                continue
            tok, suf = m.groups()
            T = 0.0 if tok is None else _PERIOD_TOKEN[tok]
            tr = _TR_SUFFIX[suf]
            for cid, v in zip(df["ID"], df[col]):
                rows.setdefault((cid, tr, stat), {})[T] = float(v)
    recs = []
    for (cid, tr, stat), d in rows.items():
        if len(d) != len(PERIODS):
            raise RuntimeError(f"incomplete periods for {cid} TR={tr} {stat}: {sorted(d)}")
        recs.append({"cell_id": cid, "tr": tr, "stat": stat,
                     **{f"T{T:.2f}": d[T] for T in PERIODS}})
    grid = (pd.DataFrame(recs)
            .sort_values(["cell_id", "tr", "stat"]).reset_index(drop=True))
    cells = (cells.rename(columns={"ID": "cell_id", "latitud": "lat", "longitud": "lon"})
             .drop_duplicates("cell_id").sort_values(["lat", "lon"]).reset_index(drop=True))
    return grid, cells


def parse_uhs_txt(text: str) -> pd.DataFrame:
    """Parse an IG-EPN UHS .txt (tolerates the stray ',,' in the 2475-yr files)."""
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"-?\d+\.\d+", line)
        if len(nums) == 5 and line.strip()[0].isdigit():
            rows.append([float(x) for x in nums])
    return pd.DataFrame(rows, columns=["T", "mean", "q16", "q50", "q84"])


def crosscheck_txt(grid: pd.DataFrame, n: int = 12) -> list[dict]:
    """Compare the attribute tables against the per-cell UHS .txt attachments."""
    report = []
    for svc, tr in ((SVC_475, 475), (SVC_2475, 2475)):
        oids = np.linspace(1, 3146, n).astype(int)
        for oid in oids:
            info = _json(layer_url(svc, 0) + f"/{oid}/attachments", {"f": "json"})
            txt = [a for a in info["attachmentInfos"] if a["contentType"] == "text/plain"][0]
            cid = txt["name"].split("_")[0]
            uhs = parse_uhs_txt(_get(layer_url(svc, 0) + f"/{oid}/attachments/{txt['id']}"))
            sub = grid[(grid.cell_id == cid) & (grid.tr == tr)].set_index("stat")
            err = 0.0
            for stat in ("mean", "q16", "q50", "q84"):
                ours = sub.loc[stat, [f"T{T:.2f}" for T in PERIODS]].to_numpy(float)
                err = max(err, float(np.max(np.abs(ours - uhs[stat].to_numpy()))))
            report.append({"cell_id": cid, "tr": tr, "max_abs_err_g": err})
    return report


# ----------------------------------------------------------------------------- geometry

def _ring_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    xy = np.asarray(ring, float)
    x, y = xy[:, 0], xy[:, 1]
    cr = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cr.sum() / 2.0
    if abs(a) < 1e-15:
        return float(x.mean()), float(y.mean()), 0.0
    cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * a)
    cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * a)
    return float(cx), float(cy), float(a)


def polygon_centroid(geom: dict) -> tuple[float, float]:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    sx = sy = sa = 0.0
    for poly in polys:
        for k, ring in enumerate(poly):
            cx, cy, a = _ring_centroid(ring)
            sx += cx * a
            sy += cy * a
            sa += a
    return (sy / sa, sx / sa) if sa else (np.nan, np.nan)


def geojson_layer(svc: str, layer: int, offset_deg: float | None = None) -> dict:
    feats = query_all(svc, layer, geometry=True, fmt="geojson", offset_deg=offset_deg)
    return {"type": "FeatureCollection", "features": feats}


def write_geojson(fc: dict, path: Path) -> None:
    for f in fc["features"]:
        f.pop("id", None)
        for k in ("Shape__Area", "Shape__Length", "Shape_Leng", "Shape_Area",
                  "Shape_STAr", "Shape_STLe"):
            f["properties"].pop(k, None)
    raw = json.dumps(fc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=9) as fh:
            fh.write(raw)
    else:
        path.write_bytes(raw)


# ----------------------------------------------------------------------------- tables

def esri_date(ms) -> str | None:
    if ms is None or (isinstance(ms, float) and np.isnan(ms)):
        return None
    t = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(milliseconds=int(ms))
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def catalog(svc: str, layer: int) -> pd.DataFrame:
    df = attrs_df(svc, layer).drop(columns=["OBJECTID", "FID"], errors="ignore")
    if "Fecha" in df:
        df["Fecha"] = df["Fecha"].map(esri_date)
    ren = {"Fecha": "time", "Latitud": "lat", "Longitud": "lon", "Profundidad": "depth_km",
           "Mw": "mw", "Fuente": "source", "Catalogo": "catalog", "ID": "event_id", "M0": "m0"}
    return df.rename(columns=ren).sort_values("time", na_position="first").reset_index(drop=True)


def capitals() -> pd.DataFrame:
    df = attrs_df(SVC_CAP, 0)
    return (df.rename(columns={"ID": "cell_id", "NOMBRE": "name", "Canton": "canton",
                               "Provincia": "province", "Tipo": "type", "lat_c": "lat",
                               "lon_c": "lon", "PGA_475a": "pga_475", "PGA_2475a": "pga_2475"})
            [["OBJECTID", "cell_id", "name", "canton", "province", "type", "lat", "lon",
              "pga_475", "pga_2475"]]
            .rename(columns={"OBJECTID": "objectid"}))


def cabeceras_population() -> pd.DataFrame:
    fc = geojson_layer(SVC_CAP, 1, offset_deg=0.0005)
    recs = []
    for f in fc["features"]:
        p = f["properties"]
        lat, lon = polygon_centroid(f["geometry"])
        recs.append({
            "cell_id": p.get("ID"), "name": p.get("NOMBRE"), "type": p.get("Tipo"),
            "lat": lat, "lon": lon,
            "pop_2010": p.get("Pob_total"), "pop_2010_male": p.get("Pob_masculina"),
            "pop_2010_female": p.get("Pob_femenina"), "households_2010": p.get("hogares"),
            "dwellings_2010": p.get("viviendas"), "census_sectors": p.get("N_sectores"),
            "pop_2019_proj": p.get("Proy_Pob2019"), "pop_2022": p.get("pob_2022"),
            "pga_475": p.get("PGA0_01"), "pga_2475": p.get("PGA0_002"),
            "shaking_label": p.get("Aceleracion"),
        })
    return pd.DataFrame(recs).sort_values("name").reset_index(drop=True)


def admin_units() -> dict[str, pd.DataFrame]:
    spec = {
        "provinces": (2, {"DPA_PROVIN": "province_code", "DPA_DESPRO": "province"}),
        "cantons": (0, {"DPA_CANTON": "canton_code", "DPA_DESCAN": "canton",
                        "DPA_PROVIN": "province_code", "DPA_DESPRO": "province"}),
        "parishes": (1, {"DPA_PARROQ": "parish_code", "DPA_DESPAR": "parish",
                         "DPA_CANTON": "canton_code", "DPA_DESCAN": "canton",
                         "DPA_PROVIN": "province_code", "DPA_DESPRO": "province"}),
    }
    out = {}
    for name, (layer, cols) in spec.items():
        fc = geojson_layer(SVC_DPA, layer, offset_deg=0.002)
        recs = []
        for f in fc["features"]:
            lat, lon = polygon_centroid(f["geometry"])
            recs.append({**{v: f["properties"].get(k) for k, v in cols.items()},
                         "lat": lat, "lon": lon})
        out[name] = pd.DataFrame(recs).sort_values(list(cols.values())[0]).reset_index(drop=True)
        write_geojson(fc, OUT / f"admin_{name}.geojson.gz")
    return out


# ----------------------------------------------------------------------------- images

def download_hc_images(caps: pd.DataFrame) -> int:
    """Download the official hazard-curve JPGs of the cantonal capitals to data-raw."""
    dest = RAW / "hc"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    oids = caps["objectid"].tolist()
    for i in range(0, len(oids), 100):
        chunk = ",".join(map(str, oids[i:i + 100]))
        q = _json(layer_url(SVC_CAP, 0) + "/queryAttachments",
                  {"objectIds": chunk, "f": "json"})
        for g in q["attachmentGroups"]:
            for a in g["attachmentInfos"]:
                if not a["name"].endswith("_HC.jpg"):
                    continue
                path = dest / a["name"]
                if not path.exists():
                    url = layer_url(SVC_CAP, 0) + f"/{g['parentObjectId']}/attachments/{a['id']}"
                    path.write_bytes(_get(url, binary=True))
                n += 1
    return n


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-images", action="store_true", help="skip hazard-curve JPGs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    print("hazard grid ...")
    grid, cells = hazard_grid()
    fc_cells = geojson_layer(SVC_475, 0, offset_deg=0.0005)
    area = {f["properties"]["ID"]: f["properties"].get("Shape__Area") for f in fc_cells["features"]}
    cells["area_km2"] = cells["cell_id"].map(area).astype(float) / 1e6
    for f in fc_cells["features"]:
        f["properties"] = {"cell_id": f["properties"]["ID"]}
    write_geojson(fc_cells, OUT / "hazard_cells.geojson.gz")
    grid.to_csv(OUT / "hazard_grid.csv.gz", index=False, float_format="%.4f")
    cells.to_csv(OUT / "hazard_cells.csv", index=False, float_format="%.4f")
    counts.update(hazard_cells=len(cells), hazard_grid_rows=len(grid))

    print("cross-checking against UHS .txt attachments ...")
    check = crosscheck_txt(grid)
    worst = max(c["max_abs_err_g"] for c in check)
    print(f"  max |table - txt| = {worst:.2e} g over {len(check)} cells")
    if worst > 1e-3:
        raise RuntimeError(f"attribute tables disagree with UHS txt: {check}")

    print("cantonal capitals ...")
    caps = capitals()
    caps.drop(columns="objectid").to_csv(OUT / "capitals.csv", index=False)
    pop = cabeceras_population()
    pop.to_csv(OUT / "cabeceras_population.csv", index=False, float_format="%.5f")
    counts.update(capitals=len(caps), cabeceras_population=len(pop))

    print("seismic sources and catalogs ...")
    srcs = {"sources_crust_interface": (SVC_SRC, 0), "sources_inslab": (SVC_SRC, 1),
            "faults": (SVC_SRC, 2), "sources_beauval2018": (SVC_HIST, 1)}
    for name, (svc, layer) in srcs.items():
        fc = geojson_layer(svc, layer)
        counts[name] = len(fc["features"])
        if fc["features"]:
            write_geojson(fc, OUT / f"{name}.geojson")
    for name, (svc, layer) in {"catalog_shallow": (SVC_SRC, 3), "catalog_deep": (SVC_SRC, 4),
                               "catalog_historical": (SVC_HIST, 0)}.items():
        df = catalog(svc, layer)
        df.to_csv(OUT / f"{name}.csv.gz", index=False)
        counts[name] = len(df)

    print("administrative units ...")
    for name, df in admin_units().items():
        df.to_csv(OUT / f"admin_{name}.csv", index=False, float_format="%.5f")
        counts[f"admin_{name}"] = len(df)

    if not args.no_images:
        print("hazard-curve images (data-raw) ...")
        counts["hc_images"] = download_hc_images(caps)

    manifest = {
        "provider": "Instituto Geofísico - Escuela Politécnica Nacional (IG-EPN)",
        "product": "Mapa Digital Interactivo del Peligro Sísmico Probabilístico para el Ecuador",
        "url": STORYMAP,
        "storymap": STORYMAP_ITEM,
        "feature_services": BASE,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site_condition": "rock, Vs30 = 760 m/s",
        "periods_s": list(PERIODS),
        "return_periods_yr": [475, 2475],
        "units": "g",
        "txt_crosscheck_max_abs_err_g": worst,
        "counts": counts,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
