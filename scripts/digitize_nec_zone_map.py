"""
Digitize the NEC-SE-DS seismic zone map (Figura 1) and Table 19 into apeQuake data.

Source: NEC-SE-DS "Peligro Sismico, Diseno Sismo Resistente" (NEC-15), section 3.1.1,
Figura 1 "Ecuador, zonas sismicas para propositos de diseno y valor del factor de zona Z",
and section 10.2, Tabla 19 "Poblaciones ecuatorianas y valor del factor Z".

Pipeline
--------
1. Extract the Figura 1 raster embedded in ``NEC-SE-DS-Peligro-Sismico-parte-1.pdf``
   (741 x 503 px, WGS-84 graticule, ~1.5 km per pixel).
2. Georeference it from the graticule ticks drawn just outside the map frame
   (8 meridians 82-75 W, 6 parallels 1 N-4 S); plate carree, so a linear fit per axis.
3. Mask to continental Ecuador with the bundled province polygons (IG-EPN admin layer).
4. Classify each pixel by hue band.  The zone colours are a translucent overlay on a
   hillshade, so hue is stable where RGB and chromaticity are not:
       red < 14 deg (0.50) | orange < 37.5 (0.40) | yellow < 56 (0.35)
       56-99: pale (0.30) if saturation < 0.53, else green (0.25) | dark green < 150 (0.15)
   Labels, roads, city patches and water, and a 2-px halo around them (colour-mixed
   edges), stay unclassified and are filled from the nearest classified pixel; then a
   7 x 7 majority filter is applied three times.
5. Parse Table 19 from the PDF word layout and join each row to the bundled parish
   table on (province, canton, parish | town).  The table is used for validation and
   for the "nearest listed town" check, not for the lookup itself.

Galapagos is not digitized from the inset: the whole province is Z = 0.30 g per the
inset annotation, and ``apeQuake.nec.zoning`` handles it by province.

Usage::

    python scripts/digitize_nec_zone_map.py --nec-dir "<folder with NEC-SE-DS-Peligro-Sismico-parte-*.pdf>"

Writes ``src/apeQuake/nec/data/nec_zone_map.npz`` and ``nec_table19.csv``; the extracted
figure and a QC overlay go to ``data-raw/nec`` (git-ignored).
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from matplotlib.path import Path as MplPath
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apeQuake.hazard import _data  # noqa: E402

OUT = ROOT / "src" / "apeQuake" / "nec" / "data"
RAW = ROOT / "data-raw" / "nec"

Z_VALUES = np.array([0.15, 0.25, 0.30, 0.35, 0.40, 0.50])
HUE_EDGES = (14.0, 37.5, 56.0, 77.0, 99.0, 150.0)   # deg; class index 5 (red) .. 0 (dark green)
LON_TICKS = np.arange(-82.0, -74.0)                 # 82 W .. 75 W
LAT_TICKS = np.arange(1.0, -5.0, -1.0)              # 1 N .. 4 S
OUTSIDE = 255

PART1 = "NEC-SE-DS-Peligro-S*smico-parte-1.pdf"
TABLE19 = (("NEC-SE-DS-Peligro-S*smico-parte-2.pdf", range(47, 50)),
           ("NEC-SE-DS-Peligro-S*smico-parte-31.pdf", range(0, 21)))


def _pdf(nec_dir: Path, pattern: str) -> Path:
    hits = sorted(nec_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found in {nec_dir}")
    return hits[0]


def extract_figure(nec_dir: Path) -> np.ndarray:
    """RGB array of Figura 1 (largest image on the page that captions it)."""
    import fitz  # PyMuPDF, only needed to regenerate

    doc = fitz.open(_pdf(nec_dir, PART1))
    for page in doc:
        if "Figura 1. Ecuador, zonas" in page.get_text() and page.get_images():
            xref = max(page.get_images(full=True), key=lambda im: im[2] * im[3])[0]
            pix = fitz.Pixmap(fitz.csRGB, fitz.Pixmap(doc, xref))
            return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    raise RuntimeError("Figura 1 not found in the NEC PDF")


def georeference(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pixel-centre longitudes (per column) and latitudes (per row) from the frame ticks."""
    grey = rgb.mean(axis=2)
    h, w = grey.shape
    fx = w - 45 + int(grey[40:h - 40, w - 45:].mean(axis=0).argmin())   # right frame line
    fy = h - 40 + int(grey[h - 40:, 200:w - 100].mean(axis=1).argmin())  # bottom frame line
    xs = np.flatnonzero(grey[fy + 1, :fx] < 150)
    ys = np.flatnonzero(grey[:fy, fx + 1] < 150)
    if len(xs) != len(LON_TICKS) or len(ys) != len(LAT_TICKS):
        raise RuntimeError(f"tick detection failed: {len(xs)} meridians, {len(ys)} parallels")
    ax, bx = np.polyfit(LON_TICKS, xs, 1)
    ay, by = np.polyfit(LAT_TICKS, ys, 1)
    qc = {
        "px_per_deg_lon": ax, "px_per_deg_lat": -ay,
        "resid_lon_px": float(np.abs(np.polyval([ax, bx], LON_TICKS) - xs).max()),
        "resid_lat_px": float(np.abs(np.polyval([ay, by], LAT_TICKS) - ys).max()),
    }
    lon = (np.arange(w) - bx) / ax
    lat = (np.arange(h) - by) / ay
    return lon, lat, qc


def province_paths(skip: tuple[str, ...] = ("GALAPAGOS",)) -> list[MplPath]:
    paths = []
    for f in _data.geojson("admin_provinces.geojson.gz")["features"]:
        if _data.normalize(f["properties"]["DPA_DESPRO"]) in skip:
            continue
        g = f["geometry"]
        for poly in g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]:
            verts, codes = [], []
            for ring in poly:
                ring = np.asarray(ring, float)
                verts.append(ring)
                codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
            paths.append(MplPath(np.concatenate(verts), codes))
    return paths


def ecuador_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.c_[LON.ravel(), LAT.ravel()]
    mask = np.zeros(LON.size, bool)
    for p in province_paths():
        mask |= p.contains_points(pts)
    return mask.reshape(LON.shape)


def classify(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    hsv = rgb_to_hsv(rgb.astype(float) / 255.0)
    hue, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    # water, grey city patches and dark text; pixels next to them are colour-mixed
    # (e.g. light-blue estuary + orange reads as yellow), so they do not vote either
    blank = (sat <= 0.30) | (val <= 0.35) | ((hue > 150) & (hue < 300))
    coloured = ~ndimage.binary_dilation(blank, iterations=2)
    k = np.full(hue.shape, -1)
    k[coloured & ((hue < HUE_EDGES[0]) | (hue > 340))] = 5
    for idx, (lo, hi) in zip((4, 3, 2, 1, 0), zip(HUE_EDGES[:-1], HUE_EDGES[1:])):
        k[coloured & (hue >= lo) & (hue < hi)] = idx
    # pale (0.30) and green (0.25) overlap in hue over green lowland relief;
    # saturation separates them (pale ~0.40-0.47, green ~0.62-0.68)
    yg = coloured & (hue >= HUE_EDGES[2]) & (hue < HUE_EDGES[4])
    k[yg] = np.where((sat[yg] >= 0.53) & (hue[yg] >= 72.0), 1, 2)
    ok = mask & (k >= 0)
    frac = float(ok.sum() / mask.sum())
    near = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    k = k[near[0], near[1]]
    for _ in range(3):
        k = np.stack([ndimage.uniform_filter((k == i).astype(float), 7)
                      for i in range(len(Z_VALUES))]).argmax(axis=0)
    return k, frac


def _norm(s: str) -> str:
    s = str(s).replace("Ð", "Ñ")   # the IG-EPN admin layer carries the same font quirk
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    s = s.replace("STO.", "SANTO")
    s = re.sub(r"\(.*?\)|[^A-Z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w != "Z")


def parse_table19(nec_dir: Path) -> pd.DataFrame:
    """Rows of Tabla 19 from the PDF word layout; column edges come from each page's header."""
    import fitz

    rows = []
    for pattern, pages in TABLE19:
        doc = fitz.open(_pdf(nec_dir, pattern))
        for pi in pages:
            words = doc[pi].get_text("words")
            head = {w[4]: w for w in words if w[4] in ("PARROQUIA", "CANTÓN", "PROVINCIA")}
            if len(head) < 3:
                continue
            top = head["PARROQUIA"][3]
            zcol = [w for w in words if w[4] == "Z" and abs(w[1] - head["PARROQUIA"][1]) < 3]
            if not zcol:
                continue
            edges = (head["PARROQUIA"][0] - 3, head["CANTÓN"][0] - 3,
                     head["PROVINCIA"][0] - 3, zcol[0][0] - 12)
            zs = [w for w in words if w[1] > top and abs(w[0] - edges[3] - 12) <= 15
                  and re.fullmatch(r"\d\.\d\d", w[4])]
            if not zs:
                continue
            zy = np.array([(w[1] + w[3]) / 2 for w in zs])
            cells: list[list[list]] = [[[] for _ in range(4)] for _ in zs]
            for w in words:
                if w in zs or w[1] <= top or w[1] > zy.max() + 12:   # skip header and caption
                    continue
                col = next((c for c, edge in enumerate(edges) if w[0] < edge), None)
                if col is None:
                    continue
                yc = (w[1] + w[3]) / 2
                i = int(np.argmin(np.abs(zy - yc)))
                if abs(zy[i] - yc) <= 20:
                    cells[i][col].append((w[1], w[0], w[4]))
            for zw, c in zip(zs, cells):
                # the PDF font maps Ñ to the glyph Ð
                text = [" ".join(t for *_, t in sorted(col)).replace("Ð", "Ñ") for col in c]
                rows.append(dict(poblacion=text[0], parroquia=text[1], canton=text[2],
                                 provincia=text[3], z=float(zw[4])))
    return pd.DataFrame(rows)


def join_parishes(t19: pd.DataFrame) -> pd.DataFrame:
    par = _data.admin("parishes").astype({"parish_code": str})
    keys = {c: par[s].map(_norm) for c, s in (("kp", "parish"), ("kc", "canton"), ("kv", "province"))}
    par = par.assign(**keys)
    code, lat, lon = [], [], []
    for r in t19.itertuples():
        base = par[(par.kc == _norm(r.canton)) & (par.kv == _norm(r.provincia))]
        m = base[base.kp == _norm(r.parroquia)]
        if m.empty:
            m = base[base.kp == _norm(r.poblacion)]
        code.append(m.parish_code.iloc[0].zfill(6) if not m.empty else "")
        lat.append(float(m.lat.iloc[0]) if not m.empty else np.nan)
        lon.append(float(m.lon.iloc[0]) if not m.empty else np.nan)
    return t19.assign(parish_code=code, lat=lat, lon=lon)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nec-dir", type=Path, required=True,
                    help="folder holding NEC-SE-DS-Peligro-Sismico-parte-*.pdf")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rgb = extract_figure(args.nec_dir)
    lon, lat, qc = georeference(rgb)
    print(f"georeference: {qc}")
    mask = ecuador_mask(lon, lat)
    k, frac = classify(rgb, mask)
    print(f"directly classified: {frac:.1%} of continental Ecuador pixels")

    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    zone = np.where(mask, k, OUTSIDE).astype(np.uint8)[r0:r1, c0:c1]
    np.savez_compressed(
        OUT / "nec_zone_map.npz",
        zone=zone, lon=lon[c0:c1], lat=lat[r0:r1], z_values=Z_VALUES,
        source=np.array("NEC-SE-DS (NEC-15) Figura 1; scripts/digitize_nec_zone_map.py"),
    )

    t19 = join_parishes(parse_table19(args.nec_dir))
    t19.to_csv(OUT / "nec_table19.csv", index=False, float_format="%.5f")
    hit = t19.dropna(subset=["lat"])
    i = np.clip(np.round((hit.lat.to_numpy() - lat[r0]) / (lat[1] - lat[0])).astype(int), 0, zone.shape[0] - 1)
    j = np.clip(np.round((hit.lon.to_numpy() - lon[c0]) / (lon[1] - lon[0])).astype(int), 0, zone.shape[1] - 1)
    zmap = np.where(zone[i, j] == OUTSIDE, np.nan, Z_VALUES[np.minimum(zone[i, j], 5)])
    print(f"Table 19: {len(t19)} rows, {len(hit)} joined to parishes; "
          f"map agrees at the parish point for {np.mean(zmap == hit.z):.1%}")

    from PIL import Image
    Image.fromarray(rgb).save(RAW / "figura1.png")
    edge = (ndimage.grey_dilation(k, 3) != ndimage.grey_erosion(k, 3)) & mask
    over = rgb.copy()
    over[edge] = 0
    Image.fromarray(over).save(RAW / "figura1_qc.png")
    print(f"wrote {OUT / 'nec_zone_map.npz'} {zone.shape} and {OUT / 'nec_table19.csv'}")


if __name__ == "__main__":
    main()
