"""Digitize the NEC-SE-DS seismic-hazard curves (Figures 10-32, Section 10.3).

The official NEC-15 *Peligro sismico* PDF carries 23 raster plots, one per
provincial capital, each showing the annual exceedance rate against PGA / Sa
for five measures (PGA, Sa(0.1 s), Sa(0.2 s), Sa(0.5 s), Sa(1.0 s)).  This
script recovers the curves numerically:

1. extract the embedded plot images from PDF pages 118-129;
2. calibrate the axes from the plot frame (x linear from 0, y log10 from 1e-5
   at the bottom edge to 1 at the top edge) and cross-check with the ticks;
3. classify pixels by curve colour and extract each curve's centreline from
   both column and row scans (columns are accurate on flat segments, rows on
   steep ones);
4. order the points along the curve, fill occlusions where one curve hides
   another, smooth lightly, and resample;
5. write ``src/apeQuake/nec/data/nec_hazard_curves.json`` plus QA overlays.

Usage::

    python scripts/digitize_nec_hazard.py "<NEC_SE_DS pdf>" [--qa-dir DIR]

Requires PyMuPDF, Pillow, numpy and scipy (all are dev-time only; the
digitized JSON ships with the package).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

# Viewer pages (1-based) of the official NEC_SE_DS PDF that hold the plots.
FIRST_PAGE, LAST_PAGE = 118, 129

# (figure, city, lat, lon, x-axis maximum in g), in order of appearance.
CITIES = [
    (10, "Tulcan", 0.81, -77.72, 1.5),
    (11, "Ibarra", 0.35, -78.13, 1.5),
    (12, "Quito", -0.20, -78.51, 1.2),
    (13, "Latacunga", -0.93, -78.62, 1.2),
    (14, "Ambato", -1.25, -78.63, 1.2),
    (15, "Riobamba", -1.67, -78.65, 1.2),
    (16, "Guaranda", -1.59, -79.00, 1.0),
    (17, "Azogues", -2.74, -78.85, 1.0),
    (18, "Cuenca", -2.90, -79.00, 1.0),
    (19, "Loja", -3.98, -79.21, 1.0),
    (20, "Esmeraldas", 0.97, -79.65, 2.0),
    (21, "Portoviejo", -1.06, -80.46, 1.5),
    (22, "Santa Elena", -2.23, -80.86, 1.5),
    (23, "Santo Domingo", -0.26, -79.17, 1.2),
    (24, "Babahoyo", -1.81, -79.52, 1.0),
    (25, "Guayaquil", -2.17, -79.91, 1.0),
    (26, "Machala", -3.26, -79.96, 1.0),
    (27, "Orellana", -0.46, -76.99, 1.0),
    (28, "Tena", -0.99, -77.81, 1.0),
    (29, "Puyo", -1.49, -78.00, 1.0),
    (30, "Macas", -2.30, -78.12, 1.2),
    (31, "Zamora", -4.06, -78.95, 1.0),
    (32, "Nueva Loja", 0.09, -76.89, 0.8),
]

# Curve key -> reference RGB.  Keys are the structural period in seconds
# ("PGA" is T = 0).
COLORS = {
    "PGA": (0, 0, 0),
    "1.0": (50, 70, 155),
    "0.5": (100, 180, 65),
    "0.2": (150, 90, 40),
    "0.1": (180, 70, 150),
}
COLOR_TOL = 62.0  # max RGB distance to a reference colour
CORE_RADIUS = 1.8  # px; minimum half-thickness for a pixel to count as "line"
RUN_MAX = 15 # longest run (px) still considered a centreline sample
LOG_RANGE = 5.0  # decades between the bottom (1e-5) and top (1) frame edges


def extract_images(pdf_path: Path) -> list[np.ndarray]:
    """Return the 23 plot images (RGB arrays) in figure order."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    images = []
    for pno in range(FIRST_PAGE, LAST_PAGE + 1):
        page = doc[pno - 1]
        found = []
        for info in page.get_images(full=True):
            xref = info[0]
            rects = page.get_image_rects(xref)
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            found.append((rects[0].y0, arr.copy()))
        images.extend(a for _, a in sorted(found, key=lambda t: t[0]))
    return images


def _group_mean(idx: np.ndarray) -> list[float]:
    """Mean of each run of consecutive integers (frame lines can be 2 px)."""
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [float(g.mean()) for g in groups]


def find_frame(rgb: np.ndarray) -> tuple[float, float, float, float]:
    """Plot-frame edges (left, right, top, bottom) in pixel-centre coords."""
    dark = rgb.max(axis=2) < 110
    h, w = dark.shape
    rows = _group_mean(np.where(dark.sum(1) > 0.5 * w)[0])
    cols = _group_mean(np.where(dark.sum(0) > 0.5 * h)[0])
    if len(rows) < 2 or len(cols) < 2:
        raise RuntimeError("plot frame not found")
    return cols[0], cols[-1], rows[0], rows[-1]


def check_xmax(rgb: np.ndarray, frame, xmax: float) -> float:
    """Return the major-tick step (g) implied by *xmax*; expect 0.2 or 0.5."""
    left, right, _, bottom = frame
    dark = rgb.max(axis=2) < 110
    b = int(round(bottom))
    band = dark[b - 14 : b - 1, :].sum(axis=0)
    major = [x for x in range(int(left) + 3, int(right) - 2) if band[x] >= 7]
    # collapse adjacent columns, ignore curve pixels (wide runs)
    groups = [g for g in np.split(major, np.where(np.diff(major) > 2)[0] + 1) if 0 < len(g) <= 3]
    centres = np.array([g.mean() for g in groups])
    if len(centres) < 2:
        return float("nan")
    step_px = float(np.median(np.diff(centres)))
    return step_px * xmax / (right - left)


def color_masks(rgb: np.ndarray, frame) -> dict[str, np.ndarray]:
    """Boolean mask per curve colour, restricted to the frame interior."""
    left, right, top, bottom = frame
    h, w, _ = rgb.shape
    inside = np.zeros((h, w), bool)
    inside[int(top) + 3 : int(bottom) - 2, int(left) + 3 : int(right) - 2] = True
    px = rgb.astype(float)
    masks = {}
    for key, ref in COLORS.items():
        if key == "PGA":
            m = (rgb.max(axis=2) < 70) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 30)
        else:
            d = np.sqrt(((px - np.array(ref)) ** 2).sum(axis=2))
            m = d < COLOR_TOL
        m &= inside
        # opening removes 1-px ticks / anti-aliasing fringes, keeps 4-5 px lines
        # Keep only pixels sitting in a stretch of line at least ~2*CORE px
        # thick (curves are 4-5 px).  This drops 1-px ticks and, importantly,
        # the thin slivers of a half-hidden curve whose centroid is biased.
        core = ndimage.distance_transform_edt(m) >= CORE_RADIUS
        m = ndimage.distance_transform_edt(~core) <= CORE_RADIUS
        masks[key] = _drop_legend(m, frame)
    return masks


def _drop_legend(mask: np.ndarray, frame) -> np.ndarray:
    """Remove the small glyph blobs of the legend (upper-right of the plot)."""
    left, right, top, bottom = frame
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    out = mask.copy()
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        hgt, wid = ys.stop - ys.start, xs.stop - xs.start
        cx = 0.5 * (xs.start + xs.stop)
        cy = 0.5 * (ys.start + ys.stop)
        in_legend_zone = cx > left + 0.6 * (right - left) and cy < top + 0.62 * (bottom - top)
        if in_legend_zone and wid < 70 and hgt < 50:
            out[lab == i] = False
    return out


def centreline_points(mask: np.ndarray) -> np.ndarray:
    """(x_px, y_px) centreline samples from column and row scans."""
    pts = []
    h, w = mask.shape
    for x in range(w):
        col = mask[:, x]
        if not col.any():
            continue
        idx = np.where(col)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        if len(runs) == 1 and len(runs[0]) <= RUN_MAX:
            pts.append((x, runs[0].mean()))
    for y in range(h):
        row = mask[y, :]
        if not row.any():
            continue
        idx = np.where(row)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        if len(runs) == 1 and len(runs[0]) <= RUN_MAX:
            pts.append((runs[0].mean(), y))
    return np.array(pts, float)


def build_curve(pts_px: np.ndarray, frame, xmax: float, donors=None) -> dict:
    """Order, de-noise and resample one curve; returns arrays and a gap flag."""
    left, right, top, bottom = frame
    a = (pts_px[:, 0] - left) / (right - left) * xmax
    ly = -LOG_RANGE * (pts_px[:, 1] - top) / (bottom - top)  # log10(rate)
    # monotone arclength-like coordinate: both terms grow along the curve
    s = a / xmax + (-ly) / LOG_RANGE
    order = np.argsort(s)
    s, a, ly = s[order], a[order], ly[order]

    # bin along s (about 1 px) and take medians
    ds = 1.0 / (right - left)
    nb = int(np.ceil((s.max() - s.min()) / ds)) + 1
    bi = np.minimum(((s - s.min()) / ds).astype(int), nb - 1)
    bs, ba, bl, have = (np.full(nb, np.nan) for _ in range(4))
    for k in np.unique(bi):
        sel = bi == k
        bs[k], ba[k], bl[k] = np.median(s[sel]), np.median(a[sel]), np.median(ly[sel])
        have[k] = 1
    ok = ~np.isnan(bs)
    # Partially occluded stretches bias the centroid toward the visible edge.
    # Reject bins that stray from a rolling median of the curve by > 2 px.
    width_px, height_px = right - left, bottom - top
    idx = np.where(ok)[0]
    for _ in range(2):
        ia, il = ba[idx], bl[idx]
        win = 25
        pa = np.pad(ia, win // 2, mode="edge")
        pl = np.pad(il, win // 2, mode="edge")
        ma = np.array([np.median(pa[i : i + win]) for i in range(len(ia))])
        ml = np.array([np.median(pl[i : i + win]) for i in range(len(il))])
        dist = np.hypot((ia - ma) / xmax * width_px, (il - ml) / LOG_RANGE * height_px)
        bad = dist > 2.0
        # never drop the end points; they anchor the interpolation
        bad[:3] = bad[-3:] = False
        ok[idx[bad]] = False
        bs[idx[bad]] = ba[idx[bad]] = bl[idx[bad]] = np.nan
        idx = idx[~bad]
    # fill occlusions (a hidden curve lies under the one drawn on top)
    sg = s.min() + np.arange(nb) * ds
    # Only runs of >= 12 empty bins are real occlusions; shorter holes are
    # just sparse sampling on steep segments and are bridged by PCHIP below.
    # A hidden stretch follows the curve drawn on top of it, so fill it from
    # the nearest visible curve (offset-blended to join both ends).
    notok = np.where(~ok)[0]
    filled = np.zeros(nb, bool)
    for run in np.split(notok, np.where(np.diff(notok) > 1)[0] + 1):
        if len(run) < 12:
            continue
        filled[run] = True
        i0, i1 = run[0] - 1, run[-1] + 1
        if i0 < 0 or i1 >= nb or not (ok[i0] and ok[i1]):
            continue
        a0, a1, l0, l1 = ba[i0], ba[i1], bl[i0], bl[i1]
        t = (run - i0) / (i1 - i0)
        aa = a0 + t * (a1 - a0)
        best = None
        for da, dl in donors or []:
            d0, d1 = np.interp(a0, da, dl), np.interp(a1, da, dl)
            err = abs(d0 - l0) + abs(d1 - l1)
            if best is None or err < best[0]:
                best = (err, da, dl, d0, d1)
        if best is not None and best[0] < 0.2 and a1 - a0 > 1e-6:
            _, da, dl, d0, d1 = best
            ll = np.interp(aa, da, dl) + (l0 - d0) + t * ((l1 - d1) - (l0 - d0))
        else:
            ll = l0 + t * (l1 - l0)
        ba[run], bl[run], bs[run], ok[run] = aa, ll, sg[run], True
    fa = PchipInterpolator(bs[ok], ba[ok])(sg)
    fl = PchipInterpolator(bs[ok], bl[ok])(sg)
    # light smoothing of the centreline (window ~ 9 px)
    k = np.ones(9) / 9.0
    fa_s = np.convolve(np.pad(fa, 4, mode="edge"), k, mode="valid")
    fl_s = np.convolve(np.pad(fl, 4, mode="edge"), k, mode="valid")
    # trim the part sitting on the frame (curve exiting the plot)
    keep = (fl_s > -LOG_RANGE + 0.02) & (fl_s < -0.0)
    return {
        "a": fa_s[keep],
        "log10_rate": fl_s[keep],
        "gap_fraction": float(filled[keep].mean()) if keep.any() else 1.0,
    }


def _extend_hidden_ends(curve: dict, donor: dict, tol: float = 0.01) -> None:
    """Extend *curve* in place where its first/last stretch hides under *donor*.

    Sa(0.2 s) often runs under Sa(0.1 s) at the start and/or end of a plot, so
    its visible part begins late or stops early.  The missing ends follow the
    donor with a constant log-rate offset matching the join.
    """
    a, ly = curve["a"], curve["log10_rate"]
    da, dl = donor["a"], donor["log10_rate"]
    n_old = len(a)
    head = da < a[0] - tol
    if head.any():
        off = ly[0] - np.interp(a[0], da, dl)
        a = np.concatenate([da[head], a])
        ly = np.concatenate([dl[head] + off, ly])
    tail = da > a[-1] + tol
    if tail.any() and ly[-1] > -LOG_RANGE + 0.1:  # not a curve that left via the floor
        off = ly[-1] - np.interp(a[-1], da, dl)
        a = np.concatenate([a, da[tail]])
        ly = np.concatenate([ly, dl[tail] + off])
    added = len(a) - n_old
    if added:
        curve["gap_fraction"] = (curve["gap_fraction"] * n_old + added) / len(a)
        curve["a"], curve["log10_rate"] = a, np.minimum(ly, -1e-6)


def resample(curve: dict, n: int = 160) -> dict:
    """Resample on *n* points, uniform in arclength so steep parts stay dense."""
    a, ly = curve["a"], curve["log10_rate"]
    seg = np.hypot(np.diff(a) / max(a.max(), 1e-9), np.diff(ly) / LOG_RANGE)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    tt = np.linspace(0, t[-1], n)
    return {
        "a": np.interp(tt, t, a).round(4).tolist(),
        "rate": [float(f"{r:.4g}") for r in 10.0 ** np.interp(tt, t, ly)],
        "gap_fraction": round(curve["gap_fraction"], 3),
    }


def qa_overlay(rgb: np.ndarray, frame, xmax: float, curves: dict, path: Path) -> None:
    """Draw the recovered curves in white-outlined dots over the source image."""
    left, right, top, bottom = frame
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im)
    for key, c in curves.items():
        for a, r in zip(c["a"], c["rate"]):
            px = left + a / xmax * (right - left)
            py = top + (-np.log10(r)) / LOG_RANGE * (bottom - top)
            dr.ellipse([px - 1.6, py - 1.6, px + 1.6, py + 1.6], outline=(255, 255, 0))
    im.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path, help="official NEC_SE_DS (peligro sismico).pdf")
    ap.add_argument("--out", type=Path, default=Path("src/apeQuake/nec/data/nec_hazard_curves.json"))
    ap.add_argument("--qa-dir", type=Path, default=None)
    args = ap.parse_args()

    images = extract_images(args.pdf)
    if len(images) != len(CITIES):
        raise SystemExit(f"expected {len(CITIES)} plots, found {len(images)}")
    if args.qa_dir:
        args.qa_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "source": "NEC-SE-DS (NEC-15), Section 10.3, Figures 10-32 (IG-EPN hazard study)",
        "method": "raster digitization, scripts/digitize_nec_hazard.py",
        "axes": {"x": "acceleration (g), linear", "y": "annual exceedance rate (1/yr), log"},
        "periods": list(COLORS),
        "cities": {},
    }
    for (fig, name, lat, lon, xmax), rgb in zip(CITIES, images):
        frame = find_frame(rgb)
        step = check_xmax(rgb, frame, xmax)
        masks = color_masks(rgb, frame)
        curves = {}
        hidden = []
        points = {}
        for key, m in masks.items():
            pts = centreline_points(m)
            if len(pts) < 40:  # curve (almost) entirely hidden under another
                hidden.append(key)
                continue
            points[key] = pts
        # pass 1: no donors; pass 2: occluded stretches follow a visible curve
        first = {k: build_curve(p, frame, xmax) for k, p in points.items()}
        built = {}
        for key, pts in points.items():
            donors = [(c["a"], c["log10_rate"]) for k, c in first.items() if k != key]
            built[key] = build_curve(pts, frame, xmax, donors)
        if "0.2" in built and "0.1" in built:
            _extend_hidden_ends(built["0.2"], built["0.1"])
        curves = {k: resample(b) for k, b in built.items()}
        for key in hidden:
            # Sa(0.2 s) and Sa(0.1 s) are near-coincident; the hidden one takes
            # the other's curve and is flagged so users can see it is an alias.
            donor = "0.1" if key == "0.2" else "0.2"
            if donor in hidden or donor not in curves:
                raise SystemExit(f"{name}: cannot recover {key}")
            curves[key] = {**curves[donor], "aliased_from": donor}
        out["cities"][name] = {
            "figure": fig,
            "lat": lat,
            "lon": lon,
            "x_max_g": xmax,
            "curves": {k: curves[k] for k in COLORS},
        }
        gaps = {k: v["gap_fraction"] for k, v in curves.items()}
        print(f"Fig {fig:2d} {name:14s} tick-step={step:.3f}g gaps={gaps}")
        if args.qa_dir:
            qa_overlay(rgb, frame, xmax, curves, args.qa_dir / f"fig{fig:02d}_{name}.png")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
