"""
Digitize the IG-EPN hazard-curve images of the cantonal capitals.

IG-EPN publishes hazard curves (annual rate of exceedance vs. spectral acceleration, rock)
only as MATLAB-rendered JPGs, one per cantonal-capital cell. All 365 images share the same
figure layout:

* plot box: x in [114, 791] px, y in [90, 583] px (measured; identical in every image);
* y axis: log10, fixed 1e0 (top) .. 1e-6 (bottom);
* x axis: linear, 0 .. xmax, where xmax is a MATLAB "nice" limit that changes per site;
* legend box over the upper-right corner (masked: curves hidden behind it are lost);
* one color per period (PGA thick red, 0.05 green, 0.07 magenta, 0.10 black, 0.20 cyan,
  0.50 navy, 1.00 yellow, 2.00 steel blue).

Calibration of xmax uses no OCR. The mean hazard curve must satisfy
lambda(Sa_TR) = 1/TR at the published mean UHS ordinates (TR = 475, 2475; all 8 periods),
so xmax is the candidate nice limit that best satisfies those 16 anchors. The choice is
discrete, so it cannot deform the curves; the leftover anchor residuals are reported as an
independent quality measure per site and per period.

Output: src/apeQuake/hazard/data/igepn/hazard_curves_capitals.csv.gz with columns
    cell_id, period, sa_g, rate        (rate = annual rate of exceedance, 1/yr)
and hazard_curves_capitals_qc.csv with per-curve quality metrics.

Usage:
    python scripts/igepn/digitize_hazard_curves.py [--plot CELL_ID ...]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "apeQuake" / "hazard" / "data" / "igepn"
RAW = ROOT / "data-raw" / "igepn" / "hc"

X0, X1, Y0, Y1 = 114, 791, 90, 583           # axes box (pixel lines)
LOG_TOP, LOG_BOT = 0.0, -6.0                  # log10(rate) at Y0 / Y1
LEGEND = (553, 780, 102, 379)                 # x0, x1, y0, y1 (with margin)
INSET = 8                                     # skip axes lines + inward tick marks
PERIODS = (0.0, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0)
XMAX_CANDIDATES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4,
                   1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


# Strongest ink of each legend swatch (median over images); thin lines never get darker.
REF_RGB = np.array([
    (245, 2, 4),        # PGA red (thick)
    (93, 190, 95),      # 0.05 green
    (162, 65, 160),     # 0.07 magenta
    (0, 0, 0),          # 0.10 black
    (113, 210, 203),    # 0.20 cyan
    (16, 22, 110),      # 0.50 navy
    (240, 237, 144),    # 1.00 yellow
    (53, 99, 123),      # 2.00 steel blue
], float) / 255.0
ALPHA_MIN = 0.45
RESID_MAX = 0.12
MIN_BLEND_CHROMA = 0.12
MIN_PIXEL_CHROMA = 0.08
BLACK = 3
PGA = 0
# faint anti-aliased steel blue (2.00 s) on steep segments unmixes as cyan (0.20 s):
# cyan pixels off the traced 0.20 s curve are extra evidence for the 2.00 s curve
CONFUSED_WITH = {7: 4}
MIN_PURE_POINTS = 80
N_OUT = 40                                    # output points per curve                          # below this, trace pure | blend


def _inks() -> tuple[np.ndarray, list[frozenset[int]]]:
    """Pure inks plus 50/50 blends of every pair.

    Where two curves coincide, the thin later-drawn line sits on the earlier one and JPEG
    chroma subsampling (2x2) averages them (e.g. PGA red + 0.50 s navy -> crimson). A blend
    pixel is evidence for both curves. Blends of near-complementary inks are almost
    neutral (green + magenta ~ gray) and would match the dotted gridlines, so blends with
    chroma below MIN_BLEND_CHROMA are not used.
    """
    refs, members = list(REF_RGB), [frozenset({k}) for k in range(len(REF_RGB))]
    for i in range(len(REF_RGB)):
        for j in range(i + 1, len(REF_RGB)):
            c = 0.5 * (REF_RGB[i] + REF_RGB[j])
            if c.max() - c.min() < MIN_BLEND_CHROMA:
                continue
            refs.append(c)
            members.append(frozenset({i, j}))
    return np.asarray(refs), members


INKS, INK_MEMBERS = _inks()


def classify(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-curve (pure, blend) membership masks, each (K, H, W), by alpha-unmixing.

    An anti-aliased pixel of ink C is p = w + a (C - w). For every ink the best a and the
    residual |p - w - a (C - w)| are computed; the pixel takes the ink with the smallest
    residual if a is large enough. Light-gray gridlines are black ink at a ~ 0.13 and are
    rejected by ALPHA_MIN; any neutral pixel may only be black ink. Pure inks win ties against blends (small residual bonus).
    """
    full = rgb.shape[:2]
    rgb = rgb[Y0:Y1 + 1, X0:X1 + 1]                    # plot box only
    h, w = rgb.shape[:2]
    p = rgb.reshape(-1, 3) - 1.0                       # p - w
    d = INKS - 1.0                                     # C - w, shape (M, 3)
    a = np.clip((p @ d.T) / (d * d).sum(1), 0.0, 1.3)  # (N, M)
    resid = np.linalg.norm(p[:, None, :] - a[..., None] * d[None], axis=2)
    resid[a < ALPHA_MIN] = np.inf
    # neutral pixels (gridlines, anti-aliased black) can only be black ink
    neutral = (rgb.reshape(-1, 3).max(1) - rgb.reshape(-1, 3).min(1)) < MIN_PIXEL_CHROMA
    colored = np.ones(len(INKS), bool)
    colored[BLACK] = False
    resid[np.ix_(neutral, colored)] = np.inf
    resid[:, len(REF_RGB):] += 0.02
    m = resid.argmin(1)
    ok = resid[np.arange(m.size), m] < RESID_MAX
    K = len(REF_RGB)
    pure = np.zeros((K, *full), bool)
    blend = np.zeros((K, *full), bool)
    for idx, mem in enumerate(INK_MEMBERS):
        sel = (ok & (m == idx)).reshape(h, w)
        for k in mem:
            tgt = pure if len(mem) == 1 else blend
            tgt[k, Y0:Y1 + 1, X0:X1 + 1] |= sel
    return pure, blend


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators fit of a non-decreasing sequence (L2)."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v2], [w], [c]
    return np.repeat(vals, cnt)


def _drop_violators(xs: np.ndarray, ys: np.ndarray, tol: float = 3.0,
                    max_votes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Remove points that break monotonicity against many others.

    A real point has no later point clearly above it (rate cannot increase with Sa) and no
    earlier point clearly below it. Junk (a stray run in the dense bundle near Sa = 0,
    a crossing curve) collects many such votes; isotonic regression alone would pool it
    into a false plateau.
    """
    if xs.size < 3:
        return xs, ys
    later_above = np.triu(ys[None, :] < ys[:, None] - tol, 1).sum(1)
    earlier_below = np.tril(ys[None, :] > ys[:, None] + tol, -1).sum(1)
    keep = (later_above + earlier_below) <= max_votes
    return xs[keep], ys[keep]


def _column_rows(x: int) -> np.ndarray:
    if LEGEND[0] <= x <= LEGEND[1]:
        return np.r_[np.arange(Y0 + INSET, LEGEND[2]), np.arange(LEGEND[3], Y1 - INSET)]
    return np.arange(Y0 + INSET, Y1 - INSET)


def trace(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centerline (px) of one curve inside the plot box, legend masked.

    Each column offers candidate runs of member pixels (steep parts span many rows; the run
    median is the centerline). A Viterbi pass picks one run or "skip" per column,
    penalizing vertical gaps between consecutive runs and, strongly, upward moves (the rate
    cannot increase with Sa). Residual spikes are removed against a rolling median and the
    result is projected on a monotone sequence.
    """
    cols, cands = [], []
    for x in range(X0 + INSET, X1 - INSET + 1):
        rows = _column_rows(x)
        hit = rows[mask[rows, x]]
        if hit.size < 1:
            continue
        runs = [r for r in np.split(hit, np.where(np.diff(hit) > 3)[0] + 1) if r.size >= 1]
        if runs:
            cols.append(x)
            cands.append([(float(r[0]), float(r[-1]), float(np.median(r))) for r in runs])
    if len(cols) < 5:
        return np.empty(0), np.empty(0)

    SKIP, UP, GAP, DRIFT, GAIN, MAX_JUMP = 0.5, 6.0, 1.0, 0.75, 1.0, 15.0
    # state: (cost, lo, hi, x_last, back_pointer); lo is None for the "unstarted" state.
    # Starting and ending are free: the unstarted state costs nothing in every column and
    # the path is read back from the best real state of any column, so a junk pixel in the
    # dense bundle near Sa = 0 cannot anchor the whole curve. Each accepted run earns
    # GAIN while a skipped column costs only SKIP, so a curve briefly hidden behind another
    # one is bridged instead of restarted; jumps and upward moves still pay per pixel.
    UNSTARTED = (0.0, None, None, None, -1)
    prev = [UNSTARTED]
    history = []
    best_end = (np.inf, -1, -1)                                 # (cost, column t, state i)
    for t, (c, x) in enumerate(zip(cands, cols)):
        cur = []
        for lo, hi, _ in c:
            best = (np.inf, -1)
            for b, (cost, plo, phi, px, _) in enumerate(prev):
                if plo is None:
                    tr = 0.0
                else:
                    # skipped columns may drift a little, never by more than a jump
                    allow = min(DRIFT * max(0, x - px - 1), MAX_JUMP)
                    gap = max(0.0, lo - phi - allow, plo - hi)
                    up = max(0.0, plo - hi - 2.0)
                    # curves are continuous (steep parts give overlapping runs, gap 0):
                    # a big jump is another curve, e.g. thin magenta over thick PGA red
                    tr = np.inf if gap > MAX_JUMP else GAP * gap + UP * up
                if cost + tr < best[0]:
                    best = (cost + tr, b)
            cost = best[0] - GAIN - 0.1 * min(hi - lo + 1, 10)
            cur.append((cost, lo, hi, x, best[1]))
            if cost < best_end[0]:
                best_end = (cost, t, len(cur) - 1)
        started = [j for j, st in enumerate(prev) if st[1] is not None]
        if started:
            b = min(started, key=lambda j: prev[j][0])
            cur.append((prev[b][0] + SKIP, prev[b][1], prev[b][2], prev[b][3], b))
        cur.append(UNSTARTED)
        history.append(cur)
        prev = cur
    xs, ys = [], []
    _, t, i = best_end
    while t >= 0 and i >= 0:
        state = history[t][i]
        if state[1] is None:
            break
        if i < len(cands[t]):
            xs.append(cols[t])
            ys.append(cands[t][i][2])
        i = state[4]
        t -= 1
    xs, ys = np.asarray(xs[::-1], float), np.asarray(ys[::-1], float)
    if xs.size < 5:
        return xs, ys
    med = pd.Series(ys).rolling(11, center=True, min_periods=1).median().to_numpy()
    keep = np.abs(ys - med) <= 5.0
    xs, ys = _drop_violators(xs[keep], ys[keep])
    return xs, _isotonic_increasing(ys)


def _leftover(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, band: float = 4.0) -> np.ndarray:
    """Pixels of ``mask`` that are NOT on the traced curve (xs, ys) of their own label."""
    out = mask.copy()
    if xs.size == 0:
        return out
    rows = np.arange(mask.shape[0])[:, None]
    yline = np.interp(np.arange(mask.shape[1]), xs, ys, left=np.nan, right=np.nan)
    near = np.abs(rows - yline[None, :]) <= band
    out[near] = False
    return out


def _runs(mask: np.ndarray, x: int) -> list[float]:
    rows = _column_rows(x)
    hit = rows[mask[rows, x]]
    if hit.size < 1:
        return []
    return [float(np.median(r)) for r in np.split(hit, np.where(np.diff(hit) > 3)[0] + 1)
            if r.size >= 1]


def fill_gaps(xs: np.ndarray, ys: np.ndarray, blend: np.ndarray,
              max_extend: int = X1 - X0) -> tuple[np.ndarray, np.ndarray]:
    """Fill hidden stretches of a pure-ink trace with blend pixels that continue it.

    A blend pixel {k, j} means curves k and j coincide there. It is accepted for curve k
    only if it lies on k's own trace: within a few px of the linear interpolation across
    an interior gap, or of the local extrapolation when marching past either end.
    """
    pts = dict(zip(xs.astype(int).tolist(), ys.tolist()))
    xi = sorted(pts)
    for a, b in zip(xi[:-1], xi[1:]):                       # interior gaps
        for x in range(a + 1, b):
            yexp = pts[a] + (pts[b] - pts[a]) * (x - a) / (b - a)
            c = [y for y in _runs(blend, x) if abs(y - yexp) <= 4.0]
            if c:
                pts[x] = min(c, key=lambda y: abs(y - yexp))
    for step in (1, -1):                                    # march past the ends
        xi = sorted(pts)
        x0 = xi[-1] if step == 1 else xi[0]
        miss = 0
        x = x0
        while miss < 15 and abs(x - x0) < max_extend and X0 + INSET <= x + step <= X1 - INSET:
            x += step
            near = sorted(pts, key=lambda q: abs(q - x))[:8]
            if len(near) >= 2:
                slope = np.polyfit(near, [pts[q] for q in near], 1)[0]
            else:
                slope = 0.0
            last = max(near) if step == 1 else min(near)
            yexp = pts[last] + slope * (x - last)
            c = [y for y in _runs(blend, x) if abs(y - yexp) <= 4.0 + 0.1 * abs(x - last)]
            if c:
                pts[x] = min(c, key=lambda y: abs(y - yexp))
                miss = 0
            else:
                miss += 1
    xi = np.array(sorted(pts), float)
    xi, yi = _drop_violators(xi, np.array([pts[int(q)] for q in xi]))
    return xi, _isotonic_increasing(yi)


def px_to_rate(y: np.ndarray) -> np.ndarray:
    return 10.0 ** (LOG_TOP + (y - Y0) / (Y1 - Y0) * (LOG_BOT - LOG_TOP))


def px_to_frac(x: np.ndarray) -> np.ndarray:
    return (x - X0) / (X1 - X0)


def rate_at(frac: np.ndarray, logr: np.ndarray, sa_frac: float) -> float:
    """log10 rate interpolated at a fractional x position (nan outside the traced span)."""
    if frac.size < 2 or sa_frac < frac[0] or sa_frac > frac[-1]:
        return np.nan
    return float(np.interp(sa_frac, frac, logr))


def calibrate_xmax(curves: dict[int, tuple[np.ndarray, np.ndarray]],
                   uhs: pd.DataFrame) -> tuple[float, float, int]:
    """Pick xmax minimizing the median |misfit| of lambda(Sa_TR) to 1/TR over all anchors.

    The median keeps one badly traced curve from dragging the axis calibration.
    """
    best = (np.nan, np.inf, 0)
    for xmax in XMAX_CANDIDATES:
        errs = []
        for k, (frac, logr) in curves.items():
            for tr in (475, 2475):
                sa = uhs.loc[(tr, PERIODS[k])]
                lr = rate_at(frac, logr, sa / xmax)
                if np.isfinite(lr):
                    errs.append(lr - np.log10(1.0 / tr))
        if len(errs) < 6:
            continue
        rms = float(np.median(np.abs(errs)))
        if rms < best[1]:
            best = (xmax, rms, len(errs))
    return best


def digitize(path: Path, uhs: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], float]:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    pure, blend = classify(rgb)
    curves = {}
    traced: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(len(PERIODS)):
        mask = pure[k]
        if k != PGA and PGA in traced:
            # the thick red line's JPEG edges unmix as magenta-ish ink: its band is not
            # evidence for other curves (a curve really on top of PGA shows up as blend)
            mask = _leftover(mask, *traced[PGA], band=5.0)
        if k in CONFUSED_WITH and CONFUSED_WITH[k] in traced:
            mask = mask | _leftover(pure[CONFUSED_WITH[k]], *traced[CONFUSED_WITH[k]])
        x, y = trace(mask)
        if x.size < MIN_PURE_POINTS:         # (almost) fully coincident with another curve
            x, y = trace(pure[k] | blend[k])
        elif blend[k].any():
            x, y = fill_gaps(x, y, blend[k])
        traced[k] = (x, y)
        if x.size >= 5:
            curves[k] = (px_to_frac(x), np.log10(px_to_rate(y)))
    xmax, rms, n = calibrate_xmax(curves, uhs)
    extended = extend_coincident(curves, uhs, xmax)
    cid = path.name.split("_")[0]
    rows, qc = [], []
    for k, (frac, logr) in curves.items():
        sa = frac * xmax
        # median-5 smoothing in px columns, then 40 log-spaced Sa points per curve: the
        # pixel resolution (~0.004 g, ~0.012 decades) does not support a denser table
        lr = pd.Series(logr).rolling(5, center=True, min_periods=1).median().to_numpy()
        grid = np.geomspace(sa.min(), sa.max(), N_OUT)
        lr_g = np.interp(np.log(grid), np.log(sa), lr)
        for s_, r_ in zip(grid, lr_g):
            rows.append({"cell_id": cid, "period": PERIODS[k], "sa_g": round(float(s_), 5),
                         "rate": float(f"{10 ** r_:.4g}")})
        res = {}
        for tr in (475, 2475):
            v = rate_at(frac, logr, uhs.loc[(tr, PERIODS[k])] / xmax)
            res[tr] = v - np.log10(1.0 / tr) if np.isfinite(v) else np.nan
        qc.append({"cell_id": cid, "period": PERIODS[k], "n_points": int(frac.size),
                   "sa_min": float(sa.min()), "sa_max": float(sa.max()),
                   "dlog10_rate_475": res[475], "dlog10_rate_2475": res[2475],
                   "coincident_with": PERIODS[extended[k]] if k in extended else np.nan})
    for k in set(range(len(PERIODS))) - set(curves):
        qc.append({"cell_id": cid, "period": PERIODS[k], "n_points": 0})
    for q in qc:
        q.update(xmax=xmax, site_rms=rms)
    return pd.DataFrame(rows), qc, xmax


def extend_coincident(curves: dict[int, tuple[np.ndarray, np.ndarray]], uhs: pd.Series,
                      xmax: float, tol_px: float = 10.0, tol_anchor: float = 0.05
                      ) -> dict[int, int]:
    """Extend curves that end ON another curve along that curve, if anchors confirm it.

    A thin line drawn exactly over another one (0.50 s over PGA, 0.07 s under 0.20 s)
    leaves no pixels of its own after JPEG. If curve k's trace ends within ``tol_px`` of
    (the two separate gradually, so the visible end is already a few px off)
    curve j, the hidden stretch is taken from j, but only when at least one of k's own
    published UHS anchors (1/475, 1/2475) falls in that stretch and lies on j within
    ``tol_anchor`` decades. Otherwise the curve is left truncated. Returns {k: j} for the
    curves that were extended (recorded in the QC table).
    """
    tol = tol_px / (Y1 - Y0) * (LOG_TOP - LOG_BOT)        # px -> decades
    used: dict[int, int] = {}
    for k in list(curves):
        for side in ("left", "right"):
            fk, lk = curves[k]
            xe, ye = (fk[0], lk[0]) if side == "left" else (fk[-1], lk[-1])
            for j, (fj, lj) in curves.items():
                if j == k or not (fj[0] <= xe <= fj[-1]):
                    continue
                if abs(np.interp(xe, fj, lj) - ye) > tol:
                    continue
                part = fj < xe if side == "left" else fj > xe
                if part.sum() < 5:
                    continue
                lo, hi = fj[part].min(), fj[part].max()
                anchors = [(uhs.loc[(tr, PERIODS[k])] / xmax, np.log10(1.0 / tr))
                           for tr in (475, 2475)]
                inside = [(a, r) for a, r in anchors if lo <= a <= hi]
                if not inside or any(abs(np.interp(a, fj, lj) - r) > tol_anchor
                                     for a, r in inside):
                    continue
                f_new = np.r_[fj[part], fk] if side == "left" else np.r_[fk, fj[part]]
                l_new = np.r_[lj[part], lk] if side == "left" else np.r_[lk, lj[part]]
                curves[k] = (f_new, np.minimum.accumulate(l_new))   # rate non-increasing
                used[k] = j
                break
    return used


def _digitize_job(job):
    return digitize(*job)


def load_uhs() -> pd.DataFrame:
    g = pd.read_csv(OUT / "hazard_grid.csv.gz")
    g = g[g.stat == "mean"]
    long = g.melt(id_vars=["cell_id", "tr"], value_vars=[f"T{T:.2f}" for T in PERIODS],
                  var_name="period", value_name="sa")
    long["period"] = long["period"].str[1:].astype(float)
    return long.set_index(["cell_id", "tr", "period"])["sa"].sort_index()


def plot_check(cell_id: str, curves: pd.DataFrame, uhs: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.4))
    ax0.imshow(Image.open(RAW / f"{cell_id}_HC.jpg"))
    ax0.set_axis_off()
    ax0.set_title("IG-EPN original")
    colors = ["red", "limegreen", "m", "k", "c", "navy", "gold", "steelblue"]
    for k, T in enumerate(PERIODS):
        c = curves[curves.period == T]
        if len(c):
            ax1.semilogy(c.sa_g, c.rate, color=colors[k], lw=2.5 if k == 0 else 1.2,
                         label="PGA" if T == 0 else f"{T:.2f} s")
        for tr, mk in ((475, "o"), (2475, "s")):
            ax1.semilogy(uhs.loc[(cell_id, tr, T)], 1 / tr, mk, mfc="none", color=colors[k])
    ax1.set_ylim(1e-6, 1)
    ax1.set_xlim(0, None)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlabel("Sa [g]")
    ax1.set_ylabel("annual rate of exceedance [1/yr]")
    ax1.set_title(f"digitized {cell_id}  (markers: published mean UHS at 475 / 2475)")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", nargs="*", default=[], help="cell ids to render check plots")
    ap.add_argument("--only", nargs="*", default=[], help="digitize only these cell ids")
    ap.add_argument("--plot-dir", default=str(ROOT / "data-raw" / "igepn" / "check"))
    args = ap.parse_args()
    uhs = load_uhs()
    files = sorted(RAW.glob("*_HC.jpg"))
    if args.only:
        files = [f for f in files if f.name.split("_")[0] in args.only]
    jobs = [(p, uhs.loc[p.name.split("_")[0]]) for p in files]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(_digitize_job, jobs, chunksize=4))
    frames = [r[0] for r in results]
    qcs = [q for r in results for q in r[1]]
    curves = pd.concat(frames, ignore_index=True)
    qc = pd.DataFrame(qcs).sort_values(["cell_id", "period"])
    if not args.only:
        curves.to_csv(OUT / "hazard_curves_capitals.csv.gz", index=False)
        qc.to_csv(OUT / "hazard_curves_capitals_qc.csv", index=False, float_format="%.4f")
    _report(curves, qc, files)
    if args.plot:
        d = Path(args.plot_dir)
        d.mkdir(parents=True, exist_ok=True)
        for cid in args.plot:
            plot_check(cid, curves[curves.cell_id == cid], uhs, d / f"{cid}.png")


def _report(curves: pd.DataFrame, qc: pd.DataFrame, files: list) -> None:
    a = qc[["dlog10_rate_475", "dlog10_rate_2475"]].abs().stack()
    print(f"{len(files)} images, {curves.groupby(['cell_id', 'period']).ngroups} curves")
    print("xmax used:", qc.drop_duplicates("cell_id").xmax.value_counts().to_dict())
    print(f"|dlog10 rate| at UHS anchors: median {a.median():.3f}, p95 {a.quantile(.95):.3f}, "
          f"max {a.max():.3f}  (0.043 = 10% in rate)")
    print("missing curves:", int((qc.n_points == 0).sum()))


if __name__ == "__main__":
    main()
