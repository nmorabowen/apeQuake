"""Design tokens for the hazard plots and maps (light / dark).

Every color is a documented step of the reference data-viz palette; the sets below were
checked with its validator (``validate_palette.js``) rather than picked by eye:

* **Hazard (sequential)**: one hue, blue 100 -> 700. Low hazard is the light end on the
  light surface; the anchor flips on the dark surface so low values recede into it.
* **Ordered series (ordinal)**: return periods and spectral periods are ordered, so they
  take one-hue blue steps, never categorical hues. Sets of 1-5 pass the ordinal checks
  (monotone L, adjacent dL >= 0.06, light end >= 2:1) in both modes; more than five
  cannot pass on one hue, so those plots add direct labels as secondary encoding.
* **Source-zone types (categorical)**: orange / aqua / violet, the only identity hues on
  a map. Blue is reserved for the hazard ramp. All-pairs CVD dE 9.2 light / 9.4 dark;
  aqua is under 3:1 on the light surface, so zones are always named in the legend,
  popups and ``EcuadorHazard.sources()`` (relief rule). Background zones are neutral.
* Faults, epicenters and sites are **ink** with a **surface halo**, so they read on any
  cell color; catalogs differ by fill and outline, not by hue (a map carries at most
  three identity hues).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
        350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
        600: "#184f95", 650: "#104281", 700: "#0d366b"}

# validated ordinal sets, listed low -> high value
_ORDINAL = {
    "light": {1: [450], 2: [250, 550], 3: [250, 450, 650], 4: [250, 400, 550, 700],
              5: [250, 350, 450, 550, 700]},
    "dark": {1: [400], 2: [500, 100], 3: [600, 350, 100], 4: [600, 400, 250, 100],
             5: [600, 500, 350, 200, 100]},
}


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    secondary: str
    muted: str
    grid: str
    baseline: str
    zones: dict
    background_zone: str


LIGHT = Theme("light", surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", secondary="#52514e",
              muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
              zones={"crustal": "#eb6834", "interface": "#1baf7a", "inslab": "#4a3aa7"},
              background_zone="#898781")
DARK = Theme("dark", surface="#1a1a19", page="#0d0d0d", ink="#ffffff", secondary="#c3c2b7",
             muted="#898781", grid="#2c2c2a", baseline="#383835",
             zones={"crustal": "#d95926", "interface": "#199e70", "inslab": "#9085e9"},
             background_zone="#898781")


def theme(name: str = "light") -> Theme:
    if name not in ("light", "dark"):
        raise ValueError("theme must be 'light' or 'dark'")
    return LIGHT if name == "light" else DARK


def hazard_steps(mode: str = "light") -> list[str]:
    """Sequential hazard ramp, low -> high value."""
    steps = [BLUE[k] for k in sorted(BLUE)]
    return steps if mode == "light" else steps[::-1]


def hazard_cmap(mode: str = "light"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"igepn_hazard_{mode}", hazard_steps(mode))


def ordinal(n: int, mode: str = "light") -> tuple[list[str], bool]:
    """``n`` ordered colors (low -> high) and whether direct labels are required.

    Up to five steps come from validated sets. Beyond five no one-hue ramp keeps the
    adjacent lightness gap, so the steps are spread over the full usable range and the
    caller must label the series directly (secondary encoding).
    """
    sets = _ORDINAL[mode]
    if n <= 5:
        return [BLUE[k] for k in sets[max(n, 1)]][:max(n, 1)], False
    lo, hi = (250, 700) if mode == "light" else (600, 100)
    keys = sorted(BLUE)
    usable = [k for k in keys if min(lo, hi) <= k <= max(lo, hi)]
    if mode == "dark":
        usable = usable[::-1]
    idx = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    return [BLUE[usable[i]] for i in idx], True


def style_axes(ax, t: Theme) -> None:
    """Recessive chrome: hairline solid grid, baseline-colored spines, muted ticks."""
    ax.set_facecolor(t.surface)
    ax.figure.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.baseline)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t.muted, labelcolor=t.secondary, labelsize=8, length=3, width=0.6)
    ax.grid(True, which="major", color=t.grid, linewidth=0.6, linestyle="-")
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t.secondary)
    ax.yaxis.label.set_color(t.secondary)


def title(ax, t: Theme, main: str, sub: str | None = None) -> None:
    """Left-aligned title in ink with an optional secondary subtitle."""
    ax.set_title(main, loc="left", fontsize=11, color=t.ink, fontweight="semibold",
                 pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=t.secondary,
                va="bottom", ha="left")


def halo(t: Theme, width: float = 2.0):
    """Surface-colored ring under a mark so it reads over any fill."""
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=width, foreground=t.surface), pe.Normal()]


def end_labels(ax, t: Theme, ends: list[tuple[float, float, str]], gap_px: float = 11.0,
               dx_px: float = 18.0) -> None:
    """Direct labels at line ends, spread vertically with hairline leaders.

    Converging lines would stack their labels on top of each other; nudging labels away
    without a connector detaches them from their lines, so each label keeps a thin
    leader back to its end point.
    """
    if not ends:
        return
    ax.figure.canvas.draw()                       # transforms need a laid-out figure
    to_px = ax.transData.transform
    from_px = ax.transData.inverted().transform
    pts = sorted(((to_px((x, y)), lab) for x, y, lab in ends), key=lambda p: p[0][1])
    ys = [p[0][1] for p in pts]
    for i in range(1, len(ys)):                   # push up until every gap >= gap_px
        ys[i] = max(ys[i], ys[i - 1] + gap_px)
    shift = (np.mean([p[0][1] for p in pts]) - np.mean(ys))
    ys = [y + shift for y in ys]                  # re-centre the stack on the ends
    x_lab = max(p[0][0] for p in pts) + dx_px
    for (xy, lab), y in zip(pts, ys):
        tx, ty = from_px((x_lab, y))
        ex, ey = from_px(xy)
        ax.annotate(lab, (ex, ey), xytext=(tx, ty), textcoords="data", va="center",
                    fontsize=7.5, color=t.secondary, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=t.baseline, lw=0.6,
                                    shrinkA=0, shrinkB=3))


def legend(ax, t: Theme, **kw):
    lg = ax.legend(frameon=True, fontsize=8, labelcolor=t.secondary, **kw)
    fr = lg.get_frame()
    fr.set_facecolor(t.surface)
    fr.set_edgecolor(t.grid)
    fr.set_linewidth(0.6)
    fr.set_alpha(0.92)
    return lg
