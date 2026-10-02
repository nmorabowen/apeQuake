"""SVG figures of the site report (matplotlib, text as paths so Typst needs no fonts)."""
from __future__ import annotations

import io
from typing import Any

import numpy as np

from ..hazard import _data
from ..nec.zoning import _zone_map

# Zone colours, close to NEC-SE-DS Figura 1 (0.15 dark green ... >= 0.50 red)
ZONE_COLORS = ("#2a8c3c", "#7cbf55", "#e2e69a", "#f3cf3d", "#ef8a46", "#de3c32")
NEC, A16, A22, IG, IGX, NECUHS = "#c0392b", "#2c3e50", "#8e44ad", "#16a085", "#16a085", "#e67e22"


def _svg(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _new(figsize):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "path", "font.size": 9, "axes.titlesize": 9})
    return plt.subplots(figsize=figsize)


def location_map(d: dict[str, Any]) -> str:
    """NEC zone map with the provinces and the site."""
    from matplotlib.colors import ListedColormap

    zone, lon, lat, zv = _zone_map()
    fig, ax = _new((5.2, 5.0))
    z = np.ma.masked_equal(zone, 255)
    ext = (lon[0] - (lon[1] - lon[0]) / 2, lon[-1] + (lon[1] - lon[0]) / 2,
           lat[-1] + (lat[0] - lat[1]) / 2, lat[0] - (lat[0] - lat[1]) / 2)
    ax.imshow(z, extent=ext, cmap=ListedColormap(ZONE_COLORS), vmin=-0.5, vmax=5.5,
              interpolation="nearest", aspect="equal")
    for f in _data.geojson("admin_provinces.geojson.gz")["features"]:
        g = f["geometry"]
        for poly in g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]:
            ring = np.asarray(poly[0])
            ax.plot(ring[:, 0], ring[:, 1], color="#555", lw=0.35)
    la, lo = d["inputs"]["lat"], d["inputs"]["lon"]
    ax.plot(lo, la, marker="*", ms=14, mfc="white", mec="black", mew=1.0, zorder=5)
    ax.set_xlim(-81.2, -75.1)
    ax.set_ylim(-5.1, 1.6)
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    import matplotlib.patches as mpatches

    ax.legend(handles=[mpatches.Patch(color=c, label=f"Z = {v:.2f} [g]" if v < 0.5 else "Z ≥ 0.50 [g]")
                       for c, v in zip(ZONE_COLORS, zv)], loc="lower left", fontsize=7,
              frameon=True)
    ax.grid(alpha=0.25, lw=0.4)
    return _svg(fig)


def _plot(ax, c, **kw):
    ax.plot(c["T"], c["Sa"], **kw)


def rock_spectra(d: dict[str, Any]) -> str:
    r = d["rock"]
    fig, ax = _new((6.4, 3.8))
    _plot(ax, r["nec_475"], color=NEC, lw=2, label="NEC-SE-DS 2015, perfil B")
    _plot(ax, r["asce_design"], color=A16, lw=1.6, ls="--", label="ASCE/SEI 7, roca de referencia")
    _plot(ax, r["igepn_475"], color=IG, marker="o", ms=4, label="IG-EPN, 475 años")
    _plot(ax, r["igepn_2475_x2_3"], color=IGX, marker="s", ms=4, ls=":",
          label="IG-EPN, 2/3 × 2475 años")
    _plot(ax, r["nec_uhs_475"], color=NECUHS, marker="^", ms=5, ls="none",
          label=f"NEC-SE-DS 2015, curvas de peligro ({d['nec_uhs']['city']}), 475 años")
    return _finish(fig, ax)


def site_spectra(d: dict[str, Any]) -> str:
    s, cls = d["site"], d["site_classes"]
    fig, ax = _new((6.4, 3.8))
    _plot(ax, s["nec"], color=NEC, lw=2, label=f"NEC-SE-DS 2015, perfil {cls['nec']}")
    if s["asce7_16"] is not None:
        _plot(ax, s["asce7_16"], color=A16, lw=1.6, ls="--",
              label=f"ASCE/SEI 7-16, clase {cls['asce7_16']}")
    if s["asce7_22"] is not None:
        _plot(ax, s["asce7_22"], color=A22, lw=1.6, ls="-.",
              label=f"ASCE/SEI 7-22, clase {cls['asce7_22']} (aproximado)")
    _plot(ax, s["igepn_475_scaled"], color=IG, marker="o", ms=4,
          label="IG-EPN 475 años × amplificación NEC (aproximado)")
    return _finish(fig, ax)


def _finish(fig, ax) -> str:
    ax.set_xlim(0, 3.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Período T [s]")
    ax.set_ylabel("Sa [g]")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="upper right")
    return _svg(fig)
