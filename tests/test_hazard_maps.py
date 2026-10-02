"""Hazard maps: static overlays and the interactive explorer."""
import json
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from apeQuake.hazard import STATS, EcuadorHazard  # noqa: E402
from apeQuake.hazard.maps import explore_data, resolve_points  # noqa: E402


@pytest.fixture(scope="module")
def hz():
    return EcuadorHazard()


def _legend_labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


# ---------------------------------------------------------------- points

def test_resolve_points_accepts_every_form(hz):
    s = hz.site("Manta")
    df = pd.DataFrame({"lat": [-1.0], "lon": [-78.5], "label": ["from df"]})
    pts = resolve_points(hz, ["Quito", (-0.25, -78.45), (-0.25, -78.45, "Proyecto"), s])
    assert [p.label for p in pts][2:] == ["Proyecto", s.label]
    assert resolve_points(hz, df)[0].label == "from df"
    assert len(resolve_points(hz, (-0.25, -78.45))) == 1        # a single tuple
    assert resolve_points(hz, None) == []


# ---------------------------------------------------------------- static map

def test_plot_map_overlays(hz):
    ax = hz.plot_map(475, faults=True, sources=["crustal", "interface"],
                     catalog=["historical", "deep"], min_mw=6.5, capitals=True,
                     points=["Quito", (-0.95, -80.73, "Site A")])
    labels = _legend_labels(ax)
    n_hist = int((hz.catalog("historical").mw >= 6.5).sum())
    assert f"historical catalog ({n_hist})" in labels
    assert {"crustal source zones", "interface source zones", "faults", "sites"} <= set(labels)
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("Site A\n") and t.endswith(" g") for t in texts)


def test_plot_map_point_label_matches_site(hz):
    ax = hz.plot_map(975, period=0.2, points=[(-0.25, -78.45, "P")], legend=False)
    val = hz.site(-0.25, -78.45).uhs(975).Sa.iloc[4]
    assert ax.texts[0].get_text().endswith(f"{val:.2f} g")


def test_plot_map_extent_points(hz):
    ax = hz.plot_map(points=[(-0.2, -78.5), (-0.4, -78.3)], extent="points")
    x0, x1 = ax.get_xlim()
    assert -79.0 < x0 < -78.5 and -78.3 < x1 < -77.8
    with pytest.raises(ValueError):
        hz.plot_map(extent="points")


def test_plot_map_custom_catalog_and_errors(hz):
    ev = pd.DataFrame({"lat": [-1.0, -2.0], "lon": [-78.0, -79.0], "magnitude": [4.0, 6.0]})
    ax = hz.plot_map(catalog=ev, min_mw=5)
    assert "custom catalog (1)" in _legend_labels(ax)
    with pytest.raises(ValueError, match="unknown source"):
        hz.plot_map(sources="volcanic")


# ---------------------------------------------------------------- interactive map

def _payload(path):
    html = path.read_text(encoding="utf-8")
    m = re.search(r'<script id="hazard-data" type="application/json">(.*?)</script>', html,
                  re.S)
    return html, json.loads(m.group(1))


def test_explore_writes_standalone_page(hz, tmp_path):
    out = hz.explore(tmp_path / "sub" / "map.html", tr=975, period=0.2,
                     points=["Quito", (-0.25, -78.45, "Proyecto")])
    assert out.exists()                                    # parent folder created
    html, d = _payload(out)
    assert "/*__DATA__*/" not in html
    assert len(d["cells"]["features"]) == 3146
    assert d["init"] == {"tr": 975, "period": 0.2, "stat": "mean"}
    assert set(d["catalogs"]) == {"shallow", "deep", "historical"}
    assert len(d["faults"]["features"]) == 8 and len(d["sources"]["features"]) == 22
    assert [p["label"] for p in d["points"]][1] == "Proyecto"


def test_explore_value_layout_matches_published(hz):
    # the page indexes v[(t * n_stats + s) * n_periods + p]
    d = explore_data(hz)
    f = next(f for f in d["cells"]["features"] if f["properties"]["id"] == "S0.0478.52")
    v = np.asarray(f["properties"]["v"])
    site = hz.site("Quito")
    n_s, n_p = len(STATS), len(d["periods"])
    for t, tr in enumerate(d["trs"]):
        for s, stat in enumerate(STATS):
            np.testing.assert_allclose(v[(t * n_s + s) * n_p:(t * n_s + s + 1) * n_p],
                                       site.published(tr, stat), atol=1e-4)
    assert f["properties"]["cap"] is True


def test_explore_point_popups_use_site_uhs(hz):
    d = explore_data(hz, points=["Quito"], point_trs=(475, 975))
    p = d["points"][0]
    np.testing.assert_allclose(p["uhs"][1]["sa"], hz.site("Quito").uhs(975).Sa, atol=1e-4)
    assert p["uhs"][0]["source"] == "published"


def test_explore_escapes_script_breakout(hz, tmp_path):
    out = hz.explore(tmp_path / "x.html", points=[(-0.25, -78.45, "</script><b>x")],
                     catalogs=())
    html, d = _payload(out)
    assert html.count("</script>") == 3                    # leaflet.js, data, app code
    assert d["points"][0]["label"] == "</script><b>x"


# ---------------------------------------------------------------- design tokens

def test_style_uses_validated_steps_only():
    from apeQuake.hazard import _style

    documented = set(_style.BLUE.values())
    for mode in ("light", "dark"):
        for n in range(1, 6):
            cols, needs_labels = _style.ordinal(n, mode)
            assert len(cols) == n and not needs_labels and set(cols) <= documented
        cols, needs_labels = _style.ordinal(8, mode)
        assert len(set(cols)) == 8 and needs_labels      # beyond 5: direct labels required
    # dark mode flips the sequential anchor: low hazard recedes into the dark surface
    assert _style.hazard_steps("dark") == _style.hazard_steps("light")[::-1]
    assert _style.theme("dark").zones["inslab"] == "#9085e9"


def test_plots_render_in_both_themes(hz):
    import matplotlib.pyplot as plt

    q = hz.site("Quito")
    for theme in ("light", "dark"):
        ax = hz.plot_map(475, faults=True, sources=True, catalog="deep", min_mw=6,
                         points=["Quito"], theme=theme)
        assert ax.get_facecolor()[:3] != (1.0, 1.0, 1.0) or theme == "light"
        q.plot_uhs([225, 475, 975, 2475], theme=theme)
        q.plot_hazard_curves(list(q.periods), theme=theme)     # 8 periods: end labels
        plt.close("all")
    with pytest.raises(ValueError):
        hz.plot_map(theme="sepia")


def test_explore_page_has_both_themes_and_table_view(hz, tmp_path):
    html = hz.explore(tmp_path / "m.html", catalogs=()).read_text(encoding="utf-8")
    assert ':root[data-theme="dark"]' in html and "prefers-color-scheme: dark" in html
    assert 'id="csv"' in html and 'id="mw"' in html                  # table view, Mw filter
    for hexcode in ("#eb6834", "#1baf7a", "#4a3aa7", "#d95926", "#199e70", "#9085e9"):
        assert hexcode in html                                      # validated zone hues


# ---------------------------------------------------------------- self-contained page

def test_explore_embeds_leaflet_by_default(hz, tmp_path):
    html = hz.explore(tmp_path / "m.html", catalogs=()).read_text(encoding="utf-8")
    # works where external scripts are blocked (file previews, attachments, offline)
    assert not re.findall(r'<(?:script|link)[^>]+(?:src|href)="https?://', html)
    assert "Leaflet 1.9.4" in html and "data:image/png;base64," in html
    assert "if (!window.L)" in html                                 # graceful fallback


def test_explore_cdn_mode_has_integrity(hz, tmp_path):
    from apeQuake.hazard.maps import _LEAFLET_SRI

    html = hz.explore(tmp_path / "c.html", catalogs=(), inline_leaflet=False).read_text(
        encoding="utf-8")
    for sri in _LEAFLET_SRI.values():
        assert f'integrity="{sri}"' in html


def test_vendored_leaflet_matches_published_hashes():
    import base64
    import hashlib
    from importlib.resources import files

    from apeQuake.hazard.maps import _LEAFLET_SRI

    v = files("apeQuake.hazard").joinpath("templates", "vendor", "leaflet")
    for name, sri in _LEAFLET_SRI.items():
        digest = base64.b64encode(hashlib.sha256(v.joinpath(name).read_bytes()).digest())
        assert f"sha256-{digest.decode()}" == sri
    assert "BSD 2-Clause" in v.joinpath("LICENSE").read_text(encoding="utf-8")


# ---------------------------------------------------------------- pick, copy, underlay, 3D

def test_sites_and_uhs_at_keep_labels(hz):
    pts = [(-0.14557, -78.27484, "P1"), (-0.13184, -78.58521, "P2")]
    assert [s.label for s in hz.sites(pts)] == ["P1", "P2"]
    assert hz.uhs_at(pts, tr=2475).place.tolist() == ["P1", "P2"]


def test_outline_underlay_is_embedded_and_simplified(hz):
    from apeQuake.hazard import _data
    from apeQuake.hazard.maps import _outlines, _simplify

    o = _outlines()
    n_simple = sum(len(r) for f in o["features"] for poly in f["geometry"]["coordinates"]
                   for r in poly)
    n_full = sum(len(r) for f in _data.geojson("admin_provinces.geojson.gz")["features"]
                 for poly in (f["geometry"]["coordinates"]
                              if f["geometry"]["type"] == "MultiPolygon"
                              else [f["geometry"]["coordinates"]])
                 for r in poly[:1])
    assert len(o["features"]) == 25 and n_simple < 0.6 * n_full
    ring = [[0, 0], [1, 0.001], [2, 0], [2, 2], [0, 2], [0, 0]]
    out = _simplify(ring, 0.01)
    assert out[0] == out[-1] and [1, 0.001] not in out          # closed, collinear point gone
    assert len(explore_data(hz, catalogs=())["provinces"]["features"]) == 25


def test_explore_page_has_pick_copy_and_3d(hz, tmp_path):
    html = hz.explore(tmp_path / "p.html", catalogs=()).read_text(encoding="utf-8")
    for needle in ('id="pick"', 'id="iso"', 'root.id = "isoview"', 'data-cp=', 'id="pcopy"',
                   "Province outlines", "tileerror", "parseHash"):
        assert needle in html, needle
    # pins must be restored before the first refresh() saves state
    assert html.index("restorePicks();   //") < html.rindex("applyTheme();")


# ---------------------------------------------------------------- static isometric view

def test_plot_iso_returns_axes3d_with_site(hz):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ax = hz.plot_iso(points=[(-0.94168, -80.73405, "Urban Tower")])
    texts = [t.get_text() for t in ax.texts]
    assert isinstance(ax, Axes3D)
    assert "Urban Tower" in texts and "PGA, TR = 475 yr (mean)" in texts
    assert any(ln.get_marker() == "x" for ln in ax.lines)
    plt.close(ax.figure)


def test_plot_iso_dark_odd_tr_and_point_forms(hz):
    import matplotlib.pyplot as plt

    ax = hz.plot_iso(1000, 0.2, theme="dark", points=["Quito", hz.site("Manta")],
                     provinces=False, elev=25, azim=-120)
    assert "Sa(0.2 s), TR = 1000 yr (mean)" in [t.get_text() for t in ax.texts]
    assert ax.figure.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    plt.close(ax.figure)
    fig = plt.figure()
    with pytest.raises(ValueError):
        hz.plot_iso(ax=fig.add_subplot())                      # not a 3D axes
    plt.close(fig)


def test_plot_iso_periods_panels(hz):
    import matplotlib.pyplot as plt

    fig = hz.plot_iso_periods((0.0, 1.0), colorbar="shared", points=[(-0.25, -78.45)])
    assert sum(a.name == "3d" for a in fig.axes) == 2 and len(fig.axes) == 3
    plt.close(fig)
    with pytest.raises(ValueError):
        hz.plot_iso_periods(colorbar="none")


@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg", ".PNG"])
def test_export_iso_writes_file(hz, tmp_path, ext):
    out = hz.export_iso(tmp_path / "sub" / f"iso{ext}", 475, 0.2, dpi=60,
                        points=[(-0.94168, -80.73405, "Urban Tower")])
    assert out.exists() and out.stat().st_size > 1000 and out.suffix == ext
    if ext.lower() == ".png":
        assert out.read_bytes()[1:4] == b"PNG"


def test_export_iso_periods_and_bad_extension(hz, tmp_path):
    out = hz.export_iso_periods(tmp_path / "p.png", (0.0, 0.2), dpi=50)
    assert out.exists() and out.stat().st_size > 1000
    for fn in (hz.export_iso, hz.export_iso_periods):
        with pytest.raises(ValueError, match="extension"):
            fn(tmp_path / "bad.gif")
    assert not (tmp_path / "bad.gif").exists()
