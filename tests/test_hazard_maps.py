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
