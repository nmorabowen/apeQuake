"""assess_site: NEC-SE-DS vs ASCE 7-16 / 7-22 vs IG-EPN at a point in Ecuador."""
from __future__ import annotations

import json

import numpy as np
import pytest

from apeQuake.hazard import EcuadorHazard
from apeQuake.site_assessment import (
    PERIODS,
    assess_site,
    site_class_asce7_22,
    site_class_nec,
)

QUITO = (-0.2200, -78.5125)          # Z = 0.40, Sierra (eta 2.48)
GUAYAQUIL = (-2.1894, -79.8891)      # Z = 0.40, Costa (eta 1.80)


@pytest.mark.parametrize("vs, cls", [(1600, "A"), (1500, "A"), (1000, "B"), (760, "B"),
                                     (759.9, "C"), (360, "C"), (250, "D"), (180, "D"),
                                     (179, "E")])
def test_site_class_nec(vs, cls):
    assert site_class_nec(vs) == cls


@pytest.mark.parametrize("vs, cls", [(1600, "A"), (1000, "B"), (700, "BC"), (500, "C"),
                                     (400, "CD"), (250, "D"), (200, "DE"), (150, "E")])
def test_site_class_asce7_22(vs, cls):
    assert site_class_asce7_22(vs) == cls


def test_nec_rule_for_ss_s1():
    r = assess_site(*QUITO, site_class="D")
    assert r.ss_s1["method"] == "nec"
    assert r.ss_s1["ss"] == pytest.approx(1.5 * 2.48 * 0.40)          # plateau of class B
    assert r.ss_s1["s1"] == pytest.approx(1.5 * r.ss_s1["nec_rule"]["sa_b_10"])
    # on rock, by construction, ASCE design = NEC 475 at 0.2 s and 1.0 s
    for row in r.comparison:
        assert row["rock"]["asce_design"] == pytest.approx(row["rock"]["nec_475"], rel=1e-9)


def test_igepn_rule_for_ss_s1():
    r = assess_site(*QUITO, site_class="D", method="igepn")
    pub = EcuadorHazard().site(*QUITO, interp="bilinear").published(2475, "mean")
    assert r.ss_s1["ss"] == pytest.approx(pub[4])     # T = 0.2 s
    assert r.ss_s1["s1"] == pytest.approx(pub[6])     # T = 1.0 s
    assert set(r.ss_s1["candidates"]) == {"nec", "igepn"}


def test_manual_matches_office_sheet():
    """Office sheet (Rumipamba): Ss = 2.1, S1 = 0.7, site D -> SDS 1.40, SD1 0.793."""
    r = assess_site(*QUITO, site_class="D", method="manual", ss=2.1, s1=0.7)
    p = r.asce7_16["parameters"]
    assert (p["Fa"], p["Fv"]) == pytest.approx((1.0, 1.7))
    assert p["SDS"] == pytest.approx(1.4)
    assert p["SD1"] == pytest.approx(2 / 3 * 1.7 * 0.7)


def test_asce7_16_trigger_reported_and_exception_applied():
    r = assess_site(*QUITO, site_class="D")
    assert "D_S1" in r.asce7_16["exceptions"]
    assert any("11.4.8" in w for w in r.warnings)


def test_asce7_22_is_approximate_from_7_16_factors():
    r = assess_site(*GUAYAQUIL, vs30=200)
    assert r.site_classes == {"nec": "D", "asce7_16": "D", "asce7_22": "DE"}
    p16, p22 = r.asce7_16["parameters"], r.asce7_22["parameters"]
    assert r.asce7_22["approximate"] is True
    assert p22["SDS"] == pytest.approx(p16["SDS"]) and p22["SD1"] == pytest.approx(p16["SD1"])
    assert p22["site_class"] == "DE"


def test_rock_site_class_b_equals_rock_view():
    r = assess_site(*QUITO, site_class="B")       # dropdown B: Fa = Fv = 1 (11.4.3)
    for row in r.comparison:
        assert row["site"]["nec"] == pytest.approx(row["rock"]["nec_475"])
        assert row["site"]["asce7_16"] == pytest.approx(row["rock"]["asce_design"])
        assert row["site"]["igepn_475_scaled"] == pytest.approx(row["rock"]["igepn_475"])


def test_site_class_f_skips_site_view():
    r = assess_site(*QUITO, site_class="F")
    assert r.site is None and r.asce7_16 is None and r.asce7_22 is None
    assert r.rock is not None and "site" not in r.comparison[0]
    assert any("class F" in w for w in r.warnings)


def test_overrides():
    r = assess_site(*QUITO, site_class="C", z=0.35, region="costa")
    assert r.zone["z_used"] == 0.35 and r.zone["eta_used"] == 1.80
    assert any("overridden" in w for w in r.warnings)


def test_spectra_shapes():
    r = assess_site(*QUITO, vs30=300)
    for curve in (r.nec["spectrum"], r.asce7_16["spectrum"], r.asce7_22["spectrum"]):
        assert np.allclose(curve["T"], PERIODS) and len(curve["Sa"]) == len(PERIODS)
        assert np.all(np.asarray(curve["Sa"]) > 0)
    assert len(r.igepn["uhs"]["475"]["mean"]) == len(r.igepn["periods"]) == 8
    assert r.nec_uhs["city"] and set(r.nec_uhs) >= {"475", "2500"}


def test_class_e_with_large_s1_has_no_asce_site_spectrum():
    # ASCE 7-16 Table 11.4-2: no Fv for class E with S1 > 0.1 ("See 11.4.8")
    r = assess_site(*GUAYAQUIL, site_class="E")
    assert r.asce7_16 is None and r.asce7_22 is None
    assert r.site["nec"] is not None and r.site["asce7_16"] is None
    assert r.comparison[0]["site"]["asce7_16"] is None
    assert any("not available" in w for w in r.warnings)


def test_to_dict_is_strict_json():
    d = assess_site(*GUAYAQUIL, site_class="E").to_dict()
    s = json.dumps(d, allow_nan=False)            # no NaN / inf leaks
    assert '"comparison"' in s


@pytest.mark.parametrize("kw", [dict(), dict(vs30=300, site_class="D"),
                                dict(site_class="G"), dict(site_class="D", method="manual"),
                                dict(site_class="D", method="usgs"), dict(vs30=-5)])
def test_bad_inputs(kw):
    with pytest.raises(ValueError):
        assess_site(*QUITO, **kw)


def test_outside_ecuador():
    with pytest.raises(ValueError):
        assess_site(-12.05, -77.04, site_class="D")
