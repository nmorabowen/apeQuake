"""apeQuake/1 JSON API: dispatch contract and schema conformance.

The schemas are the contract; the runtime checks must agree with them:
valid fixtures pass both, shape errors are rejected by both (``bad_request``),
domain errors pass the schema and are refused by the domain.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from apeQuake.api import API, API_VERSION, ERROR_CODES, commands, dispatch, schema, schema_names

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

QUITO = {"lat": -0.22, "lon": -78.51}
ASCE16 = {"code": "ASCE7-16", "ss": 1.5, "s1": 0.6, "siteClass": "C", "tl": 8}
MPRS = {"code": "ASCE7-22-MPRS", "siteClass": "D", "tl": 8,
        "saMcer": [0.6, 0.6, 0.62, 0.7, 0.9, 1.1, 1.3, 1.45, 1.5, 1.5, 1.45, 1.35, 1.2,
                   0.95, 0.8, 0.6, 0.48, 0.32, 0.23, 0.18, 0.11, 0.07]}
NEC = {"z": 0.4, "siteClass": "D", "region": "sierra"}
ELF_ASCE = {"code": "asce7", "w": 1000, "r": 8, "sds": 1.0, "sd1": 0.6, "s1": 0.6, "tl": 8}
ELF_NEC = {"code": "nec", "w": 1000, "r": 8, **NEC}
STORIES = [{"weight": 400, "height": 3.5}, {"weight": 400, "height": 7.0},
           {"weight": 200, "height": 10.5}]


def _has_report_toolchain() -> bool:
    import os
    import shutil
    from pathlib import Path

    pkg = Path(os.environ.get("APPDATA", "")) / "typst" / "packages" / "local" / "ape-informes" / "0.1.0"
    return shutil.which("typst") is not None and (pkg / "images" / "APE_LOGO.png").is_file()


needs_typst = pytest.mark.skipif(not _has_report_toolchain(),
                                 reason="typst CLI or @local/ape-informes not installed")


@pytest.fixture(scope="module")
def registry():
    from referencing.jsonschema import DRAFT202012

    resources = [(schema(n)["$id"], DRAFT202012.create_resource(schema(n))) for n in schema_names()]
    return referencing.Registry().with_resources(resources)


def validator(name, registry):
    return jsonschema.Draft202012Validator(schema(name), registry=registry)


def req(command, args=None):
    r = {"api": API, "command": command}
    if args is not None:
        r["args"] = args
    return r


# --------------------------------------------------------------------- schemas
def test_every_schema_is_valid_draft_2020_12():
    for name in schema_names():
        s = schema(name)
        jsonschema.Draft202012Validator.check_schema(s)
        assert s["$id"].endswith(f"/{name}.json")


def test_refs_name_their_file():
    """Every $ref names its file (``common.json#/...``), even inside common.json.

    A bare ``#/$defs/x`` in common.json is valid JSON Schema, but the TypeScript
    generator in ape-tools (json-schema-ref-parser) resolves it against the schema
    that pulled common.json in. apeLoads writes its refs the same way.
    """
    for name in schema_names():
        text = json.dumps(schema(name))
        assert '"$ref": "#' not in text, name


def test_schema_names_cover_commands():
    for cmd in commands():
        assert f"{cmd}.args" in schema_names() and f"{cmd}.result" in schema_names()
    with pytest.raises(KeyError):
        schema("../../secrets")


def test_describe(registry):
    r = dispatch(req("api.describe"))
    assert r["ok"] and r["result"]["version"] == API_VERSION
    assert [c["name"] for c in r["result"]["commands"]] == list(commands())
    validator("api.describe.result", registry).validate(r["result"])
    validator("envelope.response", registry).validate(r)


# --------------------------------------------------------------- valid fixtures
VALID = [
    ("zoning.at", QUITO),
    ("zoning.at", {"lat": -0.7432, "lon": -90.3135}),                         # Galapagos
    ("site.assess", {**QUITO, "siteClass": "D"}),
    ("site.assess", {**QUITO, "vs30": 250, "method": "igepn"}),
    ("site.assess", {**QUITO, "siteClass": "D", "method": "manual", "ss": 2.1, "s1": 0.7}),
    ("site.assess", {"lat": -2.19, "lon": -79.89, "siteClass": "E"}),         # no ASCE site
    ("site.assess", {**QUITO, "siteClass": "F"}),                             # rock only
    ("site.assess", {**QUITO, "siteClass": "C", "z": 0.35, "region": "costa", "tlAsce": 4}),
    ("map.layer", {"layer": "necZones"}),
    ("map.layer", {"layer": "igepnHazard"}),
    ("map.layer", {"layer": "igepnHazard", "tr": 2475, "period": 0.2, "stat": "q84"}),
    ("map.layer", {"layer": "faults"}),
    ("map.layer", {"layer": "sourceZones"}),
    ("map.layer", {"layer": "capitals"}),
    ("map.layer", {"layer": "provinces"}),
    pytest.param("report.build", {**QUITO, "siteClass": "D",
                                  "report": {"project": "Prueba", "siteName": "Quito",
                                             "authors": [{"name": "N. Mora"}]}},
                 marks=needs_typst),
    ("spectrum.code", ASCE16),
    ("spectrum.code", {**ASCE16, "code": "ASCE7-10", "level": "mcer", "tMax": 6, "n": 50}),
    ("spectrum.code", {"code": "ASCE7-22", "sms": 1.8, "sm1": 1.2, "siteClass": "CD", "tl": 8}),
    ("spectrum.code", MPRS),
    ("spectrum.code", {**MPRS, "sms": 1.4, "sm1": 0.9}),
    ("spectrum.code", {"code": "NEC-15", **NEC, "export": {"format": "etabs", "units": "m/s2"}}),
    ("spectrum.code", {**ASCE16, "export": {"format": "opensees-python"}}),
    ("spectrum.code", {**ASCE16, "export": {"format": "csv"}, "n": 10.0}),
    ("design.sdc", {"sds": 1.0, "sd1": 0.6, "s1": 0.8, "riskCategory": "IV"}),
    ("design.sdc", {"sds": 0.1, "sd1": 0.05}),
    ("design.elf", {**ELF_ASCE, "hn": 30, "structureType": "concrete_moment_frame"}),
    ("design.elf", {**ELF_ASCE, "edition": "7-22", "t": 1.2, "hn": 30, "stories": STORIES}),
    ("design.elf", {**ELF_ASCE, "edition": "7-10", "t": 0.5, "riskCategory": "III",
                    "applyUpperLimit": False}),
    ("design.elf", {**ELF_NEC, "hn": 10.5, "stories": STORIES, "importance": "esencial",
                    "phiP": 0.9, "phiE": 0.9}),
    ("design.elf", {**ELF_NEC, "t": 0.8, "structureType": "hormigon_con_muros"}),
]


@pytest.mark.parametrize("command, args", VALID)
def test_valid_requests(command, args, registry):
    validator(f"{command}.args", registry).validate(args)
    validator("envelope.request", registry).validate(req(command, args))
    r = dispatch(req(command, args))
    assert r["ok"], r
    validator(f"{command}.result", registry).validate(r["result"])
    validator("envelope.response", registry).validate(r)
    json.dumps(r, allow_nan=False)


def test_office_sheet_numbers_through_the_api():
    r = dispatch(req("site.assess", {**QUITO, "siteClass": "D", "method": "manual",
                                     "ss": 2.1, "s1": 0.7}))["result"]
    p = r["asce716"]["parameters"]
    assert (p["fa"], p["fv"], round(p["sds"], 4), round(p["sd1"], 4)) == (1.0, 1.7, 1.4, 0.7933)
    assert r["asce722"]["approximate"] is True


# ---------------------------------------------------------------- shape errors
BAD_ARGS = [
    ("zoning.at", {"lat": -0.22}),
    ("zoning.at", {**QUITO, "extra": 1}),
    ("zoning.at", {"lat": "0", "lon": -78.5}),
    ("zoning.at", {"lat": 95, "lon": -78.5}),
    ("zoning.at", {"lat": True, "lon": -78.5}),
    ("site.assess", QUITO),                                                   # no class
    ("site.assess", {**QUITO, "vs30": 250, "siteClass": "D"}),               # both
    ("site.assess", {**QUITO, "siteClass": "G"}),
    ("site.assess", {**QUITO, "vs30": 0}),
    ("site.assess", {**QUITO, "siteClass": "D", "method": "manual"}),       # ss/s1 missing
    ("site.assess", {**QUITO, "siteClass": "D", "method": "manual", "ss": 2.1}),
    ("site.assess", {**QUITO, "siteClass": "D", "ss": 2.1, "s1": 0.7}),     # not manual
    ("site.assess", {**QUITO, "siteClass": "D", "method": "usgs"}),
    ("site.assess", {**QUITO, "siteClass": "D", "z": 0.6}),
    ("site.assess", {**QUITO, "siteClass": "D", "region": "amazonia"}),
    ("site.assess", {**QUITO, "siteClass": "D", "tlAsce": -1}),
    ("site.assess", {**QUITO, "siteClass": "D", "siteclass": "D"}),
    ("api.describe", {"x": 1}),
    ("map.layer", {}),
    ("map.layer", {"layer": "roads"}),
    ("map.layer", {"layer": "faults", "tr": 475}),
    ("map.layer", {"layer": "igepnHazard", "tr": 975}),
    ("map.layer", {"layer": "igepnHazard", "period": 0.3}),
    ("map.layer", {"layer": "igepnHazard", "stat": "max"}),
    ("map.layer", {"layer": "igepnHazard", "tr": True}),
    ("report.build", {**QUITO}),
    ("report.build", {**QUITO, "siteClass": "D", "report": {"projet": "x"}}),
    ("report.build", {**QUITO, "siteClass": "D", "report": {"authors": [{"email": "a@b"}]}}),
    ("report.build", {**QUITO, "siteClass": "D", "report": {"project": ""}}),
    ("spectrum.code", {}),
    ("spectrum.code", {**ASCE16, "code": "ASCE7-05"}),
    ("spectrum.code", {k: v for k, v in ASCE16.items() if k != "tl"}),
    ("spectrum.code", {**ASCE16, "siteClass": "F"}),
    ("spectrum.code", {**ASCE16, "z": 0.4}),                                # other code's field
    ("spectrum.code", {**ASCE16, "level": "service"}),
    ("spectrum.code", {**ASCE16, "n": 1}),
    ("spectrum.code", {**ASCE16, "n": 2.5}),
    ("spectrum.code", {**ASCE16, "export": {"format": "xlsx"}}),
    ("spectrum.code", {**ASCE16, "export": {"format": "csv", "units": "ft/s2"}}),
    ("spectrum.code", {**MPRS, "saMcer": MPRS["saMcer"][:-1]}),
    ("spectrum.code", {**MPRS, "sms": 1.4}),
    ("spectrum.code", {"code": "ASCE7-22", "sms": 1.8, "sm1": 1.2, "siteClass": "F", "tl": 8}),
    ("spectrum.code", {"code": "NEC-15", **NEC, "region": "amazonia"}),
    ("design.sdc", {"sds": 1.0}),
    ("design.sdc", {"sds": 1.0, "sd1": 0.6, "riskCategory": "V"}),
    ("design.sdc", {"sds": -0.1, "sd1": 0.6}),
    ("design.elf", {"w": 1000}),
    ("design.elf", {**ELF_ASCE, "code": "nch433", "t": 1.0}),
    ("design.elf", ELF_ASCE),                                               # no t / hn
    ("design.elf", {**ELF_ASCE, "t": 1.0, "edition": "7-05"}),
    ("design.elf", {**ELF_ASCE, "t": 1.0, "importance": "esencial"}),       # NEC field
    ("design.elf", {**ELF_ASCE, "t": 1.0, "applyUpperLimit": 1}),
    ("design.elf", {**ELF_ASCE, "t": 1.0, "stories": []}),
    ("design.elf", {**ELF_ASCE, "t": 1.0, "stories": [{"weight": 1}]}),
    ("design.elf", {**ELF_NEC, "t": 1.0, "phiP": 1.2}),
    ("design.elf", {**ELF_NEC, "t": 1.0, "sds": 1.0}),                      # ASCE field
]


@pytest.mark.parametrize("command, args", BAD_ARGS)
def test_shape_errors_rejected_by_schema_and_dispatch(command, args, registry):
    assert not validator(f"{command}.args", registry).is_valid(args)
    r = dispatch(req(command, args))
    assert not r["ok"] and r["error"]["code"] == "bad_request", r
    validator("envelope.response", registry).validate(r)


# --------------------------------------------------------------- domain errors
@pytest.mark.parametrize("command, args", [
    ("zoning.at", {"lat": -12.05, "lon": -77.04}),
    ("site.assess", {"lat": -1.0, "lon": -82.0, "siteClass": "D"}),
])
def test_outside_ecuador_is_out_of_area(command, args, registry):
    assert validator(f"{command}.args", registry).is_valid(args)
    r = dispatch(req(command, args))
    assert r["error"]["code"] == "out_of_area"


# ------------------------------------------------------------------- envelope
@pytest.mark.parametrize("request_, code", [
    (None, "bad_request"),
    ([], "bad_request"),
    ({"command": "zoning.at"}, "bad_request"),
    ({"api": "apeLoads/1", "command": "zoning.at", "args": QUITO}, "bad_request"),
    ({"api": API, "command": "zoning.at", "args": [1]}, "bad_request"),
    ({"api": API, "command": "zoning.at", "args": QUITO, "x": 1}, "bad_request"),
    ({"api": API, "command": "nope"}, "unknown_command"),
    ({"api": API, "command": ""}, "bad_request"),
    ({"api": API, "command": "zoning.at", "args": {"lat": math.nan, "lon": 0}}, "bad_request"),
    ({"api": API, "command": "zoning.at", "args": {"lat": math.inf, "lon": 0}}, "bad_request"),
])
def test_envelope_errors_never_raise(request_, code, registry):
    r = dispatch(request_)
    assert r["ok"] is False and r["error"]["code"] == code
    assert r["error"]["code"] in ERROR_CODES
    validator("envelope.response", registry).validate(r)


def test_messages_are_printable():
    r = dispatch({"api": API, "command": "x‮\x00" * 50})
    msg = r["error"]["message"]
    assert "‮" not in msg and "\x00" not in msg and len(msg) <= 600


# ------------------------------------------------------------------------ CLI
def _cli(text):
    import os
    from pathlib import Path

    import apeQuake

    src = str(Path(apeQuake.__file__).resolve().parents[1])     # works installed or from src/
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, "-m", "apeQuake.api"], input=text, text=True,
                          capture_output=True, timeout=120, env=env)


def test_cli_ok_and_errors():
    p = _cli(json.dumps(req("zoning.at", QUITO)))
    assert p.returncode == 0 and json.loads(p.stdout)["result"]["z"] == 0.4
    p = _cli("{not json")
    assert p.returncode == 1 and json.loads(p.stdout)["error"]["code"] == "bad_request"
    p = _cli('{"api": "apeQuake/1", "command": "zoning.at", "args": {"lat": NaN, "lon": 0}}')
    assert p.returncode == 1 and json.loads(p.stdout)["error"]["code"] == "bad_request"


def test_nec_zone_grid_decodes_to_the_zone_map():
    from apeQuake.nec import zone_at, zone_grid

    g = dispatch(req("map.layer", {"layer": "necZones"}))["result"]
    zone, lon, lat, _ = zone_grid()
    assert (g["ny"], g["nx"]) == zone.shape
    assert all(sum(n for _, n in row) == g["nx"] for row in g["rows"])
    # decode the row through Quito and compare with zone_at
    i = round((-0.22 - g["lat0"]) / g["dlat"])
    j = round((-78.51 - g["lon0"]) / g["dlon"])
    k, seen = None, 0
    for code, n in g["rows"][i]:
        if seen + n > j:
            k = code
            break
        seen += n
    assert g["classes"][k]["z"] == zone_at(-0.22, -78.51).z


def test_hazard_layer_matches_hazard_map():
    from apeQuake.hazard import EcuadorHazard

    r = dispatch(req("map.layer", {"layer": "igepnHazard", "tr": 475, "period": 0.0}))["result"]
    df = EcuadorHazard.hazard_map(475, 0.0).set_index("cell_id")
    f = r["features"]["features"][0]["properties"]
    assert f["sa"] == pytest.approx(df.loc[f["cellId"], "sa"], abs=1e-4)
    assert r["min"] <= f["sa"] <= r["max"]


def test_report_unavailable_without_typst(monkeypatch):
    import apeQuake.report as rep

    monkeypatch.setattr(rep.shutil, "which", lambda _name: None)
    r = dispatch(req("report.build", {**QUITO, "siteClass": "D"}))
    assert r["ok"] is False and r["error"]["code"] == "report_unavailable"


@needs_typst
def test_report_pdf_is_a_pdf():
    import base64

    r = dispatch(req("report.build", {**QUITO, "siteClass": "F"}))["result"]
    assert base64.b64decode(r["pdfBase64"])[:5] == b"%PDF-"
    assert "fig-sitio.svg" not in r["figures"] and r["typ"].startswith("#import")


# ------------------------------------------------- spectra and design (1.4.0)
def _result(command, args):
    r = dispatch(req(command, args))
    assert r["ok"], r
    return r["result"]


def test_spectrum_code_matches_the_library():
    import numpy as np

    from apeQuake.code_spectrum.codes import ASCE7_16Spectrum, ASCE7_22MultiPeriodSpectrum
    from apeQuake.code_spectrum.codes.asce7_22_mprs import MPRS_PERIODS

    r = _result("spectrum.code", {**ASCE16, "n": 40})
    m = ASCE7_16Spectrum(1.5, 0.6, "C", 8.0)
    np.testing.assert_allclose(r["spectrum"]["sa"], m.sa(r["spectrum"]["t"]))
    assert r["parameters"]["sds"] == m.sds and r["parameters"]["siteClass"] == "C"
    assert set(m.knee_periods()) <= set(r["kneePeriods"])
    r = _result("spectrum.code", MPRS)
    m = ASCE7_22MultiPeriodSpectrum(MPRS_PERIODS, MPRS["saMcer"], 8.0, "D")
    np.testing.assert_allclose(r["spectrum"]["sa"], m.sa(r["spectrum"]["t"]))
    assert r["parameters"]["smsSM1Source"].startswith("inferred")


def test_spectrum_code_export_is_the_file_text(tmp_path):
    from apeQuake.code_spectrum import to_etabs
    from apeQuake.code_spectrum.codes import NECSpectrum

    e = _result("spectrum.code", {"code": "NEC-15", **NEC,
                                  "export": {"format": "etabs", "units": "cm/s2"}})["export"]
    model = NECSpectrum(z=0.4, site_class="D", region="sierra")
    ref = to_etabs(model, tmp_path / "x.txt", units="cm/s2").read_text(encoding="utf-8")
    assert e["text"] == ref and e["fileName"].endswith(".txt") and e["units"] == "cm/s2"
    py = _result("spectrum.code", {**ASCE16, "export": {"format": "opensees-python"}})["export"]
    assert py["fileName"].endswith(".py") and "timeSeries('Path'" in py["text"]


def test_design_sdc_matches_section_11_6():
    r = _result("design.sdc", {"sds": 0.4, "sd1": 0.1, "s1": 0.2})
    assert (r["sdc"], r["fromSds"], r["fromSd1"], r["s1Override"]) == ("C", "C", "B", False)
    assert r["importanceFactor"] == 1.0
    assert _result("design.sdc", {"sds": 0.4, "sd1": 0.1})["s1Override"] is None


def test_design_elf_matches_the_library():
    import numpy as np

    from apeQuake.code_spectrum.codes import NECSpectrum
    from apeQuake.design import asce7, nec

    r = _result("design.elf", {**ELF_ASCE, "edition": "7-22", "t": 1.2, "hn": 30,
                               "stories": STORIES})
    ref = asce7.base_shear(W=1000, R=8, sds=1.0, sd1=0.6, s1=0.6, tl=8, T=1.2, hn=30,
                           edition="7-22", weights=[400, 400, 200], heights=[3.5, 7.0, 10.5])
    assert (r["v"], r["cs"], r["t"], r["k"], r["method"]) == (ref.V, ref.Cs, ref.T, ref.k, 2)
    roles = {lim["role"] for lim in r["limits"]}
    assert roles == {"base", "cap", "min", "minS1"}
    assert r["governing"] in {lim["equation"] for lim in r["limits"]}
    np.testing.assert_allclose([s["fx"] for s in r["stories"]], ref.Fx)
    assert r["stories"][0]["vx"] == pytest.approx(r["v"])

    r = _result("design.elf", {**ELF_NEC, "hn": 10.5, "importance": "esencial", "phiP": 0.9})
    ref = nec.base_shear(NECSpectrum(z=0.4, site_class="D", region="sierra"), W=1000, R=8,
                         I=1.5, phi_p=0.9, hn=10.5)
    assert (r["v"], r["ta"], r["i"]) == (ref.V, ref.Ta, 1.5) and "stories" not in r


def test_design_elf_cites_each_editions_equations():
    for edition, cap in (("7-16", "12.8-3"), ("7-22", "12.8-4")):
        r = _result("design.elf", {**ELF_ASCE, "edition": edition, "t": 1.0})
        assert r["governing"] == cap


@pytest.mark.parametrize("command, args", [
    ("spectrum.code", {**ASCE16, "siteClass": "D"}),     # 11.4.8: S1 >= 0.2 needs site-specific
    ("spectrum.code", {**MPRS, "saMcer": [0.0] * 22}),
])
def test_spectrum_domain_refusals_are_value_errors(command, args, registry):
    assert validator(f"{command}.args", registry).is_valid(args)
    r = dispatch(req(command, args))
    assert r["ok"] is False and r["error"]["code"] == "value_error", r
