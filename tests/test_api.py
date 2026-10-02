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
