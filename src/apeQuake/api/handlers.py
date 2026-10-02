"""Command handlers: check args, call the domain, build camelCase payloads.

Units on the wire (one per quantity): latitude / longitude in decimal degrees
(WGS-84), accelerations in g, periods in s, Vs30 in m/s, distances in km.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import checks as c
from ..nec import zone_at
from ..site_assessment import assess_site

Handler = Callable[[dict[str, Any]], dict[str, Any]]

REGIONS = ("costa", "sierra", "oriente", "esmeraldas", "galapagos")
SITE_CLASSES = ("A", "B", "C", "D", "E", "F")
METHODS = ("nec", "igepn", "manual")


@dataclass(frozen=True)
class Command:
    name: str
    summary: str
    handler: Handler


# --------------------------------------------------------------------------- args
def _point(args: dict[str, Any]) -> tuple[float, float]:
    return (c.number(args["lat"], "args.lat", minimum=-90, maximum=90),
            c.number(args["lon"], "args.lon", minimum=-180, maximum=180))


# ------------------------------------------------------------------------ payloads
def _curve(d: dict[str, Any] | None) -> dict[str, list[float]] | None:
    if d is None:
        return None
    return {"t": [float(v) for v in d["T"]], "sa": [float(v) for v in d["Sa"]]}


def _listed(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {"town": d["town"], "parish": d["parish"], "canton": d["canton"],
            "province": d["province"], "z": d["z"], "distanceKm": d["distance_km"]}


def _zone_payload(z: dict[str, Any]) -> dict[str, Any]:
    return {
        "lat": z["lat"], "lon": z["lon"], "z": z["z"], "zone": z["zone"],
        "province": z["province"], "region": z["region"], "eta": z["eta"],
        "boundaryKm": z["boundary_km"], "zAcrossBoundary": z["z_across_boundary"],
        "nearestListed": _listed(z["nearest_listed"]), "source": z["source"],
        "warnings": list(z["warnings"]),
        "notices": list(z.get("notices", [])),
    }


def _nec_params(p: dict[str, Any] | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {"z": p["Z"], "eta": p["eta"], "fa": p["Fa"], "fd": p["Fd"], "fs": p["Fs"],
            "r": p["r"], "t0": p["T0"], "tc": p["Tc"], "tl": p["TL"],
            "siteClass": p["site_class"], "region": p["region"]}


def _asce16_params(p: dict[str, Any]) -> dict[str, Any]:
    return {"ss": p["ss"], "s1": p["s1"], "siteClass": p["site_class"], "fa": p["Fa"],
            "fv": p["Fv"], "sms": p["SMS"], "sm1": p["SM1"], "sds": p["SDS"],
            "sd1": p["SD1"], "t0": p["T0"], "ts": p["Ts"], "tl": p["TL"]}


def _asce22_params(p: dict[str, Any]) -> dict[str, Any]:
    return {"siteClass": p["site_class"], "sms": p["SMS"], "sm1": p["SM1"], "sds": p["SDS"],
            "sd1": p["SD1"], "t0": p["T0"], "ts": p["Ts"], "tl": p["TL"],
            "fa716": p["Fa_7_16"], "fv716": p["Fv_7_16"]}


def _uhs(d: dict[str, Any]) -> dict[str, list[float]]:
    return {k: [float(v) for v in d[k]] for k in ("mean", "q16", "q84")}


def _comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    r = row["rock"]
    out: dict[str, Any] = {"t": row["T"], "rock": {
        "nec475": r["nec_475"], "asceDesign": r["asce_design"],
        "igepn475": r["igepn_475"], "igepn2475x23": r["igepn_2475_x2_3"]}}
    if "site" in row:
        s = row["site"]
        out["site"] = {"nec": s["nec"], "asce716": s["asce7_16"], "asce722": s["asce7_22"],
                       "igepn475Scaled": s["igepn_475_scaled"]}
    return out


def assessment_payload(d: dict[str, Any]) -> dict[str, Any]:
    """camelCase payload of :meth:`SiteAssessment.to_dict`."""
    zone = d["zone"]
    a16, a22, site = d["asce7_16"], d["asce7_22"], d["site"]
    ig = d["igepn"]
    return {
        "zone": {**_zone_payload(zone), "zUsed": zone["z_used"],
                 "regionUsed": zone["region_used"], "etaUsed": zone["eta_used"]},
        "siteClasses": {"nec": d["site_classes"]["nec"], "asce716": d["site_classes"]["asce7_16"],
                        "asce722": d["site_classes"]["asce7_22"]},
        "ssS1": {
            "method": d["ss_s1"]["method"], "ss": d["ss_s1"]["ss"], "s1": d["ss_s1"]["s1"],
            "candidates": d["ss_s1"]["candidates"],
            "necRule": {"factor": d["ss_s1"]["nec_rule"]["factor"],
                        "saB02": d["ss_s1"]["nec_rule"]["sa_b_02"],
                        "saB10": d["ss_s1"]["nec_rule"]["sa_b_10"]},
        },
        "nec": {"parameters": _nec_params(d["nec"]["parameters"]),
                "rockParameters": _nec_params(d["nec"]["rock_parameters"]),
                "spectrum": _curve(d["nec"]["spectrum"]),
                "rockSpectrum": _curve(d["nec"]["rock_spectrum"])},
        "necUhs": {"city": d["nec_uhs"]["city"], "distanceKm": d["nec_uhs"]["distance_km"],
                   "tr475": _curve(d["nec_uhs"]["475"]), "tr2500": _curve(d["nec_uhs"]["2500"])},
        "asce716": None if a16 is None else {
            "parameters": _asce16_params(a16["parameters"]), "exceptions": a16["exceptions"],
            "tlSource": a16["tl_source"], "spectrum": _curve(a16["spectrum"])},
        "asce722": None if a22 is None else {
            "parameters": _asce22_params(a22["parameters"]), "approximate": a22["approximate"],
            "note": a22["note"], "spectrum": _curve(a22["spectrum"])},
        "igepn": {"cellId": ig["cell_id"], "distanceKm": ig["distance_km"],
                  "periods": [float(v) for v in ig["periods"]],
                  "uhs": {"tr475": _uhs(ig["uhs"]["475"]), "tr2475": _uhs(ig["uhs"]["2475"])},
                  "siteScaledNote": ig.get("site_scaled_note")},
        "rock": {"nec475": _curve(d["rock"]["nec_475"]),
                 "asceDesign": _curve(d["rock"]["asce_design"]),
                 "igepn475": _curve(d["rock"]["igepn_475"]),
                 "igepn2475x23": _curve(d["rock"]["igepn_2475_x2_3"]),
                 "necUhs475": _curve(d["rock"]["nec_uhs_475"])},
        "site": None if site is None else {
            "nec": _curve(site["nec"]), "asce716": _curve(site["asce7_16"]),
            "asce722": _curve(site["asce7_22"]),
            "igepn475Scaled": _curve(site["igepn_475_scaled"]),
            "igepn2475x23Scaled": _curve(site["igepn_2475_x2_3_scaled"]),
            "necAmplification": _curve(site["nec_amplification"])},
        "comparison": [_comparison_row(r) for r in d["comparison"]],
        "warnings": list(d["warnings"]),
        "notices": list(d.get("notices", [])),
    }


# ------------------------------------------------------------------------ handlers
def zoning_at(args: dict[str, Any]) -> dict[str, Any]:
    c.obj(args, "args", required=("lat", "lon"))
    lat, lon = _point(args)
    return _zone_payload(zone_at(lat, lon).to_dict())


_ASSESS_OPTIONAL = ("vs30", "siteClass", "method", "ss", "s1", "z", "region", "tlAsce")


def site_assess_args(args: dict[str, Any]) -> dict[str, Any]:
    """Checked keyword arguments for :func:`assess_site`."""
    c.obj(args, "args", required=("lat", "lon"), optional=_ASSESS_OPTIONAL)
    lat, lon = _point(args)
    has_vs, has_sc = "vs30" in args, "siteClass" in args
    if has_vs == has_sc:
        raise c.ArgError("args: give exactly one of «vs30» or «siteClass»")
    kw: dict[str, Any] = {"lat": lat, "lon": lon}
    if has_vs:
        kw["vs30"] = c.number(args["vs30"], "args.vs30", exclusive_minimum=0, maximum=5000)
    else:
        kw["site_class"] = c.enum(args["siteClass"], "args.siteClass", SITE_CLASSES)
    method = c.enum(args.get("method", "nec"), "args.method", METHODS)
    kw["method"] = method
    has_manual = "ss" in args or "s1" in args
    if method == "manual":
        if not ("ss" in args and "s1" in args):
            raise c.ArgError("args: method «manual» needs «ss» and «s1»")
        kw["ss"] = c.number(args["ss"], "args.ss", exclusive_minimum=0, maximum=10)
        kw["s1"] = c.number(args["s1"], "args.s1", exclusive_minimum=0, maximum=10)
    elif has_manual:
        raise c.ArgError("args: «ss» / «s1» are only accepted with method «manual»")
    if "z" in args:
        kw["z"] = c.number(args["z"], "args.z", minimum=0.15, maximum=0.5)
    if "region" in args:
        kw["region"] = c.enum(args["region"], "args.region", REGIONS)
    if "tlAsce" in args:
        kw["tl_asce"] = c.number(args["tlAsce"], "args.tlAsce", exclusive_minimum=0, maximum=20)
    return kw


def site_assess(args: dict[str, Any]) -> dict[str, Any]:
    kw = site_assess_args(args)
    result = assessment_payload(assess_site(**kw).to_dict())
    result["input"] = {k: v for k, v in args.items()}
    return result


# --------------------------------------------------------------------- map layers
_PERIODS = (0.0, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0)


def map_layer(args: dict[str, Any]) -> dict[str, Any]:
    from . import layers as L

    c.obj(args, "args", required=("layer",), optional=("tr", "period", "stat"))
    layer = c.enum(args["layer"], "args.layer", L.LAYERS)
    extra = [k for k in ("tr", "period", "stat") if k in args]
    if layer != "igepnHazard" and extra:
        raise c.ArgError(f"args: «{extra[0]}» is only accepted with layer «igepnHazard»")
    if layer == "necZones":
        return L.nec_zones()
    if layer == "igepnHazard":
        tr = args.get("tr", 475)
        if isinstance(tr, bool) or tr not in (475, 2475):
            raise c.ArgError("args.tr: expected one of 475, 2475")
        period = c.number(args.get("period", 0.0), "args.period")
        if not any(abs(period - p) < 1e-9 for p in _PERIODS):
            raise c.ArgError("args.period: expected one of " + ", ".join(f"{p:g}" for p in _PERIODS))
        stat = c.enum(args.get("stat", "mean"), "args.stat", ("mean", "q16", "q50", "q84"))
        return L.igepn_hazard(int(tr), float(period), stat)
    return {"faults": L.faults, "sourceZones": L.source_zones, "capitals": L.capitals,
            "provinces": L.provinces}[layer]()


# ------------------------------------------------------------------------ report
_REPORT_FIELDS = ("title", "project", "client", "documentCode", "revision", "date", "siteName")


def _report_meta(x: object):
    from ..report import ReportMeta

    r = c.obj(x, "args.report", optional=(*_REPORT_FIELDS, "authors"))
    kw: dict[str, Any] = {}
    names = {"documentCode": "document_code", "siteName": "site_name"}
    for k in _REPORT_FIELDS:
        if k in r:
            kw[names.get(k, k)] = c.string(r[k], f"args.report.{k}")
    if "authors" in r:
        if not isinstance(r["authors"], list) or len(r["authors"]) > 10:
            raise c.ArgError("args.report.authors: expected a list of at most 10 authors")
        authors = []
        for i, a in enumerate(r["authors"]):
            a = c.obj(a, f"args.report.authors[{i}]", required=("name",),
                      optional=("email", "affiliation"))
            authors.append({k: c.string(v, f"args.report.authors[{i}].{k}") for k, v in a.items()})
        kw["authors"] = authors
    return ReportMeta(**kw)


def report_build(args: dict[str, Any]) -> dict[str, Any]:
    import base64

    from ..report import build_report

    c.obj(args, "args", required=("lat", "lon"), optional=(*_ASSESS_OPTIONAL, "report"))
    meta = _report_meta(args["report"]) if "report" in args else None
    kw = site_assess_args({k: v for k, v in args.items() if k != "report"})
    rep = build_report(assess_site(**kw), meta)
    assert rep.pdf is not None
    return {"pdfBase64": base64.b64encode(rep.pdf).decode("ascii"), "typ": rep.typ,
            "figures": dict(rep.figures), "fileName": "informe-sismico.pdf"}
