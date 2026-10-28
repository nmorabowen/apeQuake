"""Code spectra and seismic design commands: ``spectrum.code``, ``design.sdc``, ``design.elf``.

Same conventions as :mod:`.handlers`: accelerations in g, periods in s, heights in m.
Forces (``w``, ``v``, story weights, ``fx`` / ``vx``) are in whatever unit ``w`` is
given in; the result uses the same unit.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from . import checks as c

SPECTRUM_CODES = ("ASCE7-10", "ASCE7-16", "ASCE7-22", "ASCE7-22-MPRS", "NEC-15")
ASCE_SITE_CLASSES = ("A", "B", "C", "D", "E")            # 7-10 / 7-16 (F: site-specific)
ASCE22_SITE_CLASSES = ("A", "B", "BC", "C", "CD", "D", "DE", "E")
NEC_SITE_CLASSES = ("A", "B", "C", "D", "E")
LEVELS = ("design", "mcer")
EXPORT_FORMATS = ("csv", "etabs", "sap2000", "opensees-tcl", "opensees-python")
EXPORT_UNITS = ("g", "m/s2", "cm/s2")
RISK_CATEGORIES = ("I", "II", "III", "IV")
EDITIONS = ("7-10", "7-16", "7-22")
ASCE_STRUCTURE_TYPES = ("steel_moment_frame", "concrete_moment_frame",
                        "steel_eccentrically_braced_frame",
                        "steel_buckling_restrained_braced_frame", "other")
NEC_STRUCTURE_TYPES = ("acero_sin_arriostramientos", "acero_con_arriostramientos",
                       "hormigon_porticos", "hormigon_con_muros")
NEC_IMPORTANCE = ("esencial", "especial", "otra")
_MAX_STORIES = 200
_MAX_POINTS = 2000


def camel(key: str) -> str:
    """``"site_class"`` -> ``"siteClass"``, ``"SDS"`` -> ``"sds"``, ``"Fa_7_16"`` -> ``"fa716"``."""
    head, *rest = key.split("_")
    return head.lower() + "".join(p[:1].upper() + p[1:] for p in rest)


def _wire(v: Any) -> Any:
    """A JSON value: numpy scalars to Python, tuples / arrays to lists."""
    if isinstance(v, (str, bool)) or v is None:
        return v
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_wire(x) for x in v]
    if isinstance(v, (int, float)):
        return v
    return str(v)


# --------------------------------------------------------------------- spectra
_SPECTRUM_FIELDS = {
    "ASCE7-10": (("ss", "s1", "siteClass", "tl"), ("level",)),
    "ASCE7-16": (("ss", "s1", "siteClass", "tl"), ("level",)),
    "ASCE7-22": (("sms", "sm1", "siteClass", "tl"), ("level",)),
    "ASCE7-22-MPRS": (("saMcer", "siteClass", "tl"), ("sms", "sm1", "level")),
    "NEC-15": (("z", "siteClass", "region"), ()),
}


def _accel(args: dict[str, Any], key: str, name: str) -> float:
    return c.number(args[key], f"{name}.{key}", exclusive_minimum=0, maximum=10)


def spectrum_model(args: dict[str, Any]):
    """The :class:`CodeSpectrumModel` the ``spectrum.code`` args describe."""
    from ..code_spectrum.codes import (
        ASCE7_10Spectrum,
        ASCE7_16Spectrum,
        ASCE7_22MultiPeriodSpectrum,
        ASCE7_22Spectrum,
        NECSpectrum,
    )
    from ..code_spectrum.codes.asce7_22_mprs import MPRS_PERIODS
    from .handlers import REGIONS

    name = "args"
    if "code" not in args:
        raise c.ArgError("args: missing field «code»")
    code = c.enum(args["code"], "args.code", SPECTRUM_CODES)
    req, opt = _SPECTRUM_FIELDS[code]
    a = c.obj(args, name, required=("code", *req), optional=(*opt, "tMax", "n", "export"))
    level = c.enum(a.get("level", "design"), f"{name}.level", LEVELS)
    if code == "NEC-15":
        return NECSpectrum(z=c.number(a["z"], f"{name}.z", minimum=0.15, maximum=0.5),
                           site_class=c.enum(a["siteClass"], f"{name}.siteClass",
                                             NEC_SITE_CLASSES),
                           region=c.enum(a["region"], f"{name}.region", REGIONS))
    tl = c.number(a["tl"], f"{name}.tl", exclusive_minimum=0, maximum=20)
    if code in ("ASCE7-10", "ASCE7-16"):
        cls = ASCE7_10Spectrum if code == "ASCE7-10" else ASCE7_16Spectrum
        return cls(_accel(a, "ss", name), _accel(a, "s1", name),
                   c.enum(a["siteClass"], f"{name}.siteClass", ASCE_SITE_CLASSES), tl,
                   level=level)
    sc = c.enum(a["siteClass"], f"{name}.siteClass", ASCE22_SITE_CLASSES)
    if code == "ASCE7-22":
        return ASCE7_22Spectrum(_accel(a, "sms", name), _accel(a, "sm1", name), tl, sc,
                                level=level)
    sa = a["saMcer"]
    if not isinstance(sa, list) or len(sa) != len(MPRS_PERIODS):
        raise c.ArgError(f"{name}.saMcer: expected {len(MPRS_PERIODS)} MCE_R ordinates, "
                         "one per Section 11.4.5.1 period")
    sa = [c.number(v, f"{name}.saMcer[{i}]", minimum=0, maximum=10) for i, v in enumerate(sa)]
    if ("sms" in a) != ("sm1" in a):
        raise c.ArgError(f"{name}: give both «sms» and «sm1», or neither")
    kw = {k: _accel(a, k, name) for k in ("sms", "sm1") if k in a}
    return ASCE7_22MultiPeriodSpectrum(MPRS_PERIODS, sa, tl, sc, level=level, **kw)


def _export(model, x: object, t_max: float, n: int) -> dict[str, Any]:
    from ..code_spectrum import export as ex

    e = c.obj(x, "args.export", required=("format",), optional=("units",))
    fmt = c.enum(e["format"], "args.export.format", EXPORT_FORMATS)
    units = c.enum(e.get("units", "g"), "args.export.units", EXPORT_UNITS)
    suffix = {"csv": ".csv", "opensees-tcl": ".tcl", "opensees-python": ".py"}.get(fmt, ".txt")
    stem = model.code.lower().replace(" ", "-")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{stem}{suffix}"
        kw: dict[str, Any] = {"t_max": t_max, "n": n, "units": units}
        if fmt == "csv":
            ex.to_csv(model, path, **kw)
        elif fmt == "etabs":
            ex.to_etabs(model, path, **kw)
        elif fmt == "sap2000":
            ex.to_sap2000(model, path, **kw)
        else:
            ex.to_opensees(model, path, style="tcl" if fmt == "opensees-tcl" else "python", **kw)
        text = path.read_text(encoding="utf-8")
    return {"format": fmt, "units": units, "fileName": f"{stem}{suffix}", "text": text}


def spectrum_code(args: dict[str, Any]) -> dict[str, Any]:
    model = spectrum_model(args)
    t_max = c.number(args.get("tMax", 4.0), "args.tMax", exclusive_minimum=0, maximum=20)
    n = c.number(args.get("n", 200), "args.n", minimum=2, maximum=_MAX_POINTS)
    if not n.is_integer():                     # JSON Schema "integer" admits 5.0
        raise c.ArgError(f"args.n: expected an integer from 2 to {_MAX_POINTS}")
    n = int(n)
    T = model.default_periods(t_max, n)
    result: dict[str, Any] = {
        "code": model.code,
        "parameters": {camel(k): _wire(v) for k, v in model.parameters().items()},
        "kneePeriods": [float(t) for t in model.knee_periods()],
        "spectrum": {"t": [float(t) for t in T], "sa": [float(s) for s in model.sa(T)]},
    }
    if "export" in args:
        result["export"] = _export(model, args["export"], t_max, n)
    return result


# ---------------------------------------------------------------------- design
def _stories(x: object) -> tuple[list[float], list[float]]:
    if not isinstance(x, list) or not 1 <= len(x) <= _MAX_STORIES:
        raise c.ArgError(f"args.stories: expected 1 to {_MAX_STORIES} stories")
    weights, heights = [], []
    for i, s in enumerate(x):
        s = c.obj(s, f"args.stories[{i}]", required=("weight", "height"))
        weights.append(c.number(s["weight"], f"args.stories[{i}].weight", minimum=0))
        heights.append(c.number(s["height"], f"args.stories[{i}].height", exclusive_minimum=0))
    return weights, heights


def _story_payload(weights, heights, fx, vx) -> list[dict[str, float]]:
    return [{"weight": w, "height": h, "fx": float(f), "vx": float(v)}
            for w, h, f, v in zip(weights, heights, fx, vx)]


def design_sdc(args: dict[str, Any]) -> dict[str, Any]:
    from ..design import asce7

    c.obj(args, "args", required=("sds", "sd1"), optional=("s1", "riskCategory"))
    sds = c.number(args["sds"], "args.sds", minimum=0, maximum=10)
    sd1 = c.number(args["sd1"], "args.sd1", minimum=0, maximum=10)
    s1 = c.number(args["s1"], "args.s1", minimum=0, maximum=10) if "s1" in args else None
    rc = c.enum(args.get("riskCategory", "II"), "args.riskCategory", RISK_CATEGORIES)
    r = asce7.seismic_design_category(sds, sd1, s1=s1, risk_category=rc)
    return {"sdc": r.sdc, "fromSds": r.from_sds, "fromSd1": r.from_sd1,
            "riskCategory": r.risk_category, "s1Override": r.s1_override,
            "importanceFactor": asce7.importance_factor(rc)}


_ELF_COMMON = ("w", "r", "t", "hn", "structureType", "stories")
_ELF_ASCE = ("edition", "sds", "sd1", "s1", "tl", "riskCategory", "ie", "applyUpperLimit")
_ELF_NEC = ("z", "siteClass", "region", "importance", "phiP", "phiE")


def _limits(edition: str, breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    from ..design.asce7 import _EQ

    eq = _EQ[edition]
    roles = {eq["method1"]: "method1", eq["method2"]: "base", eq["cap"]: "cap",
             eq["cap_TL"]: "capTL", eq["min"]: "min", eq["min_S1"]: "minS1"}
    out = []
    for key, value in breakdown.items():
        if key.startswith("Cs_") and key != "Cs":
            number = key.rsplit("_", 1)[1]
            out.append({"equation": number, "role": roles[number], "cs": float(value)})
    return out


def _elf_asce(args: dict[str, Any]) -> dict[str, Any]:
    from ..design import asce7

    c.obj(args, "args", required=("code", "w", "r", "sds", "sd1", "s1", "tl"),
          optional=(*_ELF_COMMON, *_ELF_ASCE))
    kw: dict[str, Any] = {
        "W": c.number(args["w"], "args.w", exclusive_minimum=0),
        "R": c.number(args["r"], "args.r", exclusive_minimum=0, maximum=20),
        "sds": c.number(args["sds"], "args.sds", minimum=0, maximum=10),
        "sd1": c.number(args["sd1"], "args.sd1", minimum=0, maximum=10),
        "s1": c.number(args["s1"], "args.s1", minimum=0, maximum=10),
        "tl": c.number(args["tl"], "args.tl", exclusive_minimum=0, maximum=20),
        "edition": c.enum(args.get("edition", "7-16"), "args.edition", EDITIONS),
        "risk_category": c.enum(args.get("riskCategory", "II"), "args.riskCategory",
                                RISK_CATEGORIES),
        "structure_type": c.enum(args.get("structureType", "other"), "args.structureType",
                                 ASCE_STRUCTURE_TYPES),
    }
    if "ie" in args:
        kw["Ie"] = c.number(args["ie"], "args.ie", exclusive_minimum=0, maximum=2)
    _period_args(args, kw)
    if "applyUpperLimit" in args:
        if not isinstance(args["applyUpperLimit"], bool):
            raise c.ArgError("args.applyUpperLimit: expected a boolean")
        kw["apply_upper_limit"] = args["applyUpperLimit"]
    if "stories" in args:
        kw["weights"], kw["heights"] = _stories(args["stories"])
    r = asce7.base_shear(**kw, metric=True)
    out = {"code": "asce7", "edition": r.edition, "v": r.V, "cs": r.Cs, "t": r.T, "ta": r.Ta,
           "cu": r.Cu, "k": r.k, "w": r.W, "r": r.R, "ie": r.Ie,
           "method": r.breakdown["method"],
           "governing": r.breakdown["governing"].split(" ")[-1],
           "limits": _limits(r.edition, r.breakdown)}
    if r.Fx is not None:
        out["stories"] = _story_payload(kw["weights"], kw["heights"], r.Fx, r.Vx)
    return out


def _period_args(args: dict[str, Any], kw: dict[str, Any]) -> None:
    if "t" not in args and "hn" not in args:
        raise c.ArgError("args: give «t», «hn», or both")
    if "t" in args:
        kw["T"] = c.number(args["t"], "args.t", exclusive_minimum=0, maximum=20)
    if "hn" in args:
        kw["hn"] = c.number(args["hn"], "args.hn", exclusive_minimum=0, maximum=1000)


def _elf_nec(args: dict[str, Any]) -> dict[str, Any]:
    from ..code_spectrum.codes import NECSpectrum
    from ..design import nec
    from .handlers import REGIONS

    c.obj(args, "args", required=("code", "w", "r", "z", "siteClass", "region"),
          optional=(*_ELF_COMMON, *_ELF_NEC))
    spectrum = NECSpectrum(z=c.number(args["z"], "args.z", minimum=0.15, maximum=0.5),
                           site_class=c.enum(args["siteClass"], "args.siteClass",
                                             NEC_SITE_CLASSES),
                           region=c.enum(args["region"], "args.region", REGIONS))
    category = c.enum(args.get("importance", "otra"), "args.importance", NEC_IMPORTANCE)
    kw: dict[str, Any] = {
        "W": c.number(args["w"], "args.w", exclusive_minimum=0),
        "R": c.number(args["r"], "args.r", exclusive_minimum=0, maximum=20),
        "I": nec.importance_factor(category),
        "phi_p": c.number(args.get("phiP", 1.0), "args.phiP", exclusive_minimum=0, maximum=1),
        "phi_e": c.number(args.get("phiE", 1.0), "args.phiE", exclusive_minimum=0, maximum=1),
        "structure_type": c.enum(args.get("structureType", "hormigon_porticos"),
                                 "args.structureType", NEC_STRUCTURE_TYPES),
    }
    _period_args(args, kw)
    if "stories" in args:
        kw["weights"], kw["heights"] = _stories(args["stories"])
    r = nec.base_shear(spectrum, **kw)
    out = {"code": "nec", "v": r.V, "cs": r.shear_coefficient, "saTa": r.Sa_Ta, "ta": r.Ta,
           "taMethod1": r.Ta_method_1, "k": r.k, "w": r.W, "r": r.R, "phiP": r.phi_p,
           "phiE": r.phi_e, "importance": category, "i": r.I}
    if r.Fx is not None:
        out["stories"] = _story_payload(kw["weights"], kw["heights"], r.Fx, r.Vx)
    return out


def design_elf(args: dict[str, Any]) -> dict[str, Any]:
    if "code" not in args:
        raise c.ArgError("args: missing field «code»")
    code = c.enum(args["code"], "args.code", ("asce7", "nec"))
    return _elf_asce(args) if code == "asce7" else _elf_nec(args)
