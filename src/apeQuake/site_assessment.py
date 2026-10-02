"""Seismic parameters of a site in Ecuador: NEC-SE-DS vs ASCE 7-16 / 7-22 vs IG-EPN.

``assess_site(lat, lon, ...)`` gathers, for one point:

* **NEC-SE-DS**: Z / zone / region / eta from the digitized zone map
  (:func:`apeQuake.nec.zone_at`), the elastic spectrum for the site class, and the
  code UHS from the digitized NEC hazard curves of the nearest capital (475 and
  2500 yr).
* **Ss and S1 on rock** (reference Vs30 = 760 m/s), by one of three methods:

  - ``"nec"``: ``Ss = 1.5 Sa_B(0.2 s)``, ``S1 = 1.5 Sa_B(1.0 s)``, with ``Sa_B`` the
    NEC elastic spectrum for site class B (Fa = Fd = 1) at the point's Z and eta.
    The factor 1.5 takes the 475-yr NEC design level to an MCE_R-type level,
    mirroring ASCE's ``SD = 2/3 SM``.  A convention, not a risk-targeted value.
  - ``"igepn"``: Sa(0.2 s) and Sa(1.0 s) of the IG-EPN 2475-yr mean UHS (rock,
    uniform hazard, not risk-targeted).
  - ``"manual"``: values supplied by the engineer.

* **ASCE 7-16**: Fa / Fv (Tables 11.4-1/2) for the site class, two-period spectrum;
  the 11.4.8 site-specific triggers (common in Ecuador: class D/E with S1 >= 0.2)
  are reported and the 11.4.8 exception applied.
* **ASCE 7-22 (approximate)**: 7-22 has no Fa / Fv tables (SMS / SM1 come from the
  USGS geodatabase, which does not cover Ecuador).  SMS / SM1 are taken from the
  7-16 tables and the 7-22 two-period spectrum (11.4.5.2) is built from them.
* **IG-EPN** probabilistic hazard at the point (rock, 475 / 2475 yr, mean and
  16 / 84 % fractiles), and an *approximate* site version scaled by the NEC site
  amplification ``Sa_NEC,site(T) / Sa_NEC,B(T)``.

Two comparisons are returned: on **rock** (NEC class B, ASCE reference rock with
Fa = Fv = 1, IG-EPN as published) and for the **site class**.  Spectra are design
level: NEC elastic (475 yr), ASCE ``2/3 MCE_R``, IG-EPN 475 yr and ``2/3 x 2475 yr``.

ASCE has no TL map for Ecuador; by default TL is the NEC TL of the same site class.
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .code_spectrum.codes.asce7_16 import ASCE7_16Spectrum
from .code_spectrum.codes.asce7_22 import ASCE7_22Spectrum
from .code_spectrum.codes.nec import NECSpectrum
from .hazard import EcuadorHazard
from .nec import load_hazard_database, zone_at
from .nec.zoning import _km

__all__ = [
    "SsS1Method",
    "SiteAssessment",
    "assess_site",
    "site_class_nec",
    "site_class_asce7_22",
    "PERIODS",
]

SsS1Method = Literal["nec", "igepn", "manual"]

#: Period grid of the returned spectra [s]: 0-4 s every 0.02 s plus the IG-EPN periods.
PERIODS = np.unique(np.round(np.r_[np.arange(0.0, 4.0001, 0.02), 0.05, 0.07], 4))

MCE_FACTOR = 1.5       # NEC 475-yr design level -> MCE_R-type level (method "nec")
REF_PERIODS = (0.2, 1.0)

_NEC_VS = ((1500.0, "A"), (760.0, "B"), (360.0, "C"), (180.0, "D"))       # m/s, >= bound
_ASCE22_VS = ((1524.0, "A"), (914.4, "B"), (640.1, "BC"), (442.0, "C"),
              (304.8, "CD"), (213.4, "D"), (152.4, "DE"))                 # m/s, > bound


def site_class_nec(vs30: float) -> str:
    """NEC-SE-DS Tabla 2 (also ASCE 7-16 Table 20.3-1) site class from Vs30 [m/s]."""
    return next((c for b, c in _NEC_VS if vs30 >= b), "E")


def site_class_asce7_22(vs30: float) -> str:
    """ASCE 7-22 Table 20.2-1 site class from Vs30 [m/s] (bounds converted from ft/s)."""
    return next((c for b, c in _ASCE22_VS if vs30 > b), "E")


@dataclass
class SiteAssessment:
    """Result of :func:`assess_site`.  All accelerations in g, periods in s."""

    inputs: dict[str, Any]
    zone: dict[str, Any]
    site_classes: dict[str, str]
    ss_s1: dict[str, Any]
    nec: dict[str, Any]
    nec_uhs: dict[str, Any]
    asce7_16: dict[str, Any] | None
    asce7_22: dict[str, Any] | None
    igepn: dict[str, Any]
    rock: dict[str, Any]
    site: dict[str, Any] | None
    comparison: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (numpy arrays become lists)."""
        return _jsonable(self.__dict__)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else v
    if isinstance(x, np.integer):
        return int(x)
    return x


def _curve(T: np.ndarray, sa: np.ndarray) -> dict[str, list]:
    return {"T": np.asarray(T, float), "Sa": np.asarray(sa, float)}


def assess_site(
    lat: float,
    lon: float,
    *,
    vs30: float | None = None,
    site_class: str | None = None,
    method: SsS1Method = "nec",
    ss: float | None = None,
    s1: float | None = None,
    z: float | None = None,
    region: str | None = None,
    tl_asce: float | None = None,
) -> SiteAssessment:
    """Compare NEC-SE-DS, ASCE 7-16, ASCE 7-22 (approximate) and IG-EPN at a point.

    Parameters
    ----------
    lat, lon : float
        WGS-84 degrees, inside Ecuador.
    vs30 : float, optional
        Average shear-wave velocity of the upper 30 m [m/s]; each code classifies it
        with its own table.  Give either ``vs30`` or ``site_class``.
    site_class : {"A", "B", "C", "D", "E", "F"}, optional
        Used for all three codes.  ``"F"`` requires a site response analysis: the
        site-class comparison is skipped (the rock comparison is still returned).
    method : {"nec", "igepn", "manual"}
        How Ss and S1 (rock) are obtained; ``"manual"`` needs ``ss`` and ``s1``.
    z, region : optional
        Override the NEC zone factor and region (eta) read from the map.
    tl_asce : float, optional
        TL for the ASCE spectra; defaults to the NEC TL of the same site class.
    """
    if (vs30 is None) == (site_class is None):
        raise ValueError("give exactly one of vs30 or site_class")
    if method not in ("nec", "igepn", "manual"):
        raise ValueError(f"method must be 'nec', 'igepn' or 'manual', got {method!r}")
    if method == "manual" and (ss is None or s1 is None):
        raise ValueError("method='manual' needs ss and s1")
    warns: list[str] = []

    # --- site classes --------------------------------------------------------
    if vs30 is not None:
        if vs30 <= 0:
            raise ValueError("vs30 must be > 0")
        classes = {"nec": site_class_nec(vs30), "asce7_16": site_class_nec(vs30),
                   "asce7_22": site_class_asce7_22(vs30)}
    else:
        sc = str(site_class).strip().upper()
        if sc not in ("A", "B", "C", "D", "E", "F"):
            raise ValueError(f"site_class must be A-F, got {site_class!r}")
        classes = {"nec": sc, "asce7_16": sc, "asce7_22": sc}
    is_f = classes["nec"] == "F"
    if is_f:
        warns.append("Site class F: a site response analysis is required (NEC-SE-DS 3.2; "
                     "ASCE 7 11.4.7 / 21.1). Only the rock comparison is reported.")

    # --- NEC zoning ----------------------------------------------------------
    zn = zone_at(lat, lon)
    warns += list(zn.warnings)
    z_used = float(z) if z is not None else zn.z
    region_used = region if region is not None else zn.region
    if z is not None and z != zn.z:
        warns.append(f"Z overridden: {z_used:.2f} (map gives {zn.z:.2f}).")
    if region is not None and region != zn.region:
        warns.append(f"Region overridden: {region_used} (province gives {zn.region}).")

    nec_rock = NECSpectrum(z=z_used, site_class="B", region=region_used)
    nec_site = None if is_f else NECSpectrum(z=z_used, site_class=classes["nec"], region=region_used)

    # --- IG-EPN --------------------------------------------------------------
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ig = EcuadorHazard().site(lat, lon, interp="bilinear")
    warns += [f"IG-EPN: {w.message}" for w in caught]
    ig_T = np.asarray(ig.periods, float)
    ig_uhs = {f"{tr}": {s: ig.published(tr, s) for s in ("mean", "q16", "q84")} for tr in (475, 2475)}

    # --- Ss / S1 on rock -----------------------------------------------------
    i02, i10 = (int(np.flatnonzero(np.isclose(ig_T, t))[0]) for t in REF_PERIODS)
    candidates = {
        "nec": (MCE_FACTOR * float(nec_rock.sa(0.2)[0]), MCE_FACTOR * float(nec_rock.sa(1.0)[0])),
        "igepn": (float(ig_uhs["2475"]["mean"][i02]), float(ig_uhs["2475"]["mean"][i10])),
    }
    if method == "manual":
        candidates["manual"] = (float(ss), float(s1))
    ss_used, s1_used = candidates[method]
    ss_s1 = {
        "method": method, "ss": ss_used, "s1": s1_used,
        "candidates": {k: {"ss": a, "s1": b} for k, a, b in ((k, *v) for k, v in candidates.items())},
        "nec_rule": {"factor": MCE_FACTOR, "sa_b_02": float(nec_rock.sa(0.2)[0]),
                     "sa_b_10": float(nec_rock.sa(1.0)[0])},
    }

    # --- NEC code UHS (nearest digitized capital) ------------------------------
    city = load_hazard_database().nearest(lat, lon)
    nec_uhs = {"city": city.name, "distance_km": float(_km(city.lat - lat, city.lon - lon, lat))}
    for tr in (475, 2500):
        T_, sa_ = city.uhs(tr)
        nec_uhs[str(tr)] = _curve(T_, sa_)
    if nec_uhs["distance_km"] > 25:
        warns.append(f"NEC hazard curves exist only for capitals; using {city.name}, "
                     f"{nec_uhs['distance_km']:.0f} km away.")

    # --- ASCE -----------------------------------------------------------------
    tl_rock = float(tl_asce) if tl_asce is not None else float(nec_rock.tl)
    asce_rock = ASCE7_16Spectrum.from_sds_sd1(2 / 3 * ss_used, 2 / 3 * s1_used, tl=tl_rock)

    a16 = a22 = s16 = s22 = None
    if not is_f:
        tl_site = float(tl_asce) if tl_asce is not None else float(nec_site.tl)
        try:
            s16 = ASCE7_16Spectrum(ss_used, s1_used, classes["asce7_16"], tl_site,
                                   allow_exception=True, vs_measured=vs30 is not None)
        except ValueError as exc:      # e.g. class E with S1 > 0.1: Table 11.4-2 has no Fv
            warns.append(f"ASCE 7-16 / 7-22 site spectra not available: {exc}")
    if s16 is not None:
        if s16.exceptions:
            warns.append("ASCE 7-16 11.4.8 requires a site-specific analysis "
                         f"({', '.join(s16.exceptions)}); the 11.4.8 exception was applied.")
            if "E_S1" in s16.exceptions:
                warns.append("ASCE 7-16 11.4.8 exception 3 (class E, S1 >= 0.2) is valid only "
                             "for T <= Ts with the ELF procedure: verify.")
        a16 = {"parameters": s16.parameters(), "exceptions": list(s16.exceptions),
               "tl_source": "input" if tl_asce is not None else "NEC TL of the site class",
               "spectrum": _curve(PERIODS, s16.sa(PERIODS))}
        s22 = ASCE7_22Spectrum(s16.sms, s16.sm1, tl_site, classes["asce7_22"]
                               if classes["asce7_22"] != "F" else "E")
        a22 = {"parameters": {**s22.parameters(), "Fa_7_16": s16.fa, "Fv_7_16": s16.fv},
               "approximate": True,
               "note": "SMS/SM1 from ASCE 7-16 Fa/Fv (no USGS geodatabase for Ecuador); "
                       "7-22 two-period spectrum (11.4.5.2).",
               "spectrum": _curve(PERIODS, s22.sa(PERIODS))}

    # --- spectra ----------------------------------------------------------------
    nec = {"parameters": None if is_f else nec_site.parameters(),
           "rock_parameters": nec_rock.parameters(),
           "spectrum": None if is_f else _curve(PERIODS, nec_site.sa(PERIODS)),
           "rock_spectrum": _curve(PERIODS, nec_rock.sa(PERIODS))}

    igepn = {"cell_id": ig.cell_id, "distance_km": float(ig.distance_km), "periods": ig_T,
             "uhs": ig_uhs}

    rock = {
        "nec_475": nec["rock_spectrum"],
        "asce_design": _curve(PERIODS, asce_rock.sa(PERIODS)),
        "igepn_475": _curve(ig_T, ig_uhs["475"]["mean"]),
        "igepn_2475_x2_3": _curve(ig_T, 2 / 3 * ig_uhs["2475"]["mean"]),
        "nec_uhs_475": nec_uhs["475"],
    }

    site = None
    if not is_f:
        amp = nec_site.sa(ig_T) / nec_rock.sa(ig_T)
        site = {
            "nec": nec["spectrum"],
            "asce7_16": a16["spectrum"] if a16 else None,
            "asce7_22": a22["spectrum"] if a22 else None,
            "igepn_475_scaled": _curve(ig_T, ig_uhs["475"]["mean"] * amp),
            "igepn_2475_x2_3_scaled": _curve(ig_T, 2 / 3 * ig_uhs["2475"]["mean"] * amp),
            "nec_amplification": _curve(ig_T, amp),
        }
        igepn["site_scaled_note"] = ("Approximate: rock UHS x NEC amplification "
                                     "Sa_NEC,site(T) / Sa_NEC,B(T).")

    # --- comparison table at 0.2 s and 1.0 s -------------------------------------
    comparison = []
    for T in REF_PERIODS:
        k = int(np.flatnonzero(np.isclose(ig_T, T))[0])
        row = {"T": T,
               "rock": {"nec_475": float(nec_rock.sa(T)[0]),
                        "asce_design": float(asce_rock.sa(T)[0]),
                        "igepn_475": float(ig_uhs["475"]["mean"][k]),
                        "igepn_2475_x2_3": float(2 / 3 * ig_uhs["2475"]["mean"][k])}}
        if site is not None:
            row["site"] = {"nec": float(nec_site.sa(T)[0]),
                           "asce7_16": float(s16.sa(T)[0]) if s16 else None,
                           "asce7_22": float(s22.sa(T)[0]) if s22 else None,
                           "igepn_475_scaled": float(site["igepn_475_scaled"]["Sa"][k])}
        comparison.append(row)

    inputs = {"lat": float(lat), "lon": float(lon), "vs30": vs30, "site_class": site_class,
              "method": method, "ss": ss, "s1": s1, "z": z, "region": region, "tl_asce": tl_asce}
    zone = {**zn.to_dict(), "z_used": z_used, "region_used": region_used,
            "eta_used": float(nec_rock.eta)}
    return SiteAssessment(inputs, zone, classes, ss_s1, nec, nec_uhs, a16, a22, igepn,
                          rock, site, comparison, warns)
