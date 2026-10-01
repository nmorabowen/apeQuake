"""Live queries to IG-EPN services (not part of the bundled snapshot)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

_RECENT = ("https://services8.arcgis.com/WFUbp7xZ6MJpsSuI/arcgis/rest/services/"
           + urllib.parse.quote("sismos_180_días_WFL1") + "/FeatureServer/0/query")


def fetch_recent_events(timeout: float = 30.0) -> pd.DataFrame:
    """Earthquakes located by IG-EPN in the last 180 days (requires internet).

    Returns one row per event: ``event_id``, ``time`` (UTC), ``lat``, ``lon``,
    ``depth_km``, ``magnitude``, ``magnitude_type``, ``status`` plus the network
    quality fields reported by IG-EPN.
    """
    feats, start = [], 0
    while True:
        params = {"where": "1=1", "outFields": "*", "returnGeometry": "false",
                  "resultOffset": start, "resultRecordCount": 2000, "f": "json"}
        with urllib.request.urlopen(_RECENT + "?" + urllib.parse.urlencode(params),
                                    timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        if "error" in d:
            raise RuntimeError(f"IG-EPN service error: {d['error']}")
        feats += [f["attributes"] for f in d["features"]]
        if not d["features"] or not d.get("exceededTransferLimit"):
            break
        start += len(d["features"])
    df = pd.DataFrame(feats)
    ren = {"etec_evento": "event_id", "etec_latitud": "lat", "etec_longitud": "lon",
           "etec_profundidad": "depth_km", "etec_magnitud_M": "magnitude",
           "etec_tipo_magnitud_P": "magnitude_type", "etec_estadoEvaluacion": "status",
           "etec_azimuthalGap": "azimuthal_gap", "etec_fasesUsadas": "phases_used",
           "etec_estacionesUsadas": "stations_used", "etec_infoAdicional": "info"}
    df = df.rename(columns=ren)
    if "etec_tiempo" in df:
        t = df["etec_tiempo"]
        df["time"] = (pd.to_datetime(t, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(t)
                      else pd.to_datetime(t, utc=True, errors="coerce"))
    keep = ["event_id", "time", "lat", "lon", "depth_km", "magnitude", "magnitude_type",
            "status", "azimuthal_gap", "phases_used", "stations_used", "info"]
    return (df[[c for c in keep if c in df]]
            .sort_values("time", ascending=False).reset_index(drop=True))
