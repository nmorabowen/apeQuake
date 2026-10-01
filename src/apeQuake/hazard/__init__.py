"""Seismic hazard of Ecuador from the IG-EPN probabilistic model (Beauval et al., 2018)."""
from ._data import PERIODS, RETURN_PERIODS, STATS
from .curves import poe_from_tr, powerlaw_fit, tr_from_poe
from .ecuador import EcuadorHazard
from .live import fetch_recent_events
from .site import HazardSite

__all__ = [
    "EcuadorHazard",
    "HazardSite",
    "fetch_recent_events",
    "tr_from_poe",
    "poe_from_tr",
    "powerlaw_fit",
    "PERIODS",
    "RETURN_PERIODS",
    "STATS",
]
