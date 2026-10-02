"""Structured warnings: a stable ``code``, its ``params`` and an English ``text``.

Clients render ``code`` + ``params`` in their own language (ape-tools: Spanish) and
fall back to ``text`` for codes they do not know.  ``warnings`` lists elsewhere in
apeQuake are the ``text`` of these notices, kept for compatibility.

Codes
-----
zoning (``zone_at``):
    ``near_boundary`` {boundaryKm, zAcross}; ``table19_differs`` {town, canton,
    distanceKm, zTable, zMap}; ``zona_no_delimitada`` {}.
site assessment (``assess_site``):
    ``site_class_f`` {}; ``z_override`` {z, zMap}; ``region_override`` {region,
    regionMap}; ``igepn_note`` {message}; ``nec_city_far`` {city, distanceKm};
    ``asce_unavailable`` {reason}; ``asce716_trigger`` {triggers};
    ``asce716_exception3`` {}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODES = (
    "near_boundary", "table19_differs", "zona_no_delimitada",
    "site_class_f", "z_override", "region_override", "igepn_note", "nec_city_far",
    "asce_unavailable", "asce716_trigger", "asce716_exception3",
)


@dataclass(frozen=True)
class Notice:
    code: str
    text: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "params": dict(self.params), "text": self.text}
