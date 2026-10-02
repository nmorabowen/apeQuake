"""The JSON Schemas of the API, shipped as package data (``api/schemas/*.json``).

The schema files are the source of truth for the contract: the ape-tools TypeScript
types are generated from them.  Every schema has an absolute ``$id`` under
:data:`SCHEMA_BASE` (an identifier, never fetched); cross-file references are
relative (``"common.json#/$defs/curve"``) and resolve against it.
"""
from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

SCHEMA_BASE = "https://nmorabowen.github.io/apeQuake/schemas/apeQuake-1/"
COMMON = "common"
ENVELOPE = ("envelope.request", "envelope.response")


@cache
def _text(name: str) -> str:
    return resources.files("apeQuake.api").joinpath("schemas", f"{name}.json").read_text("utf-8")


def load_schema(name: str, known: frozenset[str]) -> dict[str, Any]:
    """A fresh copy of schema ``name``; ``KeyError`` for an unknown name (never a path)."""
    if name not in known:
        raise KeyError(name)
    return json.loads(_text(name))
