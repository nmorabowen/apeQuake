"""Shape checks for request arguments, mirroring the JSON Schemas without jsonschema.

The schemas under ``api/schemas/`` are the contract; these checks enforce the same
shapes at runtime so the package needs no ``jsonschema`` dependency.  The tests prove
both agree (every fixture the schema rejects comes back ``bad_request``).
"""
from __future__ import annotations

import math
import unicodedata
from typing import Any, Iterable

_ECHO = 60


class ArgError(ValueError):
    """The request or its arguments do not match the schema (``bad_request``)."""


def printable(text: str) -> str:
    """Replace control, bidi-override and lone-surrogate characters with U+FFFD."""
    out = []
    for ch in str(text):
        cat = unicodedata.category(ch)
        bad = cat in ("Cc", "Cs") and ch not in "\n\t" or ch in "‪‫‬‭‮⁦⁧⁨⁩"
        out.append("�" if bad else ch)
    return "".join(out)


def echo(value: object) -> str:
    """A short, printable rendering of a request value for error messages."""
    s = printable(str(value))
    return s if len(s) <= _ECHO else s[: _ECHO - 1] + "…"


def obj(x: object, name: str, *, required: Iterable[str] = (),
        optional: Iterable[str] = ()) -> dict[str, Any]:
    """A closed JSON object with the given required and optional keys."""
    if not isinstance(x, dict):
        raise ArgError(f"{name}: expected an object")
    req, opt = tuple(required), tuple(optional)
    for k in x:
        if not isinstance(k, str):
            raise ArgError(f"{name}: keys must be strings")
        if k not in req and k not in opt:
            raise ArgError(f"{name}: unknown field «{echo(k)}»")
    for k in req:
        if k not in x:
            raise ArgError(f"{name}: missing field «{k}»")
    return x


def number(x: object, name: str, *, minimum: float | None = None,
           maximum: float | None = None, exclusive_minimum: float | None = None) -> float:
    """A finite JSON number (bool is not a number) within the bounds."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ArgError(f"{name}: expected a finite number")
    v = float(x)
    if minimum is not None and v < minimum:
        raise ArgError(f"{name}: must be >= {minimum:g}")
    if maximum is not None and v > maximum:
        raise ArgError(f"{name}: must be <= {maximum:g}")
    if exclusive_minimum is not None and v <= exclusive_minimum:
        raise ArgError(f"{name}: must be > {exclusive_minimum:g}")
    return v


def enum(x: object, name: str, values: Iterable[str]) -> str:
    vals = tuple(values)
    if not isinstance(x, str) or x not in vals:
        raise ArgError(f"{name}: expected one of {', '.join(vals)}")
    return x


def string(x: object, name: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise ArgError(f"{name}: expected a non-blank string")
    return x
