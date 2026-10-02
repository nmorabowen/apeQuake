"""The dispatcher: one request object in, one response object out, never raises."""
from __future__ import annotations

import json
from typing import Any, Literal, get_args

from . import checks as c
from . import handlers as h
from .schema_files import COMMON, ENVELOPE, load_schema
from ..nec.zoning import OutsideEcuadorError

API = "apeQuake/1"
"""Contract identifier carried by every request and response."""

API_VERSION = "1.0.2"
"""Semantic version of the ``apeQuake/1`` contract (additive changes bump the minor)."""

ErrorCode = Literal["bad_request", "unknown_command", "out_of_area", "value_error",
                    "internal_error"]
ERROR_CODES: tuple[ErrorCode, ...] = get_args(ErrorCode)

_MAX_MESSAGE = 600

# Most specific first: ArgError and OutsideEcuadorError are ValueError subclasses.
_ERROR_MAP: tuple[tuple[type[Exception], ErrorCode], ...] = (
    (c.ArgError, "bad_request"),
    (OutsideEcuadorError, "out_of_area"),
    (ValueError, "value_error"),
)


def _describe(args: dict[str, Any]) -> dict[str, Any]:
    c.obj(args, "args")
    return {
        "api": API,
        "version": API_VERSION,
        "commands": [
            {"name": cmd.name, "summary": cmd.summary,
             "argsSchema": schema(f"{cmd.name}.args"),
             "resultSchema": schema(f"{cmd.name}.result")}
            for cmd in REGISTRY.values()
        ],
        "common": schema(COMMON),
    }


REGISTRY: dict[str, h.Command] = {
    cmd.name: cmd
    for cmd in (
        h.Command("api.describe", "The API version, its commands and their JSON Schemas.",
                  _describe),
        h.Command("zoning.at",
                  "NEC-SE-DS zone factor Z, zone, region and eta at a point (Figura 1), "
                  "with the nearest Tabla 19 town.", h.zoning_at),
        h.Command("site.assess",
                  "NEC-SE-DS vs ASCE 7-16 / 7-22 (approximate) vs IG-EPN at a point: "
                  "parameters, spectra and rock / site comparisons.", h.site_assess),
    )
}

_SCHEMA_NAMES: frozenset[str] = frozenset(
    {COMMON, *ENVELOPE} | {f"{n}.{part}" for n in REGISTRY for part in ("args", "result")}
)


def commands() -> tuple[str, ...]:
    """Names of the registered commands, in registry order."""
    return tuple(REGISTRY)


def schema_names() -> tuple[str, ...]:
    """Names accepted by :func:`schema`, sorted."""
    return tuple(sorted(_SCHEMA_NAMES))


def schema(name: str) -> dict[str, Any]:
    """A JSON Schema (draft 2020-12) of the contract, as a fresh dict.

    ``name`` is ``common``, ``envelope.request``, ``envelope.response``, or
    ``<command>.args`` / ``<command>.result``.  Raises ``KeyError`` for others.
    """
    return load_schema(name, _SCHEMA_NAMES)


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"api": API, "ok": True, "result": result}


def error(code: ErrorCode, message: str) -> dict[str, Any]:
    message = c.printable(message)
    if len(message) > _MAX_MESSAGE:
        message = message[: _MAX_MESSAGE - 1] + "…"
    return {"api": API, "ok": False, "error": {"code": code, "message": message}}


def _run(request: object) -> dict[str, Any]:
    req = c.obj(request, "request", required=("api", "command"), optional=("args",))
    if req["api"] != API:
        raise c.ArgError(f'request.api: expected "{API}"')
    name = c.string(req["command"], "request.command")
    command = REGISTRY.get(name)
    if command is None:
        return error("unknown_command", f"unknown command «{c.echo(name)}»; see api.describe")
    args = req.get("args", {})
    if not isinstance(args, dict):
        raise c.ArgError("args: expected an object")
    result = command.handler(args)
    try:
        json.dumps(result, allow_nan=False)
    except ValueError:
        raise ValueError("a result value is not a finite number") from None
    return _ok(result)


def dispatch(request: object) -> dict[str, Any]:
    """Run one request and return its response.  Never raises.

    A domain error comes back as ``ok: false`` with a stable ``code``; an unexpected
    failure as ``internal_error`` carrying only the exception type.
    """
    try:
        return _run(request)
    except Exception as exc:  # the contract: dispatch never raises
        for kind, code in _ERROR_MAP:
            if isinstance(exc, kind):
                return error(code, str(exc))
        return error("internal_error",
                     f"internal error ({type(exc).__name__}); report the request that caused it")
