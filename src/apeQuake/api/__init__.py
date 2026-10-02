"""The JSON API that ape-tools calls (``apeQuake/1``), modelled on apeLoads ADR-0006.

One pure function, :func:`dispatch`, takes a request object and returns a response
object; it never raises.  Every command's ``args`` and ``result`` has a JSON Schema
(draft 2020-12) shipped under ``api/schemas/``::

    >>> from apeQuake.api import dispatch
    >>> r = dispatch({"api": "apeQuake/1", "command": "zoning.at",
    ...               "args": {"lat": -0.22, "lon": -78.51}})
    >>> r["ok"], r["result"]["z"], r["result"]["region"], r["result"]["eta"]
    (True, 0.4, 'sierra', 2.48)

Requests and responses::

    request   {"api": "apeQuake/1", "command": str, "args": object}
    response  {"api": "apeQuake/1", "ok": true,  "result": {...}}
           or {"api": "apeQuake/1", "ok": false, "error": {"code": str, "message": str}}

Error codes: ``bad_request`` (shape does not match the schema), ``unknown_command``,
``out_of_area`` (point outside Ecuador / the NEC zone map), ``value_error`` (the
domain refuses a value), ``internal_error`` (a bug; the message carries only the
exception type).

Units on the wire: degrees (WGS-84), g, s, m/s (Vs30), km.  Nothing in a request
names a file.  Command line: ``python -m apeQuake.api`` reads a request from stdin
and writes the response to stdout (exit 0 ok, 1 not ok).
"""
from .dispatcher import API, API_VERSION, ERROR_CODES, ErrorCode, commands, dispatch, schema, schema_names

__all__ = ["API", "API_VERSION", "ERROR_CODES", "ErrorCode", "commands", "dispatch",
           "schema", "schema_names"]
