# JSON API (`apeQuake/1`)

`apeQuake.api.dispatch(request) -> response` is the entry point ape-tools uses. It
is one pure function that never raises. Every command's arguments and result
are described by a JSON Schema (draft 2020-12) shipped in `apeQuake/api/schemas/`,
and those files are the contract. The design follows apeLoads ADR-0006.

```python
from apeQuake.api import dispatch

dispatch({"api": "apeQuake/1", "command": "site.assess",
          "args": {"lat": -0.22, "lon": -78.51, "vs30": 250, "method": "nec"}})
```

| Command | What it returns |
|---|---|
| `api.describe` | The version, the commands, and their schemas. |
| `zoning.at` | NEC Z, zone, province, region, η, distance to the nearest zone boundary, and the nearest Tabla 19 town. |
| `site.assess` | The NEC-SE-DS, ASCE 7-16, ASCE 7-22 (approximate) and IG-EPN comparison: parameters, spectra, the rock and site views, and warnings. |

**Error codes:**

- `bad_request`: the request doesn't match the schema.
- `unknown_command`: no command by that name.
- `out_of_area`: the point is outside Ecuador or the NEC zone map.
- `value_error`: the domain refused a value.
- `internal_error`: a bug. The message carries only the exception type.

**Units:** degrees (WGS-84), g, s, m/s for Vs30, and km.

**Command line:** `python -m apeQuake.api` reads one request on stdin and writes
the response to stdout.

::: apeQuake.api.dispatch
