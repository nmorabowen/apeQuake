"""``python -m apeQuake.api``: one request on stdin, the response on stdout.

Exit 0 when ``ok`` is true, 1 otherwise.  Invalid JSON (including NaN / Infinity) is
answered as ``bad_request``.  Output is ASCII JSON.
"""
from __future__ import annotations

import json
import sys

from .dispatcher import dispatch, error


def _reject_constant(name: str) -> None:
    raise ValueError(f"invalid JSON constant {name}")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read(), parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        response = error("bad_request", "request is not valid JSON")
    else:
        response = dispatch(request)
    sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
