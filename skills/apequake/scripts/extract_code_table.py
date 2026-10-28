"""Extract a fixed-column table from a design-code PDF, and validate it.

Wraps the `pdftotext -table` recipe with the failure modes apeQuake (and codeSpectra before it) has
already hit baked in (see references/pdf-extraction.md):

* the value column is located by regex at end-of-line, not by the header's
  column offset -- headers and data rows are padded differently on some pages,
  which silently drops entire pages;
* cell wraps are merged from the line before *and* the line after, with
  trailing wraps attached only to fields left empty so captions are not
  absorbed as data;
* every non-ASCII character is audited, because embedded fonts mis-map
  accented capitals on some pages but not others.

It prints a validation report and refuses to write output when a check fails
unless --force is given. Silence is the enemy here: an extraction that
"works" while dropping a page looks exactly like one that worked.

Usage
-----
    python extract_code_table.py PDF --first 98 --last 116 \\
        --columns POBLACIÓN PARROQUIA CANTÓN PROVINCIA \\
        --value-pattern '0\\.\\d{2}' --value-name Z \\
        --expect-values 0.15 0.25 0.30 0.35 0.40 0.50 \\
        --stop-at 'Tabla 19' --repair-glyphs --out tabla19.json

Column names must match the printed header exactly, accents included (the
header line is matched literally).  On the office NEC-SE-DS PDF this recipe
gives 515 rows and passes every check (verified 2026-10).

Run with --dry-run first and read the report before trusting anything.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

#: Accented capitals that legitimately appear in Spanish place names.
EXPECTED_NON_ASCII = set("ÑÁÉÍÓÚÜ")

#: Glyph mis-mappings observed in the NEC-SE-DS PDF font. Only applied with
#: --repair-glyphs, and only ever where context proves the substitution.
KNOWN_GLYPH_REPAIRS = {"Ð": "Ñ", "┌": "Ú", "Ë": "Ó"}


def page_text(pdf: Path, page: int, mode: str = "-table") -> list[str]:
    """Return one page as lines, via Xpdf pdftotext."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "p.txt"
        subprocess.run(
            ["pdftotext", mode, "-enc", "UTF-8", "-f", str(page), "-l", str(page),
             str(pdf), str(out)],
            check=True, capture_output=True,
        )
        return out.read_text(encoding="utf-8").splitlines()


def parse_page(
    lines: list[str],
    columns: list[str],
    value_re: re.Pattern[str],
    stop_at: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse one page. Returns (rows, problems)."""
    rows: list[dict[str, Any]] = []
    problems: list[str] = []

    header_idx = None
    offsets: list[int] = []
    for i, line in enumerate(lines):
        if all(c in line for c in columns):
            header_idx, offsets = i, [line.index(c) for c in columns]
            break
    if header_idx is None:
        return rows, ["no header found"]

    bounds = list(zip(offsets, [*offsets[1:], 10_000], strict=True))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    buf = [""] * len(columns)

    for line in lines[header_idx + 1:]:
        if stop_at and stop_at in line:
            break
        if not line.strip() or line.strip().isdigit():
            continue
        if all(c in line for c in columns):        # header repeats mid-page
            continue

        match = value_re.search(line)
        if match:
            body = line[: match.start()]
            cells = [norm(body[a:b]) for a, b in bounds]
            merged = [norm(f"{buf[i]} {cells[i]}") for i in range(len(columns))]
            row = dict(zip(columns, merged, strict=True))
            row["_value"] = match.group(1)
            rows.append(row)
            buf = [""] * len(columns)
        else:
            cells = [norm(line[a:b]) for a, b in bounds]
            if any(cells):
                for i, cell in enumerate(cells):
                    if cell:
                        buf[i] = norm(f"{buf[i]} {cell}")

    # A trailing wrap belongs to the row above -- but only in a field that is
    # still empty, otherwise the table caption gets absorbed as data.
    if any(buf) and rows:
        for i, col in enumerate(columns):
            if buf[i] and not rows[-1][col]:
                rows[-1][col] = buf[i]
                buf[i] = ""
    if any(buf):
        problems.append(f"unconsumed buffer {buf}")
    return rows, problems


def validate(
    rows: list[dict[str, Any]],
    columns: list[str],
    expect_values: list[str] | None,
    expect_rows: int | None,
) -> list[str]:
    """Return a list of failures. Empty means every check passed."""
    failures: list[str] = []

    if not rows:
        return ["no rows extracted"]

    if expect_rows is not None and len(rows) != expect_rows:
        failures.append(f"row count {len(rows)}, expected {expect_rows}")

    blank = [c for c in columns if any(not r[c] for r in rows)]
    if blank:
        failures.append(f"empty cells in columns: {blank}")

    seen = sorted({r["_value"] for r in rows})
    if expect_values is not None:
        unexpected = set(seen) - set(expect_values)
        if unexpected:
            failures.append(
                f"values not in the standard's discrete set: {sorted(unexpected)}"
            )

    stray = {
        ch for r in rows for c in columns for ch in r[c]
        if not ch.isascii() and ch not in EXPECTED_NON_ASCII
    }
    if stray:
        detail = ", ".join(
            f"{ch!r} U+{ord(ch):04X} ({unicodedata.name(ch, '?')})" for ch in sorted(stray)
        )
        failures.append(f"unexpected non-ASCII, likely font mis-mapping: {detail}")

    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdf", type=Path)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--columns", nargs="+", required=True,
                   help="header labels, left to right, excluding the value column")
    p.add_argument("--value-pattern", default=r"0\.\d{2}",
                   help="regex for the value column, anchored at end-of-line")
    p.add_argument("--value-name", default="value")
    p.add_argument("--expect-values", nargs="*", default=None,
                   help="the discrete set the standard permits, e.g. 0.15 0.25 ...")
    p.add_argument("--expect-rows", type=int, default=None)
    p.add_argument("--stop-at", default=None, help="stop at a line containing this")
    p.add_argument("--repair-glyphs", action="store_true",
                   help="apply the known NEC font repairs (see module docstring)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="write output even if validation fails")
    args = p.parse_args()

    value_re = re.compile(rf"\s({args.value_pattern})\s*$")
    rows: list[dict[str, Any]] = []
    problems: list[str] = []

    for page in range(args.first, args.last + 1):
        page_rows, page_problems = parse_page(
            page_text(args.pdf, page), args.columns, value_re, args.stop_at
        )
        rows.extend(page_rows)
        problems.extend(f"page {page}: {m}" for m in page_problems)
        print(f"  page {page:>4}: {len(page_rows):>4} rows", file=sys.stderr)

    if args.repair_glyphs:
        n = 0
        for row in rows:
            for col in args.columns:
                fixed = row[col]
                for bad, good in KNOWN_GLYPH_REPAIRS.items():
                    fixed = fixed.replace(bad, good)
                if fixed != row[col]:
                    row[col], n = fixed, n + 1
        print(f"\nglyph repairs applied: {n}", file=sys.stderr)

    print(f"\n{'=' * 60}\nrows extracted: {len(rows)}")
    print(f"distinct {args.value_name}: "
          f"{sorted(Counter(r['_value'] for r in rows).items())}")
    chars = Counter(ch for r in rows for c in args.columns for ch in r[c]
                    if not ch.isascii())
    print(f"non-ASCII: {dict(chars) or 'none'}")
    print(f"parse problems: {problems or 'none'}")

    failures = validate(rows, args.columns, args.expect_values, args.expect_rows)
    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nDo not ship this table. Read references/pdf-extraction.md; the "
              "usual causes are a page whose header pads the value column "
              "differently, or a font mis-mapping accented capitals.")
    else:
        print("\nvalidation: all checks passed")

    print("\nSpot-check these against the printed table by hand before shipping:")
    for row in rows[:5] + rows[len(rows) // 2: len(rows) // 2 + 3] + rows[-3:]:
        cells = " | ".join(row[c] for c in args.columns)
        print(f"  {cells} | {args.value_name}={row['_value']}")
    print("=" * 60)

    if args.dry_run:
        return 1 if failures else 0
    if failures and not args.force:
        print("\nRefusing to write output. Pass --force to override.", file=sys.stderr)
        return 1
    if args.out:
        payload = {
            "source": f"{args.pdf.name}, pages {args.first}-{args.last}",
            "note": "Extracted with extract_code_table.py. Verify against the "
                    "printed table before use."
                    + (" Known font glyphs repaired." if args.repair_glyphs else ""),
            "fields": [*args.columns, args.value_name],
            "rows": [[row[c] for c in args.columns] + [row["_value"]] for row in rows],
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=0),
                            encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
