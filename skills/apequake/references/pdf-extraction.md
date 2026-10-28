# Design-code PDFs: verifying clauses and extracting tables

Every coefficient, equation and table in apeQuake must be transcribed from the
standard itself, never from memory or from another library. This is how to do
that against the office PDFs, and the failures already hit doing it. They are
all **silent**: the extraction succeeds, looks plausible, and quietly drops or
corrupts rows. The validation section is the important half.

Moved here from codeSpectra's `codespectra-helper` skill (retired), plus what
the 2026-10 port of codeSpectra into apeQuake found.

## 1. Where the PDFs are

`C:\Users\nmora\seadrive_root\nmb\My Libraries\Libros\Codigos\`, one folder per
body: `ASCE\ASCE 7-10.pdf`, `ASCE\ASCE 7-16.pdf`, `ASCE\ASCE 7-22.pdf`,
`Ecuador\NEC\NEC_SE_DS_(peligro sismico).pdf`, `Chile\NCh433-...pdf`, and so on.

## 2. Which tool

| Tool | Use for | Notes |
|---|---|---|
| `pdftotext -layout -enc UTF-8` (Xpdf 4.00, `/mingw64/bin`) | Reading clauses; grepping a whole standard | **No `-bbox-layout`** (Xpdf, not Poppler). `-table` exists and is best for simple tabular pages |
| PyMuPDF `page.get_text("words")` (`import fitz`) | Multi-page tables, positional parsing | What `scripts/digitize_nec_zone_map.py` uses for Tabla 19: column edges from each page's header words |
| `scripts/extract_code_table.py` (this skill) | Fixed-column tables via `pdftotext -table`, with the checks below built in | Run with `--dry-run` first. `--columns` must match the printed header **with accents** (`POBLACIÓN … CANTÓN`); unaccented names find no header. The Tabla 19 recipe in its docstring gives 515 rows, all checks passing |

If the output looks mangled in the terminal, suspect the console code page
before the file. Check the bytes, or set `PYTHONIOENCODING=utf-8`.

## 3. Verifying a clause (the routine check)

Extract the whole standard once into the scratchpad, then grep it:

```bash
pdftotext -layout -enc UTF-8 "…/ASCE/ASCE 7-22.pdf" asce722.txt
grep -n "12.8.1.1 Calculation of Seismic Response" asce722.txt   # first hits are the TOC
sed -n 12318,12470p asce722.txt
```

- The **first matches are the table of contents** (and the commentary,
  "C12.8…"). Take the body occurrence.
- Pages are **two-column**. A clause and its equations may sit in different
  columns of the same line range; read the whole page, or `cut -c` a column.
- **Table cells come out misaligned against their row labels.** ASCE 7-16
  Table 12.8-1 prints Cu shifted one row against SD1, ASCE 7-22 Table 12.8-2
  prints Ct and x out of line with the structure types, and ASCE 7-16 Table
  11.9-1 interleaves its rows with the clause text. Rebuild each row from
  the order of values and check it against a second edition or a raw
  (`pdftotext` without `-layout`) extraction.
- Check **every edition you implement**. Numbering and content change: 7-22
  added §12.8.1.1 Method 1 and renumbered Eqs. 12.8-2 to 12.8-8. Copying 7-16
  numbering into 7-22 results would cite the wrong equations.
- Read the **whole clause, not only the equations**. ASCE 7-16 §11.9.2 ends
  with "SaMv shall not be less than one-half of the corresponding SaM for
  horizontal components". codeSpectra implemented Eqs. 11.9-1 to 11.9-4 and
  missed that floor, which governs most of 0–2 s on stiff sites.

## 4. Finding the pages of a table

Don't trust printed page numbers (front matter). Probe by what the data
looks like, not by the caption, which also appears in the TOC:

```bash
for p in $(seq 90 130); do
  n=$(pdftotext -table -enc UTF-8 -f $p -l $p "$PDF" - 2>/dev/null \
      | grep -cE "0\.(15|25|30|35|40|50)$")
  [ "$n" -gt 0 ] && echo "page $p: $n candidate rows"
done
```

## 5. Multi-page tables with wrapped cells (NEC Tabla 19)

- **Locate the value column by pattern, not by the header's offset.** One page
  of Tabla 19 pads its header differently from its data rows. Slicing by
  header offset silently dropped that page: 21 towns, including Cuenca.
  With `pdftotext`, use a regex at end of line (`\s(0\.\d{2})\s*$`). With
  PyMuPDF, match the value words by pattern near the Z header's x.
- **Wraps go both ways.** A wrapped name usually sits on the line before its
  data row, sometimes after. Attach a trailing wrap only to fields still
  empty, or the caption is absorbed as data. Stop at the caption.

## 6. Character corruption

The NEC-SE-DS font mis-maps three accented capitals:

| Extracted | Actual | Example |
|---|---|---|
| `U+00D0` Ð | Ñ | `BAÐOS` → BAÑOS |
| `U+250C` ┌ | Ú | `SUC┌A` → SUCÚA |
| `U+00CB` Ë | Ó | `SIMËN BOLIVAR` → SIMÓN BOLIVAR |

The same PDF renders Ñ correctly elsewhere: the corruption is per font, not
per document, so spot checks miss it. apeQuake's `nec_table19.csv` shipped
`SUC┌A` and `SIMËN` until 2026-10 (now `_FONT_FIX` in
`digitize_nec_zone_map.py`, with a regression test). **Audit every non-ASCII
character** of an extracted table:

```python
from collections import Counter
Counter(ch for row in rows for f in fields for ch in row[f] if not ch.isascii())
```

Repair a glyph only where context proves it (the same word spelled correctly
in a neighbouring column, an unmistakable place name). Otherwise flag it.

## 7. Validation (run all before shipping a table)

| Check | Catches |
|---|---|
| Row count vs an independent count of value-bearing lines | dropped pages |
| Distinct values vs the standard's own discrete set | column desync, stray captures |
| No empty required fields | wrap handling |
| No leftover parse buffer at page end | truncated rows |
| Non-ASCII audit | font corruption |
| 10–15 hand-checked well-known rows | everything else |
| Uniqueness of the lookup key | silently collapsing distinct rows |

The last check matters. Tabla 19 has 30 repeated `poblacion` names; 24 span
provinces and 20 of those carry a **different Z**. A name-only key returns
the wrong design value. Even `(name, province)` is not unique: `PUEBLO NUEVO`
in Guayas sits in two cantones with Z 0.40 and 0.50. Widen the key or refuse
the ambiguous entry; never pick one.

## 8. Storing the result

- Small coefficient tables go in module-level dict or tuple literals next to
  the code that uses them, with a comment naming the table, edition and page
  (see `code_spectrum/codes/asce7_16.py`, `design/asce7.py`).
- Large row tables go in `src/apeQuake/<pkg>/data/`, read with
  `importlib.resources`. Keep the script that generated them under
  `scripts/`, so a regeneration reproduces the repairs.
- When a table is cross-checked against another implementation, say so in
  the test module's docstring and pin any intended difference with a test.

## 9. Deferred values and escape hatches

Where a table defers to a site-specific study (ASCE 7-16 Fa for Site Class E
at Ss ≥ 1.0, Fv for E with S1 > 0.1), store `None` and **raise** with the
clause, rather than interpolating toward an undefined cell. Escape hatches are
explicit and named after the provision they invoke (`allow_exception=True`
for §11.4.8; `from_sds_sd1()` for site-specific values), never a generic
`strict=False`.

## 10. External data and licensing

Ship the reader, not the data, unless the licence clearly permits
redistribution:

| Source | Licence | What may ship |
|---|---|---|
| ASCE 7, NEC-SE-DS tables | transcribed provisions | the values, cited |
| GeoNames | CC BY 4.0 | derived subset, with attribution |
| Palacios, Celi & Poveda 2023 PSHA contours | no LICENSE (all rights reserved) | nothing; a reader for a local copy |
| PEER NGA, RENAC records | provider terms | not bundled; tests use synthetic copies of the headers |

Citation is not a licence.
