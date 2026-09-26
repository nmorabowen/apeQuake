# apeQuake — working rules for agents

apeQuake is a small Python library for seismic ground-motion processing
(`github.com/nmorabowen/apeQuake`, default branch `main`). Everything hangs off
one `Record` (a pandas-backed X/Y/Z time history) with composite processors
attached to it (`rec.filter`, `rec.spectrum`, `rec.response_spectra`, ...). It
is a sibling of apeGmsh (`nmorabowen/apeGmsh`) in the ape* suite. User docs:
https://nmorabowen.github.io/apeQuake/.

`CLAUDE.md` is one line (`@AGENTS.md`): this file is the single source for
Claude, Codex and any other agent.

## Task guides

None yet. Nearly all work in this repo is one kind — adding or changing a
composite on `Record` — and its checklist is the "Composite conventions"
section below. Add a guide under `.claude/skills/apequake-<task>/SKILL.md`
(tracked; the rest of `.claude/` is ignored) only when a second kind of work
recurs. Why there is no guide or lint yet: `docs/agent-surface.md`.

## Where things are

- `src/apeQuake/core/record.py` — `Record`. The composites are attached in
  `Record.__init__`; read that, don't copy the list here.
- `src/apeQuake/<composite>/` — one subpackage per composite; its
  `__init__.py` exports the class in `__all__`.
- `tests/` — pytest. The root `conftest.py` puts `src/` on `sys.path`, so the
  suite runs without installing the package.
- `docs/` + `mkdocs.yml` — the published user manual (MkDocs Material +
  mkdocstrings). Each `docs/api/<module>.md` renders docstrings through a
  `:::` block, so a docstring edit in `src/` is a docs change.
- `docs/agent-surface.md` — the plan doc for this agent surface. It is
  excluded from the site (`exclude_docs` in `mkdocs.yml`): the manual documents
  usage, not internals.
- The README has the project tree and roadmap; don't duplicate them here.

## Setup, tests, docs — exact commands

```bash
pip install -e ".[dev]"            # pytest, ruff, mypy (+ runtime deps)
python -m pytest -q                # 53 passed, ~20 s (2026-09-25)
pip install -e ".[docs]"
python -m mkdocs build --strict    # must be clean before a PR that touches docs/ or docstrings
```

Traps:

1. **No CI runs the tests.** The only workflow, `.github/workflows/docs.yml`,
   deploys the site on push to `main` (paths include `src/**`) and does not run
   on PRs. It runs `mkdocs gh-deploy --force`, which is *not* strict, so a docs
   warning ships silently. Run pytest and the strict build locally.
2. **obspy is a hard runtime dependency.** `from apeQuake import Record` works
   without it, but constructing a `Record` fails: `Record.__init__` attaches
   `Filter`, whose module imports obspy at the top.
3. **numba is optional at runtime.** `response_spectra.py` falls back to plain
   Python loops; `test_numba_kernel_matches_python_fallback` pins that both
   paths agree. Keep them equivalent.
4. **Plot methods default to `show=True`.** Tests that plot select a headless
   backend first (`matplotlib.use("Agg")` at the top of the file, as in
   `tests/test_spectrogram.py`).
5. **ruff and mypy are in `[dev]` but not enforced, and the tree is not clean**
   (2026-09-25: `ruff check .` 11 findings; `ruff format --check .` would
   rewrite 24 files; `mypy src` 26 errors). Don't mass-reformat or mass-fix in
   a feature PR; keep the files you add clean (`python -m ruff check <files>`).
6. **GitHub Pages had to be enabled once by hand** (done 2026-05-29). If the
   site 404s after a successful deploy, check `gh api repos/nmorabowen/apeQuake/pages`
   before debugging the workflow.

## Composite conventions (each one was learned from a fix)

When adding or changing a composite, keep all of these. The commit and test
named on each item are the record of why.

- [ ] **Registration.** New subpackage `src/apeQuake/<name>/` with `__all__`;
      attach it in `Record.__init__`; add it to both parametrize lists in
      `tests/test_exports.py` and to `test_composites_attached` in
      `tests/test_record.py` (those lists are hand-written); add
      `docs/api/<name>.md` and a `nav` entry in `mkdocs.yml`.
      `__all__` once named a class that did not exist (`b095710`,
      `test_all_entries_actually_exist`).
- [ ] **Never mutate `record.df`.** Work on a copy: the composite's own
      `df_<name>` (`set_df` / `reset_df`) or a caller's `df=` (copied).
      `test_filter_does_not_mutate_baseline`.
- [ ] **One filter pipeline, applied once.** Two shapes exist: (a) stateful
      `apply_*` methods that rewrite `df_<name>` and call `_invalidate()`
      (Spectrum, Spectrogram, IntensityMeasures); (b) ResponseSpectra's
      `add_filter` steps, applied inside `compute()` — `df=None` means filters
      on by default, a caller's `df` means off unless `use_filters=True`.
      Convenience wrappers delegate to `compute()`; they must not pre-copy
      `record.df` and pass it as a caller `df`, which silently drops the
      filters (`b095710`, `tests/test_response_spectra_filters.py`).
- [ ] **Plot from the cache.** A `plot_*` method consumes `compute()`'s cached
      result and never re-preprocesses. `plot_spectrogram` used to rebuild its
      own ObsPy stream and re-apply detrend/taper/band on top of the `apply_*`
      filters — silent double processing (`80a8c01`, `tests/test_spectrogram.py`).
- [ ] **No exact float comparisons on grids.** Uniformity of a time grid is a
      tolerance test, not `np.unique(np.diff(t))`; round-off in an `np.arange`
      grid once triggered a silent re-interpolation with a wrong `dt`
      (`b095710`, `Record._get_dt`, `tests/test_record.py`).
- [ ] **Validate numerics against an independent reference.** The bar is
      `tests/test_newmark.py`: Newmark against `scipy.integrate.solve_ivp`,
      a closed-form steady state, and the `Sa = ω²·Sd` identity.
- [ ] **No dead parameters.** A parameter that is accepted and documented must
      be read (`Spectrum.compare(combined=)` was not; removed in `9e9e416`).

## Docs site

- It mirrors apeGmsh's MkDocs setup (Material theme, warm-paper / deep-navy
  palette with a seismic-red accent, mkdocstrings API pages). When extending
  the docs, copy apeGmsh's conventions rather than inventing new ones.
- New guide pages go under `docs/guides/` and need a `nav` entry.

## Lessons and plans

There is no gotchas doc or ledger yet. Lessons live in the fix commits'
messages and in "Composite conventions" above. When you learn one, add an item
there with its fix commit and regression test. If a lesson that is already
written down bites again, record the recurrence in `docs/agent-surface.md` —
a recurrence of a greppable pattern is what earns a CI lint.

## PRs and branches

- Branch from fresh `main`; base every PR on `main`, never on another feature
  branch (stacked `--base` PRs merge into the dead branch, not `main`).
- One topic per PR, merged with a merge commit, as PRs #1–#4 were. Leave
  merging to the owner.
