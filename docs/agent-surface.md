# Agent surface — AGENTS.md for apeQuake

Revision 1. Not yet adversarially reviewed.

Status: **built; draft PR.** Branch `claude/agent-surface`, cut from `main` @ `346d122`
(2026-09-25). Port of the agent-surface method piloted in the Ladruno OpenSees fork (WP-115: an
`AGENTS.md`, short task guides, and a quirk lint in CI). The method ports; the rules do not. Every
rule here has to come from apeQuake's own incidents.

This file is excluded from the published site (`exclude_docs` in `mkdocs.yml`).

## Problem

apeQuake had no agent-facing documentation: no `CLAUDE.md`/`AGENTS.md`, no CONTRIBUTING, no
CHANGELOG, no gotchas doc. What an agent needs before changing the library was spread across three
places. None of them is in front of the agent when it starts work:

- **Fix-commit messages and PR bodies.** `b095710` (four correctness bugs), `80a8c01` (spectrogram
  double processing), `9e9e416` (dead parameter, license). These come from a 2026-05-29 "deep
  review" whose numbered backlog (#4, #6, #7) is quoted in PRs #1 and #4. The review itself is not
  in the repo.
- **Machine-local memory.** `~/.claude/projects/C--Users-nmora-Github-apeQuake/memory/`
  (`apequake-overview.md`, `github-pages-manual-enable.md`), which only Claude sessions opened
  in the older clone `C:\Users\nmora\Github\apeQuake` load.
- **Nowhere.** For example: no CI runs the tests, the ruff/mypy baseline is not clean, and new
  composites must be added to hand-written test lists.

### Phase 0 baseline (the gate)

| Source | Found |
|---|---|
| In-repo lessons archive (gotchas, ledger, ADRs, CHANGELOG) | none |
| Fix commits on `main` | `b095710`, `80a8c01`, `9e9e416`; small Dec-2025 fixes `caf345f`, `c34c004` (feature tweaks, not lessons) |
| Memory entries about apeQuake | 2 files, 4 lessons (the only memory dir that mentions the repo) |
| Recurrences (a lesson already written down that bit again) | **0** |
| Commit/PR text matching again / recur / rediscover / regress / revert | none, apart from the words "regression test" |

The two pipeline incidents (response spectra silently dropping the registered filters, `b095710`,
and the spectrogram re-filtering at plot time, `80a8c01`) are one family: a composite bypasses the
filter pipeline. But the same review found both on the same day, and PR #1 lists the second as a
follow-up. Neither was written down before the other bit, so this is not a recurrence.

**Gate: fail.** There are no recurrences and no archive, so this is Phase 1 only.

## Shape

1. **`AGENTS.md` is the single source.** `CLAUDE.md` is the one line `@AGENTS.md`. There was no
   old `CLAUDE.md`, so nothing moves verbatim. `AGENTS.md` is written as the missing map: where
   things are, the exact setup/test/docs commands and their traps, the conventions each composite
   must keep (each tied to its fix commit and regression test), docs-site rules, where lessons go,
   and PR rules.
   *Accept:* every command in it was run on this branch, and every commit, test and file it names
   exists (see Results).
2. **Memory lessons move into the repo.** The lessons in memory about this repo are now in
   `AGENTS.md`: obspy is a hard runtime dependency, the docs mirror apeGmsh, the Newmark validation
   bar, and GitHub Pages needs a one-time manual enable. The memory files themselves are unchanged.
   *Accept:* each lesson has an `AGENTS.md` home (table below).
3. **`.gitignore`**: `.claude/*` + `!.claude/skills/`. `.claude/` was not ignored, so a worktree
   under `.claude/worktrees/` or a `settings.local.json` showed as untracked in the main checkout
   (observed: `?? .claude/` once this branch's worktree existed). `.claude/skills/` stays trackable
   for a future guide.
   *Accept:* `git check-ignore` ignores `.claude/settings.local.json` and `.claude/worktrees/x` but
   not `.claude/skills/probe/SKILL.md`.
4. **This plan doc** at `docs/agent-surface.md`. The repo has no plans/ADR folder, so it goes here.
   It is excluded from the MkDocs build so the user manual stays user-facing.
   *Accept:* `mkdocs build --strict` is clean and emits no `agent-surface/` page.

## Rejected approaches

- **A task guide (`.claude/skills/apequake-new-composite/`).** The history has one recurring kind
  of work: adding or changing a composite on `Record`. Examples: IM `de8900b`, plot_record
  `76d59b8`, response spectra `1d50565`, and the unmerged SSI-COV `modal_analysis` on
  `origin/add-modal-analysis` (`50ae3d1`). But a guide's checklist must *point into* an archive,
  and none exists. The library is about 2.9k lines and this is essentially its only kind of work.
  A separate guide would repeat `AGENTS.md`, which every agent already loads, and would be a
  second copy to keep in sync. The checklist lives in `AGENTS.md` under "Composite conventions".
  Revisit when a second kind of work recurs, or when the checklist outgrows `AGENTS.md`.
- **A quirk lint.** The gate failed: zero recurrences. The greppable candidates each have exactly
  one incident:
  - exact-equality grid uniformity (`np.unique(np.diff(...))`, `b095710`);
  - an `__all__` naming a missing symbol (`b095710`, already pinned by
    `tests/test_exports.py::test_all_entries_actually_exist`, so a lint adds nothing);
  - a pre-copied `record.df` passed as a caller `df` (`b095710`, not mechanically distinct from
    legitimate copies).
  Without a recurrence these stay checklist items. No rule was invented to have a lint.
- **Adding pytest/ruff/mypy to CI.** It would be useful, since nothing runs the tests on PRs, but it
  is a CI policy decision for the owner and out of scope for this port. `ruff`/`mypy` are not clean
  on `main` (11 / 26 findings), so gating them needs a cleanup first. Recorded as an open question.
- **Publishing this plan doc on the site.** The site is the end-user manual. Internals and agent
  notes stay out of it, which is why the doc is excluded.
- **Copying the README project tree or the composite list into `AGENTS.md`.** Both would drift.
  `AGENTS.md` points to the README and to `Record.__init__` instead.

## Results (2026-09-25)

| Check | Result |
|---|---|
| `python -m pytest -q` (worktree, Python 3.11.9) | 53 passed |
| `python -m mkdocs build --strict` | clean; `agent-surface.md` not in the built site |
| `git check-ignore` on `.claude/settings.local.json`, `.claude/worktrees/x` | ignored |
| `git check-ignore` on `.claude/skills/probe/SKILL.md` | not ignored (trackable) |
| obspy claim: `sys.modules['obspy']=None`, then import and construct | `from apeQuake import Record` ok; `Record(...)` raises `ModuleNotFoundError` |
| Baseline `ruff check .` / `ruff format --check .` / `mypy src` on `main` | 11 findings / 24 files to reformat / 26 errors |
| Named commits (`b095710`, `80a8c01`, `9e9e416`) and tests in `AGENTS.md` | all exist on `main` |
| New Python files added | none (no lint, so ruff/mypy have nothing new to check) |

Memory lessons moved:

| Memory file | Lesson | Home in `AGENTS.md` |
|---|---|---|
| `apequake-overview.md` | obspy is a hard dependency | Trap 2, refined: *importing* `Record` works without obspy; *constructing* one does not |
| `apequake-overview.md` | docs mirror the apeGmsh MkDocs setup | "Docs site" |
| `apequake-overview.md` | keep the Newmark-vs-`solve_ivp` bar | "Composite conventions", numerics item |
| `github-pages-manual-enable.md` | the Pages site 404s until enabled once | Trap 6 |

## Live incidents — merge order

None. No lint shipped, so there is nothing to find. No production code was changed.

## Phase 6 — measurement

Baseline: 0 recurrences, 2026-09-25. After about 10 PRs, count review or post-merge findings that
match an item already in "Composite conventions". A recurrence of a greppable item earns a CI lint
(`ci/check_quirk_patterns.py` shape from the Ladruno reference, with a mutation check against the
pre-fix and fix commits). Anything else stays a checklist item.

## Open questions

- **CI for tests.** Should a PR workflow run `pytest` (and later ruff/mypy, after a cleanup)? Today
  a broken test can only be caught locally.
- **`origin/add-modal-analysis` (`50ae3d1`, 2026-03-28).** It adds an SSI-COV `modal_analysis`
  composite on top of `1d50565`, before the May review fixes. It merges textually clean with
  `main` (`git merge-tree`), but it would not be in the hand-written lists in
  `tests/test_exports.py` / `test_composites_attached`, and it has no `docs/api` page. The older
  clone `C:\Users\nmora\Github\apeQuake` is checked out on that branch with uncommitted
  `modal_degradation/` work. Which checkout is canonical is the owner's call. The Claude memory
  for this repo is keyed to the older clone's path.
