---
name: trtutils-pr-pipeline
description: Use when taking a trtutils PR from implementation to merge, or reviving an old one - "run the PR pipeline", "review and fix this PR", "get PR #N ready to merge", "rebuild this stale PR on main", "simplify loop", "test on 5080 and Orin then report".
---

# trtutils PR pipeline

Review -> fix -> simplify -> test -> report -> merge (on the user's word only).
Work in a dedicated worktree (`trtutils-worktree` skill).

## 0. Stale PRs: rebuild, don't rebase

When a PR (or a stack of PRs) is behind `main`, rebuild it on current `main` instead of
rebasing the old stack:

1. List its commits: `git log --oneline origin/main..<branch>`.
2. For each commit, check whether `main` already has the change (`git log -S`, read the
   current code). Drop superseded commits; note which PR superseded them.
3. Create a fresh branch from `origin/main` and re-apply only what is still needed
   (`git cherry-pick` or rewrite by hand when the code moved).
4. Push to the PR's branch (`git push --force-with-lease`) and say in the PR what was dropped.

## 1. Review

Run a correctness review of the diff at a chosen effort level (e.g. `/code-review high`).
For each finding, reproduce it (a failing test or a short script) before accepting it.
Discard findings you cannot trigger.

## 2. Fix

Fix the confirmed findings. Each fix gets a test that fails without it. Then
`make fix && make typecheck`.

## 3. Simplify loop

Review the changed code from four angles, apply what holds up, and repeat until a round
finds nothing:

- **reuse**: existing helpers in `core/`, `image/`, `postprocessors/` instead of new ones
- **simplification**: fewer branches, fewer files, no speculative options
- **efficiency**: no extra copies or per-image Python loops on hot paths
- **altitude**: the change sits in the shared function all callers route through

Run the affected tests after every round.

## 4. Test

- RTX 5080 (local): `PYTHONPATH=src <main>/.venv/bin/python -m pytest -q -p no:cacheprovider tests`
- Jetson AGX Orin (TRT 8.6): same suite via the `trtutils-jetson-testing` skill
- Perf-sensitive changes: `trtutils-benchmarking`; memory-sensitive: `trtutils-memcheck`

Compare every failure against `main` on the same machine and data; only differences count.

## 5. Report to the user

PR URL, what changed, review findings and their fixes, simplify rounds, pass/fail/skip
counts per machine, any failures that also occur on `main`. Stop and wait.

## 6. Merge (only after the user says "merge")

```bash
gh pr merge <N> --squash --delete-branch
```

## Conventions

- Branch names: `feat/`, `fix/`, `perf/`, `docs/`, `tests/`, `chore/` + short name.
- Every commit message ends with:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: <session url>
  ```

- PR body: `## What`, `## Why`, `## Tests` (new tests, which fail without the fix, suite
  counts, `make fix`/`make typecheck` clean), ending with the Claude Code footer and session url.
  See `gh pr view 241 --json body` for an example.
- Delegating implementation to subagents: give each one the worktree path, exact test
  command, the files it may touch, and the commit command; review every diff yourself.
