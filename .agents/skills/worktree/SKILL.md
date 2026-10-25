---
name: worktree
description: Use when starting work on a trtutils PR or branch and you need an isolated, runnable checkout - "set up a worktree", "new branch for PR", "make this worktree runnable", "tests import the wrong trtutils", "missing ONNX/engines in the worktree".
---

# trtutils worktree setup

One worktree per PR. Never edit the main checkout or a worktree another agent is using.
Below, `<main>` is the main checkout and `<wt>` the new worktree.

## Steps

1. Create the worktree from fresh `origin/main`:

   ```bash
   git -C <main> fetch origin
   git -C <main> worktree add <wt> -b <type>/<short-name> origin/main
   ```

   For an existing PR branch: `git -C <main> worktree add <wt> <branch>`.

2. Link the lint/type tools (used by `make fix`, `make typecheck`, pre-commit):

   ```bash
   ln -s <main>/.venv-ci <wt>/.venv-ci
   ```

   `.venv-ci` is not in `.gitignore`; keep it in `.git/info/exclude` (shared by all worktrees)
   so it is never committed.

3. Link test data. `data/` is mostly gitignored (only the test images and `simple.onnx` are
   tracked), so link each missing entry from the main checkout:

   ```bash
   cd <wt>
   for e in <main>/data/*; do n=$(basename "$e"); [ -e "data/$n" ] || ln -s "$e" "data/$n"; done
   ```

   For a new model family, make a real `data/<family>/` directory instead of a symlink so new
   ONNX files do not land in the main checkout. Engines cached through a linked
   `data/engines` are shared; they carry the TensorRT version in their filename.

4. Use the shared venv's python with `PYTHONPATH=src`. The `.venv` install of trtutils is
   NOT editable, so without it every run silently tests the stale installed copy:

   ```bash
   PYTHONPATH=src <main>/.venv/bin/python -m pytest -q -p no:cacheprovider tests/download/test_config.py
   ```

   In ad-hoc GPU scripts, log `trtutils.__file__` first to prove which source ran.

5. Check the toolchain: `make fix && make typecheck` should be clean on a fresh worktree.

## Gotchas

- GPU results that alias pinned host memory segfault when pickled or `np.save`d after the
  model is deleted; deep-copy arrays when you capture them.
- When the disk fills, the usual culprits are stale `/tmp/tmp*` download dirs and
  `~/.cache/uv` (`uv cache clean`). Chain cleanup steps with `;`, not `&&`.
- Remove the worktree when the PR merges: `git -C <main> worktree remove <wt>`.
