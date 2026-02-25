# Phase 0 Plan Audit

## Status: RESOLVED — All issues addressed

## Summary

The Phase 0 plan is **well-structured and comprehensive**. It correctly identifies the infrastructure pieces needed. The original audit found **7 issues** — 3 bugs, 2 design problems, and 2 missing pieces. After review, 2 were false positives and the remaining 5 have been fixed.

---

## BUG 1: `data/` path is wrong in conftest.py

**Severity: Critical — tests won't find images/engines**

The plan's root `conftest.py` (Step 7) defines:
```python
DATA_DIR = BASE_DIR / "tests" / "data"
```

But the actual data lives at `data/` (project root), **not** `tests/data/`:
- `tests/paths.py:34` — `DATA_DIR = BASE_DIR / "data"` (goes to project root `/data/`)
- `tests/helpers.py:14` — `DATA_DIR = BASE_DIR / "data"` (same)

The plan's `conftest.py` would resolve to `<project>/tests/data/` which **does not exist**. The actual path is `<project>/data/`.

**Fix:** Change to `DATA_DIR = BASE_DIR / "data"`.

---

## BUG 2: Python 3.8 incompatible type hints

**Severity: Critical — SyntaxError on target Python**

The plan's `conftest.py` uses bare `dict[str, float]`, `list[np.ndarray]`, and `Path | None` which are **not valid at runtime on Python 3.8**. While `from __future__ import annotations` defers evaluation of function annotations, it does **not** defer `dict[...]` used as a return type hint in the `ci/check_coverage.py` script (which has no `__future__` import annotation deferral applied to runtime code):

In `ci/check_coverage.py` line 154:
```python
def get_coverage_by_module() -> dict[str, float]:
```
This uses PEP 585 syntax that requires Python 3.9+. And `module_coverage: dict[str, float] = {}` at line 171 is a **runtime** annotation that will fail on 3.8.

The root `conftest.py` is protected by `from __future__ import annotations`, but the coverage script is not.

**Fix:** In `ci/check_coverage.py`, add `from __future__ import annotations` and use `Dict` from typing for any runtime annotation (`module_coverage: Dict[str, float]` or just remove the annotation).

---

## BUG 3: `version_engine_path` behavior differs from existing code

**Severity: Medium — engine cache miss, rebuilds every time**

The plan's `version_engine_path`:
```python
def version_engine_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_{TRT_VERSION}{path.suffix}")
```

The existing `tests/paths.py` version handles `None` gracefully:
```python
def version_engine_path(base_path: Path, trt_version: str | None = None) -> Path:
    if trt_version is None:
        return base_path
```

The plan's version will produce filenames like `simple_unknown.engine` when TRT is not importable (returns `"unknown"` string), rather than falling back to the unversioned path. This is a subtle regression — the existing code returns the unversioned path when TRT is unavailable.

**Fix:** Match existing behavior — return `"unknown"` string still creates a valid path, but consider whether you want `simple_unknown.engine` or just `simple.engine` when TRT is missing.

---

## DESIGN 1: `make test` target conflict

**Severity: Medium — confusing existing workflows**

The plan says to add new targets but the existing `make test` target calls `./ci/run_tests.sh`, which runs `python3 -m pytest -rP -v tests/`. After moving tests to `tests/legacy/`, this script will discover **both** legacy and new tests.

The plan doesn't mention updating `ci/run_tests.sh`. Step 8 shows new Makefile targets but doesn't say what happens to the existing `test:` target. It will silently change behavior — the existing `run_tests.sh` will now run legacy tests from `tests/legacy/` and 0 new tests.

**Fix:** Either:
1. Update `ci/run_tests.sh` to `--ignore=tests/legacy/` (breaking change for CI), or
2. Keep `test:` running everything (legacy + new) and add `test-new:` as planned, or
3. Explicitly document what `make test` does post-Phase 0.

---

## DESIGN 2: Coverage ratcheting script runs full test suite

**Severity: Low — impractical for pre-commit**

`ci/check_coverage.py` calls `subprocess.run(["pytest", "--cov", ...])` which runs the **entire** test suite. The plan notes this is slow (Step 9: `always_run: false`), but:

1. The script hardcodes `tests/` as the test directory — this will run legacy tests too (no `--ignore=tests/legacy/`).
2. A pre-commit hook that runs the full GPU test suite is impractical. Even with `always_run: false`, when it **does** trigger, it blocks the commit for minutes.
3. The script mixes concerns: running tests + checking thresholds. Better to split: pytest generates the report, script only checks thresholds from an existing `.coverage.json`.

**Fix:** Separate the concerns:
- `make test-cov` generates `.coverage.json`
- `ci/check_coverage.py` only reads and validates — no test execution

---

## MISSING 1: No `.gitignore` updates

The plan creates `htmlcov/`, `.coverage`, `.coverage.json` — none of these are mentioned for `.gitignore`. They'll show up in `git status` immediately.

**Fix:** Add to `.gitignore`:
```
htmlcov/
.coverage
.coverage.*
.coverage.json
```

---

## MISSING 2: Step 5/6 `data/` directory mental model is wrong

The plan says "Keep tests/data/ in place (shared test data)" but `tests/data/` **does not exist**. The data directory is at the project root: `data/`. The `git mv` commands reference files that exist, but the plan's mental model of where data lives is wrong.

The **directory structure diagram** (Step 6) shows:
```
tests/
└── data/                    # Existing test data (not moved)
    ├── engines/
    └── ...
```

This directory doesn't exist. The actual data path is `BASE_DIR / "data"` (project root `/data/`). The new conftest.py fixtures need to point there.

---

## Minor Issues

| Issue | Location | Note |
|-------|----------|------|
| `batch_size` adds `8` not in existing tests | conftest.py fixture | Existing conftest has `params=[1, 2, 4]`, plan adds `8`. Fine but intentional? May slow test suite. |
| `test_images` fixture scope change | conftest.py | Existing is unscoped (function), plan makes it `session`. Fine for read-only images, but note the behavior change. |
| `random_images` scope change | conftest.py | Existing is unscoped, plan makes it `session` with seed `42`. The seed change is good (reproducibility), but session scope means RNG state is shared across all tests — later tests get different random data than if run alone. |
| `@abstractmethod` in coverage exclusions | pyproject.toml | Plan includes `@abstractmethod` exclusion. This is fine but the codebase also has `@abc.abstractmethod` — check if the exclusion regex catches both. |
| No `addopts` in plan's pyproject.toml | Step 2 vs task_plan.md | The task_plan.md shows `addopts = "--timeout=300"` but the Phase 0 detailed plan (Step 2) only has `timeout = 300`. These are different — `timeout` is the pytest-timeout config key, `addopts` passes CLI args. Both are needed if you want timeout to apply without `-p timeout`. |

---

## Resolution Status

| Issue | Status | Resolution |
|-------|--------|------------|
| BUG 1: Wrong data path | **FIXED** | Changed to `DATA_DIR = BASE_DIR / "data"`. Also fixed in Phases 1, 2, 4 conftest files. |
| BUG 2: Python 3.8 hints | **FALSE POSITIVE** | Script already has `from __future__ import annotations`. PEP 563 defers all annotations (function + variable) to strings on 3.8. |
| BUG 3: version_engine_path | **KEPT AS-IS** | User chose "always version" — `_unknown.engine` suffix when TRT missing is the desired behavior. |
| DESIGN 1: make test conflict | **FIXED** | `ci/run_tests.sh` updated to `--ignore=tests/legacy/`. New only. Legacy via `make test-legacy`. |
| DESIGN 2: Coverage script concerns | **FIXED** | Rewritten as validate-only (reads `.coverage.json`, no pytest execution). `make test-cov` generates report. |
| MISSING 1: .gitignore | **FALSE POSITIVE** | `.gitignore` already has `htmlcov/`, `.coverage`, `.coverage.*` entries. |
| MISSING 2: data/ mental model | **FIXED** | Same as BUG 1. Directory diagram updated to remove nonexistent `tests/data/`. |

### Additional fix found during review
- **ONNX filename**: All phase plans referenced `data/test.onnx` but the tracked test model is `data/simple.onnx` (per `tests/paths.py`). Fixed in Phases 1, 2, 4.

### Cascade impacts on other phases
- **Phase 1**: Fixed `simple_onnx_path` fixture path (`.parent.parent.parent` + `simple.onnx`)
- **Phase 2**: Fixed `ONNX_PATH` constant (same fix)
- **Phase 3**: Rewritten for validate-only coverage script design, updated hook behavior docs
- **Phase 4**: Fixed `ONNX_PATH` constant (same fix)
- **Phases 5-6**: No direct path issues (use fixtures from parent conftest)
