# Phase 3: Pre-commit Coverage Check (Core + Engine)

**Branch:** `test-overhaul/phase-3-precommit-core-engine`
**Goal:** Finalize the ratcheting coverage mechanism, record baseline thresholds for core + engine, validate end-to-end.

---

## Prerequisites
- Phase 0 (infrastructure, coverage script skeleton)
- Phase 1 (core tests achieving 100% branch coverage)
- Phase 2 (engine tests achieving 100% branch coverage)

## Deliverables
- [ ] `ci/check_coverage.py` — Production-ready ratcheting script
- [ ] `.coverage-threshold.json` — Updated with actual Phase 1+2 baselines
- [ ] Pre-commit hook validated end-to-end
- [ ] Coverage workflow documented
- [ ] CI script for coverage (`ci/run_coverage.sh`)

---

## Step 1: Finalize ci/check_coverage.py

The Phase 0 skeleton established the validate-only design. The script:

1. Reads an existing `.coverage.json` report (does NOT run pytest)
2. Parses the JSON report to extract per-module branch coverage
3. Compares against `.coverage-threshold.json`
4. Fails if any module dropped below its threshold
5. Updates thresholds if coverage increased (with `--update` flag)

Phase 3 work: validate the script against actual `pytest-cov` JSON output, fix any parsing issues, and confirm module grouping logic is correct.

### Key Validation Steps

**Verify JSON report format** by running `make test-cov` and inspecting `.coverage.json`:
```bash
make test-cov
python3 -c "import json; d=json.load(open('.coverage.json')); print(json.dumps(list(d['files'].keys())[:5], indent=2))"
```

**Verify module grouping** produces expected keys:
```bash
python3 ci/check_coverage.py --verbose
# Should show: trtutils/core, trtutils/_engine, trtutils/_flags, etc.
```

**Verify branch coverage aggregation** uses `covered_branches / num_branches` (not `percent_covered` averaging, which would weight small files equally with large files).

### Script Modes

| Mode | Command | Behavior |
|------|---------|----------|
| Check | `python ci/check_coverage.py` | Fail if any module below threshold |
| Update | `python ci/check_coverage.py --update` | Update thresholds to current values |
| Verbose | `python ci/check_coverage.py --verbose` | Show per-module breakdown |
| Custom report | `python ci/check_coverage.py --report FILE` | Use specific report file |

### Exit Codes
- `0` — All modules at or above threshold
- `1` — One or more modules below threshold
- `2` — Missing report file or threshold file

---

## Step 2: Record Baseline Thresholds

After Phase 1+2 tests are passing, run:

```bash
python ci/check_coverage.py --update
```

This updates `.coverage-threshold.json` with actual coverage numbers:

```json
{
    "trtutils/core": 100.0,
    "trtutils/_engine": 100.0,
    "trtutils/_flags": 95.0,
    "trtutils/builder": 0,
    "trtutils/download": 0,
    "trtutils/inspect": 0,
    "trtutils/models": 0,
    "trtutils/image": 0
}
```

**Note:** `_flags` may not be exactly 100% if some version-specific branches can't be reached on the current TRT version. That's OK — the ratchet preserves whatever the achieved value is.

---

## Step 3: Create CI Coverage Script

**File:** `ci/run_coverage.sh`

This script runs both halves: generate the report, then validate thresholds.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Step 1: Run new tests with branch coverage (generates .coverage.json)
python3 -m pytest \
    tests/ \
    --ignore=tests/legacy/ \
    --cov=src/trtutils \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report=json:.coverage.json \
    -v \
    "$@"

# Step 2: Validate coverage against ratcheting thresholds
python3 ci/check_coverage.py
```

Make executable: `chmod +x ci/run_coverage.sh`

**Note:** `make test-cov` generates the report only. `ci/run_coverage.sh` generates AND validates. The pre-commit hook runs `ci/check_coverage.py` alone (validates an existing report).

---

## Step 4: Validate Pre-commit Hook

The `.pre-commit-config.yaml` already has a `coverage-check` hook from Phase 0. Validate it:

```bash
# Stage some changes to trigger pre-commit
echo "# test" >> tests/core/test_cache.py
git add tests/core/test_cache.py

# Run pre-commit
pre-commit run coverage-check

# Should pass (coverage >= thresholds)
# Clean up
git checkout -- tests/core/test_cache.py
```

### Hook Behavior
- The coverage hook runs `ci/check_coverage.py` which **only reads** `.coverage.json` — it does NOT run tests
- This is fast (milliseconds) since it's just JSON parsing
- Developers run `make test-cov` to generate the report, then the hook validates thresholds on commit
- If `.coverage.json` doesn't exist, the hook exits with code 2 and a helpful message

### Hook Stage

Use manual stage so developers opt in explicitly:
```yaml
      - id: coverage-check
        name: Coverage threshold check
        entry: bash -c 'python3 ci/check_coverage.py'
        language: system
        pass_filenames: false
        always_run: false
        stages: [manual]  # Only runs with: pre-commit run coverage-check
```

The validate-only design means the hook is fast enough for pre-commit stage, but manual is still recommended since the developer needs to have run `make test-cov` first to have a current report.

---

## Step 5: Document Coverage Workflow

Add a section to the repo documenting the coverage workflow. This could go in CONTRIBUTING.md or the README:

### Coverage Workflow Summary

```
# Run tests with coverage report
make test-cov

# Run tests with HTML coverage report
make test-cov-html
open htmlcov/index.html

# Check coverage against thresholds (ratchet check)
python ci/check_coverage.py

# Update thresholds after adding tests
python ci/check_coverage.py --update

# Run coverage check via pre-commit
pre-commit run coverage-check
```

### Coverage Ratcheting Rules

1. Coverage thresholds are stored in `.coverage-threshold.json`
2. Thresholds are per-module (e.g., `trtutils/core`, `trtutils/_engine`)
3. Coverage can only go UP — the `--update` flag raises thresholds, never lowers
4. If you need to lower a threshold (e.g., after removing test-only code), manually edit `.coverage-threshold.json`
5. CI runs `ci/check_coverage.py` to enforce thresholds

---

## Step 6: Verify Everything End-to-End

1. Run full test suite with coverage:
   ```bash
   ci/run_coverage.sh
   ```

2. Verify thresholds recorded:
   ```bash
   cat .coverage-threshold.json
   # core and engine should be ~100%
   ```

3. Simulate a regression — comment out a test, verify check fails:
   ```bash
   # Comment out a test in tests/core/test_cache.py
   python ci/check_coverage.py
   # Should exit 1 with "FAIL: trtutils/core coverage X% < threshold Y%"
   ```

4. Restore the test, verify check passes again.

5. Run pre-commit hook:
   ```bash
   pre-commit run coverage-check
   ```

---

## Files Created/Modified Summary

| File | Action | Purpose |
|------|--------|---------|
| `ci/check_coverage.py` | Finalized | Production-ready ratcheting script |
| `ci/run_coverage.sh` | Created | CI coverage execution wrapper |
| `.coverage-threshold.json` | Updated | Baseline thresholds from Phase 1+2 |
| `.pre-commit-config.yaml` | Updated | Coverage hook stage adjustment |
| `Makefile` | Updated | Add `test-cov` target if not already present |

---

## Estimated Effort
- ~1-2 hours
- Most time spent debugging JSON report parsing and threshold calculation
