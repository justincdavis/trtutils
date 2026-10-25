---
name: test-style
description: Use when writing, reviewing or refactoring trtutils tests - "add tests", "write tests for", "make the tests thorough", "parametrize this", "skip on Blackwell/Jetson", "add a test marker". Summarizes .agents/docs/TEST_CODE_STYLE.md.
---

# trtutils test style

The full standard is `.agents/docs/TEST_CODE_STYLE.md`; read it before writing tests. The rules
that matter most:

## Structure

- Copyright header, module docstring, `from __future__ import annotations` first.
- All imports at the top of the file, including ones only fixtures use.
- Flat `test_*` functions. A class only to parametrize a group, with an autouse fixture.
- One-sentence present-tense docstring; `-> None` on every test; no type hints on fixtures.

## Coverage shape

- Prefer one comprehensive parametrized test over many small ones; round-trip validation
  over checking intermediate state.
- Build matrices from `pytest.param(..., id="kebab-case")` tables; stack `parametrize`
  decorators or use `itertools.product` for exhaustive combos.
- Factory fixtures (`build_test_engine`, `build_model_engine`, `random_images`) with
  `yield` teardown; session scope for engine builds and image loads.
- Prove thoroughness: `--cov=src/trtutils --cov-branch`, and a quick mutation check (break
  the source line under test, confirm a test fails, revert).

## Real hardware, no mocks

- Use real CUDA/TensorRT. Mock only to check cleanup routing, with `patch.object(...,
  wraps=...)`.
- GPU tests that need a model ONNX must skip when it is missing (`build_model_engine` does).
- Hardware-specific failure: skip that `pytest.param` with
  `marks=pytest.mark.skipif(FLAGS.IS_BLACKWELL, reason=...)` and add a focused new test for
  the replacement path. Do not branch inside a test body on platform.
- Use `pytest.skip()` (not `xfail`) for features that may be unavailable.

## Markers (registered in `pytest.ini` only)

`cpu`, `jetson`, `dla`, `performance`, `slow`, `download`, `cuda_graph`, `correctness`,
`regression`, `research`. Every test touching CUDA graphs gets `cuda_graph`. There is no
`gpu` marker. To gate a class of tests, register a marker and exclude it via `addopts`
(like `research`), not a custom `--run-X` flag.

## Don't test

Verbose/debug flags in isolation, "doesn't crash" calls, TRT version branching, vendor error
message wording, flag detection, thread-lock behavior.

## Run

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/<area>/test_<x>.py
make fix && make typecheck
```
