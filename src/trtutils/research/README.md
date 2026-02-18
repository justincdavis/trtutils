# Research Submodule Developer Guide

This guide describes how to implement a new research submodule in the
`trtutils.research` package. Each submodule is a self-contained implementation
of a research paper that plugs into trtutils' build system, CLI, and
lazy-loading infrastructure.

The `axonn/` submodule (AxoNN: Energy-Aware Execution of Neural Network
Inference on Multi-Accelerator Heterogeneous SoCs, DAC 2022) is the canonical
example. This guide is derived from the patterns it establishes.

---

## Module Naming Convention

- Use the paper or method name in lowercase: `axonn`, not `energy_optimizer`.
- Use underscores for multi-word names: `layer_fusion`, not `layerfusion`.
- The name must be a valid Python identifier (no hyphens, no leading digits).
- The name becomes the CLI subcommand (`trtutils research <name>`) and the
  import path (`from trtutils.research import <name>`).

---

## Required File Structure

Every research submodule must have:

- `__init__.py` — public API, `__all__`, module docstring with paper reference
- `_cli.py` — CLI registration (required if the module produces artifacts)

Everything else is flexible. Organize internal files as the problem demands.
Use underscore-prefixed filenames for private implementation modules (the
standard Python convention for "not part of the public API").

### AxoNN's Layout (for reference)

```
src/trtutils/research/axonn/
    __init__.py     # Public API: exports build_engine()
    _build.py       # Main orchestration logic
    _cli.py         # CLI subcommands
    _cost.py        # Cost model utilities
    _profile.py     # Engine profiling and layer extraction
    _solver.py      # Z3 optimization solver
    _types.py       # Dataclasses and enums
```

This is one way to organize — a module that doesn't need a solver or profiler
would look different. The key requirement is `__init__.py` and `_cli.py`.

---

## Registration Checklist

Every file that needs to be created or modified when adding a new research
submodule called `<name>`:

### 1. Create the submodule directory

```
src/trtutils/research/<name>/
    __init__.py
    _cli.py
    # ... other implementation files
```

### 2. Register in `src/trtutils/research/__init__.py`

Four changes in this file:

**a)** Add to the `_SUBMODULES` list (controls discovery order):

```python
_SUBMODULES = ["axonn", "<name>"]
```

**b)** Add to the `_LAZY_SUBMODULES` set (enables lazy import via `__getattr__`):

```python
_LAZY_SUBMODULES = {"axonn", "<name>"}
```

**c)** Add to `__all__`:

```python
__all__ = [
    "axonn",
    "<name>",
    "discover_submodules",
    "register_cli",
]
```

**d)** Add a `TYPE_CHECKING` import (for static analysis / IDE support):

```python
if TYPE_CHECKING:
    from . import axonn as axonn
    from . import <name> as <name>
```

### 3. Add optional dependency extra in `pyproject.toml`

Under `[project.optional-dependencies]`, add the module's dependencies:

```toml
<name> = [
    "some-dep>=x.y.z,<a.b.c",
]
```

Then add it to the `dev` aggregate extra:

```toml
dev = [
    # ... existing entries ...
    "trtutils[<name>]",
]
```

### 4. Create `_cli.py` with `register_cli()` function

See the [CLI Integration](#cli-integration) section for the full contract.

### 5. Create tests

```
tests/research/<name>/
    __init__.py
    test_<name>.py
```

See the [Testing](#testing) section for guidance.

---

## How Discovery & Lazy Loading Works

The research package uses three mechanisms that work together:

### Discovery: `_SUBMODULES` list + `discover_submodules()`

`discover_submodules()` iterates over `_SUBMODULES` and tries to import each
one. If the import succeeds, the module name is included in the returned list.
If the import fails (e.g., missing optional dependencies), the module is
silently skipped.

```python
# From research/__init__.py
def discover_submodules() -> list[str]:
    available: list[str] = []
    for name in _SUBMODULES:
        with contextlib.suppress(ImportError):
            __import__(f"trtutils.research.{name}")
            available.append(name)
    return available
```

This means: if a user doesn't have `z3-solver` installed, `axonn` simply
won't appear in `discover_submodules()` output. No error, no warning.

### Lazy Loading: `_LAZY_SUBMODULES` set + `__getattr__`

The `__getattr__` hook in `research/__init__.py` intercepts attribute access
for names in `_LAZY_SUBMODULES` and imports the submodule on demand:

```python
def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(...)
```

This allows `from trtutils.research import <name>` to work without eagerly
importing every research submodule (and their heavy dependencies) when
`trtutils.research` is first loaded.

### CLI Registration

The `register_cli()` function in `research/__init__.py` iterates discovered
modules, imports each module's `_cli.py`, and calls its `register_cli()`:

```python
def register_cli(subparsers, parents) -> None:
    for name in discover_submodules():
        with contextlib.suppress(ImportError):
            mod = __import__(f"trtutils.research.{name}._cli", fromlist=["register_cli"])
            mod.register_cli(subparsers, parents)
```

The main CLI (`__main__.py`) calls this only when at least one research module
is discovered, creating the `research` subcommand dynamically.

---

## Public API Design

### `__init__.py` Requirements

The submodule's `__init__.py` must:

1. Export public functions via `__all__`
2. Include a module docstring with the paper reference

#### Prescribed Docstring Format

Follow the pattern established by AxoNN:

```python
"""
Implementation for the research paper <PaperName>.

<One-paragraph description of what the module does and what problem it solves.>

**Goal**: <One-sentence statement of the optimization objective.>

Reference
---------
<Full paper title> (<Venue Year>)
<DOI URL>

Example:
-------
>>> from trtutils.research.<name> import <main_function>
>>>
>>> result = <main_function>(
...     onnx="model.onnx",
...     output="model.engine",
...     # ... key parameters with comments
... )
>>>
>>> print(f"Result: {result}")

"""

from __future__ import annotations

from ._build import <main_function>

__all__ = [
    "<main_function>",
]
```

### Artifact Generation

If the module builds TensorRT engines or other artifacts, expose a top-level
function (e.g., `build_engine()`) as the primary public API. This function
should:

- Accept an ONNX model path and output path
- Accept a calibration batcher if INT8 is needed
- Return meaningful results (metrics, layer assignments, etc.)
- Handle the full pipeline internally (profile, optimize, build)

### Usage of Built Artifacts

The module does **not** need to provide its own engine loading wrapper if
the artifact is a standard TensorRT engine. Users load it with `TRTEngine`:

```python
from trtutils import TRTEngine

engine = TRTEngine("model.engine")
outputs = engine(inputs)
```

Only provide a custom usage wrapper if the module produces non-standard
artifacts or requires special post-processing logic.

---

## CLI Integration

### Available Parent Parsers

The main CLI creates four parent parsers and passes all of them to each
research module's `register_cli()`. Use them as `parents=` in your
subcommand's `add_parser()` call. Only include the parsers your subcommand
actually needs.

| Parser | Key Arguments | Use When |
|---|---|---|
| `parents[0]`: `general_parser` | `--log_level`, `--verbose`, `--nvtx` | Always. All commands need logging. |
| `parents[1]`: `dla_parser` | `--dla_core` | Module uses DLA acceleration. |
| `parents[2]`: `build_common_parser` | `--onnx` (required), `--output` (required), `--workspace`, `--optimization_level`, `--shape`, `--timing_cache`, `--calibration_cache`, `--direct_io`, `--prefer_precision_constraints`, `--reject_empty_algorithms`, `--ignore_timing_mismatch`, `--cache` | Module builds TensorRT engines from ONNX. |
| `parents[3]`: `calibration_parser` | `--calibration_dir`, `--input_shape`, `--input_dtype`, `--batch_size`, `--data_order`, `--max_images`, `--resize_method`, `--input_scale` | Module needs INT8 calibration data. |

**Important**: The parent parsers are passed as a flat list in the order
shown above. Reference them by index: `parents[0]` is `general_parser`,
`parents[1]` is `dla_parser`, etc.

The parent parsers are defined in `src/trtutils/__main__.py` in the `_main()`
function, and the list is assembled at line 1851:

```python
research_register_cli(
    research_subparsers,
    [general_parser, dla_parser, build_common_parser, calibration_parser],
)
```

### `register_cli()` Function Contract

Your `_cli.py` must export a `register_cli()` function with this signature:

```python
def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
```

This function should:

1. Create a parser for the module: `subparsers.add_parser("<name>", help="...")`
2. Create sub-subparsers for each command the module offers
3. Set the default action (no subcommand) to print help
4. For each subcommand, select which parent parsers to inherit

#### Example Structure

```python
def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register <Name> CLI subcommands under the research parser."""
    # Top-level parser for this module
    mod_parser = subparsers.add_parser(
        "<name>",
        help="<Short description of the module.>",
    )
    mod_subparsers = mod_parser.add_subparsers(
        title="<name> commands",
        dest="<name>_command",
        required=False,
    )
    mod_parser.set_defaults(func=lambda _args: mod_parser.print_help())

    # "build" subcommand — inherits all 4 parent parsers
    build_parser = mod_subparsers.add_parser(
        "build",
        help="Build a <Name>-optimized engine from an ONNX file.",
        parents=parents,  # all 4 parents
    )
    # Add module-specific arguments
    build_parser.add_argument("--my_param", type=float, default=0.5, help="...")
    build_parser.set_defaults(func=_my_build_handler)
```

### CLI Subcommand Naming

Subcommand names should mirror the public API function names:

- `build_engine()` in Python → `build` subcommand in CLI
- `analyze_model()` in Python → `analyze` subcommand in CLI

### Handler Function Pattern

Each CLI subcommand handler receives a `SimpleNamespace` from argparse:

```python
def _my_build_handler(args: SimpleNamespace) -> None:
    """Build a <Name>-optimized engine from an ONNX file."""
    # Extract arguments
    onnx_path = Path(args.onnx)         # from build_common_parser
    output_path = Path(args.output)     # from build_common_parser
    my_param = args.my_param            # module-specific arg

    # Call the public API
    result = my_build_function(
        onnx=onnx_path,
        output=output_path,
        my_param=my_param,
        verbose=args.verbose,           # from general_parser
    )

    # Log results
    LOG.info("Build Complete")
    LOG.info(f"Engine saved to: {output_path}")
    LOG.info(f"Result: {result}")
```

Use `trtutils._log.LOG` for all output. Import it as:
```python
from trtutils._log import LOG
```

---

## Optional Dependencies

Every research module **must** declare its own extra in `pyproject.toml`.

### Rules

1. The extra name matches the module name:
   ```toml
   axonn = ["z3-solver>=4.12.0,<5.0.0"]
   ```

2. Add to the `dev` aggregate extra:
   ```toml
   dev = [
       "trtutils[axonn]",
       "trtutils[<name>]",
       # ...
   ]
   ```

3. The module's imports of optional dependencies happen at module load time.
   If the dependency is missing, the `import` in `__init__.py` or internal
   files will raise `ImportError`, which `discover_submodules()` catches
   with `contextlib.suppress(ImportError)`.

4. There is no need for explicit availability checking or custom error
   messages at the module level — the discovery mechanism handles it.
   If a user tries to import directly without the dependency, they get a
   standard Python `ImportError`.

---

## Error Handling

- Return `None` or sentinel values for infeasible results rather than raising
  exceptions. For example, if an optimization problem has no feasible solution,
  return `None` and log the reason.
- Provide clear log messages explaining why something failed, using `LOG.warning`
  or `LOG.error` as appropriate.
- Validate inputs at the public API boundary and raise `ValueError` with
  descriptive messages for invalid parameters.
- Internal functions can assume inputs are valid (validated by the public API).

---

## Core Module Integration Reference

These trtutils modules are useful building blocks for research submodules:

| Module | Import Path | What It Provides |
|---|---|---|
| Builder | `trtutils.builder` | `build_engine()` for ONNX-to-TRT conversion, `can_run_on_dla()` for DLA layer analysis, `build_dla_engine()` for mixed GPU/DLA builds, calibration batchers (`AbstractBatcher`, `ImageBatcher`, `SyntheticBatcher`) |
| Engine | `trtutils._engine` | `TRTEngine` class for loading and executing built engines |
| Jetson | `trtutils.jetson` | `profile_engine()` for tegrastats-based profiling with power/energy measurement, `benchmark_engine()` for Jetson benchmarks |
| Core CUDA | `trtutils.core` | Memory allocation/free/copy, CUDA streams, CUDA graph capture, `Binding` wrappers |
| Flags | `trtutils._flags` | Runtime feature detection: `TENSORRT_VERSION`, `IS_JETSON`, available execution APIs |
| Benchmark | `trtutils._benchmark` | `BenchmarkResult` and `Metric` dataclasses for performance measurement |
| Image | `trtutils.image` | `Detector`, `Classifier`, preprocessors and postprocessors for image models |
| Profiling | `trtutils.profile_engine` | Top-level `profile_engine()` for layer-by-layer profiling with optional Jetson power metrics |
| Log | `trtutils._log` | `LOG` logger instance — use this for all module output |

---

## Testing

- Tests go in `tests/research/<name>/`.
- Include an `__init__.py` in both `tests/research/` (already exists) and
  `tests/research/<name>/`.
- **Unit tests**: test dataclasses, utility functions, cost models, and solvers
  with synthetic data. These run anywhere without hardware.
- **Integration tests**: engine building, profiling, and end-to-end runs that
  require a GPU/DLA. These are run manually on target hardware.
- Import internal modules directly for testing:
  ```python
  from trtutils.research.<name>._types import MyDataclass
  from trtutils.research.<name>._cost import my_cost_function
  ```
- Use `pytest.approx` for floating-point comparisons.
- Use `pytest.raises` to test error conditions.
- Run tests with: `python3 -m pytest -rP -v tests/research/<name>/`

---

## AxoNN as Reference

Quick reference mapping of AxoNN's files to their roles:

| File | Role | Key Exports |
|---|---|---|
| `__init__.py` | Public API, paper docstring | `build_engine` |
| `_types.py` | Dataclasses and enums | `Layer`, `LayerCost`, `Schedule`, `ProcessorType`, `AxoNNConfig` |
| `_cost.py` | Cost model utilities | `compute_total_time`, `compute_total_energy`, `compute_gpu_only_costs`, `create_gpu_only_schedule`, `estimate_transition_cost` |
| `_profile.py` | Engine profiling and layer extraction | Builds GPU/DLA engines, profiles layers, extracts per-layer costs |
| `_solver.py` | Z3 optimization solver | Finds optimal GPU/DLA assignments under energy and transition constraints |
| `_build.py` | Main orchestration | `build_engine()` — profiles, solves, builds final engine |
| `_cli.py` | CLI registration | `register_cli()`, handler `_axonn_build()` |

The flow is: `_cli.py` parses args → calls `_build.py:build_engine()` →
which calls `_profile.py` to get per-layer costs → `_solver.py` to find
optimal assignment → `trtutils.builder` to build the final engine.
