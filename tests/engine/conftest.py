# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine test fixtures -- built engines, test data, memory mode configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Callable, Generator


# tests/engine/conftest.py -> tests/engine -> tests -> project_root -> data/
ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def engine_path(build_test_engine: Callable[..., Path]) -> Path:
    """Session-scoped built engine for general engine tests."""
    return build_test_engine(ONNX_PATH)


@pytest.fixture
def engine(engine_path: Path) -> Generator:
    """Create a fresh TRTEngine instance per test (no warmup)."""
    from trtutils import TRTEngine

    eng = TRTEngine(engine_path, warmup=False)
    yield eng
    del eng


@pytest.fixture
def engine_with_warmup(engine_path: Path) -> Generator:
    """TRTEngine with warmup enabled."""
    from trtutils import TRTEngine

    eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
    yield eng
    del eng


@pytest.fixture
def engine_verbose(engine_path: Path) -> Generator:
    """TRTEngine with verbose=True."""
    from trtutils import TRTEngine

    eng = TRTEngine(engine_path, warmup=False, verbose=True)
    yield eng
    del eng


@pytest.fixture
def engine_no_pagelocked(engine_path: Path) -> Generator:
    """TRTEngine with pagelocked_mem=False."""
    from trtutils import TRTEngine

    eng = TRTEngine(engine_path, warmup=False, pagelocked_mem=False)
    yield eng
    del eng


@pytest.fixture
def random_input(engine) -> list:
    """Generate random input matching engine spec."""
    return engine.get_random_input()  # type: ignore[union-attr]
