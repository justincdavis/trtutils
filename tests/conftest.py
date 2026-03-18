# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# CPU-only stub injection
# When TRTUTILS_IGNORE_MISSING_CUDA=1, inject lightweight stub modules for
# tensorrt and cuda so that trtutils can be imported without GPU hardware.
# This MUST run before any trtutils imports.
# ---------------------------------------------------------------------------
_CPU_ONLY = os.environ.get("TRTUTILS_IGNORE_MISSING_CUDA", "0") == "1"

if _CPU_ONLY:
    if "tensorrt" not in sys.modules:
        _trt = types.ModuleType("tensorrt")
        _trt.__version__ = "0.0.0"
        # base classes used in class inheritance at module level
        _trt.ILogger = type(
            "ILogger",
            (),
            {
                "Severity": type(
                    "Severity",
                    (),
                    {
                        "INTERNAL_ERROR": 0,
                        "ERROR": 1,
                        "WARNING": 2,
                        "INFO": 3,
                        "VERBOSE": 4,
                    },
                ),
            },
        )
        _trt.IProgressMonitor = type("IProgressMonitor", (), {})
        _trt.IProfiler = type("IProfiler", (), {})
        _trt.IInt8EntropyCalibrator2 = type("IInt8EntropyCalibrator2", (), {})
        # classes accessed via hasattr in _flags.py
        _trt.ICudaEngine = type("ICudaEngine", (), {})
        _trt.DataType = type("DataType", (), {})
        _trt.IBuilderConfig = type("IBuilderConfig", (), {})
        _trt.Builder = type("Builder", (), {})
        _trt.IExecutionContext = type("IExecutionContext", (), {})
        # classes with attrs used as default parameter values
        _trt.TensorFormat = type("TensorFormat", (), {"LINEAR": 0})
        _trt.DeviceType = type("DeviceType", (), {"GPU": 0, "DLA": 1})
        sys.modules["tensorrt"] = _trt
    if "cuda" not in sys.modules:
        sys.modules["cuda"] = types.ModuleType("cuda")


from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ENGINES_DIR = DATA_DIR / "engines"

if not _CPU_ONLY:
    from tests._gpu_fixtures import *  # noqa: F403
