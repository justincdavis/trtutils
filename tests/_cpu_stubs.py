# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Stub modules for tensorrt and cuda.

Injects lightweight fakes into sys.modules so trtutils can be imported
without GPU hardware. Must be called before any trtutils imports.
"""

from __future__ import annotations

import sys
import types


def inject() -> None:
    """Inject stub tensorrt and cuda modules into sys.modules."""
    if "tensorrt" not in sys.modules:
        trt = types.ModuleType("tensorrt")
        trt.__version__ = "0.0.0"
        # base classes used in class inheritance at module level
        trt.ILogger = type(
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
        trt.IProgressMonitor = type("IProgressMonitor", (), {})
        trt.IProfiler = type("IProfiler", (), {})
        trt.IInt8EntropyCalibrator2 = type("IInt8EntropyCalibrator2", (), {})
        # classes accessed via hasattr in _flags.py
        trt.ICudaEngine = type("ICudaEngine", (), {})
        trt.DataType = type("DataType", (), {})
        trt.IBuilderConfig = type("IBuilderConfig", (), {})
        trt.Builder = type("Builder", (), {})
        trt.IExecutionContext = type("IExecutionContext", (), {})
        # classes with attrs used as default parameter values
        trt.TensorFormat = type("TensorFormat", (), {"LINEAR": 0})
        trt.DeviceType = type("DeviceType", (), {"GPU": 0, "DLA": 1})
        sys.modules["tensorrt"] = trt
    if "cuda" not in sys.modules:
        sys.modules["cuda"] = types.ModuleType("cuda")
