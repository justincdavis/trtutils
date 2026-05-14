# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply the ``research`` marker to every test under ``tests/research/``."""
    research_marker = pytest.mark.research
    for item in items:
        if "tests/research/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(research_marker)
