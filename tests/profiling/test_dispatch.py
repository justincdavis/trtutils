# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the profile_engine dispatcher's iterations handling."""

from __future__ import annotations

import pytest

import trtutils._profile as dispatch

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("jetson", [False, True], ids=["standard", "jetson"])
@pytest.mark.parametrize(
    ("iterations", "expected"),
    [pytest.param(None, None, id="default"), pytest.param(7, 7, id="explicit")],
)
def test_dispatch_forwards_iterations_only_when_given(
    monkeypatch, jetson, iterations, expected
) -> None:
    """Without iterations the backend default applies; an explicit value is forwarded."""
    received: list[dict[str, object]] = []

    def fake(**kwargs: object) -> None:
        received.append(kwargs)

    monkeypatch.setattr(dispatch, "jetson_profile_engine", fake)
    monkeypatch.setattr(dispatch, "inspect_profile_engine", fake)

    dispatch.profile_engine("engine.engine", iterations=iterations, jetson=jetson)

    assert len(received) == 1
    assert received[0].get("iterations") == expected
