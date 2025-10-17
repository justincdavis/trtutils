# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from .common import rtdetr_pagelocked_perf


def test_rtdetrv1_pagelocked_perf() -> None:
    """Test the performance of the RT-DETRv1 model."""
    rtdetr_pagelocked_perf("rtdetrv1", use_dla=False)


def test_rtdetrv1_pagelocked_perf_dla() -> None:
    """Test the performance of the RT-DETRv1 model with DLA."""
    rtdetr_pagelocked_perf("rtdetrv1", use_dla=True)


def test_rtdetrv2_pagelocked_perf() -> None:
    """Test the performance of the RT-DETRv2 model."""
    rtdetr_pagelocked_perf("rtdetrv2", use_dla=False)


def test_rtdetrv2_pagelocked_perf_dla() -> None:
    """Test the performance of the RT-DETRv2 model with DLA."""
    rtdetr_pagelocked_perf("rtdetrv2", use_dla=True)


def test_rtdetrv3_pagelocked_perf() -> None:
    """Test the performance of the RT-DETRv3 model."""
    rtdetr_pagelocked_perf("rtdetrv3", use_dla=False)


def test_rtdetrv3_pagelocked_perf_dla() -> None:
    """Test the performance of the RT-DETRv3 model with DLA."""
    rtdetr_pagelocked_perf("rtdetrv3", use_dla=True)


def test_dfine_pagelocked_perf() -> None:
    """Test the performance of the D-FINE model."""
    rtdetr_pagelocked_perf("dfine", use_dla=False)


def test_dfine_pagelocked_perf_dla() -> None:
    """Test the performance of the D-FINE model with DLA."""
    rtdetr_pagelocked_perf("dfine", use_dla=True)


def test_deim_pagelocked_perf() -> None:
    """Test the performance of the DEIM model."""
    rtdetr_pagelocked_perf("deim", use_dla=False)


def test_deim_pagelocked_perf_dla() -> None:
    """Test the performance of the DEIM model with DLA."""
    rtdetr_pagelocked_perf("deim", use_dla=True)


def test_deimv2_pagelocked_perf() -> None:
    """Test the performance of the DEIMv2 model."""
    rtdetr_pagelocked_perf("deimv2", use_dla=False)


def test_deimv2_pagelocked_perf_dla() -> None:
    """Test the performance of the DEIMv2 model with DLA."""
    rtdetr_pagelocked_perf("deimv2", use_dla=True)


def test_rfdetr_pagelocked_perf() -> None:
    """Test the performance of the RF-DETR model."""
    rtdetr_pagelocked_perf("rfdetr", use_dla=False)


def test_rfdetr_pagelocked_perf_dla() -> None:
    """Test the performance of the RF-DETR model with DLA."""
    rtdetr_pagelocked_perf("rfdetr", use_dla=True)

