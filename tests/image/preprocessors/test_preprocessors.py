# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/ -- every backend against CPUPreprocessor, and their shared behavior."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.image.preprocessors._matrix import (
    BAD_IMAGES,
    BATCH_SIZE,
    BATCHES,
    BUFFER_KINDS,
    CPU_RESIZE,
    DTYPES,
    FLOATS,
    FRAMES,
    GEOMETRIES,
    KINDS,
    RESIZES,
    Case,
    matrix,
)
from tests.image.preprocessors._tolerance import assert_close, reference
from trtutils import Buffer, MemoryLocation
from trtutils.core import stream_synchronize

NORMALIZATIONS = ("range", "imagenet")
TRIO = ("people-1920x1080", "people-1280x720", "horse-1920x1080")
UNKNOWN_RESIZE = r"Unknown method|not area"
BAD_IMAGE = r"uint8|\(height, width, channels\)"
UNALIGNED = r"whole factor|multiple of 16"
UNSUPPORTED_DTYPE = r"float32 or float16|float16 or float32"
NO_BATCH_SIZE = {
    "cpu": "CPUPreprocessor has no batch size: it preprocesses any number of images",
    "cuda": "CUDAPreprocessor has no batch size: its buffers grow to any batch",
}
CPU_ACCEPTS_BAD_IMAGES = (
    "CPUPreprocessor does not validate images and hands them to cv2, which resizes float32 "
    "and 4-channel images (and grayscale from OpenCV 5) instead of raising ValueError"
)
TRT_IGNORES_DTYPE = (
    "TRTPreprocessor does not check dtype: every dtype but float16 builds a float32 engine, "
    "so uint8, int8 and float64 return float32 instead of raising"
)


def _images(frames, names) -> list[np.ndarray]:
    return [frames(*FRAMES[name]) for name in names]


def _as_kind(image: np.ndarray, kind: str) -> np.ndarray | Buffer:
    if kind == "numpy":
        return image
    if kind == "host":
        return Buffer.from_array(image, pinned=False)
    if kind == "pinned":
        return Buffer.from_array(image, pinned=True, mapped=False)
    if kind == "mapped":
        return Buffer.from_array(image, pinned=True, mapped=True)
    return Buffer.from_array(image, MemoryLocation.DEVICE)


def _submit(inputs: list, batch: str) -> object:
    return inputs[0] if batch == "single" else inputs


def _assert_same(result: tuple, expected: tuple) -> None:
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]


def _check(case: Case, preprocessor, inputs, images, preprocessors) -> None:
    result, ratios, padding = preprocessor.preprocess(inputs)
    if case.backend == "cpu":
        expected = reference(images, case)
    else:
        cpu = preprocessors(replace(case, backend="cpu", cpu_resize=None))
        expected = cpu.preprocess(images)
    width, height = case.output_shape
    assert result.shape == (len(images), 3, height, width)
    assert result.dtype == DTYPES[case.dtype]
    assert ratios == expected[1]
    assert padding == expected[2]
    assert_close(result, expected[0], case, images)


class TestPreprocessor:
    """CPU, CUDA, TensorRT and DLA preprocessors: CPUPreprocessor parity and shared behavior."""

    @pytest.mark.parametrize(
        "case",
        matrix(
            resize=RESIZES,
            dtype=FLOATS,
            norm=NORMALIZATIONS,
            cpu_resize=CPU_RESIZE,
            kind=KINDS,
            batch=tuple(BATCHES),
        ),
    )
    def test_matches_cpu(self, case, frames, preprocessors) -> None:
        """Shape, dtype, ratios and padding match CPUPreprocessor, the values within tolerance."""
        images = _images(frames, BATCHES[case.batch])
        inputs = [_as_kind(image, case.kind) for image in images]
        _check(case, preprocessors(case), _submit(inputs, case.batch), images, preprocessors)

    @pytest.mark.parametrize(
        "case",
        matrix(
            resize=RESIZES,
            antialias=(True,),
            cpu_resize=CPU_RESIZE,
            kind=KINDS,
            batch=tuple(BATCHES),
        )
        + matrix(geometry=GEOMETRIES, antialias=(True,)),
    )
    def test_antialias_matches_cpu(self, case, frames, preprocessors) -> None:
        """Antialiased resizes match CPUPreprocessor's for every input kind, batch and geometry."""
        if case.geometry is not None:
            geometry = case.geometry
            images = [frames(geometry.photo, geometry.height, geometry.width)]
        else:
            images = _images(frames, BATCHES[case.batch])
        inputs = [_as_kind(image, case.kind) for image in images]
        _check(case, preprocessors(case), _submit(inputs, case.batch), images, preprocessors)

    @pytest.mark.parametrize("case", matrix(geometry=GEOMETRIES, dtype=FLOATS, norm=NORMALIZATIONS))
    def test_geometry_matches_cpu(self, case, frames, preprocessors) -> None:
        """Other scales, aspect ratios and DLA factors match CPUPreprocessor within tolerance."""
        geometry = case.geometry
        image = frames(geometry.photo, geometry.height, geometry.width)
        _check(case, preprocessors(case), image, [image], preprocessors)

    @pytest.mark.parametrize(
        "case",
        matrix(resize=RESIZES, kind=BUFFER_KINDS, batch=("single", "batch2", "mixed")),
    )
    def test_inputs_match_numpy(self, case, frames, preprocessors) -> None:
        """Buffers, alone or alternating with arrays in a batch, preprocess exactly like arrays."""
        preprocessor = preprocessors(case)
        images = _images(frames, BATCHES[case.batch])
        inputs = [
            _as_kind(image, case.kind) if i % 2 == 0 else image for i, image in enumerate(images)
        ]
        result = preprocessor.preprocess(_submit(inputs, case.batch))
        _assert_same(result, preprocessor.preprocess(images))

    @pytest.mark.parametrize(
        "case", matrix(resize=RESIZES, antialias=(False, True), cpu_resize=CPU_RESIZE)
    )
    def test_batch_matches_singles(self, case, frames, preprocessors) -> None:
        """Growing, shrinking and mixed-size batches reproduce each image's single result exactly."""
        preprocessor = preprocessors(case)
        images = _images(frames, TRIO)
        singles = [preprocessor.preprocess(image) for image in images]
        for order in ((0,), (1, 0, 2), (0, 2), (2, 1)):
            batch, ratios, padding = preprocessor.preprocess([images[i] for i in order])
            assert batch.shape[0] == len(order)
            for slot, i in enumerate(order):
                tensor, single_ratios, single_padding = singles[i]
                np.testing.assert_array_equal(batch[slot], tensor[0])
                assert ratios[slot] == single_ratios[0]
                assert padding[slot] == single_padding[0]

    @pytest.mark.parametrize("case", matrix(dtype=FLOATS))
    def test_no_copy(self, case, frames, preprocessors) -> None:
        """no_copy returns a view the next call overwrites; a default call returns a private array."""
        preprocessor = preprocessors(case)
        images = _images(frames, TRIO)
        first, _, _ = preprocessor.preprocess(images, no_copy=True)
        snapshot = first.copy()
        second, _, _ = preprocessor.preprocess(images[::-1][:2], no_copy=True)
        assert np.shares_memory(first, second)
        assert not np.array_equal(first, snapshot)
        np.testing.assert_array_equal(first[:2], second)
        copied, _, _ = preprocessor.preprocess(images)
        np.testing.assert_array_equal(copied, snapshot)
        preprocessor.preprocess(images[::-1][:2], no_copy=True)
        assert not np.shares_memory(copied, second)
        np.testing.assert_array_equal(copied, snapshot)

    @pytest.mark.parametrize("case", matrix(resize=RESIZES, size=[(640, 640)]))
    def test_call_matches_preprocess(self, case, frames, preprocessors) -> None:
        """__call__ matches preprocess, and a per-call resize matches the constructor's resize."""
        images = _images(frames, BATCHES["mixed"])
        own = preprocessors(case)
        expected = own.preprocess(images)
        _assert_same(own(images), expected)
        letterbox = preprocessors(replace(case, resize="letterbox"))
        _assert_same(letterbox(images, case.resize), expected)
        _assert_same(letterbox(images, resize=case.resize), expected)
        _assert_same(letterbox.preprocess(images, case.resize), expected)
        smooth = preprocessors(replace(case, antialias=True))
        antialiased = smooth.preprocess(images)
        _assert_same(own(images, antialias=True), antialiased)
        _assert_same(own.preprocess(images, antialias=True), antialiased)
        _assert_same(smooth(images, antialias=False), expected)
        _assert_same(letterbox(images, case.resize, antialias=True), antialiased)
        assert antialiased[1:] == expected[1:]
        assert not np.array_equal(antialiased[0], expected[0])

    @pytest.mark.parametrize(
        "case",
        matrix(
            resize=RESIZES,
            norm=NORMALIZATIONS,
            cpu_resize=CPU_RESIZE,
            kind=("numpy", "device"),
            skip=lambda case: (
                "TensorRT may compute in float16 when its engine writes float16, "
                "so its float16 output need not be the rounded float32 output"
                if case.backend == "trt"
                else None
            ),
        ),
    )
    def test_fp16_matches_fp32(self, case, frames, preprocessors) -> None:
        """float16 output is the float32 output rounded to float16 (the DLA's float32 is its cast)."""
        inputs = [_as_kind(image, case.kind) for image in _images(frames, BATCHES["mixed"])]
        fp32 = preprocessors(replace(case, dtype="fp32")).preprocess(inputs)
        fp16 = preprocessors(replace(case, dtype="fp16")).preprocess(inputs)
        assert fp16[0].dtype == np.float16
        np.testing.assert_array_equal(fp16[0], fp32[0].astype(np.float16))
        assert fp16[1:] == fp32[1:]

    @pytest.mark.parametrize(
        "case",
        matrix(
            dtype=FLOATS,
            batch=("single", "mixed"),
            skip=lambda case: (
                "CPUPreprocessor has no direct_preproc: its batch stays on the host"
                if case.backend == "cpu"
                else None
            ),
        ),
    )
    def test_direct_preproc(self, case, frames, preprocessors) -> None:
        """direct_preproc leaves exactly the given images on the device, as preprocess returns them."""
        preprocessor = preprocessors(case)
        images = _images(frames, BATCHES[case.batch])
        buffer, ratios, padding = preprocessor.direct_preproc(images, no_warn=True)
        stream_synchronize(preprocessor._stream)
        width, height = case.output_shape
        assert isinstance(buffer, Buffer)
        assert buffer.is_device
        assert buffer.shape == (len(images), 3, height, width)
        assert buffer.dtype == DTYPES[case.dtype]
        _assert_same((buffer.numpy(), ratios, padding), preprocessor.preprocess(images))

    @pytest.mark.parametrize("case", matrix(cpu_resize=CPU_RESIZE))
    def test_warmup_keeps_outputs(self, case, frames, build_preprocessor) -> None:
        """warmup, before or between calls, leaves the outputs as a fresh preprocessor's."""
        images = _images(frames, TRIO)
        expected = build_preprocessor(case).preprocess(images)
        preprocessor = build_preprocessor(case)
        preprocessor.warmup()
        _assert_same(preprocessor.preprocess(images), expected)
        preprocessor.warmup()
        _assert_same(preprocessor.preprocess(images), expected)

    @pytest.mark.parametrize(
        "case",
        matrix(
            resize=RESIZES,
            cpu_resize=CPU_RESIZE,
            skip=lambda case: (
                "CPUPreprocessor always resizes with cv2: there is nothing to route"
                if case.backend == "cpu"
                else "the DLA cannot pad resize: host images always go to the CPU"
                if case.dla and case.resize == "pad"
                else None
            ),
        ),
    )
    def test_cpu_resize_routing(self, case, frames, preprocessors) -> None:
        """Host images resize where cpu_resize says (DLA: resizes_on_cpu), device Buffers on the device."""
        aligned = frames(*FRAMES["people-1920x1080"])
        unaligned = frames(*FRAMES["people-1000x750"])
        preprocessor = preprocessors(case)
        on_device = preprocessors(replace(case, cpu_resize=False if case.dla else None))
        on_cpu = preprocessors(replace(case, cpu_resize=True))
        unaligned_on_cpu = case.cpu_resize is True or (case.dla and case.cpu_resize is None)
        if case.dla:
            assert preprocessor.resizes_on_cpu(*aligned.shape[:2]) is bool(case.cpu_resize)
            assert preprocessor.resizes_on_cpu(*unaligned.shape[:2]) is unaligned_on_cpu
            with pytest.raises(ValueError, match=UNALIGNED):
                preprocessor.preprocess(Buffer.from_array(unaligned, MemoryLocation.DEVICE))
        batch = [Buffer.from_array(aligned, MemoryLocation.DEVICE), aligned]
        expected = [
            on_device.preprocess(aligned),
            (on_cpu if case.cpu_resize else on_device).preprocess(aligned),
        ]
        if case.dla and case.cpu_resize is False:
            with pytest.raises(ValueError, match=UNALIGNED):
                preprocessor.preprocess(unaligned)
        else:
            batch.append(Buffer.from_array(unaligned, pinned=True, mapped=True))
            expected.append((on_cpu if unaligned_on_cpu else on_device).preprocess(unaligned))
        result, ratios, padding = preprocessor.preprocess(batch)
        for slot, (tensor, slot_ratios, slot_padding) in enumerate(expected):
            np.testing.assert_array_equal(result[slot], tensor[0])
            assert ratios[slot] == slot_ratios[0]
            assert padding[slot] == slot_padding[0]
        if not case.dla:
            cv2_resized = on_cpu.preprocess(unaligned)[0]
            assert not np.array_equal(cv2_resized, on_device.preprocess(unaligned)[0])
        elif case.resize == "center_crop":
            bilinear = on_cpu.preprocess(aligned)[0]
            assert np.abs(bilinear - on_device.preprocess(aligned)[0]).max() > 1e-3

    @pytest.mark.parametrize(
        "case",
        matrix(
            resize=RESIZES,
            antialias=(True,),
            cpu_resize=(None, True),
            skip=lambda case: None if case.dla else "only the DLA leaves antialiasing to the CPU",
        ),
    )
    def test_dla_antialias_routing(self, case, frames, preprocessors, build_preprocessor) -> None:
        """The DLA center crops itself; other antialiased resizes go to the CPU or raise."""
        aligned = frames(*FRAMES["people-1920x1080"])
        device = Buffer.from_array(aligned, MemoryLocation.DEVICE)
        preprocessor = preprocessors(case)
        never_cpu = build_preprocessor(case, cpu_resize=False)
        if case.resize == "center_crop":
            assert preprocessor.resizes_on_cpu(*aligned.shape[:2]) is bool(case.cpu_resize)
            _check(case, preprocessor, device, [aligned], preprocessors)
            return
        assert preprocessor.resizes_on_cpu(*aligned.shape[:2]) is True
        assert preprocessor.resizes_on_cpu(*aligned.shape[:2], antialias=False) is bool(
            case.cpu_resize
        )
        _check(case, preprocessor, aligned, [aligned], preprocessors)
        for bad, owner in ((device, preprocessor), (aligned, never_cpu), (device, never_cpu)):
            with pytest.raises(ValueError, match="cannot antialias"):
                owner.preprocess(bad)

    @pytest.mark.parametrize("case", matrix(cpu_resize=CPU_RESIZE))
    def test_rejects_unknown_resize(self, case, frames, preprocessors, build_preprocessor) -> None:
        """An unknown resize method raises ValueError at construction and per call."""
        with pytest.raises(ValueError, match="Unknown method"):
            build_preprocessor(replace(case, size=(640, 640)), resize="area")
        with pytest.raises(ValueError, match=UNKNOWN_RESIZE):
            preprocessors(case).preprocess(frames(*FRAMES["people-1920x1080"]), resize="area")

    @pytest.mark.parametrize("case", matrix(skip=lambda case: NO_BATCH_SIZE.get(case.backend)))
    def test_rejects_batch_over_size(self, case, frames, preprocessors) -> None:
        """More images than the configured batch size raise ValueError."""
        images = [frames(*FRAMES["people-1920x1080"])] * (BATCH_SIZE + 1)
        with pytest.raises(ValueError, match="exceeds configured batch size"):
            preprocessors(case).preprocess(images)

    @pytest.mark.parametrize(
        "case",
        matrix(
            cpu_resize=CPU_RESIZE,
            kind=("numpy", "device"),
            bad=tuple(BAD_IMAGES),
            position=("single", "second"),
            xfail=lambda case: CPU_ACCEPTS_BAD_IMAGES if case.backend == "cpu" else None,
        ),
    )
    def test_rejects_bad_image(self, case, frames, preprocessors) -> None:
        """Images that are not uint8 with three channels raise ValueError, alone or after a good one."""
        good = frames(*FRAMES["people-1920x1080"])
        bad = _as_kind(BAD_IMAGES[case.bad](good), case.kind)
        with pytest.raises(ValueError, match=BAD_IMAGE):
            preprocessors(case).preprocess(bad if case.position == "single" else [good, bad])

    @pytest.mark.parametrize(
        "case",
        matrix(
            dtype=("uint8", "int8", "float64"),
            skip=lambda case: (
                "CPUPreprocessor writes any numpy dtype" if case.backend == "cpu" else None
            ),
            xfail=lambda case: TRT_IGNORES_DTYPE if case.backend == "trt" else None,
        ),
    )
    def test_rejects_unsupported_dtype(self, case, build_preprocessor) -> None:
        """Backends that normalize on the device raise ValueError for dtypes but float32/float16."""
        with pytest.raises(ValueError, match=UNSUPPORTED_DTYPE):
            build_preprocessor(case)
