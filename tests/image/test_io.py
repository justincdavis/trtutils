# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for trtutils.io -- GPU image and video decode into device Buffers."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest

from trtutils import FLAGS
from trtutils.core import Buffer
from trtutils.image import Detector

cv2 = pytest.importorskip("cv2")
io = pytest.importorskip("trtutils.io")

needs_images = pytest.mark.skipif(not FLAGS.FOUND_NVIMGCODEC, reason="nvImageCodec not installed")
needs_video = pytest.mark.skipif(
    not FLAGS.FOUND_PYNVVIDEOCODEC, reason="PyNvVideoCodec not installed"
)

if TYPE_CHECKING:
    from pathlib import Path

SIZES = [
    pytest.param((1080, 1920), id="1080p"),
    pytest.param((2160, 3840), id="4k"),
]
NUM_FRAMES = 12


@pytest.fixture(scope="module")
def jpeg_paths(tmp_path_factory, images) -> dict[tuple[int, int], Path]:
    """JPEGs of a real photo resized to each size."""
    root = tmp_path_factory.mktemp("jpeg")
    paths = {}
    for height, width in (p.values[0] for p in SIZES):
        img = cv2.resize(images["horse"].array, (width, height), interpolation=cv2.INTER_CUBIC)
        path = root / f"{height}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths[(height, width)] = path
    return paths


@pytest.fixture(scope="module")
def video_path(tmp_path_factory) -> Path:
    """A small H.264 clip made with ffmpeg."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc=size=640x360:rate=30", "-frames:v", str(NUM_FRAMES),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path),
    ]  # fmt: skip
    if subprocess.run(cmd, check=False).returncode != 0:  # noqa: S603
        pytest.skip("ffmpeg cannot encode H.264")
    return path


@needs_images
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("as_bytes", [False, True], ids=["path", "bytes"])
def test_decode_image_matches_cv2(jpeg_paths, size, as_bytes) -> None:
    """decode_image is a uint8 HWC device Buffer matching cv2.imread to within IDCT rounding."""
    path = jpeg_paths[size]
    buf = io.decode_image(path.read_bytes() if as_bytes else path)
    assert isinstance(buf, Buffer)
    assert buf.is_device
    assert buf.dtype == np.uint8
    assert buf.shape == (*size, 3)
    diff = np.abs(buf.numpy().astype(int) - cv2.imread(str(path)).astype(int))
    # nvJPEG and libjpeg round a few pixels differently; a channel swap would fail by far more
    assert diff.max() <= 1
    assert (diff > 0).mean() < 1e-3


@needs_images
def test_decode_image_exact_on_photo(images) -> None:
    """On the stock horse photo the decode is bit-identical to cv2.imread."""
    image = images["horse"]
    np.testing.assert_array_equal(io.decode_image(image.path).numpy(), image.array)


@needs_images
def test_decode_image_bad_data_raises() -> None:
    """Undecodable bytes raise ValueError."""
    with pytest.raises(ValueError, match="could not decode"):
        io.decode_image(b"not an image")


@needs_images
def test_end2end_on_decoded_buffer_matches_cv2(yolov10_engine, images) -> None:
    """end2end on the decoded device Buffer equals end2end on the cv2 array."""
    model = Detector(yolov10_engine, warmup=False)
    path = images["horse"].path
    expected = model.end2end(cv2.imread(str(path)))
    assert expected
    assert model.end2end(io.decode_image(path)) == expected


@needs_video
def test_video_reader_frames(video_path) -> None:
    """VideoReader yields every frame as device BGR close to cv2's decode."""
    cap = cv2.VideoCapture(str(video_path))
    with io.VideoReader(video_path) as reader:
        assert len(reader) == NUM_FRAMES
        count = 0
        for frame in reader:
            ok, ref = cap.read()
            assert ok
            assert isinstance(frame, Buffer)
            assert frame.is_device
            assert frame.dtype == np.uint8
            assert frame.shape == (360, 640, 3) == reader.shape
            diff = np.abs(frame.numpy().astype(int) - ref.astype(int))
            # different colour matrix than cv2; a channel swap would give a far larger diff
            assert diff.mean() < 4
            count += 1
    assert count == NUM_FRAMES


@pytest.mark.parametrize(
    ("flag", "call"),
    [
        pytest.param("FOUND_NVIMGCODEC", lambda _p: io.decode_image(b""), id="image"),
        pytest.param("FOUND_PYNVVIDEOCODEC", io.VideoReader, id="video"),
    ],
)
def test_missing_library_raises(monkeypatch, flag, call, tmp_path) -> None:
    """Without the decoder libraries the entry points raise a clear ImportError."""
    monkeypatch.setattr(FLAGS, flag, False)
    with pytest.raises(ImportError, match="not installed"):
        call(tmp_path / "x.mp4")
