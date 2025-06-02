# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING

import cv2
import numpy as np
from cv2ext.image import letterbox, rescale

from trtutils._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self


class AbstractBatcher(ABC):
    """Abstract base class for data batching classes."""

    @abstractmethod
    def __init__(self: Self) -> None:
        """Initialize the batcher."""

    @property
    @abstractmethod
    def num_batches(self: Self) -> int:
        """Get the number of batches."""

    @property
    @abstractmethod
    def batch_size(self: Self) -> int:
        """Get the batch size."""

    @abstractmethod
    def get_next_batch(self: Self) -> np.ndarray | None:
        """Get the batch of data."""


class ImageBatcher(AbstractBatcher):
    """Creates image batches for calibrating TensorRT engines."""

    def __init__(
        self: Self,
        image_dir: Path | str,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        batch_size: int = 8,
        order: str = "NCHW",
        max_images: int | None = None,
        resize_method: str = "letterbox",
        input_scale: tuple[float, float] = (0.0, 1.0),
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Create batches of images for TensorRT calibration.

        Parameters
        ----------
        image_dir : Path, str
            The directory containing the images to calibrate the model with.
        shape : tuple[int, int, int]
            The expected input shape of the network in format HWC
            (height, width, channels)
        dtype : np.dtype
            The expected datatype input of the network
        batch_size : int, optional
            The batch size to group images in.
            Default is 8
        order : str, optional
            The ordering of data elements expected by the network.
            Options are: ['NCHW', 'NHWC'], default is 'NCHW'
        max_images : int, optional
            Optionally, specify the maximum number of images to calibrate with
        resize_method : str, optional
            The method by which to resize the images
            Options are: ['letterbox', 'linear'], default is 'letterbox'
        input_scale : tuple[float, float], optional
            The range with which the image should have values in
            Examples are: (0.0, 255.0), (0.0, 1.0), (-1.0, 1.0)
            The default is (0.0, 1.0)
        verbose : bool, optional
            Whether to print verbose output, by default None

        Raises
        ------
        FileNotFoundError
            If the input_dir cannot be found
        NotADirectoryError
            If the input_dir is a file
        ValueError
            If the max_images is set less than one
        ValueError
            If no images could be found in the input_dir
        ValueError
            If no valid batches could be formed

        """
        self._verbose = verbose

        # verify resize method and input scale
        valid_resize_methods = ["letterbox", "linear"]
        if resize_method not in valid_resize_methods:
            err_msg = (
                f"Invalid resize method found, options are: {valid_resize_methods}"
            )
            raise ValueError(err_msg)
        self._resize_method = resize_method
        self._input_scale = input_scale

        # handle the shape and dtype
        self._dtype = dtype
        self._batch = batch_size
        self._height = shape[0]
        self._width = shape[1]
        self._channel = shape[2]

        # assign order
        valid_orders: list[str] = ["NCHW", "NHWC"]
        if order not in valid_orders:
            err_msg = f"Invalid order found, {order}, options are: {valid_orders}"
            raise ValueError(err_msg)
        self._order = order
        self._data_shape: tuple[int, int, int, int] = (
            self._batch,
            self._channel,
            self._height,
            self._width,
        )
        if order == "NHWC":
            self._data_shape = (self._batch, self._height, self._width, self._channel)

        # verify image directory
        image_dir_path: Path = Path(image_dir)
        if not image_dir_path.exists():
            err_msg = f"Could not find image directory: {image_dir}"
            raise FileNotFoundError(err_msg)
        if image_dir_path.is_file():
            err_msg = f"Image directory: {image_dir} is a file."
            raise NotADirectoryError(err_msg)

        # get the paths of all images
        image_exts: list[str] = [".jpg", ".jpeg", ".png"]
        self._images: list[Path] = []
        for extension in image_exts:
            for file in image_dir_path.glob(f"*{extension}"):
                self._images.append(file)

        # ensure we found some images
        if len(self._images) == 0:
            err_msg = f"Could not find any images in directory: {image_dir}"
            raise ValueError(err_msg)

        # make the list deterministic
        self._images.sort()

        # trim to max_images
        if max_images is not None:
            if not max_images > 1:
                err_msg = f"max_images must be greated than 1, found: {max_images}"
                raise ValueError(err_msg)

            self._images = self._images[0:max_images]

        # generate batches list
        self._batches: list[list[Path]] = []
        for i in range(0, len(self._images), self._batch):
            with contextlib.suppress(IndexError):
                batch: list[Path] = [self._images[i + j] for j in range(self._batch)]
                self._batches.append(batch)

        if len(self._batches) == 0:
            err_msg = "Could not form any valid batches."
            raise ValueError(err_msg)

        LOG.debug(f"ImageBatcher found images: {len(self._images)}")
        LOG.debug(f"ImageBatcher formed batches: {len(self._batches)}")

        # tracking indices for iteration
        self._current_batch: int = 0

        # threading setup to speedup loading
        cpu_cores = os.cpu_count()
        if cpu_cores is None:
            cpu_cores = 1
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._batch, cpu_cores - 1),
        )
        self._event: Event = Event()
        self._queue: Queue[np.ndarray] = Queue(maxsize=3)
        self._thread: Thread = Thread(target=self._run, daemon=True)

        atexit.register(self._close)

        self._thread.start()

    @property
    def num_batches(self: Self) -> int:
        """Get the number of batches."""
        return len(self._batches)

    @property
    def batch_size(self: Self) -> int:
        """Get the batch size."""
        return self._batch

    def _close(self: Self) -> None:
        self._event.set()
        self._thread.join()

    def _get_image(self: Self, image_path: Path) -> np.ndarray:
        # read image
        img = cv2.imread(str(image_path.resolve()))

        # resize and rescale the image
        if self._resize_method == "letterbox":
            resized_img, _, _ = letterbox(img, (self._width, self._height))
        else:
            resized_img = cv2.resize(
                img,
                (self._width, self._height),
                interpolation=cv2.INTER_LINEAR,
            )
        rescaled_img = rescale(resized_img, self._input_scale)

        # run the transpose operations and make contiguous
        new_img = rescaled_img[np.newaxis, :]
        if self._order == "NCHW":
            new_img = np.transpose(new_img, (0, 3, 1, 2))
        else:
            new_img = np.transpose(new_img, (0, 1, 2, 3))

        new_img = new_img.astype(self._dtype)

        if not new_img.flags["C_CONTIGUOUS"]:
            new_img = np.ascontiguousarray(new_img)

        return new_img

    def _run(self: Self) -> None:
        # for each batch get the images
        for idx, image_paths in enumerate(self._batches):
            if self._event.is_set():
                return

            LOG.debug(f"ImageBatcher getting batch: {idx}")

            # get the batch
            results = list(self._pool.map(self._get_image, image_paths))
            data = np.zeros(self._data_shape, dtype=self._dtype)

            for i, img in enumerate(results):
                data[i] = img

            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

            # keep trying to put images into the queue
            while not self._event.is_set():
                try:
                    self._queue.put(data, timeout=0.1)

                    if self._verbose:
                        LOG.debug(
                            f"ImageBatcher put batch: {idx} / {len(self._batches)}"
                        )

                    break
                except Full:
                    continue

    def get_next_batch(self: Self) -> np.ndarray | None:
        """
        Get a batch of images which have been preprocessed.

        Returns
        -------
        np.ndarray | None
            The batch of images if one exists

        """
        if self._current_batch == len(self._batches):
            return None

        while not self._event.is_set():
            with contextlib.suppress(Empty):
                batch = self._queue.get(timeout=0.1)
                self._current_batch += 1

                if self._verbose:
                    LOG.debug(
                        f"ImageBatcher get batch: {self._current_batch} / {len(self._batches)}"
                    )

                return batch

        return None
