# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from cv2ext.image import letterbox, resize_linear, rescale

if TYPE_CHECKING:
    from typing_extensions import Self


class ImageBatcher:
    """Creates image batches for calibrating TensorRT engines."""

    def __init__(
        self: Self,
        input_dir: Path | str,
        shape: tuple[int, int, int, int],
        dtype: np.dtype,
        max_images: int | None = None,
        resize_method: str = "letterbox",
        input_scale: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """
        Create batches of images for TensorRT calibration.

        Parameters
        ----------
        input_dir : Path, str
            The directory containing the images to calibrate the model with.
        shape : tuple[int, int, int]
            The expected input shape of the network in format BCHW
            (batch size, channels, height, width)
        dtype : np.dtype
            The expected datatype input of the network
        max_images : int, optional
            Optionally, specify the maximum number of images to calibrate with
        resize_method : str, optional
            The method by which to resize the images
            Options are: ['letterbox', 'linear'], default is 'letterbox'
        input_scale : tuple[float, float], optional
            The range with which the image should have values in
            Examples are: (0.0, 255.0), (0.0, 1.0), (-1.0, 1.0)
            The default is (0.0, 1.0) since that is the YOLO standard

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
        # verify resize method and input scale
        valid_resize_methods = ["letterbox", "linear"]
        if resize_method not in valid_resize_methods:
            err_msg = (
                f"Invalid resize method found, options are: {valid_resize_methods}"
            )
            raise ValueError(err_msg)
        self._resize_method = resize_method
        self._input_scale = input_scale

        # verify image directory
        image_dir = Path(input_dir)
        if not image_dir.exists():
            err_msg = f"Could not find image directory: {image_dir}"
            raise FileNotFoundError(err_msg)
        if image_dir.is_file():
            err_msg = f"Image directory: {image_dir} is a file."
            raise NotADirectoryError(err_msg)

        # get the paths of all images
        image_exts: list[str] = [".jpg", ".jpeg", ".png"]
        self._images: list[Path] = []
        for extension in image_exts:
            for file in image_dir.glob(f"*{extension}"):
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

        # handle the shape and dtype
        self._dtype = dtype
        self._batch = shape[0]
        self._channel = shape[1]
        self._height = shape[2]
        self._width = shape[3]

        # generate batches list
        batches: list[list[Path]] = []
        for i in range(len(self._images)):
            batch: list[Path] = []
            for _ in range(self._batch):
                batch.append(self._images[i])
            if len(batch) != self._batch:
                continue
            batches.append(batch)
        
        if len(batches) == 0:
            err_msg = "Could not form any valid batches."
            raise ValueError(err_msg)

    def get_batch(self: Self) -> list[np.ndarray]:
        """
        Get a batch of images which have been preprocessed.

        Returns
        -------
        list[np.ndarray]
            The batch of images

        """
        def _preprocess(img: np.ndarray) -> np.ndarray:
            # resize and rescale the image
            if self._resize_method == "letterbox":
                resized_img, _, _ = letterbox(img, (self._width, self._height))
            else:
                resized_img, _ = resize_linear(img, (self._width, self._height))
            rescaled_img = rescale(resized_img, self._input_scale)

            # run the transpose operations and make contiguous
            new_img = rescaled_img[np.newaxis, :]
            new_img = np.transpose(new_img, (0, 3, 1, 2))
            new_img = new_img.astype(self._dtype)
            if not new_img.flags["C_CONTIGUOUS"]:
                new_img = np.ascontiguousarray(new_img)
            return new_img

    # def preprocess_image(self, image_path):
    #     """
    #     The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
    #     resizing, normalization, data type casting, and transposing.
    #     This Image Batcher implements one algorithm for now:
    #     * Resizes and pads the image to fit the input size.
    #     :param image_path: The path to the image on disk to load.
    #     :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
    #     batch, and the resize scale used, if any.
    #     """

    #     def resize_pad(image, pad_color=(0, 0, 0)):
    #         """
    #         A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
    #         size, and pads the remaining bottom-right portions with the value provided.
    #         :param image: The PIL image object
    #         :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
    #         :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
    #         """
    #         # Get characteristics.
    #         width, height = image.size
    #         width_scale = width / self.width
    #         height_scale = height / self.height

    #         # Depending on preprocessor, box scaling will be slightly different.
    #         if self.preprocessor == "fixed_shape_resizer":
    #             scale = [self.width / width, self.height / height]
    #             image = image.resize((self.width, self.height), resample=Image.BILINEAR)
    #             return image, scale
    #         if self.preprocessor == "keep_aspect_ratio_resizer":
    #             scale = 1.0 / max(width_scale, height_scale)
    #             image = image.resize(
    #                 (round(width * scale), round(height * scale)),
    #                 resample=Image.BILINEAR,
    #             )
    #             pad = Image.new("RGB", (self.width, self.height))
    #             pad.paste(pad_color, [0, 0, self.width, self.height])
    #             pad.paste(image)
    #             return pad, scale

    #     scale = None
    #     image = Image.open(image_path)
    #     image = image.convert(mode="RGB")
    #     if (
    #         self.preprocessor == "fixed_shape_resizer"
    #         or self.preprocessor == "keep_aspect_ratio_resizer"
    #     ):
    #         # Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
    #         image, scale = resize_pad(image, (124, 116, 104))
    #         image = np.asarray(image, dtype=self.dtype)
    #     else:
    #         print(f"Preprocessing method {self.preprocessor} not supported")
    #         sys.exit(1)
    #     if self.format == "NCHW":
    #         image = np.transpose(image, (2, 0, 1))
    #     # return image/255., scale
    #     return image, scale

    # def get_batch(self):
    #     """
    #     Retrieve the batches. This is a generator object, so you can use it within a loop as:
    #     for batch, images in batcher.get_batch():
    #        ...
    #     Or outside of a batch with the next() function.
    #     :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
    #     paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
    #     """
    #     for i, batch_images in enumerate(self.batches):
    #         batch_data = np.zeros(self.shape, dtype=self.dtype)
    #         batch_scales = [None] * len(batch_images)
    #         for i, image in enumerate(batch_images):
    #             self.image_index += 1
    #             batch_data[i], batch_scales[i] = self.preprocess_image(image)
    #         self.batch_index += 1
    #         yield batch_data, batch_images, batch_scales
