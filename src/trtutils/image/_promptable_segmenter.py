# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import nvtx
from typing_extensions import Literal

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer, MemoryLocation
from trtutils.core._device import Device
from trtutils.core._stream import stream_synchronize

from ._image_model import ImageModel
from ._schema import (
    ImageEncoderOutputSchema,
    MaskDecoderInputSchema,
    MaskDecoderOutputSchema,
    resolve_schema,
)
from .interfaces import ImageEncoderInterface
from .postprocessors import (
    get_image_embeddings,
    postprocess_image_embeddings,
    postprocess_prompt_masks,
)
from .preprocessors import encode_prompts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from trtutils.compat._libs import cuda

    from ._image_model import _PostFn
    from .types import PromptMask

# whether prompt sets with a box get the padding point too: SAM's predictor adds it only
# without a box, SAM2ImagePredictor always does (box masks lose up to 0.1 IoU without it)
_PAD_BOXES = {MaskDecoderInputSchema.SAM: False, MaskDecoderInputSchema.SAM2: True}


class ImageEncoder(ImageModel, ImageEncoderInterface):
    """Implementation of image encoders, the first stage of a promptable segmenter."""

    _OUTPUT_SCHEMA = ImageEncoderOutputSchema

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cuda",
        resize_method: str = "pad",
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        output_schema: ImageEncoderOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        cpu_resize: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create an ImageEncoder object.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_range : tuple[float, float]
            The range of input values which should be passed to
            the model. By default [0.0, 1.0].
        preprocessor : str
            The type of preprocessor to use.
            The options are ['cpu', 'cuda', 'trt', 'dla0', 'dla1'], default is 'cuda'.
        resize_method : str
            The type of resize algorithm to use.
            The options are ['letterbox', 'linear', 'center_crop', 'pad'], default is 'pad',
            the top-left resize SAM encoders are trained with.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, (0.485, 0.456, 0.406) (ImageNet mean).
            Pass None to disable normalization.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, (0.229, 0.224, 0.225) (ImageNet std).
            Pass None to disable normalization.
        output_schema : ImageEncoderOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be an ImageEncoderOutputSchema enum value or a string matching
            the enum name (e.g., "SAM"). By default None, which means the schema
            is detected from the engine's output names.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this image encoder. Default is None,
            which uses the current device.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        cuda_graph : bool, optional
            Whether or not to enable CUDA graph capture for optimized execution.
            When enabled, the engine records a CUDA graph for each input
            configuration (batch size, resolution, input buffers) once it
            repeats and replays it afterwards, for run() and end2end() alike.
            Preprocessing and postprocessing run outside the graphs.
            Only effective with async_v3 backend. Default is True.
        postprocessor : str, optional
            Where to postprocess, 'cpu' or 'cuda'. Default is None, which
            keeps 'cpu' unless the task's CUDA postprocessor is the default.
        cpu_resize : bool, optional
            Whether to resize images on the CPU and leave the rest of the
            preprocessing to the preprocessor. Default is None, which resizes on
            the CPU only what the preprocessor cannot resize, see :class:`ImageModel`.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If the output schema string is invalid, or the engine outputs
            do not match the requested schema.

        """
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            postprocessor=postprocessor,
            cpu_resize=cpu_resize,
            no_warn=no_warn,
            verbose=verbose,
            output_schema=output_schema,
        )

        # prepend with 'enc_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "enc_postprocess": f"image_encoder::postprocess [{self._tag}]",
                "enc_get_image_embeddings": f"image_encoder::get_image_embeddings [{self._tag}]",
                "enc_encode": f"image_encoder::encode [{self._tag}]",
            }
        )

    @property
    def output_schema(self: Self) -> ImageEncoderOutputSchema:
        """Get the output schema used by this image encoder."""
        return self._output_schema

    def postprocess(
        self: Self,
        outputs: list[np.ndarray] | list[Buffer],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs from the engine to postprocess.
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            Per image, its (1, ...) outputs in the order of :attr:`output_schema`, the
            (1, C, H, W) embeddings first.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["enc_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        t0 = time.perf_counter()
        data = postprocess_image_embeddings(
            self._host_outputs(outputs), no_copy=no_copy, verbose=verbose
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(self: Self, *, verbose: bool | None) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, _r, _p, nc: self.postprocess(o, no_copy=nc, verbose=verbose)

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def __call__(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE by
            future inferences.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            The outputs. For single image input with postprocess=True,
            returns list[np.ndarray]. For batch input, returns batch results.

        """
        return self.run(
            images,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    # run overloads - single image input (3 overloads)
    @overload
    def run(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def run(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE by
            future inferences.
            In special case where, preprocessing and
            postprocessing will occur during run and no_copy
            was not passed (is None), then no_copy will be used
            for preprocessing and inference stages.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            For single image with postprocess=True: list[np.ndarray] (single image outputs).
            For batch with postprocess=True: list[list[np.ndarray]] (per-image outputs).
            For postprocess=False: list[np.ndarray] (raw outputs).

        Raises
        ------
        ValueError
            If preprocessed inputs are not a single batch tensor.

        """
        return self._run_core(
            images,
            None,
            None,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(verbose=verbose),
            needs_ratios=False,
        )

    # get_image_embeddings overloads
    @overload
    def get_image_embeddings(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> np.ndarray: ...

    @overload
    def get_image_embeddings(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def get_image_embeddings(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Get the image embeddings from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        np.ndarray | list[np.ndarray]
            For single image: np.ndarray (embeddings of shape (1, C, H, W)).
            For batch: list[np.ndarray] (embeddings per image).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["enc_get_image_embeddings"])

        if verbose:
            LOG.debug(f"{self._tag}: get_image_embeddings")

        batch, single = self._as_batch(outputs)
        result = get_image_embeddings(batch, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_image_embeddings

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        verbose: bool | None = ...,
    ) -> np.ndarray: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_image_embeddings in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        np.ndarray | list[np.ndarray]
            For single image: np.ndarray (embeddings of shape (1, C, H, W)).
            For batch: list[np.ndarray] (embeddings per image).

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(verbose=verbose),
            get=lambda pp: get_image_embeddings(pp, verbose=verbose),
        )

    def encode(
        self: Self,
        image: np.ndarray | Buffer,
        *,
        verbose: bool | None = None,
    ) -> tuple[list[Buffer], tuple[float, float], tuple[float, float]]:
        """
        Preprocess one image and run the engine, leaving its outputs on the device.

        The engine work is only enqueued on the stream of :attr:`engine`: synchronize it
        (or keep reading on that stream) before using the outputs.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image (HWC format) to encode.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[list[Buffer], tuple[float, float], tuple[float, float]]
            The device outputs in the order of :attr:`output_schema`, the (1, C, H, W)
            embeddings first, as views of the engine outputs that the next run
            overwrites; and the image's (x, y) ratios and padding, which map image
            pixels to the model input as ``p * ratio + padding``.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["enc_encode"])

        if verbose:
            LOG.debug(f"{self._tag}: encode")

        inputs, ratios, padding = self.prepare_inputs([image], verbose=verbose)
        t0 = time.perf_counter()
        outputs = self._order_outputs(self._engine.raw_exec(inputs, graph=True))
        self._infer_profile = (t0, time.perf_counter())

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # encode

        # images that are not preprocessed always come back with their ratios and padding
        return outputs, cast("list", ratios)[0], cast("list", padding)[0]


class MaskDecoder:
    """
    Implementation of SAM-style prompt and mask decoders.

    The engine takes the embeddings of one image and B prompt sets of N points each,
    and returns candidate masks with their predicted IoU per set. It has no image input,
    so it is not an :class:`ImageModel`; inputs are bound by name.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_schema: MaskDecoderInputSchema | str | None = None,
        output_schema: MaskDecoderOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        stream: cuda.cudaStream_t | None = None,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        shape_contexts: int | None = 4,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a MaskDecoder object.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_schema : MaskDecoderInputSchema, str, optional
            Manually specify the input schema instead of auto-detection, as an enum
            member or its name (e.g., "SAM"). By default None, which detects it
            from the engine's input names.
        output_schema : MaskDecoderOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection, as an enum
            member or its name (e.g., "SAM"). By default None, which detects it
            from the engine's output names.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this decoder. Default is None,
            which uses the current device.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        stream : cuda.cudaStream_t, optional
            The CUDA stream to run on, e.g. the image encoder's. By default None,
            which creates a new stream.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        cuda_graph : bool, optional
            Whether or not to enable CUDA graph capture, one graph per prompt shape
            that repeats. Only effective with async_v3 backend. Default is True.
        shape_contexts : int, optional
            The most per-shape execution contexts the engine keeps, see
            :class:`~trtutils.TRTEngine`. The number of points changes from prompt
            to prompt, which a context per shape keeps cheap. Before TensorRT 10
            a single context serves every shape. Default is 4.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If the engine inputs or outputs match no schema, or the schema strings
            are invalid.

        """
        self._tag: str = f"{Path(engine_path).stem}"
        self._engine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            backend=backend,
            stream=stream,
            warmup=warmup,
            dla_core=dla_core,
            device=device,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            shape_contexts=shape_contexts,
            no_warn=no_warn,
            verbose=verbose,
        )
        self._input_schema: MaskDecoderInputSchema = resolve_schema(
            self._engine, MaskDecoderInputSchema, input_schema, inputs=True
        )
        self._output_schema: MaskDecoderOutputSchema = resolve_schema(
            self._engine, MaskDecoderOutputSchema, output_schema
        )
        spec = dict(zip(self._engine.input_names, self._engine.input_spec))
        # the image encoder outputs come first, the prompt inputs last
        coords_name, _, mask_name, has_mask_name = self._input_schema.value[-4:]
        # the profile's largest (prompt sets, points per set)
        self._max_prompts: int = spec[coords_name][0][0]
        self._max_points: int = spec[coords_name][0][1]
        # no mask prompt: zeros on the device, bound in place at stable addresses
        with Device(device):
            self._mask_input = Buffer.from_array(np.zeros(*spec[mask_name]), MemoryLocation.DEVICE)
            self._has_mask_input = Buffer.from_array(
                np.zeros(*spec[has_mask_name]), MemoryLocation.DEVICE
            )
        outputs = self._engine.output_names
        self._output_order = [outputs.index(name) for name in self._output_schema.value]

        self._nvtx_tags: dict[str, str] = {
            "infer": f"mask_decoder::infer [{self._tag}]",
            "postprocess": f"mask_decoder::postprocess [{self._tag}]",
        }
        self._infer_profile: tuple[float, float] = (0.0, 0.0)
        self._post_profile: tuple[float, float] = (0.0, 0.0)

        if verbose:
            LOG.debug(
                f"{self._tag}: MaskDecoder, schema {self._input_schema.name}, "
                f"up to {self._max_prompts} prompt sets of {self._max_points} points"
            )

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

    @property
    def name(self: Self) -> str:
        """Get the name of the engine."""
        return self._engine.name

    @property
    def input_schema(self: Self) -> MaskDecoderInputSchema:
        """Get the input schema used by this decoder."""
        return self._input_schema

    @property
    def output_schema(self: Self) -> MaskDecoderOutputSchema:
        """Get the output schema used by this decoder."""
        return self._output_schema

    @property
    def max_prompts(self: Self) -> int:
        """The most prompt sets one :meth:`infer` call takes."""
        return self._max_prompts

    @property
    def max_points(self: Self) -> int:
        """The most points (box corners and padding point included) per prompt set."""
        return self._max_points

    @property
    def infer_profile(self: Self) -> tuple[float, float]:
        """The start and end times of the latest engine execution."""
        return self._infer_profile

    def infer(
        self: Self,
        features: Sequence[Buffer],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the decoder on prompts encoded by :func:`~trtutils.image.preprocessors.encode_prompts`.

        Parameters
        ----------
        features : Sequence[Buffer]
            The image encoder outputs of one image, as :meth:`ImageEncoder.encode` returns
            them: the (1, C, H, W) embeddings, then any other features of the input schema.
            Usually on the device; if they were produced on another stream, that stream
            must be synchronized first.
        point_coords : np.ndarray
            The (B, N, 2) float32 points in model input pixels.
        point_labels : np.ndarray
            The (B, N) float32 point labels.
        no_copy : bool, optional
            If True, return the engine's host output memory without a copy. It is
            overwritten by the next call.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray]
            The predicted IoU (B, M) and low-res mask logits (B, M, h, w).

        Raises
        ------
        ValueError
            If B or N exceed the engine profile, see :attr:`max_prompts` and
            :attr:`max_points`.

        """
        num_prompts, num_points = point_coords.shape[:2]
        if num_prompts > self._max_prompts or num_points > self._max_points:
            err_msg = (
                f"{self._tag}: got {num_prompts} prompt sets of {num_points} points, the "
                f"engine takes at most {self._max_prompts} sets of {self._max_points} points"
            )
            raise ValueError(err_msg)

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["infer"])

        if verbose:
            LOG.debug(f"{self._tag}: infer {point_coords.shape}")

        inputs = (
            *features,
            Buffer.wrap(np.ascontiguousarray(point_coords, np.float32)),
            Buffer.wrap(np.ascontiguousarray(point_labels, np.float32)),
            self._mask_input,
            self._has_mask_input,
        )
        t0 = time.perf_counter()
        outputs = self._engine.execute(dict(zip(self._input_schema.value, inputs)), no_copy=no_copy)
        self._infer_profile = (t0, time.perf_counter())

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # infer

        return [outputs[i] for i in self._output_order]

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        input_size: tuple[int, int],
        image_shape: tuple[int, int],
        ratios: tuple[float, float],
        padding: tuple[float, float],
        *,
        multimask: bool = True,
        mask_thres: float = 0.0,
        verbose: bool | None = None,
    ) -> list[PromptMask]:
        """
        Turn the outputs of :meth:`infer` into one mask per prompt set.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs of :meth:`infer`.
        input_size : tuple[int, int]
            The (width, height) of the image encoder input.
        image_shape : tuple[int, int]
            The (height, width) of the original image.
        ratios : tuple[float, float]
            The image's (x, y) ratios from the image encoder.
        padding : tuple[float, float]
            The image's (x, y) padding from the image encoder.
        multimask : bool
            Whether to pick the best of the multimask outputs instead of the single
            mask. By default True.
        mask_thres : float
            The logit above which a pixel is in the mask. By default 0.0.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[PromptMask]
            Per prompt set, the (H, W) uint8 mask and its predicted IoU.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["postprocess"])

        t0 = time.perf_counter()
        masks = postprocess_prompt_masks(
            outputs,
            input_size,
            image_shape,
            ratios,
            padding,
            multimask=multimask,
            mask_thres=mask_thres,
            verbose=verbose,
        )
        self._post_profile = (t0, time.perf_counter())

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return masks


class PromptableSegmenter:
    """
    Promptable (SAM-style) segmentation: an image encoder followed by a mask decoder.

    :meth:`set_image` encodes an image once, then each :meth:`predict` call turns point
    and box prompts in image pixels into masks with only the (fast) decoder.
    """

    def __init__(
        self: Self,
        encoder: ImageEncoderInterface,
        decoder: MaskDecoder,
        mask_thres: float = 0.0,
        *,
        multimask: bool = True,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a PromptableSegmenter object.

        Parameters
        ----------
        encoder : ImageEncoderInterface
            The image encoder.
        decoder : MaskDecoder
            The mask decoder.
        mask_thres : float
            The logit above which a pixel is in the mask. By default 0.0.
        multimask : bool
            Whether to pick the best of the decoder's multimask outputs (by predicted
            IoU) instead of its single mask. Multimask suits ambiguous prompts such as
            one point; the single mask suits boxes and several points. By default True.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        """
        self._nvtx_tags = {
            "set_image": "promptable_segmenter::set_image",
            "predict": "promptable_segmenter::predict",
        }
        self._encoder = encoder
        self._decoder = decoder
        self._multimask = multimask
        self._mask_thres = mask_thres
        # the encoder outputs of the current image, copied out of the encoder so a later
        # encoder run cannot overwrite them and decoder CUDA graphs see stable addresses
        self._features: list[Buffer] | None = None
        self._image_shape: tuple[int, int] = (0, 0)
        self._ratios: tuple[float, float] = (1.0, 1.0)
        self._padding: tuple[float, float] = (0.0, 0.0)
        if verbose:
            LOG.debug(f"Creating PromptableSegmenter: {encoder.name} + {decoder.name}")

    @property
    def encoder(self: Self) -> ImageEncoderInterface:
        """Get the image encoder."""
        return self._encoder

    @property
    def decoder(self: Self) -> MaskDecoder:
        """Get the mask decoder."""
        return self._decoder

    def set_image(self: Self, image: np.ndarray | Buffer, *, verbose: bool | None = None) -> None:
        """
        Encode the image that the following :meth:`predict` calls segment.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image (HWC format).
        verbose : bool, optional
            Whether or not to log additional information.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["set_image"])

        if verbose:
            LOG.debug(f"PromptableSegmenter: set_image {image.shape}")

        features, self._ratios, self._padding = self._encoder.encode(image, verbose=verbose)
        stream = self._encoder.engine.stream
        if self._features is None:
            self._features = [Buffer.device(f.shape, f.dtype) for f in features]
        for buffer, feature in zip(self._features, features):
            buffer.copy_from(feature, stream=stream)
        # the decoder may run on another stream
        stream_synchronize(stream)
        self._image_shape = (int(image.shape[0]), int(image.shape[1]))

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # set_image

    def predict(
        self: Self,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        *,
        multimask: bool | None = None,
        mask_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[PromptMask]:
        """
        Segment the image of the latest :meth:`set_image` call from prompts.

        Parameters
        ----------
        points : np.ndarray, optional
            The (N, 2) points as (x, y) in image pixels, shared by every prompt set.
        labels : np.ndarray, optional
            The (N,) point labels, 1 for foreground and 0 for background.
            By default None, which labels every point foreground.
        boxes : np.ndarray, optional
            One (4,) box or (K, 4) boxes as (x1, y1, x2, y2) in image pixels. Each box
            is one prompt set, together with the points.
        multimask : bool, optional
            Override of the constructor's ``multimask`` for this call.
        mask_thres : float, optional
            Override of the constructor's ``mask_thres`` for this call.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[PromptMask]
            One (mask, predicted IoU) per box, or a single one without boxes. Each mask
            is (H, W) uint8 (0 or 1) at the image resolution.

        Raises
        ------
        RuntimeError
            If no image was set.
        ValueError
            If neither points nor boxes are given, or a prompt set has more points than
            the decoder takes.

        """
        if self._features is None:
            err_msg = "PromptableSegmenter: call set_image before predict"
            raise RuntimeError(err_msg)

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["predict"])

        multimask = multimask if multimask is not None else self._multimask
        mask_thres = mask_thres if mask_thres is not None else self._mask_thres
        coords, point_labels = encode_prompts(
            points,
            labels,
            boxes,
            self._ratios,
            self._padding,
            pad_boxes=_PAD_BOXES[self._decoder.input_schema],
            verbose=verbose,
        )
        # every prompt set has the same number of points, so they batch in chunks of the
        # profile's max instead of being padded with extra points
        step = self._decoder.max_prompts
        masks: list[PromptMask] = []
        for start in range(0, len(coords), step):
            outputs = self._decoder.infer(
                self._features,
                coords[start : start + step],
                point_labels[start : start + step],
                no_copy=True,
                verbose=verbose,
            )
            masks.extend(
                self._decoder.postprocess(
                    outputs,
                    self._encoder.input_shape,
                    self._image_shape,
                    self._ratios,
                    self._padding,
                    multimask=multimask,
                    mask_thres=mask_thres,
                    verbose=verbose,
                )
            )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # predict

        return masks

    def end2end(
        self: Self,
        image: np.ndarray | Buffer,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        *,
        multimask: bool | None = None,
        mask_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[PromptMask]:
        """
        Encode an image and segment it from prompts: :meth:`set_image`, then :meth:`predict`.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image (HWC format).
        points : np.ndarray, optional
            The (N, 2) points as (x, y) in image pixels, see :meth:`predict`.
        labels : np.ndarray, optional
            The (N,) point labels, see :meth:`predict`.
        boxes : np.ndarray, optional
            One (4,) box or (K, 4) boxes in image pixels, see :meth:`predict`.
        multimask : bool, optional
            Override of the constructor's ``multimask`` for this call.
        mask_thres : float, optional
            Override of the constructor's ``mask_thres`` for this call.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[PromptMask]
            One (mask, predicted IoU) per box, or a single one without boxes.

        """
        self.set_image(image, verbose=verbose)
        return self.predict(
            points, labels, boxes, multimask=multimask, mask_thres=mask_thres, verbose=verbose
        )

    def __call__(
        self: Self,
        image: np.ndarray | Buffer,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        *,
        multimask: bool | None = None,
        mask_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[PromptMask]:
        """
        Encode an image and segment it from prompts, see :meth:`end2end`.

        Parameters
        ----------
        image : np.ndarray | Buffer
            The image (HWC format).
        points : np.ndarray, optional
            The (N, 2) points as (x, y) in image pixels, see :meth:`predict`.
        labels : np.ndarray, optional
            The (N,) point labels, see :meth:`predict`.
        boxes : np.ndarray, optional
            One (4,) box or (K, 4) boxes in image pixels, see :meth:`predict`.
        multimask : bool, optional
            Override of the constructor's ``multimask`` for this call.
        mask_thres : float, optional
            Override of the constructor's ``mask_thres`` for this call.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[PromptMask]
            One (mask, predicted IoU) per box, or a single one without boxes.

        """
        return self.end2end(
            image,
            points,
            labels,
            boxes,
            multimask=multimask,
            mask_thres=mask_thres,
            verbose=verbose,
        )
