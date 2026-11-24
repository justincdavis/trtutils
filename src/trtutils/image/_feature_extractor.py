# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel
from ._schema import FeatureExtractorOutputSchema
from .interfaces import FeatureExtractorInterface
from .postprocessors import get_embeddings, get_feature_maps, postprocess_features

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self

    from trtutils.core._buffer import Buffer
    from trtutils.image.types import Embedding, FeatureMap

    from ._image_model import _PostFn


class FeatureExtractor(ImageModel, FeatureExtractorInterface):
    """
    Implementation of feature extractors (image backbones).

    A feature extractor turns each image into an embedding, a (C,) vector, and,
    when the engine has one, a dense (C, h, w) feature map on its patch grid.
    """

    _OUTPUT_SCHEMA = FeatureExtractorOutputSchema

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cuda",
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        output_schema: FeatureExtractorOutputSchema | str | None = None,
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
        normalize: bool = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a FeatureExtractor object.

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
            The options are ['letterbox', 'linear', 'center_crop'], default is 'linear'.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, (0.485, 0.456, 0.406) (ImageNet mean).
            Pass None to disable normalization.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, (0.229, 0.224, 0.225) (ImageNet std).
            Pass None to disable normalization.
        output_schema : FeatureExtractorOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be a FeatureExtractorOutputSchema enum value or a string matching
            the enum name (e.g., "EMBEDDING_MAP", "GENERIC").
            By default None, which means the schema will be auto-detected from
            the engine's output names, falling back to GENERIC for any other engine.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this feature extractor. Default is None,
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
        normalize : bool, optional
            Whether to L2 normalize the embeddings during postprocessing, so a dot
            product is a cosine similarity. The feature maps are never normalized.
            Default is False. :meth:`postprocess` and :meth:`end2end` can override it.
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

        self._normalize = normalize

        # prepend with 'feat_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "feat_postprocess": f"feature_extractor::postprocess [{self._tag}]",
                "feat_get_embeddings": f"feature_extractor::get_embeddings [{self._tag}]",
                "feat_get_feature_maps": f"feature_extractor::get_feature_maps [{self._tag}]",
            }
        )

    @property
    def output_schema(self: Self) -> FeatureExtractorOutputSchema:
        """Get the output schema used by this feature extractor."""
        return self._output_schema

    def postprocess(
        self: Self,
        outputs: list[np.ndarray] | list[Buffer],
        *,
        normalize: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs from the engine to postprocess.
        normalize : bool, optional
            Whether to L2 normalize the embeddings.
            By default None, which uses the constructor's ``normalize``.
        no_copy : bool, optional
            If True, do not copy the feature maps from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
            The embeddings are always copied.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image: the (C,) embedding, then the
            (C, h, w) feature map when the engine has one.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["feat_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        normalize = normalize if normalize is not None else self._normalize
        t0 = time.perf_counter()
        data = postprocess_features(
            self._host_outputs(outputs),
            normalize=normalize,
            no_copy=no_copy,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(self: Self, *, normalize: bool | None = None, verbose: bool | None) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, _r, _p, nc: self.postprocess(
            o, normalize=normalize, no_copy=nc, verbose=verbose
        )

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

    # get_embeddings overloads
    @overload
    def get_embeddings(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> Embedding: ...

    @overload
    def get_embeddings(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[Embedding]: ...

    def get_embeddings(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> Embedding | list[Embedding]:
        """
        Get the embeddings from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        Embedding | list[Embedding]
            For single image: the (C,) float32 embedding.
            For batch: list[Embedding] (embedding per image).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["feat_get_embeddings"])

        if verbose:
            LOG.debug(f"{self._tag}: get_embeddings")

        batch, single = self._as_batch(outputs)
        result = get_embeddings(batch, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_embeddings

        return result[0] if single else result

    # get_feature_maps overloads
    @overload
    def get_feature_maps(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> FeatureMap: ...

    @overload
    def get_feature_maps(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[FeatureMap]: ...

    def get_feature_maps(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> FeatureMap | list[FeatureMap]:
        """
        Get the dense feature maps from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        FeatureMap | list[FeatureMap]
            For single image: the (C, h, w) float32 feature map.
            For batch: list[FeatureMap] (feature map per image).

        Raises
        ------
        ValueError
            If the engine has no feature map output, i.e. its output schema is
            not EMBEDDING_MAP.

        """
        expected = FeatureExtractorOutputSchema.EMBEDDING_MAP
        if self._output_schema is not expected:
            err_msg = (
                f"{self._tag}: output schema {self._output_schema.name} has no feature map, "
                f"get_feature_maps needs {expected.name} outputs {expected.value}"
            )
            raise ValueError(err_msg)

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["feat_get_feature_maps"])

        if verbose:
            LOG.debug(f"{self._tag}: get_feature_maps")

        batch, single = self._as_batch(outputs)
        result = get_feature_maps(batch, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_feature_maps

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        normalize: bool | None = ...,
        verbose: bool | None = ...,
    ) -> Embedding: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        normalize: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[Embedding]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        normalize: bool | None = None,
        verbose: bool | None = None,
    ) -> Embedding | list[Embedding]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_embeddings in that order. Makes some memory transfer
        optimizations under the hood to improve performance.
        Use :meth:`run` and :meth:`get_feature_maps` for the dense maps.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        normalize : bool, optional
            Whether to L2 normalize the embeddings.
            By default None, which uses the constructor's ``normalize``.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        Embedding | list[Embedding]
            For single image: the (C,) float32 embedding.
            For batch: list[Embedding] (embedding per image).

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(normalize=normalize, verbose=verbose),
            get=lambda pp: get_embeddings(pp, verbose=verbose),
        )
