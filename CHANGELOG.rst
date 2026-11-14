1.0.0 (Unreleased)
------------------

Added
^^^^^
* Image pipeline in ``trtutils.image``: ``Detector``, ``Classifier``, ``DepthEstimator``,
  ``HandInteractionDetector``, ``Segmenter``, ``PoseEstimator``, ``OBBDetector``,
  ``TextDetector``, ``TextRecognizer``, ``OCR`` (two-stage detect + recognize) and ``SAHI``
  (sliced inference). All are ``ImageModel`` subclasses with ``preprocess`` / ``run`` /
  ``postprocess`` / ``end2end``
* ``ImageModel`` helpers: ``prepare_inputs``, ``engine_inputs``, ``infer``, ``infer_profile``,
  ``get_preprocessor``, ``mock_run``, ``shape_contexts``
* ``output_schema`` override (enum or name) on the task models, with an ``output_schema``
  property. Outputs are matched to the schema by name, so engines whose outputs TensorRT 8.6
  reorders still work
* ``image.interfaces`` protocols (``SegmenterInterface``, ``PoseEstimatorInterface``,
  ``OBBDetectorInterface``, ``TextDetectorInterface``, ``TextRecognizerInterface``) and
  ``image.types`` result aliases (``Bbox``, ``Detection``, ``Classification``,
  ``HandInteraction``, ``RBox``, ``OBBDetection``, ``Segmentation``, ``Pose``, ``TextBox``,
  ``Text``, ``TextLine``)
* ``postprocessor="cpu" | "cuda"`` on the image models. Defaults: ``"cuda"`` for ``Segmenter``
  (when the engine outputs are float32), ``OBBDetector``, ``PoseEstimator``, ``TextDetector``,
  ``TextRecognizer``, ``OCR`` and the ``models.PPOCR*`` wrappers; ``"cpu"`` for ``Detector``,
  ``Classifier`` and ``HandInteractionDetector``. ``DepthEstimator`` is CPU only. ``Detector``
  CUDA postprocessing covers YOLO-v10 and EfficientNMS engines only
* ``preprocessor`` on the image models: ``"cuda"`` (default), ``"cpu"``, ``"trt"`` and
  ``"hybrid"`` (CPU resize, GPU normalize). ``trtutils detect`` / ``classify`` take the same
  option
* ``resize_method="center_crop"`` with ``crop_pct`` (default 0.875, the torchvision/timm
  ImageNet eval transform). ``CPUPreprocessor(resize=fn)`` takes a whole-batch preprocess
  function for models whose input shape changes per batch
* Image models accept ``Buffer`` inputs (device, mapped or host), alone or mixed with arrays.
  ``Buffer.wrap`` views CuPy and PyTorch tensors without copying
* ``Detector(class_map=...)`` and ``image.COCO91_TO_COCO80`` for mapping model class ids to
  80-class indices
* ``Classifier(softmax=...)``: skip the softmax when the ONNX head already applies one
* ``Model.build(..., embed_sst=True)`` and ``builder.onnx.embed_sst``: move BGR-to-RGB, normalize
  and HWC-to-CHW into the ONNX graph, so the engine takes uint8 ``(batch, imgsz, imgsz, 3)``.
  ``input_range`` / ``mean`` / ``std`` are fixed at build time
* ``trtutils.models``: pre-configured wrappers with ``download()`` / ``build()``. Families:
  detectors (YOLOX, YOLOv3-v13, YOLOv26, RT-DETR / RT-DETRv3, D-FINE, DEIM, RF-DETR),
  classifiers (ResNet, AlexNet, VGG, DenseNet, Inception, GoogLeNet, EfficientNet, MobileNet,
  RegNet, ShuffleNet, SqueezeNet, ConvNeXt, MaxViT, DeiT / DeiT3, YOLO-cls), depth estimators
  (Depth-Anything v1, v2, v3), segmenters, pose estimators and OBB detectors (YOLOv8, v11,
  v26), hand interaction (``Hands23``, ``HOIDETR``), and OCR (``PPOCRDetector``,
  ``PPOCRRecognizer``, ``PPOCR``)
* ``download`` / ``download_model`` and ``trtutils download``, with keyword-only ``batch`` and
  ``dynamic`` (CLI ``--batch`` / ``--dynamic``) for static-batch or symbolic-batch exports
* ``trtutils.Buffer`` / ``MemoryLocation`` (also ``core.Buffer``): one typed allocation on the
  host (pageable, pinned, or pinned and mapped) or the device, reference counted so views never
  outlive it
* ``core.KernelArgs``: ``Kernel.create_args`` returns an argument array that owns the buffers
  its pointers reference. Build one with ``KernelArgs.from_values(*args)``
* ``TRTEngine(shape_contexts=N)``: opt-in pool of up to ``N`` execution contexts, one per input
  shape tuple, sharing device memory. Creating a context takes ~65 ms on first use of each
  shape, so warm up every shape in use and size ``N`` above the number of shapes. Off by default
* ``TRTEngine`` CUDA graphs (``cuda_graph``): one graph per input configuration (shapes and
  device addresses), captured on the second run at unchanged shapes. The 8 most recent are kept
* ``TRTEngine.active_input_shapes`` / ``active_output_shapes``
* ``TRTEngine.execute`` / ``raw_exec`` / ``__call__`` / ``mock_execute`` take a mapping of input
  name to ``Buffer`` as well as a sequence in ``input_names`` order. Outputs come back in
  ``output_names`` order, which is not necessarily the ONNX order: TensorRT 8.6 can order
  same-shape outputs differently, so label them with ``zip(engine.output_names, outputs)``
* ``QueuedTRTEngine.retrieve_result``: returns the next output, or the error the engine raised
  for that input, without raising it
* ``parallel.image.ParallelDetector``: runs image detectors in parallel threads
* ``TextDetector`` / ``TextRecognizer`` / ``OCR``: DB text detection and CTC recognition on
  dynamic-shape engines, with ``get_text_boxes`` / ``get_texts``. ``OCR`` with
  ``postprocessor="cuda"`` overlaps the pages of a batch. ``TextRecognizer.end2end_regions``
  cuts text lines out of a device image. ``width_bucket`` (opt-in, default off) rounds recognizer
  widths up to a multiple of N pixels
* ``models.PPOCRDetector`` / ``PPOCRRecognizer`` / ``PPOCR`` (bundled PP-OCR dictionary) and
  OCR pre/postprocessing functions in ``preprocessors`` and ``postprocessors``
* ``models.DeiT`` / ``models.DeiT3`` and a timm exporter for DeiT and DeiT III
* ``models.DepthAnythingV1`` / ``V3`` aliases (``DepthAnythingV2`` also present)
* ``models.HOIDETR`` and ``models.Hands23`` aliases of ``HandInteractionDetector``
* ``trtutils.profile_engine``, and the CLI commands ``detect``, ``classify``, ``download``,
  ``profile``, ``build_yolo``, ``quantize``, ``generate_calibration`` and ``clear_cache``
* ``trtutils.Device``, ``get_device``, ``set_device`` (top-level exports)
* ``Binding.nbytes``, ``strides``, ``vectorized_dim`` and ``components_per_element`` describe an
  I/O tensor's layout in its TensorRT format. Bindings hold I/O in that format, unconverted; a
  layout larger than the shape is allocated flat, and outputs at the tensor shape hold its
  leading bytes

Changed
^^^^^^^
* **Breaking:** ``TRTEngine`` inputs are ``Buffer`` objects. ``execute``, ``__call__``,
  ``mock_execute``, ``raw_exec``, ``QueuedTRTEngine`` and ``ParallelTRTEngines.submit`` take a
  list of Buffers; wrap numpy arrays with ``Buffer.wrap(array)``. ``get_random_input`` returns
  host Buffers
* **Breaking:** ``TRTEngine.execute`` runs at the shapes of the submitted Buffers and validates
  every input (dtype, rank, static dims, profile bounds). Mismatches raise ``ValueError`` instead
  of copying wrong bytes, and rejected ``set_input_shape`` calls now raise. Dynamic engines return
  outputs at the executed shape
* **Breaking:** ``TRTEngine.raw_exec`` returns device ``Buffer`` views of the outputs instead of
  pointers. Its ``no_warn`` argument is removed
* **Breaking:** ``core.Binding`` wraps a host and a device ``Buffer`` (``host`` / ``device``). Its
  fields are read-only properties. ``min_shape`` / ``max_shape``, ``check``, ``bind``, ``stage``
  and ``fetch`` are new
* **Breaking:** ``Kernel`` no longer takes ``max_arg_cache``
* **Breaking:** ``QueuedTRTEngine`` and ``ParallelTRTEngines`` are no longer exported from the
  top-level ``trtutils``. Import them from ``trtutils.parallel``
* **Breaking:** ``core.cache`` functions are renamed: ``clear_cache`` is ``clear``, ``query_cache``
  is ``query`` and ``store_in_cache`` is ``store``
* ``build_engine(shapes=...)`` accepts ``(min, opt, max)`` triples per input to build a dynamic
  optimization profile. A plain shape still sets a fixed shape, as in 0.6.1
* ``create_stream(non_blocking=True)`` creates a stream that does not synchronize with the
  legacy default stream

Performance
^^^^^^^^^^^
* Buffers already on the device skip the host-to-device copy: 1080p ``Detector.end2end`` on an
  RTX 5080 goes from 1.22 to 0.88 ms
* CPU preprocessing packs a batch in one pass: 8 x 1080p takes 24-34 ms on a Jetson AGX Orin
  and 7-8 ms on an RTX 5080
* ``TRTPreprocessor`` runs at the submitted batch size: one 1080p image takes 0.63 ms on an
  RTX 5080
* ``hybrid`` preprocessing: yolo11n end2end 0.86 ms on 1080p (RTX 5080), vs 0.98 ms with
  ``"cuda"``
* PPOCR with the default CUDA postprocessors: 14-line 1080p page about 7.0 ms on an RTX 5080

Fixed
^^^^^
* ``build_engine`` writes engine files atomically (temp file, then replace), so readers never see
  a half-written engine
* ``build_engine(default_device="dla")`` without ``dla_core`` uses DLA core 0. TensorRT had
  selected no core, so the build failed
* ``read_onnx`` raises ``RuntimeError`` on a file TensorRT 8.6 "parses" into an empty network
* ``build_engine`` no longer enables ``DIRECT_IO`` when ``input_tensor_formats`` /
  ``output_tensor_formats`` are given. It forbade I/O reformats, so DLA builds whose rows are not
  64-byte aligned (e.g. fp16 LINEAR with W=48) failed with "no conformant implementation exists".
  Pass ``direct_io=True`` to keep the old behavior
* ``memcpy_host_to_device*`` gathers non-contiguous arrays before copying. They copied raw strided
  bytes, which corrupted crops and other views
* ``memcpy_device_to_host*`` raises ``ValueError`` on a non-contiguous destination instead of
  writing raw bytes over its parent array
* ``TRTEngine``: an enqueue rejected by TensorRT raises instead of returning the previous outputs
* ``TRTEngine`` / ``allocate_bindings`` size each I/O binding by the bytes its TensorRT format
  spans, not ``prod(shape) * itemsize``. Formats that pad the tensor (``CHW16``, ``CHW32``,
  ``HWC8`` and other vectorized formats with a partial vector, ``DLA_LINEAR`` / ``DLA_HWC4``
  with unaligned rows) made GPU engines write past their bindings into neighboring allocations
  and DLA engines fail to enqueue. An input Buffer smaller than its layout raises ``ValueError``,
  and the failed-enqueue error names that DLA cause
* ``TRTEngine``: inputs with dynamic non-batch dimensions are allocated at their max profile shape
* ``TRTEngine``, ``CUDAPreprocessor`` and ``TRTPreprocessor`` destroy the stream they created
  once. A second ``__del__`` (an explicit call, then garbage collection) destroyed the stale handle
  again, which segfaults on Jetson (CUDA 12.2) or frees another stream that reused the handle
* ``TRTEngine(backend="async_v2")`` / ``"async_v3"`` raises ``ValueError`` at construction when the
  installed TensorRT lacks the backend, instead of failing at the first execute
* Pinned host memory is freed only when no Buffer view or numpy array references it, fixing a
  use-after-free on arrays held past a reallocation
* ``Kernel.create_args``: cached argument arrays no longer point at freed buffers after other
  argument sets are created
* Negative Python int kernel arguments are packed as int64. They were packed as uint64
* ``trtexec`` build and run helpers split the command on spaces, breaking paths with spaces. The
  arguments are passed as a list now
* ``trtexec`` builds use ``--memPoolSize=workspace`` for the workspace size on TensorRT 8.4 and
  later. TensorRT 10 removed ``--workspace``
* ``trtutils build --device`` was parsed but not passed to ``build_engine``
* ``core.cache.clear`` (``clear_cache`` in 0.6.1) no longer raises ``FileNotFoundError`` when the
  cache directory does not exist
* INT8 calibration no longer hangs when an image fails to load. The error is raised from the
  batcher's ``get_next_batch``
* INT8 calibration no longer fails with zero worker threads on a 1-CPU machine
* INT8 calibration reuses one device buffer for its batches. It previously leaked one per batch
* INT8 calibration casts each batch to the dtype of its network input. TensorRT reads the batch
  as raw bytes of that dtype, so float32 batches for a float16 input (set by a hook or
  ``input_tensor_formats``) calibrated on garbage: infinite scales, layers moved off the DLA, or a
  crash during the build
* A ``QueuedTRTEngine`` that is not stopped is now garbage collected. Its worker thread used to
  hold a reference to the engine
* ``get_random_input`` no longer fails on float16 inputs
* ``QueuedTRTEngine`` / ``ParallelTRTEngines``: an error in a worker thread no longer leaves
  ``retrieve`` waiting forever. It is raised by ``retrieve`` in the caller's thread, and the worker
  keeps serving later inputs. ``ParallelTRTEngines.retrieve`` takes every engine's result first
* ``benchmark_engine`` / ``benchmark_engines``: ``warmup=None`` warms up as documented; 0.6.1
  skipped warmup
* ``register_jit`` functions run Numba-compiled (the ``jit`` extra previously had no effect), and
  ``with JIT:`` restores the previous JIT state on exit

Removed
^^^^^^^
* ``trtutils.impls`` package: ``impls.yolo.YOLO``, ``ParallelYOLO``, the YOLO variant classes and
  their preprocessors. Use ``image.Detector`` or the ``trtutils.models`` YOLO wrappers
* ``TRTModel``, ``QueuedTRTModel`` and ``ParallelTRTModels``. Use ``TRTEngine`` for raw engines or
  the ``trtutils.image`` models
* ``trtutils yolo`` CLI subcommand. Use ``trtutils detect`` / ``trtutils classify``
* ``TRTEngine.direct_exec``. ``execute`` reads device Buffers in place
* ``cv2ext`` dependency. The few helpers trtutils used now live in ``trtutils.image``
  (``draw_bboxes``, ``draw_text``) and the CPU preprocessing/postprocessing modules

0.6.1 (2025-06-17)
------------------

Added
^^^^^
* ``core.cuda_free`` and ``core.cuda_host_free`` for freeing CUDA device and host memory
* ``core.allocate_to_device`` and ``core.free_device_ptrs`` for managing CUDA device memory

Fixed
^^^^^
* ``TRTEngine``: Correctly reset input bindings when using ``direct_exec`` or ``raw_exec``

0.6.0 (2025-06-16)
------------------

Added
^^^^^
* TensorRT-based image preprocessing pipeline (``impls.yolo._preprocessors._trt``)
  * Automatically falls back to CUDA or CPU preprocessors when unsupported
* Engine cache utilities (``core.cache``) enabling on-disk reuse of timing caches and faster start-up
* Just-In-Time kernel generation via ``_jit`` module and ``--jit`` CLI flag
* Managed (Unified) CUDA memory option for ``TRTEngine``
* Device-specific DLA core selection through ``--dla-core`` flag
* Faster scale-swap-transpose (SST) CUDA kernel implementation
* Multi-stream and live-webcam demo applications residing in ``demos/``
* Alternative preprocessing methods measured in benchmarking
* Additional verbose / debug CLI flags and richer ``inspect`` command output

Changed
^^^^^^^
* Lazy loading of TensorRT plugins and YOLO preprocessors reduces import overhead
* Default host-side memory allocator uses pagelocked memory for all allocations; managed memory enabled by additional flag.

Fixed
^^^^^
* Resolved occasional overwrite of benchmark result files
* Miscellaneous stability fixes identified by the expanded CI test-suite


0.5.0 (2025-04-24)
------------------

Added
^^^^^
* Comprehensive engine builder subpackage (``builder``) and ``trtutils build`` CLI command
  * Layer-wise progress bar, timing cache support, and INT8 calibration via ``ImageBatcher`` / ``Calibrator``
  * Dedicated DLA build helpers (``builder._dla``) with per-layer precision and chunk-size specification
* Support for TensorRT ``execute_async_v3`` backend
* Unified logging facility exported through ``trtutils.get_logger``
* Extensive benchmarking scripts with automatic documentation generation
* Continuous Integration pipelines and greatly expanded automated test-suite (#40)

Improved
^^^^^^^^
* Enhanced YOLO CLI (image batching, new convenience flags)
* Build and benchmarking documentation now generated automatically in CI
* Utilization of timing caches when building engines with ``trtexec`` for faster rebuilds

Fixed
^^^^^
* Memory leak in ``CUDAPreprocessor`` due to incorrect explicit free call
* Numerous minor bugs uncovered by the new tests


0.4.1 (2025-03-04)
------------------

Added
^^^^^
* core.init_cuda
  * Use to start CUDA if not using a TRTEngine and only the core setup.
* benchmark_engines and jetson.benchmark_engines
  * Can benchmark TRTEngines in concurrently running mode
* Example yolo CLI program
  * Currently only support video file input and display.
* General CLI fixes for trtexec
* Experimental non-pagelocked memory addressing for TRTengines
  * Unstable, should be used with caution. Will be refined in the future
  * Does not provide performance improvement, simply for testing speedup of pagelocked memory utilization. As such, low-priority
* Basic internal profiling setup for YOLO objects.
  * No current public access, but accessible through: (_pre)(_infer)(_post)_profile attributes
  * Only stores last timestamp tuple
  * No end2end method support yet

Fixed
^^^^^
* yolo.CUDAPreprocessor using the wrong block size during resize call
* Various fixes and extensions to ParallelTRTEngines

0.4.0 (2024-12-05)
------------------

Added
^^^^^
* CUDA-based resize kernels

  * Perform linear or letterbox resizing

* core.create_kernel_args and core.Kernel

0.3.4 (2024-11-12)
------------------

Added
^^^^^
* CUDA-based preprocessing for YOLO:

  * Introduced ``CUDAPreprocessor`` and ``CPUPreprocessor``
  * Additional parameters in YOLO constructor and methods:

    * ``conf_thres``
    * ``extra_nms``, ``agnostic_nms``
    * ``resize_method``, ``preprocessing_unit``

* Runtime CUDA kernel generation with NVRTC:

  * Final transform (transpose from HWC to BCHW) reduced from 50ms to 5ms for 1280x1280, achieving a 10x speedup

Improved
^^^^^^^^
* Multi-threading safety:

  * ``ParallelYOLO`` enforces serial deserialization of engine files
  * ``CUDAProcessor`` now serializes initialization
  * Core CUDA/NVRTC calls use mutexes

0.3.3 (2024-10-31)
------------------

Added
^^^^^
* ``impls.yolo.YOLO``:

  * Added ``input_range`` parameter for specifying the input range
  * ``YOLOX`` uses ``[0:255]``, all others use ``[0:1]``

0.3.2 (2024-10-31)
------------------

Added
^^^^^
* Variations of ``impls.yolo.YOLO``: YOLOv7, YOLOv8, YOLOv9, YOLOv10, and YOLOX

Changed
^^^^^^^
* ``impls.yolo.YOLO``:

  * Version inference is now automatic
  * Postprocessing determined from outputs

0.3.1 (2024-10-29)
------------------

Improved
^^^^^^^^
* Outputs from ``impls.yolo.YOLO`` now use standard Python types:

  * Improved compatibility with JIT compilers like ``numba``

0.3.0 (2024-10-25)
------------------

Added
^^^^^
* ``impls.yolo.ParallelYOLO``: Enables running multiple YOLO models simultaneously

Improved
^^^^^^^^
* ``TRTEngine``:

  * Uses async memory copies and execution
  * Implements pagelocked memory on host

Removed
^^^^^^^
* ``backend`` submodule: Deprecated in favor of CUDA Python engines

0.2.3 (2024-10-17)
------------------

Added
^^^^^
* ``jetson.benchmark_engine`` integrated with ``jetsontools > 0.0.3``

Improved
^^^^^^^^
* ``TRTEngine``: Enhanced threading documentation

Fixed
^^^^^
* ``trtexec.build_engine``: Correctly builds for DLA core 0

0.2.2 (2024-10-17)
------------------

Changed
^^^^^^^
* ``TRTEngine``:

  * Uses ``execute_async_v2`` for inference
  * ``core.create_engine`` now creates a ``cudaStream``

0.2.1 (2024-10-16)
------------------

Added
^^^^^
* Locks for TensorRT engine creation and CUDA memory allocation

0.2.0 (2024-10-02)
------------------

Added
^^^^^
* ``benchmark_engine``: Measures engine latency
* Submodules:

  * ``jetson``
  * ``impls``
  * ``impls.yolo``: Supports YOLO variants (V7 to V10)

Changed
^^^^^^^
* ``trtexec.build_from_onnx`` renamed to ``trtexec.build_engine``

0.1.2 (2024-10-10)
------------------

Added
^^^^^
* Async and parallel execution classes:

  * ``QueuedTRTEngine``, ``QueuedTRTModel``
  * ``ParallelTRTEngine``, ``ParallelTRTModel``

0.1.1 (2024-07-30)
------------------

Fixed
^^^^^
* Resolved ``AttributeError`` during deallocation crashes

0.1.0 (2024-07-30)
------------------

Changed
^^^^^^^
* Default ``TRTEngine`` now uses CUDA Python:

  * Improved stability and compatibility
  * Legacy PyCUDA version available via ``trtutils.backends.PyCudaTRTEngine``

0.0.8 (2024-07-21)
------------------

Added
^^^^^
* ``trtexec`` submodule:

  * Locate and run ``trtexec`` commands programmatically

0.0.3 (2024-02-22)
------------------

Fixed
^^^^^
* Correct package detection as fully typed

Improved
^^^^^^^^
* Examples, documentation, and stricter linting/typing

Added
^^^^^
* PyCUDA install script for Linux
