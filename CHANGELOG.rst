Unreleased
----------

Added
^^^^^
* ``Detector(class_map=...)`` (and every detector wrapper): optional ``Mapping[int, int]`` that
  translates model class ids to output ids after postprocessing; unmapped ids pass through. Off
  by default
* ``image.COCO91_TO_COCO80``: COCO category ids (1-90) to 80-class indices, e.g.
  ``RFDETR(engine, class_map=COCO91_TO_COCO80)`` reports horse as 17
* ``CPUPreprocessor(resize=fn)`` takes a function doing the whole preprocessing,
  ``(images, *, verbose) -> (tensor, ratios, padding)``, for models whose input shape changes per batch
* ``image.HandInteractionDetector`` for hand-object interaction models with a unified,
  fixed-K output contract shared across model families
* ``models.HOIDETR`` and ``models.Hands23`` aliases of ``HandInteractionDetector``
* ``image.types``: public ``TypeAlias``\ es for image model results (``Bbox``,
  ``Detection``, ``Classification``, ``HandInteraction``), used in all image model signatures
* ``HandInteractionDetector.postprocess`` / ``end2end`` take ``conf_thres`` and a per-call
  ``nms_iou_thres`` positionally like ``Detector``; the pair thresholds are keyword-only
* ``HandInteractionDetector`` takes an ``output_schema`` override (``HandInteractionOutputSchema``
  member or name) and exposes an ``output_schema`` property; outputs are matched by name, so
  TensorRT 8.6 engines that reorder outputs work
* ``Classifier`` and ``DepthEstimator`` take an ``output_schema`` override
  (``ClassifierOutputSchema`` / ``DepthEstimatorOutputSchema`` member or name, positionally
  before ``dla_core``) and expose an ``output_schema`` property; any other engine still loads
  as ``GENERIC``
* Download support for ``hoi_detr_vitl`` and ``hands23_x101``
* ``models.DepthAnythingV1`` and ``models.DepthAnythingV3`` aliases for the
  ``DepthEstimator`` pipeline, mirroring the existing ``DepthAnythingV2``
* Download configs and ONNX export functions for Depth-Anything-V1
  (small / base / large) and Depth-Anything-V3 (mono-large, metric-large)
* ``Buffer`` / ``MemoryLocation`` (also ``core.Buffer``): one contiguous, typed allocation
  on the host (pageable, pinned, or pinned + mapped) or the device. ``Buffer.wrap`` views
  numpy arrays and any ``__cuda_array_interface__`` object (CuPy, PyTorch, ...) without
  copying; memory is reference counted so views and numpy arrays never outlive it
* ``TRTEngine.active_input_shapes`` / ``active_output_shapes``
* ``build_engine(shapes=[(name, (min_shape, opt_shape, max_shape))])`` builds a dynamic
  optimization profile; malformed triples raise ``ValueError`` and non-integer dims
  raise ``TypeError``
* ``core.KernelArgs``: ``Kernel.create_args`` returns an argument array that owns the
  buffers its pointers reference; build one directly with ``KernelArgs.from_values(*args)``
  or ``KernelArgs(pointers, keepalive)``
* ``QueuedTRTEngine.retrieve_result`` returns the next output, or the error the engine raised
  for that input, without raising it
* ``ImageModel.prepare_inputs(images, method, *, preprocessed, ratios, padding)`` gets the
  engine inputs for images, preprocessing them unless ``preprocessed`` (GPU-preprocessed
  batches are bound in place on the device), ``ImageModel.engine_inputs(image,
  images, ratios)`` builds them from a preprocessed batch (``images=None`` when the batch
  came preprocessed), ``ImageModel.infer`` runs the engine on them and returns the raw
  outputs (``ImageModel.infer_profile`` times it), and ``ImageModel.get_preprocessor(method)``
  returns the preprocessor for a method, creating it on first use
* ``download``/``download_model`` gain keyword-only ``batch`` and ``dynamic`` (CLI
  ``--batch`` / ``--dynamic``) for exporting at a fixed batch size or with a symbolic
  batch dimension (ultralytics and YOLOv10/12/13 also leave height and width symbolic);
  honored by the ultralytics, YOLOv10/12/13, and torchvision exporters and rejected
  with ``NotImplementedError`` elsewhere. ``dynamic`` cannot be combined with a batch
  above 1 or ``make_static``. Torchvision classifiers still export batch-dynamic by
  default; passing ``batch`` (or ``dynamic=False``) pins a static batch
* Benchmark harness: ``optimize`` grid gains an input-composition axis (single /
  homogeneous batch / heterogeneous batch / alternating-resolution singles) with a
  static engine built per batch size; ``optimize``/``batch`` gain ``--stage`` to record
  results under ``benchmark/data/perf-series/<device>/`` (tagged with the trtutils git
  SHA) for comparing successive library revisions, plotted by
  ``benchmark/plotting/patch_series.py`` (``run.py plot --patch-series``)
* Image models and preprocessors take ``Buffer`` images, alone or mixed with arrays in a
  batch. The CUDA and TRT preprocessors read device and mapped Buffers in place, skipping
  the host-to-device copy (1080p ``Detector.end2end`` on an RTX 5080: 1.22 to 0.88 ms);
  ``Buffer.wrap`` brings in CuPy / PyTorch tensors. An already preprocessed Buffer is bound as is
* ``image.TextDetector`` and ``image.TextRecognizer``, ``ImageModel`` subclasses for
  PaddleOCR-style DB text detection and CTC text recognition on dynamic-shape engines (any
  image size / text line width), with the same ``preprocess`` / ``run`` / ``postprocess`` /
  ``end2end`` API as the other image models plus ``get_text_boxes`` / ``get_texts``, and
  PaddleOCR's default settings. ``image.OCR`` runs both as a two-stage pipeline through
  ``end2end``; ``interfaces.TextDetectorInterface`` and ``interfaces.TextRecognizerInterface``.
  ``TextDetector`` and ``OCR`` take a positional ``conf_thres`` (box score / recognition
  confidence) per call, both models take an ``output_schema`` override
  (``TextDetectorOutputSchema`` / ``TextRecognizerOutputSchema``: ``PADDLE`` or ``GENERIC``), and
  results use the ``image.types`` aliases ``TextBox``, ``Text`` and ``TextLine``
* ``models.PPOCRDetector`` and ``models.PPOCRRecognizer`` with dynamic-shape ``build()``
  and the bundled PP-OCR character dictionary, and ``models.PPOCR`` to run both as one pipeline
* OCR pre/postprocessing functions: ``preprocessors.preprocess_text_detection``,
  ``preprocessors.preprocess_text_recognition``, ``preprocessors.resize_text_detection``,
  ``preprocessors.crop_text_region``, ``postprocessors.postprocess_text_detection``,
  ``postprocessors.postprocess_text_recognition``, ``postprocessors.get_text_boxes``,
  ``postprocessors.get_texts`` and ``postprocessors.sort_text_boxes``
* Download support for ``ppocrv4_det`` and ``ppocrv4_rec``
* ``image.Segmenter``, ``image.PoseEstimator``, and ``image.OBBDetector`` for instance
  segmentation, pose estimation, and oriented-bounding-box models (YOLO-style heads), each an
  ``ImageModel`` with its own postprocessor and ``get_segmentations`` / ``get_poses`` / ``get_obbs``
* ``models`` aliases ``YOLOv8Seg``/``YOLOv11Seg``/``YOLOv26Seg``,
  ``YOLOv8Pose``/``YOLOv11Pose``/``YOLOv26Pose``, and
  ``YOLOv8OBB``/``YOLOv11OBB``/``YOLOv26OBB``
* ``models`` classification aliases ``YOLOv8Cls``/``YOLOv11Cls``/``YOLOv26Cls`` and a
  ``softmax`` flag on ``Classifier`` (and ``postprocess_classifications``) so heads that
  already softmax inside the ONNX graph are not double-softmaxed
* Download support for the ``-seg``, ``-cls``, ``-pose``, and ``-obb`` variants of
  YOLOv8, YOLOv11, and YOLOv26 (YOLOv12 does not publish these task heads upstream); OBB
  exports default to 1024 and classification to 224, and YOLOv26 task heads export raw
  (``end2end=False``) for the trtutils decoders
* ``max_det`` (default 300) on the segmentation, pose, and OBB postprocessors and task models;
  OBB NMS uses ultralytics' probiou fast-NMS so results match ultralytics on the same export
* ``image.interfaces.SegmenterInterface``, ``PoseEstimatorInterface``, and
  ``OBBDetectorInterface``, implemented by the task models. ``postprocess`` / ``run`` / ``__call__``
  / ``end2end`` take ``(conf_thres, nms_iou_thres, *, mask_thres, ...)`` per call, with
  ``mask_thres`` (default 0.5, also a ``Segmenter`` constructor argument) only on the segmenter
* ``SegmenterOutputSchema``, ``PoseEstimatorOutputSchema``, and ``OBBDetectorOutputSchema``
  (``YOLO``): ``Segmenter``, ``PoseEstimator``, and ``OBBDetector`` are ``ImageModel`` peers of
  ``Detector`` that take ``output_schema`` (enum or name) and expose ``output_schema``; the schema
  is matched against the engine's output names in any order, and a wrong engine raises
  ``ValueError``
* ``image.types`` aliases ``Segmentation``, ``Pose``, ``OBBDetection`` and ``RBox``

Changed
^^^^^^^
* ``Detector`` resolves its input and output schemas with the same name-matching as the other
  task models: an engine with unknown input or output names now raises a ``ValueError`` listing
  the valid schemas instead of guessing from the tensor count. Auto-detection reports ``DETR``
  for engines named ``labels``/``boxes``/``scores`` (outputs are reordered by name, results are
  unchanged; pass ``DETR_LBS`` to keep the old value) and ``EFFICIENT_NMS_2`` for
  ``num``/``boxes``/``scores``/``classes`` engines
* ``image.preprocessors.CPUPreprocessor`` packs each image straight into the batch tensor
  with one float32 multiply-add: 8 x 1080p drops from 83 to 24-34 ms on a Jetson AGX Orin
  and from 21 to 7-8 ms on an RTX 5080 (imagenet mean/std: 190 to 24-34 ms and 57 to 7-8 ms).
  ``no_copy=True`` now returns a view of a reused buffer, and ``ImageModel`` uses it for its
  own engine input. The affine runs in float32, so outputs can differ from before by a few
  float32 ULP
* ``image.preprocessors.TRTPreprocessor`` builds its preprocessing engine with a
  ``1..batch_size`` profile and runs it at the number of images submitted instead of the
  configured batch: one 1080p image through ``batch_size=32`` drops from 6.0 to 0.96 ms on a
  Jetson AGX Orin (clocks locked) and from 1.08 to 0.63 ms on an RTX 5080. Cached engines are
  named ``..._b1-{batch_size}`` so older fixed-batch engines are rebuilt
* CUDA/TRT preprocessors upload same-sized batches per image instead of packing the whole
  batch on the host first: pageable memory copies straight from the caller's arrays and
  pinned memory overlaps each image's host copy with the previous image's DMA. 8 x 1080p
  is ~1.5-1.7x faster on an RTX 5080 and ~2.2x faster on a Jetson AGX Orin
* **Breaking:** ``TRTEngine`` inputs are ``Buffer`` objects only. ``execute``, ``__call__``,
  ``mock_execute``, ``raw_exec``, and ``QueuedTRTEngine`` / ``ParallelTRTEngines.submit``
  take a list of Buffers; wrap numpy data with ``Buffer.wrap(array)``.
  ``get_random_input`` returns host Buffers
* **Breaking:** ``TRTEngine.direct_exec`` is removed; ``execute`` reads device Buffers in
  place. ``raw_exec`` takes Buffers and returns device ``Buffer`` views of the outputs
  instead of pointers; its ``set_pointers`` / ``no_warn`` arguments are removed
* **Breaking:** the engine runs at the shapes of the submitted Buffers and validates
  every input (dtype, rank, static dims, profile bounds of dynamic dims), raising
  ``ValueError`` instead of copying mismatched bytes. Dynamic engines return outputs of
  the executed shape
* **Breaking:** ``Binding`` is a class over a host and a device ``Buffer`` (``host`` /
  ``device``). Its fields (``index``, ``name``, ``dtype``, ``shape``, ``is_input``,
  ``tensor_format``, ``host``, ``device``) are read-only properties, and ``allocation``,
  ``host_allocation``, ``pagelocked_mem``, and ``unified_mem`` are derived from the buffers.
  ``min_shape`` / ``max_shape`` give the profile bounds of an input, ``check(buffer, engine)``
  validates a Buffer against them and the dtype, ``bind(src, stream)`` returns device
  memory TensorRT can read ``src`` from, and ``stage`` / ``fetch`` copy a leading prefix of
  the binding (their views are reused while the shape repeats)
* **Breaking:** ``GPUImagePreprocessor.direct_preproc`` returns a device ``Buffer`` shaped
  to the submitted images instead of a pointer. GPU preprocessors no longer hold
  orig-size / scale-factor buffers: ``orig_size_allocation``, ``scale_factor_allocation``
  and the ``orig_size_dtype`` argument are removed, and ``Detector`` builds those inputs
  per image for every preprocessor
* **Breaking:** ``Kernel`` no longer takes ``max_arg_cache``
* ``TRTEngine`` keeps one CUDA graph per input configuration (the shape and device
  address of every input), so dynamic batch sizes / resolutions and in-place device
  inputs are graphed too. A configuration is captured the second time it runs at unchanged
  shapes and replayed afterwards (a graph cannot replay on a call that changes the shapes,
  so inputs that alternate shapes every call are never captured); the least recently used
  graph is evicted beyond 8. ``graph_exec``
  replays the configuration of the latest execution. A failed capture (including one
  whose enqueue TensorRT rejects) logs a warning and disables graphs for the engine
  instead of raising out of the call that triggered it
* ``TRTEngine`` holds a reference to the Buffers bound for the latest call, so
  ``graph_exec`` and an unsynchronized ``raw_exec`` never read freed inputs
* ``CUDAPreprocessor`` keeps its buffers for batches smaller than the largest seen, so the
  ``direct_preproc`` output address (and the engine's CUDA graph for it) stays stable
* ``ImageModel.end2end`` no longer captures its own CUDA graph or locks the image size and
  batch size after the first call; it uses the engine's per-configuration graphs
* ``ImageModel.run`` and ``ParallelDetector`` (now through ``ImageModel.infer``) bind
  GPU-preprocessed batches on the device instead of round-tripping through the host, and
  skip the copy of engine outputs they postprocess right away unless ``no_copy=False``
* ``ImageModel.preprocess(method='cuda')`` falls back to 'cpu' for models whose input is not
  float32, as the constructor does, and an unknown ``method`` raises ``ValueError`` instead
  of silently using the CPU preprocessor
* ``Detector`` models with a scale-factor input treat an already preprocessed batch passed
  without ``ratios`` as unscaled, as they already did for its original size
* GPU preprocessors raise ``ValueError`` for non-uint8 images instead of reading their bytes as uint8

Fixed
^^^^^
* Detection postprocessors clip boxes to the original image, ``[0, width]`` x ``[0, height]``, so
  e.g. D-FINE no longer returns ``x2 = 448`` on a 447 px wide image
* ``PoseEstimator`` and ``HandInteractionDetector`` clip boxes (and pose keypoints, as ultralytics
  does) to the original image too; ``postprocess_yolo_pose`` and ``postprocess_hand_interactions``
  take an optional keyword ``input_size`` (default ``None``, boxes clipped at 0 only)
* Engine files (``build_engine`` output and engine cache copies) are written atomically (temp file
  then replace), so concurrent readers never see a half-written engine
* ``RFDETR`` now defaults to ``resize_method="linear"`` (plain square resize, as upstream RF-DETR
  is trained); letterbox missed or shifted many detections
* RF-DETR class ids are now the model's own ids, matching upstream ``rfdetr`` (stock COCO weights:
  horse is 19); they were previously shifted by -1. Use ``class_map=COCO91_TO_COCO80`` for 80-class
  indices
* ``trtutils download`` of ``rfdetr_n`` / ``rfdetr_s`` / ``rfdetr_m`` failed with "Failed to simplify ONNX
  model": rfdetr's own simplify check rejects ~1e-4 float drift, so the exporter now skips it
* Hands23 export squared pixel distances, overflowing fp16 and turning ``pair_probs`` into NaN
  (every NaN pair was linked, contact always 1); HOI-DETR export baked batch 1 into the head's
  padding mask, so batch slots after the first returned garbage. Existing ``hands23_x101`` and
  ``hoi_detr_vitl`` ONNX files must be re-exported with ``trtutils download``
* ``DepthEstimator`` always returns depth shaped ``(batch, H, W)``; models that output a
  ``(B, 1, H, W)`` channel axis (``DepthAnythingV3``) no longer keep the extra axis
* ``register_jit`` functions now run Numba-compiled (the ``jit`` extra previously had no
  effect), and ``with JIT:`` restores the previous JIT state on exit
* ``CUDAGraph.invalidate`` no longer calls ``cudaGraphDestroy`` / ``cudaGraphExecDestroy`` on
  null or already-destroyed handles (invalid-argument errors under compute-sanitizer when an
  engine with ``cuda_graph=True`` is deleted)
* CUDA and TRT preprocessors: letterbox geometry now matches the CPU path (rounded resize
  size, ``round(pad - 0.1)`` offsets, float padding), which was off by a row or column on
  images whose fitted size is fractional (1920x1280 to 640: mean pixel error 9.7 to 0.08 of
  255), shifting OBB centers by up to 28 px; resize kernels round instead of truncating
* ``TRTEngine(backend="async_v2")`` (and ``async_v3``) raises ``ValueError`` at construction when
  the backend is missing from the installed TensorRT, instead of failing at the first execute
  (``execute_async_v2`` was removed in TensorRT 10)
* Host-to-device copies (``memcpy_host_to_device*``) gather non-contiguous arrays first; they
  copied the raw bytes, so the CUDA and TRT preprocessors read crops and other strided views
  as garbage for single images and mixed-size batches unless unified memory was on
* Device-to-host copies (``memcpy_device_to_host*``) raise ``ValueError`` on a non-contiguous
  destination instead of writing raw bytes over its parent array
* ``Detector.postprocess`` / ``get_detections`` / ``end2end`` treated an explicit
  ``conf_thres=0.0`` or ``nms_iou_thres=0.0`` as unset and fell back to the constructor value
* RT-DETR v1/v2, D-FINE, DEIM and DEIMv2 detectors failed on every call: ``Detector`` sent
  ``orig_target_sizes`` as int32 (the engines bind int64) and as (height, width) instead of
  (width, height). The size input's dtype now comes from the engine binding
* ``RTDETRv1`` / ``RTDETRv2`` / ``RTDETRv3`` defaulted to ``input_range=(0, 255)``; the exported
  models expect ``(0, 1)`` and found nothing on the default
* ``Detector`` maps engine outputs to its output schema by name. TensorRT 8.6 does not keep
  the ONNX output order, so D-FINE / DEIM / RT-DETR engines on JetPack 6 postprocessed the
  wrong arrays; schema auto-detection also ignores output order now
* ``RTDETRv1.build`` / ``RTDETRv2.build`` build at fp32 below TensorRT 10, where fp16 engines
  lose the detections (horse 0.38 vs 0.97 at fp32). ``Model`` subclasses can set ``_fp16``
* ``FLAGS.TRT_VERSION`` was ``(0, 0)`` on JetPack; it now reads the ``tensorrt`` module
* ``read_onnx`` raises on a file TensorRT 8.6 "parses" into an empty network
* ``build_engine(default_device="dla")`` without ``dla_core`` uses core 0; TensorRT selected no
  core, so the build failed
* CUDA/TRT preprocessors reallocated their staging buffers on every input shape change;
  with ``pagelocked_mem`` each free synchronized the device, making mixed-size batches
  ~3.9x and resolution switches ~3.7x slower than pageable memory. The buffers now only
  grow, and the preprocessors synchronize their stream before the CPU rewrites a pinned
  or mapped staging buffer that an earlier copy or kernel may still read (a race the
  per-shape reallocation had hidden for ``unified_mem`` mixed-size batches)
* ``TRTEngine.is_dynamic_batch`` read the max-profile binding shape and was always False;
  it is True when the profile allows more than one batch size, and
  ``ImageModel.is_dynamic_batch`` reads it
* ``TRTEngine``: shapes rejected by TensorRT's ``set_input_shape`` were silently ignored
* ``allocate_bindings``: inputs with dynamic non-batch dimensions are allocated at the
  max profile shape
* Pinned host memory is freed only when no Buffer view or numpy array references it,
  fixing use-after-free on ``no_copy`` outputs held past a reallocation
* ``Kernel.create_args``: cached argument arrays no longer point at freed buffers after
  other argument sets are created
* ``Detector`` with a CPU preprocessor and CUDA graphs wrote RT-DETRv3 extra inputs into
  the wrong engine bindings
* ``ParallelDetector`` with a GPU preprocessor passed only the image to multi-input
  (DETR-style) engines
* ``ParallelDetector`` with a CPU preprocessor fed RT-DETR models the model input size
  instead of each image's original size
* ``ParallelDetector``: an error while processing a submitted batch (a bad
  ``preprocess_method``, a wrongly shaped preprocessed tensor, a rejected enqueue, ...)
  killed the model's worker thread and left ``retrieve`` waiting forever. It is now raised
  by ``retrieve`` / ``retrieve_model`` in the caller's thread, after every model's result is
  taken, and the worker keeps serving later batches. ``QueuedTRTEngine`` /
  ``ParallelTRTEngines`` had the same flaw and now do the same: ``retrieve`` raises the
  engine's error, and ``ParallelTRTEngines.retrieve`` takes every engine's result first
* ``ParallelTRTEngines``: ``(engine, dla_core, device)`` tuples raised ``ValueError`` and
  the device was never applied
* ``ImageModel.mock_run`` with images builds every engine input (DETR-style models failed
  the input count), and ``get_random_input`` returns the image input rather than the first
* ``ImageModel`` kept using the old default preprocessor after ``update_input_range`` /
  ``update_mean_std``
* ``TRTEngine``: an enqueue rejected by TensorRT raises instead of returning the previous
  outputs
* GPU preprocessors copy back only the submitted images, not the full allocated batch
* ``download``: ``requirements_export`` no longer writes uv's "Using Python ... environment at"
  notice into the exported requirements file


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
