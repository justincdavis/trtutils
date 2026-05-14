.. _strongly_typed:

Strongly-Typed Networks and Blackwell INT8/FP8
==============================================

TensorRT 10.x supports two ways to drive engine precision:

* **Weakly-typed** (the default, and the only form before TRT 10): the builder
  takes FP32 ONNX plus precision flags (``fp16``, ``int8``, ``fp8``) and
  decides per-layer precision at build time, optionally calibrating INT8
  layers at runtime using a calibration cache or data batcher.
* **Strongly-typed**: precision is entirely determined by the ONNX graph
  itself — FP16 casts, explicit input dtypes, and ``QuantizeLinear`` /
  ``DequantizeLinear`` (Q/DQ) nodes. The builder performs no calibration and
  ignores per-layer precision flags.

On **Blackwell (SM 10.0+, compute capability ≥ 100)**, mixed INT8+FP8
precision is only available under strongly-typed mode. Attempting it
weakly-typed raises:

.. code-block:: text

    Error Code 9: API Usage Error (INT8 and FP8 mixed precision is allowed
    only when building network with kSTRONGLY_TYPED mode on Blackwell+
    platforms.)

Platform / version support matrix
---------------------------------

+------------------------------+------------------+------------------+-----------------+
| Platform                     | INT8 only        | FP8 only         | INT8 + FP8      |
+==============================+==================+==================+=================+
| Ampere / Ada / Hopper        | weakly-typed OK  | weakly-typed OK  | weakly-typed OK |
+------------------------------+------------------+------------------+-----------------+
| Blackwell + TRT 10.x         | weakly-typed OK  | weakly-typed OK  | **strongly**    |
|                              |                  |                  | **-typed only** |
+------------------------------+------------------+------------------+-----------------+
| Blackwell + future TRT (≥11) | likely strongly  | likely strongly  | strongly-typed  |
|                              | -typed only      | -typed only      | only            |
+------------------------------+------------------+------------------+-----------------+

``trtutils`` detects both dimensions at import time:

* :py:data:`~trtutils.FLAGS.IS_BLACKWELL` — True on SM 10.0+ GPUs.
* :py:data:`~trtutils.FLAGS.STRONGLY_TYPED_SUPPORTED` — True if the installed
  TensorRT exposes ``trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED``.

When to use ``strongly_typed=True``
-----------------------------------

Pass ``strongly_typed=True`` to :py:func:`trtutils.builder.build_engine` when
you are building from a **pre-quantized ONNX** that already contains Q/DQ
nodes (or explicit FP16 / FP8 dtypes). This is required on Blackwell for
mixed INT8+FP8 and recommended any time the precision decisions have been
made ahead of time.

.. code-block:: python

    from trtutils.builder import build_engine

    build_engine(
        "model_qdq.onnx",      # pre-quantized with Q/DQ nodes
        "model.engine",
        strongly_typed=True,    # precision comes from the ONNX graph
    )

The following arguments are **mutually exclusive** with ``strongly_typed=True``
and raise :py:class:`ValueError` if combined:

* ``fp16``, ``int8``, ``fp8`` — precision is carried by the graph.
* ``calibration_cache``, ``data_batcher`` — runtime calibration does not
  apply; scales must already be in the ONNX.
* ``layer_precision`` — per-layer overrides are ignored.

``input_tensor_formats`` and ``output_tensor_formats`` may still be used,
but the dtype portion is ignored — only the tensor format (``LINEAR``,
``CHW4``, ``HWC``, …) is applied. A warning is logged in this case.

Generating a Q/DQ ONNX
----------------------

``trtutils`` ships a thin wrapper around ``nvidia-modelopt`` that covers the
common PTQ path end-to-end. Both a Python API and a CLI are provided.

**Python API**

.. code-block:: python

    from trtutils.builder import quantize, build_engine, ImageBatcher

    # 1. Produce calibration data (.npy) from an image directory.
    batcher = ImageBatcher(
        image_dir="calibration_images/",
        shape=(224, 224, 3),   # HWC
        dtype="float32",
        batch_size=8,
        order="NCHW",
    )
    batcher.save_calibration_data("calib.npy")

    # 2. Quantize the ONNX, baking Q/DQ nodes into a new file.
    quantize.quantize_onnx(
        onnx_path="model.onnx",
        output_path="model_qdq.onnx",
        calibration_data="calib.npy",
        quantize_mode="int8",          # "int4" | "int8" | "fp8"
        calibration_method="max",      # "max" | "entropy" | "percentile" | "mse"
    )

    # 3. Build a strongly-typed engine from the Q/DQ ONNX.
    build_engine(
        "model_qdq.onnx",
        "model.engine",
        strongly_typed=True,
    )

**CLI**

The same three steps are available through subcommands of ``python -m trtutils``:

.. code-block:: bash

    # 1. Generate calibration data from a directory of images
    python -m trtutils generate_calibration \
        --calibration_dir calibration_images/ \
        --input_shape 224 224 3 \
        --input_dtype float32 \
        --batch_size 8 \
        --data_order NCHW \
        --output calib.npy

    # 2. Quantize the ONNX
    python -m trtutils quantize \
        --onnx model.onnx \
        --output model_qdq.onnx \
        --calibration_data calib.npy \
        --quantize_mode int8 \
        --calibration_method max

    # 3. Build a strongly-typed engine
    python -m trtutils build \
        --onnx model_qdq.onnx \
        --output model.engine \
        --strongly_typed

Under the hood ``quantize_onnx`` calls
``modelopt.onnx.quantization.quantize``. For workflows the wrapper does not
cover — QAT, custom quantizer placement, non-image calibration — reach for
the underlying tools directly:

* `NVIDIA TensorRT Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_
  (``nvidia-modelopt``) — full PTQ/QAT toolkit.
* `ONNX Runtime quantization
  <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>`_
  — alternative INT8 PTQ.
* `pytorch_quantization
  <https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html>`_
  (legacy) — pre-modelopt pipeline.

For FP16-only models, exporting via PyTorch with ``model.half()`` before
``torch.onnx.export(...)`` is sufficient — no Q/DQ needed.

DLA + strongly-typed (Jetson Orin)
----------------------------------

Jetson Orin (SM 87, Ampere) has DLA cores but no FP8 hardware. On
**JetPack 6.1+** (TRT 10.1+) the ``STRONGLY_TYPED`` flag is available, which
makes modelopt-quantized INT8 ONNX the cleanest path for DLA deployment.
Pass ``strongly_typed=True`` to :py:func:`trtutils.builder.build_dla_engine`
or the ``build_dla`` CLI and the builder will:

* Skip the weakly-typed INT8 calibration step entirely — the Q/DQ nodes in
  the ONNX define the scales.
* Still auto-assign DLA-compatible layer chunks to DLA via ``layer_device``.
  Per-layer ``layer_precision`` overrides are **not** applied; precision
  comes from the graph, so your Q/DQ quantization must be placed on the
  DLA-bound subgraph for the INT8 hardware to be exercised.

**Python API**

.. code-block:: python

    from trtutils.builder import build_dla_engine

    build_dla_engine(
        "model_qdq.onnx",
        "model_dla.engine",
        dla_core=0,
        strongly_typed=True,
    )

**CLI**

.. code-block:: bash

    python -m trtutils build_dla \
        --onnx model_qdq.onnx \
        --output model_dla.engine \
        --dla_core 0 \
        --strongly_typed

Under strongly-typed, ``build_dla_engine`` rejects ``data_batcher``,
``calibration_cache``, and ``fp8`` — those are weakly-typed artifacts. Omit
them; precision lives in the graph.

.. note::

   Thor (Blackwell, JetPack 7) drops DLA hardware, so Orin is the last
   Jetson generation where this DLA + strongly-typed workflow applies. On
   Thor you'll use the GPU path (:py:func:`trtutils.builder.build_engine`
   with ``strongly_typed=True``) covered earlier in this page.

Common errors
-------------

``"INT8+FP8 mixed precision on Blackwell requires strongly_typed=True …"``
  You passed both ``int8=True`` and ``fp8=True`` on a Blackwell GPU without
  ``strongly_typed=True``. Re-export your ONNX with Q/DQ nodes using modelopt
  or similar, then pass ``strongly_typed=True`` and drop the ``int8``/``fp8``
  flags.

``"strongly_typed=True does not support runtime calibration …"``
  You passed ``strongly_typed=True`` together with ``calibration_cache`` or
  ``data_batcher``. Strongly-typed networks have no calibration step; scales
  must live in the ONNX Q/DQ nodes. Remove the calibration arguments.

``"strongly_typed=True derives precision from the ONNX graph; remove
fp16/int8/fp8 builder flags."``
  You passed ``strongly_typed=True`` together with precision flags. Drop the
  flags — precision is now the ONNX's job.

``"Installed TensorRT does not support strongly-typed networks"``
  The installed TensorRT version predates the ``STRONGLY_TYPED`` network
  creation flag (TRT < 10.1). Upgrade TensorRT, or build weakly-typed on a
  non-Blackwell GPU.
