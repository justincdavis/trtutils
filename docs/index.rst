Welcome to trtutils
===================

**trtutils** is a high-level Python interface for TensorRT inference, providing a simple and unified way to run arbitrary TensorRT engines. This library abstracts away the complexity of CUDA memory management, binding management, and engine execution.

Features
--------

- ``TRTEngine`` runs any TensorRT engine, with async_v2 and async_v3 backends, CUDA graphs, and pagelocked or unified memory
- ``Buffer`` inputs on the host or the device; ``Buffer.wrap`` views NumPy, CuPy, and PyTorch arrays without copying
- Image models in ``trtutils.image``: ``Detector``, ``Classifier``, ``Segmenter``, ``PoseEstimator``, ``OBBDetector``, ``DepthEstimator``, ``HandInteractionDetector``, and ``OCR`` (text detection plus recognition)
- SAHI (sliced inference) for any detector, with ``SAHI``
- Preprocessing on the CPU, CUDA, or TensorRT (``preprocessor="cpu" | "cuda" | "trt" | "hybrid"``), and CPU or CUDA postprocessing (``postprocessor="cpu" | "cuda"``)
- Pre-configured ``trtutils.models`` wrappers with ``download()`` and ``build()``
- ``Model.build`` reads the image size from the ONNX; ``embed_sst=True`` moves BGR-to-RGB, normalization, and layout into the engine, which then takes uint8 images
- Model download and ONNX export with ``trtutils download`` or ``Model.download()``
- Engine building from ONNX, including FP16, FP8, INT8, strongly typed, DLA, and dynamic-shape builds
- Benchmarking, profiling, and NVTX instrumentation (with Jetson energy/power metrics)
- Comprehensive type hints and documentation

Quick Start
-----------

.. code-block:: python

    import numpy as np

    from trtutils import Buffer, TRTEngine

    # Load your TensorRT engine
    engine = TRTEngine("yolov10n_640.engine")

    # Get input specifications
    print(engine.input_shapes)  # Expected input shapes
    print(engine.input_dtypes)  # Expected input data types

    # Run inference: every input is a Buffer, on the host or the device
    image = np.zeros(engine.input_shapes[0], dtype=engine.input_dtypes[0])
    outputs = engine.execute([Buffer.wrap(image)])  # NumPy arrays are wrapped without a copy

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation <installation>
   Supported Models <models>
   Tutorials <tutorials/overview>
   Usage Guide <usage/index>
   API Reference <api>
   CLI Reference <cli>
   Examples <examples>
   Benchmarking <benchmark/index>
   Changelog <changelog>

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Getting Help
------------

- Check out the :ref:`Tutorial Overview <tutorial_overview>` for step-by-step guides
- Browse the :ref:`Examples <examples>` for code samples
- View the :ref:`Usage Guide <usage>` for component overviews
- Explore the :ref:`API Reference <api>` for detailed documentation
- Report issues on `GitHub <https://github.com/justincdavis/trtutils/issues>`_
