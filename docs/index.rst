Welcome to trtutils
===================

**trtutils** is a high-level Python interface for TensorRT inference, providing a simple and unified way to run arbitrary TensorRT engines. This library abstracts away the complexity of CUDA memory management, binding management, and engine execution.

Features
--------

- Simple, high-level interface for TensorRT inference
- Automatic CUDA memory management
- Support for arbitrary TensorRT engines
- Built-in preprocessing and postprocessing capabilities
- Comprehensive type hints and documentation
- Support for both basic engine execution and end-to-end model inference
- Specialized support for YOLO models
- Performance benchmarking and monitoring

Quick Start
-----------

.. code-block:: python

    from trtutils import TRTEngine

    # Load your TensorRT engine
    engine = TRTEngine("path_to_engine")

    # Get input specifications
    print(engine.input_shapes)  # Expected input shapes
    print(engine.input_dtypes)  # Expected input data types

    # Run inference
    inputs = read_your_data()
    outputs = engine.execute(inputs)

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation <installation>
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
