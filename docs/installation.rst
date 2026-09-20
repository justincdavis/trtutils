.. _installation:

Installation
============

This guide will help you install trtutils and its dependencies. The recommended method is
to install trtutils into a virtual environment to ensure dependency isolation.

System Requirements
-------------------

- Python 3.8 or later
- CUDA toolkit
- TensorRT
- NVIDIA GPU or Jetson device

Basic Installation
------------------

trtutils does not pull in CUDA or TensorRT by itself. Install the extra which
matches the CUDA version present on the system:

.. code-block:: console

    $ pip install "trtutils[cu13]"  # CUDA 13
    $ pip install "trtutils[cu12]"  # CUDA 12
    $ pip install "trtutils[cu11]"  # CUDA 11

Each of these installs a matching ``cuda-python`` and ``tensorrt`` package.

On Jetson devices TensorRT is provided by Jetpack, so use the Jetpack extra
instead, which installs only ``cuda-python``:

.. code-block:: console

    $ pip install "trtutils[jp7]"  # Jetpack 7 / CUDA 13
    $ pip install "trtutils[jp6]"  # Jetpack 6 / CUDA 12
    $ pip install "trtutils[jp5]"  # Jetpack 5 / CUDA 11

If CUDA and TensorRT are already installed by other means, trtutils can be
installed on its own:

.. code-block:: console

    $ pip install trtutils

For development or to get the latest features, install from source:

.. code-block:: console

    $ git clone https://github.com/justincdavis/trtutils.git
    $ cd trtutils
    $ pip install -e ".[cu12]"

Optional Dependencies
---------------------

trtutils provides several optional dependency groups that can be installed
using pip's extras feature:

ONNX Support
~~~~~~~~~~~~

Install the ONNX utilities used by the engine builder:

.. code-block:: console

    $ pip install "trtutils[onnx]"

This installs:
- ONNX
- ONNX GraphSurgeon

Quantization Tools
~~~~~~~~~~~~~~~~~~

Install the dependencies for the quantization CLI and API:

.. code-block:: console

    $ pip install "trtutils[quantize]"

This installs:
- The ``onnx`` extra
- NVIDIA ModelOpt

SAHI Support
~~~~~~~~~~~~

Install the dependencies for sliced inference via :mod:`trtutils.compat.sahi`:

.. code-block:: console

    $ pip install "trtutils[sahi]"

This installs:
- PyTorch and Torchvision
- Ultralytics
- SAHI

JIT Compiler Support
~~~~~~~~~~~~~~~~~~~~

Install support for the JIT compiler:

.. code-block:: console

    $ pip install "trtutils[jit]"

This installs:
- Numba
- LLVM-Lite

This enables the use of :func:`trtutils.enable_jit` to accelerate some CPU operations.

Development Tools
~~~~~~~~~~~~~~~~~

For development or contributing to trtutils:

.. code-block:: console

    $ pip install "trtutils[dev]"

This installs:
- Testing frameworks
- Linting tools
- Documentation generators
- Development utilities

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **CUDA/TensorRT Not Found**
   - Ensure CUDA and TensorRT are properly installed
   - Check environment variables (LD_LIBRARY_PATH, etc.)
   - Verify CUDA version compatibility

2. **Dependency Conflicts**
   - Use a virtual environment
   - Check package versions
   - Update pip: ``pip install --upgrade pip``

3. **Jetson-Specific Issues**
   - Install Jetson-specific TensorRT version
   - Use compatible CUDA version
   - Check Jetpack installation

4. **libnvrtc.so.* Not Found**
   - Ensure the version of cuda-python installed matches the version of CUDA installed
   - If using a custom CUDA path, ensure it is correctly set in the environment variables

Getting Started
---------------

After installation, verify your setup:

.. code-block:: python

    from trtutils import TRTEngine

    # Create a test engine
    engine = TRTEngine("test.engine")
    print("Installation successful!")

For more detailed examples, see the :ref:`Examples <examples>` section.
