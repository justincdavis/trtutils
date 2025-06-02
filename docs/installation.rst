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

The simplest way to install trtutils is using pip:

.. code-block:: console

    $ pip install trtutils

For development or to get the latest features, install from source:

.. code-block:: console

    $ git clone https://github.com/justincdavis/trtutils.git
    $ cd trtutils
    $ pip install -e .

Optional Dependencies
---------------------

trtutils provides several optional dependency groups that can be installed
using pip's extras feature:

YOLO Support
~~~~~~~~~~~~

Install support for YOLO object detection models:

.. code-block:: console

    $ pip install "trtutils[yolo]"

This includes dependencies for:
- YOLO-specific preprocessing
- Object detection utilities

Jetson Support
~~~~~~~~~~~~~~

For NVIDIA Jetson devices, install additional utilities:

.. code-block:: console

    $ pip install "trtutils[jetson]"

This enables:
- Power consumption monitoring
- Energy usage tracking

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
