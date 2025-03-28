.. _usage:

Usage Guide
===========

This section provides detailed documentation for using trtutils. For step-by-step
tutorials, see the :ref:`Tutorial Overview <tutorial_overview>` section.

Core Components
---------------

The main components of trtutils are:

- :py:class:`~trtutils.TRTEngine`: Core class for running TensorRT engines
- :py:class:`~trtutils.TRTModel`: High-level interface with preprocessing/postprocessing
- :py:class:`~trtutils.impls.yolo.YOLO`: Specialized interface for YOLO models

For detailed tutorials on using these components, see:

- :ref:`Basic Usage Tutorial <tutorials_basic>`
- :ref:`Advanced Usage Tutorial <tutorials_advanced>`
- :ref:`YOLO Tutorials <tutorials_yolo>`

API Reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Core API <source/trtutils.core>
   YOLO Implementation <source/trtutils.impls.yolo>
   Jetson Utilities <source/trtutils.jetson>
   TensorRT Executable <source/trtutils.trtexec>

Additional Resources
--------------------

- :ref:`Tutorial Overview <tutorial_overview>`: Step-by-step guides
- :ref:`Examples <examples>`: Code examples and demos
- :ref:`Changelog <changelog>`: Version history and changes 
