.. _jetson_performance:

Jetson Performance
==================

Lock the clocks
---------------

The ``MAXN`` power mode raises the clock ceilings but does not lock them. The GPU
still scales its frequency with load: on a Jetson AGX Orin at ``MAXN`` it idles
around 300-500 MHz against a 1300 MHz maximum, and short bursts of work such as
preprocessing a frame finish before it ramps up.

Lock the clocks for the lowest and most stable latency:

.. code-block:: console

    $ sudo jetson_clocks --store ~/clocks.conf   # save the current settings
    $ sudo jetson_clocks                         # lock CPU, GPU and memory clocks at max
    $ sudo jetson_clocks --restore ~/clocks.conf # undo

On a Jetson AGX Orin (JetPack 6.0, ``MAXN``), locking the clocks makes GPU
preprocessing up to 2.5x faster:

.. list-table::
   :header-rows: 1

   * - Preprocessing, 1080p input to 640x640
     - Default
     - Clocks locked
   * - ``CUDAPreprocessor``, 1 image
     - 1.1-1.5 ms
     - 0.8-1.1 ms
   * - ``TRTPreprocessor``, 1 image
     - 4.2-5.1 ms
     - 2.0-2.1 ms
   * - ``TRTPreprocessor``, 8 images
     - 12.1-13.8 ms
     - 6.5-6.9 ms

Ranges span pageable, pinned and unified memory.

Benchmark with locked clocks too. Without them, run-to-run differences come
mostly from frequency scaling, which can make an unrelated change look like a
win or a regression.

Memory modes
------------

``TRTEngine``, ``ImageModel`` and the GPU preprocessors take ``pagelocked_mem`` and
``unified_mem``. On Jetson ``unified_mem`` defaults to True, which maps pinned host
memory into the GPU so no explicit host-to-device copy is issued.

With clocks locked, pageable, pinned and unified (mapped) memory preprocess
within about 10% of each other on a Jetson AGX Orin, with no consistent winner.
Mapped memory does not make input images zero-copy: the GPU cannot read memory
that numpy allocated, so every image is still copied once into the mapped
buffer, by the CPU instead of the copy engine. Pinning each frame in place costs
more than that copy, so the defaults are a reasonable choice and changing the
memory mode is not a meaningful optimization on Jetson.
