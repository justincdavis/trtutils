.. _tutorials_basic:

Basic Usage Tutorial
====================

This tutorial covers the basic usage of trtutils, focusing on the core
:py:class:`~trtutils.TRTEngine` class.

The core functionality of trtutils is implemented inside of the 
:py:class:`~trtutils.TRTEngine` class. 

TRTEngine
^^^^^^^^^

An example of using :py:class:`~trtutils.TRTEngine` is given below:

.. code-block:: python

    from trtutils import Buffer, TRTEngine

    engine = TRTEngine("engine.engine")  # pass you compiled TensorRT engine file

    # after creating, you can perform inference on random data
    engine.mock_execute()

    # or with real data
    data = read_data()  # a C-contiguous np.ndarray
    outputs = engine([Buffer.wrap(data)])

Every input is a :py:class:`~trtutils.Buffer`: one contiguous, typed allocation
on the host or the device. :py:meth:`~trtutils.Buffer.wrap` views existing data
without copying it - a numpy array, another Buffer, or any object exposing
``__cuda_array_interface__`` (CuPy, PyTorch, Numba, ...). Device Buffers are read
by TensorRT in place; host Buffers are copied to the device first.

Because a Buffer carries its shape and dtype, the engine checks each input
before running: the dtype, the rank, every static dimension, and the optimization
profile bounds of every dynamic dimension. A dynamic engine runs at exactly the
submitted shape and returns outputs of the matching shape.

.. code-block:: python

    from trtutils import Buffer, MemoryLocation

    # a device Buffer, uploaded once and reused across calls
    device_input = Buffer.from_array(data, MemoryLocation.DEVICE)
    outputs = engine([device_input])

    # CuPy / PyTorch tensors are wrapped without a copy
    outputs = engine([Buffer.wrap(torch_tensor)])

.. code-block:: python

    # get information about the expected inputs
    print(engine.input_shapes)  # get the expected shapes of inputs
    print(engine.input_dtypes)  # get the datatypes of inputs
    print(engine.input_spec)  # get a list of shapes and dtypes
    
    # repeat for outputs
    print(engine.output_shapes)
    print(engine.output_dtypes)
    print(engine.output_spec)

As mentioned in the first code block, a TRTEngine can generate random data
for itself to perform inference on. This can be accessed via:

.. code-block:: python

    rand_data = engine.get_random_input()  # a list of host Buffers
    for input_data in rand_data:
        print(input_data.shape, input_data.dtype)
    output = engine(rand_data)

    # random data is cached to speedup warmup/benchmarking
    # if you need fresh data
    rand_data = engine.get_random_input(new=True)

Benchmarking
^^^^^^^^^^^^

You can benchmark engines through trtutils through either the command line 
or through the Python interface.

Command line:

.. code-block:: console

    $ python3 -m trtutils benchmark -m engine.engine -i 1000

    # If you are on a jetson device and installed the jetson dependencies
    # pass the -j flag to measure energy and power draw

    $ python3 -m trtutils benchmark -m engine.engine -i 1000 -j    

Python:

.. code-block:: python

    from trtutils import benchmark_engine

    results = benchmark_engine("engine.engine", iterations=1000)
    print(results.latency.mean)

    # using jetson
    from trtutils.jetson import benchmark_engine

    # a smaller tegra_interval means more measurements
    results = benchmark_engine("engine.engine", iterations=1000, tegra_interval=1)
    print(results.latency.mean)
    print(results.energy.mean)
    print(results.power_draw.mean)
