.. _tutorials_basic:

Basic Usage Tutorial
====================

This tutorial covers the basic usage of trtutils, focusing on the core
:py:class:`~trtutils.TRTEngine` and :py:class:`~trtutils.TRTModel` classes.

The core functionality of trtutils is implemented inside of the 
:py:class:`~trtutils.TRTEngine` class. 

Additional functionality is provided by the :py:class:`~trtutils.TRTModel` class.
This class provides a small wrapper around :py:class:`~trtutils.TRTEngine` allowing
preprocess and postprocess functions to be defined explicity.

TRTEngine
^^^^^^^^^

An example of using :py:class:`~trtutils.TRTEngine` is given below:

.. code-block:: python

    from trtutils import TRTEngine

    engine = TRTEngine("engine.engine")  # pass you compiled TensorRT engine file

    # after creating, you can perform inference on random data
    engine.mock_execute()

    # or with real data
    data = read_data()
    outputs = engine([data])

This class implements a barebones interface over a compiled TensorRT engine.
All data inputted is required to be formatted as a list of NumPy arrays, and are
expected to be of the correct shape and size (or risk a segmentation fault).

The format and datatype of inputs can be acquired from TRTEngine directly to ensure
that inputs are of the correct form. But, do note that inputs are not checked against
these properties automatically since that could incur large overhead among other issues.

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

    rand_data = engine.get_random_input()
    for input_data in rand_data:
        print(input_data.shape, input_data.dtype)
    output = engine(rand_data)

    # random data is cached to speedup warmup/benchmarking
    # if you need fresh data
    rand_data = engine.get_random_input(new=True)

TRTModel
^^^^^^^^

An example for using :py:class:`~trtutils.TRTModel` is given below:

.. code-block:: python

    import cv2
    from trtutils import TRTModel

    def preproc(imgs: list[np.ndarray]) -> list[np.ndarray]:
        # example may be that image should be resized
        return [
            cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) for img in imgs
        ]        

    model = TRTModel("object_detector.engine", preprocess=preproc)

    # now we can perform some inference
    img = cv2.imread("example.jpg")
    output = model.run([img])  # preprocessing will happen automatically

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
