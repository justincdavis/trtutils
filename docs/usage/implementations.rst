.. _impls_usage:

Implementations
---------------

Any implementations provided by trtutils are present in the `trtutils.impls` submodule.
Usage of the implementations may require installing additional pacakges.
An example of this is: `pip3 install trtutils[yolo]`, for a detailed guide on packages
installation and the possible values, see the installation guide. Usage of the 
implementations submodules is designed to be as easy as possible.

YOLO
^^^^

For a YOLO model, once the TensorRT engine is compiled, you can perform inference
in only a few lines of Python.

.. code-block:: python

    from trtutils.impls.yolo import YOLO

    yolo = YOLO("PATH_TO_ENGINE")

    # assuming img is loaded already
    detections = yolo.end2end(img)
 