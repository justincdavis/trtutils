.. _yolo_tutorial:

YOLO
====

In order to use the YOLO interface implemented by trtutils, you need to have
an ONNX engine which requires no CPU-side NMS operations. The common pattern
to solving this is to have an ONNX model with an output layer assigned to
the TensorRT EfficientNMS plugin. However, not all implementations of YOLO
support exporting with such output. As such, there is an individual guide for
how to export each 7/8/9/10/X to get end-to-end onnx represenation.

Exporting end-to-end ONNX
-------------------------

YOLOv7
^^^^^^

YOLOv7 support end-to-end export of the onnx weights directly. The workflow is
as follows:

.. code-block:: console

    # Clone the authors repository
    $ git clone https://github.com/WongKinYiu/yolov7.git
    $ cd yolov7

    # Assuming you have downloaded the weights you want to export
    # OR trained a custom model
    # export the onnx weights as such
    # change the topk-all, iou-thres, conf-thres and img-size according to your needs
    $ python3 export.py -weigths PATH_TO_WEIGHTS --end2end --grid --simplify --topk-all 100 --iou-thres 0.5 --conf-thres 0.25 --img-size 640

YOLOv8
^^^^^^

YOLOv8 is implemented by Ultralytics, since their library performs NMS on CPU
their weights do not support direct end-to-end exporting. As such, you need
to use a multi-step approach.

First step is to get the ONNX weights from Ultralytics. One such method could be:

.. code-block:: console

    # Assuming you have ultralytics installed
    # this will save the onnx weights in the same directory as the torch weights
    $ yolo export model=TORCH_WEIGHTS format=onnx

Once you have the Ultralytics ONNX weights:

.. code-block:: console

    # Clone the YOLOv8-tensorrt repo
    $ git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git
    $ cd YOLOv8-TensorRT

    # then you need to convert the weights using their script
    # Adjust the iou-thres, conf-thres, topk, and image-size according to your needs
    $ python3 export-det.py --weights PATH_TO_WEIGHTS --iou-thres 0.5 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1,3,640,640 --device cuda:0

YOLOv9
^^^^^^

YOLOv9 is written by the same authors as YOLOv7 and support many of the same
exporting options. YOLOv9 has a quirk though, where if exporting with end-to-end
ONNX weights, the input size is explicitly marked as dynamic. This is
accounted for when building the TensorRT engine.

To export the end-to-end ONNX:

.. code-block:: console

    # clone the YOLOv9 repo
    $ git clone https://github.com/WongKinYiu/yolov9.git
    $ cd yolov9

    # export the ONNX weights
    # adjust topk-all, iou-thres, conf-thres, and img-size to your needs
    # NOTE: the img-size is not static in the weights and will have to be
    # used during TensorRT engine building as well
    $ python3 export.py --weights PATH_TO_WEIGHTS --include onnx_end2end --simplify --iou-thres 0.5 --conf-thres 0.25 --topk-all 100 --img-size 640 640

YOLOv10
^^^^^^^

The YOLOv10 authors implemented on top of Ultralytics, as such it is
recomended to create a virtual environment to install their custom build
of the software.

Install custom ultralytics build and export ONNX:

.. code-block:: console

    # clone YOLOv10 repo
    $ git clone https://github.com/THU-MIG/yolov10.git
    $ cd yolov10

    # create the virtualenv
    $ python3 -m venv yolov10
    $ source yolov10/bin/activate

    # export the weights
    $ yolo export weights=PATH_TO_WEIGTHS format=onnx opset=13 simplify imgsz=640

    # deactivate the venv
    $ deactivate

YOLOX
^^^^^

The YOLOX repo contains the nessecary tools to export ONNX weights, but similiar
to Ultralytics, their software performs NMS on CPU. As such, another two-step
sequence is required to convert the ONNX weights to full end-to-end.

Export the intial ONNX weights:

.. code-block:: console

    # clone the YOLOX repo
    $ git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    $ cd YOLOX

    # export the weights
    # VERSION is one of the following: yolox-t, yolox-n, yolox-s, yolox-m
    $ python3 tools/export-onnx.py --output-name ONNX_OUTPUT -n VERSION -c TORCH_WEIGHTS -decode_in_inference

The remaining ONNX conversion to end-to-end occurs during the engine build process.

Building TensorRT engine for YOLO
---------------------------------

Building a TensorRT engine can be done via the TensorRT API
in C++/Python or through the CLI tool trtexec. trtexec is bundled
on all Jetson devices, and will be the focus on these instructions.

You can compile trtexec on other systems (or Jetson) and the source
code can be found here: https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec

trtutils packages a small wrapper around trtexec to allow interactivity
through Python. These instructions will use that and assume that
the :ref:`find_trtexec <trtutils.trtexec.find_trtexec>` is capable
of finding the compiled trtexec binary.

YOLOv7/8/10
^^^^^^^^^^^

In order to compile YOLOv 7/8/10 engines:

.. code-block:: python

    from trtutils.trtexec import build_engine

    # build the engine
    # additional options include
    # int8 precision
    # usage of a DLA on Jetson devices
    build_engine(
        weights=ONNX_WEIGHT_PATH,
        output=OUTPUT_ENGINE_PATH,
        fp16=True,
    )

YOLOv9
^^^^^^

Compiling the engine for YOLOv9 requires some additional arguments:

.. code-block:: python

    from trtutils.trtexec import build_engine

    # build the engine
    # replace 640 with the exported imgsz from the ONNX weight export step
    build_engine(
        weights=ONNX_WEIGHT_PATH,
        output=OUTPUT_ENGINE_PATH,
        fp16=True,
        shapes=[("images", (1, 3, 640, 640))],
    )

YOLOX
^^^^^

Compiling the engine for YOLOX requires an additional repo.
Instructions are as follows:

.. code-block:: console

    # clone the YOLOx tensorrt repo
    $ git clone https://github.com/justincdavis/YOLOX-TensorRT.git
    $ cd YOLOx-TensorRT

    # export the onnx weights to end-to-end TensorRT Engine
    # adjust the conf-thres, iou-thres, max-det according to your needs
    $ python3 export.py -o ONNX_WEIGHT -e ENGINE_OUTPUT --precision 'fp16' --end2end --conf_thres 0.25 --iou_thres 0.5 --max_det 100

    # to export DLA engines on Jetson you can add the following arguments as such
    # where IMAGE_DIR is a directory of images to validate quantization againist
    $ python3 export.py -o ONNX_WEIGHT -e ENGINE_OUTPUT --precision 'fp16' --end2end --conf_thres 0.25 --iou_thres 0.5 --max_det 100 --dlacore 0 --calib_input IMAGE_DIR

    # the DLA (on ORIN series Jetson devices espcially) runs faster in int8 mode
    $ python3 export.py -o ONNX_WEIGHT -e ENGINE_OUTPUT --precision 'int8' --end2end --conf_thres 0.25 --iou_thres 0.5 --max_det 100 --dlacore 0 --calib_input IMAGE_DIR

Running TensorRT YOLO Engines
-----------------------------

All end-to-end engines can be run using the :ref:`YOLO <trtutils.impls.yolo.YOLO>` class.

Example:

.. code-block:: python

    import cv2
    from trtutils.impls.yolo import YOLO

    yolo = YOLO("PATH_TO_ENGINE")

    img = cv2.imread("PATH_TO_IMAGE")

    detections = yolo.end2end(img)

Some versions of YOLO (or your custom versions) may require other parameters or
input variations. For example, following the above instructions YOLOX requires
the input image still be in range 0, 255. 

An example of adjusting such a parameter is:

.. code-block:: python

    import cv2
    from trtutils.impls.yolo import YOLO

    # load engine without making changes
    yolox = YOLO("PATH_TO_YOLOX_ENGINE")

    img = cv2.imread("PATH_TO_IMAGE")
 
    detections = yolo.end2end(img) # will not return any valid detections

    # remake with input_range set correctly
    yolox = YOLO("PATH_TO_YOLOX_ENGINE", input_range=(0, 255))

    detections = yolo.end2end(img)  # should now have valid detections
