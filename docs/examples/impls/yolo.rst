.. _examples_impls/yolo:

Example: impls/yolo.py
======================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""File showcasing the TRTEngine class."""
	
	from __future__ import annotations
	
	from pathlib import Path
	
	import cv2
	
	from trtutils.impls.yolo import YOLO
	
	
	# This example shows how to use the TRTEngine class
	# on running a Yolo model with a single input image.
	# The Yolo model is not included in this repository.
	# This works with a yolov7 engine created by
	# using the export script locating in the yolov7 repository.
	# Then generate an engine using TensorRT by:
	#  trtexec --onnx=yolo.onnx --saveEngine=yolo.engine
	# The resulting engine can be used with this example.
	def main() -> None:
	    """Run the example."""
	    engine_dir = Path(__file__).parent.parent.parent / "data" / "engines"
	    engines = [
	        engine_dir / "trt_yolov7t.engine",
	        engine_dir / "trt_yolov8n.engine",
	        engine_dir / "trt_yolov9t.engine",
	        engine_dir / "trt_yolov10n.engine",
	        engine_dir / "trt_yolov7t_dla.engine",
	        engine_dir / "trt_yolov8n_dla.engine",
	        engine_dir / "trt_yolov9t_dla.engine",
	        engine_dir / "trt_yolov10n_dla.engine",
	    ]
	
	    img = cv2.imread(str(Path(__file__).parent.parent.parent / "data" / "horse.jpg"))
	
	    for engine in engines:
	        yolo = YOLO(engine, warmup=False)
	
	        output = yolo.run(img)
	
	        bboxes = yolo.get_detections(output)
	
	        print(bboxes)
	
	        del yolo
	
	
	if __name__ == "__main__":
	    main()

