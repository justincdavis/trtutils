.. _examples_engine:

Example: engine.py
==================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	# This program is free software: you can redistribute it and/or modify
	# it under the terms of the GNU General Public License as published by
	# the Free Software Foundation, either version 3 of the License, or
	# (at your option) any later version.
	#
	# This program is distributed in the hope that it will be useful,
	# but WITHOUT ANY WARRANTY; without even the implied warranty of
	# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	# GNU General Public License for more details.
	#
	# You should have received a copy of the GNU General Public License
	# along with this program. If not, see <https://www.gnu.org/licenses/>.
	"""File showcasing the TRTEngine class."""
	from __future__ import annotations
	
	import cv2
	import numpy as np
	
	from trtutils import TRTEngine
	
	
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
	    engine = TRTEngine("yolo.engine", warmup=True, dtype=np.float32)
	
	    img = cv2.imread("example.jpg")
	    img = cv2.resize(img, (640, 640))
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    img = img / 255.0
	    img = img[np.newaxis, :]
	    img = np.transpose(img, (0, 3, 1, 2))
	    img = np.ascontiguousarray(img, dtype=np.float32)
	
	    outputs = engine.execute([img])
	    print(outputs)
	
	
	if __name__ == "__main__":
	    main()

