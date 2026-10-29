.. _examples_io:

Example: GPU decode (trtutils.io)
=================================

``trtutils.io`` decodes images with nvImageCodec and video with PyNvVideoCodec straight into
device ``Buffer``\ s, which every trtutils image model accepts. Both libraries are optional.

.. code-block:: console

    $ pip install "trtutils[nvimgcodec-cu12,nvvideocodec]"   # or nvimgcodec-cu13 / -cu11

.. code-block:: python

    from trtutils import io
    from trtutils.models import YOLOv11

    model = YOLOv11("yolov11n.engine", preprocessor="cuda")

    # image: uint8 HWC BGR on the device, no host copy
    frame = io.decode_image("street.jpg")  # a path or the encoded bytes
    detections = model.end2end(frame)

    # video: one device BGR frame per iteration
    with io.VideoReader("clip.mp4") as reader:
        print(len(reader), reader.shape)
        for frame in reader:
            detections = model.end2end(frame)

Semantics
---------

* ``decode_image`` uses ``I_BGR`` with the default nvImageCodec backend. On the stock test
  photos it is bit-identical to ``cv2.imread``; in general JPEG decoders may round a few
  pixels differently (measured: at most 1 level on a few hundredths of a percent of pixels,
  rarely 3). The returned ``Buffer`` owns the decoded image's memory.
* Both entry points synchronize the null stream before returning, so any engine stream can
  read the result immediately.
* ``VideoReader`` frames are views of **one** buffer owned by the reader: a frame is valid
  only until the next iteration. Copy it (``Buffer.copy_to``) to keep it. Pixels differ
  slightly from OpenCV's decode (different YUV to RGB matrix, mean about 1 level).
* Without the libraries, ``decode_image`` and ``VideoReader`` raise ``ImportError`` naming
  the extra to install.
