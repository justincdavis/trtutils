.. _usage:

Usage
------------

Using trtutils is straightforward. Once you have the compiled TensorRT engine,
you can simply import TRTEngine and begin your benchmarking process.
If you need to handle the outputs of the model, then you can use TRTModel
where you specify the pre and post-processing stages of the model.

TRTEngine
^^^^^^^^^

   .. code-block:: python

        import numpy as np
        from trtutils import TRTEngine

        engine = TRTEngine("engine.engine", warmup=True, dtype=np.float32)

        timings = []
        for _ in range(test_iterations):
            t0 = time.perf_counter()
            engine.mock_execute()
            timings.append(time.perf_counter() - t0)
        print(f"Model latency is {np.mean(timings):.2f} seconds")


TRTModel
^^^^^^^^

   .. code-block:: python

        import cv2
        import numpy as np

        from trtutils import TRTModel

        def preprocess(inputs: list[np.ndarray]) -> list[np.ndarray]:
            def _process(img: np.ndarray) -> np.ndarray:
                img = cv2.resize(img, (640, 640), cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                img = img[np.newaxis, :]
                img = np.transpose(img, (0, 3, 1, 2))
                return np.ascontiguousarray(img, dtype=np.float32)

            return [_process(img) for img in inputs]

        def postprocess(outputs: np.ndarray) -> tuple[float, np.ndarray, int]:
            score = outputs[2][0][0]
            bbox = outputs[1][0][0]
            clas = outputs[0][0][0]
            return score, bbox, clas

        model = TRTModel(
            "yolo.engine",
            preprocess,
            postprocess,
            warmup=True,
            dtype=np.float32,
        )

        img = cv2.imread("example.jpg")
        outputs = model.run([img])
