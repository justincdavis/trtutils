from __future__ import annotations

from .core import (
    allocate_bindings,
    create_engine,
    memcpy_device_to_host,
    memcpy_host_to_device,
)


class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self._engine, self._context, self._logger = create_engine(engine_path)

        self._inputs, self._outputs, self._allocations = allocate_bindings(
            self._engine,
            self._context,
            self._logger,
        )

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self._inputs[0].shape, self._inputs[0].dtype

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self._outputs:
            specs.append((o.shape, o.dtype))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        memcpy_host_to_device(self._inputs[0].allocation, batch)
        self._context.execute_v2(self._allocations)
        for o_idx in range(len(self._outputs)):
            memcpy_device_to_host(
                self._outputs[o_idx].host_allocation,
                self._outputs[o_idx].allocation,
            )
        return [o.host_allocation for o in self._outputs]
