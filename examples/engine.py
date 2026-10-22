# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the TRTEngine class."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trtutils import Buffer, MemoryLocation, TRTEngine


# This example shows how to use the TRTEngine class
# with a simple engine file
def main() -> None:
    """Run the example."""
    engine = TRTEngine(
        Path(__file__).parent.parent / "data" / "engines" / "simple.engine",
        warmup=True,
    )

    # inputs are Buffers; get_random_input returns host Buffers
    rand_input = engine.get_random_input()
    outputs = engine.execute(rand_input)
    print(outputs)

    for output in outputs:
        print(output.shape)

    # wrap existing numpy data without copying it
    data = [Buffer.wrap(np.ascontiguousarray(b.array)) for b in rand_input]
    outputs = engine.execute(data)

    # device Buffers are read by TensorRT in place
    device_data = [Buffer.from_array(b.array, MemoryLocation.DEVICE) for b in rand_input]
    outputs = engine.execute(device_data)
    print([output.shape for output in outputs])


if __name__ == "__main__":
    main()
