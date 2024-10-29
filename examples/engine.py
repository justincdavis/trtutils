# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the TRTEngine class."""

from __future__ import annotations

from pathlib import Path

from trtutils import TRTEngine


# This example shows how to use the TRTEngine class
# with a simple engine file
def main() -> None:
    """Run the example."""
    engine = TRTEngine(
        Path(__file__).parent.parent / "data" / "engines" / "simple.engine",
        warmup=True,
    )

    rand_input = engine.get_random_input()
    outputs = engine.execute(rand_input)
    print(outputs)

    for output in outputs:
        print(output.shape)


if __name__ == "__main__":
    main()
