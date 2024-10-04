# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


class Tegrastats:
    """Runs tegrastats in a seperate process and stores output in a file."""

    def __init__(
        self: Self,
        output: Path | str,
        interval: int = 1000,
    ) -> None:
        """
        Create an instance of tegrastats.

        Parameters
        ----------
        output : Path | str
            The path to the output file
        interval : int, optional
            The interval to run tegrastats in milliseconds, by default 1000

        """
        self._output = Path(output)
        self._interval = interval

        self._process = mp.Process(
            target=self._run,
            args=(self._output, self._interval),
            daemon=True,
        )

    def __enter__(self: Self) -> Self:
        self._process.start()
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._process.terminate()

    def _run(
        self: Self,
        output: Path,
        interval: int,
    ) -> None:
        with output.open("w+") as f:
            subprocess.run(
                ["tegrastats", f"--interval={interval}"],
                stdout=f,
                stderr=subprocess.PIPE,
                check=True,
            )
