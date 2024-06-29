# trtutils

[![](https://img.shields.io/pypi/pyversions/trtutils.svg)](https://pypi.org/pypi/trtutils/)
![PyPI](https://img.shields.io/pypi/v/trtutils.svg?style=plastic)
[![CodeFactor](https://www.codefactor.io/repository/github/justincdavis/trtutils/badge)](https://www.codefactor.io/repository/github/justincdavis/trtutils)

![MyPy](https://github.com/justincdavis/trtutils/actions/workflows/mypy.yaml/badge.svg?branch=main)
![Ruff](https://github.com/justincdavis/trtutils/actions/workflows/ruff.yaml/badge.svg?branch=main)
![PyPi Build](https://github.com/justincdavis/trtutils/actions/workflows/build-check.yaml/badge.svg?branch=main)
<!-- ![Black](https://github.com/justincdavis/trtutils/actions/workflows/black.yaml/badge.svg?branch=main) -->

Utilities for enabling easier high-level usage of TensorRT in Python.

## TRTEngine
The TRTEngine is a high-level abstraction allowing easy use of TensorRT 
engines through Python. Once an engine is built, it is simple and easy to use:

```python
from trtutils import TRTEngine

engine = TRTEngine("path_to_engine")

inputs = read_your_data()

for i in inputs:
    print(engine.execute(i))
```

We also provide an abstraction for defining higher-level models.
The TRTModel is designed to allow a user to define a pre and post 
processing step along with the engine to create an end-to-end 
inference object.

```python
from trtutils import TRTModel

# scale some images down
def pre(inputs):
    return [i / 255 for i in inputs]

# access the output classes from object detection
def post(outputs):
    return [o[0][0] for o in outputs]

model = TRTModel("path_to_engine", pre, post)

inputs = read_your_data()

for i in inputs:
    print(model(i))
```
