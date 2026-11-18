"""minigrad from Python: tinygrad's API over minigrad's Mojo engine.

```python
from minigrad import Tensor, TinyJit, nn, dtypes

x = Tensor.rand(4, 3, requires_grad=True)
loss = (x @ Tensor.ones(3, 2)).relu().sum()
loss.backward()
print(x.grad.numpy())
```

Build the engine first: `pixi run python-build`. `docs/python.md`
describes what is covered and how it differs from tinygrad.
"""

from . import nn
from .dtype import DType, dtypes
from .jit import TinyJit
from .tensor import Tensor

__all__ = ["Tensor", "TinyJit", "nn", "dtypes", "DType"]
