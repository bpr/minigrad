"""A tensor library with automatic differentiation, modeled on tinygrad
(`minigrad`).

```mojo
from minigrad import Tensor

def main() raises:
    var x = Tensor.rand(4, 3).requires_grad_()
    var loss = (x @ Tensor.ones(3, 2)).relu().sum()
    loss.backward()
    print(x.grad().to_string())
```

Layers and optimizers are in `minigrad.nn`; the MNIST loader is in
`minigrad.datasets`. `docs/architecture.md` describes how it works.
"""

from .dtype import DT, promote
from .graph import graph
from .tensor import Tensor
