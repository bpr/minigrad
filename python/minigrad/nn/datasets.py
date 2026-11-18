"""MNIST, as `tinygrad.nn.datasets.mnist` returns it."""

from __future__ import annotations

import os

from ..dtype import dtypes
from ..tensor import Tensor


def _idx(path: str, header: int, shape: tuple) -> Tensor:
    with open(path, "rb") as f:
        data = f.read()[header:]
    return Tensor(memoryview(data)).reshape(shape)


def mnist(directory: str | None = None):
    """`X_train, Y_train, X_test, Y_test`: images as uint8 of shape
    (n, 1, 28, 28), labels as uint8. Reads the unpacked files that
    `examples/get_mnist.sh` fetches into `data/mnist` (or `directory`,
    or `$MNIST_DIR`)."""
    d = directory or os.environ.get("MNIST_DIR") or os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "mnist")
    return (
        _idx(os.path.join(d, "train-images-idx3-ubyte"), 16, (-1, 1, 28, 28)),
        _idx(os.path.join(d, "train-labels-idx1-ubyte"), 8, (-1,)),
        _idx(os.path.join(d, "t10k-images-idx3-ubyte"), 16, (-1, 1, 28, 28)),
        _idx(os.path.join(d, "t10k-labels-idx1-ubyte"), 8, (-1,)),
    )
