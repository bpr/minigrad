"""tinygrad's examples/beautiful_mnist.py, run on minigrad from Python.

    bash examples/get_mnist.sh
    pixi run python-build
    PYTHONPATH=python python examples/beautiful_mnist.py

The changes from tinygrad's file, each marked `# minigrad:` below:
imports, no `@function` (tinygrad's call caching), training mode by
`Tensor.train()`, and batch indices drawn outside the jitted step and
passed in (minigrad's `randint` draws on the host, so a replay would
reuse the captured indices).
"""

import os
import time
from typing import Callable

from minigrad import Tensor, TinyJit, nn  # minigrad: imports
from minigrad.nn.datasets import mnist


def getenv(k, d=0):
    return type(d)(os.environ.get(k, d))


class Model:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5), Tensor.relu,
            nn.Conv2d(32, 32, 5), Tensor.relu,
            nn.BatchNorm(32), Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3), Tensor.relu,
            nn.Conv2d(64, 64, 3), Tensor.relu,
            nn.BatchNorm(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576, 10)]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

    @TinyJit
    @Tensor.train()  # minigrad: for Context(TRAINING=1)
    def train_step(self, X_train: Tensor, Y_train: Tensor, samples: Tensor) -> Tensor:
        opt.zero_grad()
        loss = self(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
        return loss.realize(*opt.schedule_step())

    @TinyJit
    def get_test_acc(self, X_test: Tensor, Y_test: Tensor) -> Tensor:
        return (self(X_test).argmax(axis=1) == Y_test).mean() * 100


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist()

    Tensor.manual_seed(getenv("SEED", 0))
    model = Model()
    opt = (nn.optim.SGD if getenv("SGD") else nn.optim.Adam)(nn.state.get_parameters(model))

    test_acc = float("nan")
    start = time.perf_counter()
    for i in range(getenv("STEPS", 70)):
        t0 = time.perf_counter()
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])  # minigrad: drawn here
        loss = model.train_step(X_train, Y_train, samples)
        if i % 10 == 9:
            test_acc = model.get_test_acc(X_test, Y_test).item()
        print(f"step {i + 1:3d} loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%"
              f"  {(time.perf_counter() - t0) * 1e3:7.1f} ms")
    print(f"total {time.perf_counter() - start:.2f} s")

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(f"{test_acc=} >= {target}")
        else:
            raise ValueError(f"{test_acc=} < {target}")
