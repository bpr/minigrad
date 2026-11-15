"""Train a small convolutional network on MNIST.

    bash examples/get_mnist.sh
    mojo run -I . examples/mnist.mojo [STEPS] [BATCH]

The model and settings are tinygrad's `examples/beautiful_mnist.py`: four
convolutions with batch normalization and max pooling, Adam, batches of
512. Seventy steps reach about 98% test accuracy.
"""

from std.sys import argv
from std.time import perf_counter_ns

from minigrad.datasets import Mnist
from minigrad.dtype import DT
from minigrad.graph import graph
from minigrad.nn import Adam, BatchNorm, Conv2d, Linear, Module, collect
from minigrad.tensor import Tensor


struct Model(Module, Movable):
    var c1: Conv2d
    var c2: Conv2d
    var bn1: BatchNorm
    var c3: Conv2d
    var c4: Conv2d
    var bn2: BatchNorm
    var fc: Linear

    def __init__(out self) raises:
        self.c1 = Conv2d(1, 32, 5)
        self.c2 = Conv2d(32, 32, 5)
        self.bn1 = BatchNorm(32)
        self.c3 = Conv2d(32, 64, 3)
        self.c4 = Conv2d(64, 64, 3)
        self.bn2 = BatchNorm(64)
        self.fc = Linear(576, 10)

    def __call__(self, x: Tensor) raises -> Tensor:
        var h = self.c2(self.c1(x).relu()).relu()
        h = self.bn1(h).max_pool2d()
        h = self.c4(self.c3(h).relu()).relu()
        h = self.bn2(h).max_pool2d()
        return self.fc(h.flatten(1))

    def parameters(self) -> List[Tensor]:
        return collect(
            self.c1, self.c2, self.bn1, self.c3, self.c4, self.bn2, self.fc
        )


def accuracy(model: Model, data: Mnist) raises -> Float64:
    """The fraction of the test set classified correctly."""
    Tensor.train(False)
    var correct = 0.0
    var n = data.num_test()
    var size = 1000
    for start in range(0, n, size):
        var indices = List[Int]()
        for i in range(start, min(n, start + size)):
            indices.append(i)
        var batch = data.test_batch(indices)
        correct += model(batch.x).argmax(1).eq(batch.y).sum().item()
    return correct / Float64(n)


def main() raises:
    var args = argv()
    var steps = atol(args[1]) if len(args) > 1 else 70
    var batch_size = atol(args[2]) if len(args) > 2 else 512
    Tensor.manual_seed(0)
    var data = Mnist()
    var model = Model()
    var opt = Adam(model.parameters())
    var t_start = perf_counter_ns()
    for step in range(steps):
        Tensor.train(True)
        var t0 = perf_counter_ns()
        var batch = data.random_train_batch(batch_size)
        opt.zero_grad()
        var loss = model(batch.x).sparse_categorical_crossentropy(batch.y)
        loss.backward()
        opt.step([loss])
        var value = loss.item()
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        if step % 10 == 9 or step == steps - 1:
            print(
                "step",
                step + 1,
                " loss",
                value,
                " ",
                ms,
                "ms/step   test accuracy",
                accuracy(model, data) * 100.0,
                "%",
            )
        else:
            print("step", step + 1, " loss", value, " ", ms, "ms/step")
    var st = graph()[].stats.copy()
    print(
        "total",
        Float64(perf_counter_ns() - t_start) / 1e9,
        "s;  kernels run",
        st.kernels_run,
        "(",
        st.gemm_kernels,
        "by gemm);  plans built",
        st.kernels_compiled,
    )
