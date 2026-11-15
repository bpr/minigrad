"""Time per training step of the MNIST convolutional network on random
data.

    mojo run -I . benchmarks/convnet_step.mojo [BATCH] [STEPS]

The model is `examples/mnist.mojo`'s (tinygrad's `beautiful_mnist`): four
convolutions, two batch normalizations, two max pools, a linear layer;
Adam. Prints the median step time. The matching tinygrad program is
benchmarks/tinygrad/convnet_step.py.
"""

from std.sys import argv
from std.time import perf_counter_ns

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


def main() raises:
    var args = argv()
    var batch = atol(args[1]) if len(args) > 1 else 512
    var steps = atol(args[2]) if len(args) > 2 else 20
    Tensor.manual_seed(0)
    Tensor.train(True)
    var x = Tensor.rand(batch, 1, 28, 28)
    var y = Tensor.randint([batch], 0, 10)
    var model = Model()
    var opt = Adam(model.parameters())
    var times = List[Int]()
    var first = 0
    var loss = 0.0
    for i in range(steps):
        var t0 = perf_counter_ns()
        opt.zero_grad()
        var l = model(x).sparse_categorical_crossentropy(y)
        l.backward()
        opt.step([l])
        loss = l.item()
        var dt = Int(perf_counter_ns() - t0)
        if i == 0:
            first = dt
        times.append(dt)
    sort(times)
    var st = graph()[].stats.copy()
    print("batch", batch, "steps", steps, "final loss", loss)
    print(
        "median step:",
        Float64(times[len(times) // 2]) / 1e6,
        "ms   min:",
        Float64(times[0]) / 1e6,
        "ms   first step:",
        Float64(first) / 1e6,
        "ms",
    )
    print(
        "kernels per step:",
        st.kernels_run // steps,
        "  plans built in total:",
        st.kernels_compiled,
    )
