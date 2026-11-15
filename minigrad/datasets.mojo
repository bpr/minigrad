"""Datasets (`minigrad.datasets`).

Loaders read files into host memory and build batches as realized
tensors; nothing here goes through the lazy graph.
"""

from minigrad.dtype import DT
from minigrad.tensor import Tensor, _next_u64


@fieldwise_init
struct Batch(Copyable):
    """Inputs and their labels."""

    var x: Tensor
    """`float32` inputs."""
    var y: Tensor
    """`int32` class labels."""


struct Mnist(Movable):
    """The MNIST handwritten digits: 28 by 28 grayscale images, 60,000
    for training and 10,000 for testing.

    `examples/get_mnist.sh` downloads the four files this reads.
    """

    var train_images: List[UInt8]
    var train_labels: List[UInt8]
    var test_images: List[UInt8]
    var test_labels: List[UInt8]

    def __init__(out self, directory: String = "data/mnist") raises:
        """Reads the dataset.

        Args:
            directory: Where the four unpacked `*-ubyte` files are.

        Raises:
            If a file is missing or has the wrong size.
        """
        self.train_images = _read_idx(
            directory + "/train-images-idx3-ubyte", 16, 60000 * 784
        )
        self.train_labels = _read_idx(
            directory + "/train-labels-idx1-ubyte", 8, 60000
        )
        self.test_images = _read_idx(
            directory + "/t10k-images-idx3-ubyte", 16, 10000 * 784
        )
        self.test_labels = _read_idx(
            directory + "/t10k-labels-idx1-ubyte", 8, 10000
        )

    def num_train(self) -> Int:
        """The number of training images."""
        return len(self.train_labels)

    def num_test(self) -> Int:
        """The number of test images."""
        return len(self.test_labels)

    def _batch(
        self, images: List[UInt8], labels: List[UInt8], indices: List[Int]
    ) raises -> Batch:
        var n = len(indices)
        var x = Tensor.empty([n, 1, 28, 28], DT.float32)
        var y = Tensor.empty([n], DT.int32)
        var px = x.data_ptr().unsafe_bitcast[Float32]()
        var py = y.data_ptr().unsafe_bitcast[Int32]()
        var src = images.unsafe_ptr()
        for b in range(n):
            var row = src.unsafe_offset(indices[b] * 784)
            var dst = px.unsafe_offset(b * 784)
            for i in range(784):
                dst[unsafe_offset=i] = Float32(row[unsafe_offset=i]) * (
                    1.0 / 255.0
                )
            py[unsafe_offset=b] = Int32(labels[indices[b]])
        return Batch(x, y)

    def train_batch(self, indices: List[Int]) raises -> Batch:
        """The training images at `indices`, of shape (n, 1, 28, 28) with
        pixels scaled to [0, 1], and their labels."""
        return self._batch(self.train_images, self.train_labels, indices)

    def test_batch(self, indices: List[Int]) raises -> Batch:
        """The test images at `indices` and their labels; see
        `train_batch`."""
        return self._batch(self.test_images, self.test_labels, indices)

    def random_train_batch(self, n: Int) raises -> Batch:
        """`n` training images drawn uniformly with replacement, using
        the generator `Tensor.manual_seed` seeds."""
        var indices = List[Int](capacity=n)
        for _ in range(n):
            indices.append(Int(_next_u64() % UInt64(self.num_train())))
        return self.train_batch(indices)


def _read_idx(path: String, header: Int, count: Int) raises -> List[UInt8]:
    """The `count` bytes after the `header` bytes of an IDX file."""
    var bytes: List[UInt8]
    with open(path, "r") as f:
        bytes = f.read_bytes()
    if len(bytes) != header + count:
        raise Error(
            t"{path}: expected {header + count} bytes, found {len(bytes)}"
        )
    var out = List[UInt8](capacity=count)
    for i in range(count):
        out.append(bytes[header + i])
    return out^
