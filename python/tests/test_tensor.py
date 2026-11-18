"""The Python bindings against NumPy: construction, operators and their
type rules, reductions, movement and indexing, layers, gradients,
optimizers, TinyJit, and data interchange.

    pixi run python-test
"""

import os
import tempfile
import unittest

import numpy as np

from minigrad import Tensor, TinyJit, dtypes, nn
from minigrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save


def close(test, got, want, tol=1e-5):
    got = got.numpy() if isinstance(got, Tensor) else np.asarray(got)
    want = want.numpy() if isinstance(want, Tensor) else np.asarray(want)
    test.assertEqual(got.shape, want.shape)
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def seq(*shape, scale=0.1):
    n = int(np.prod(shape))
    return ((np.arange(n) % 17 - 8) * scale).astype(np.float32).reshape(shape)


class TestCreation(unittest.TestCase):
    def test_from_python(self):
        self.assertEqual(Tensor([[1, 2], [3, 4]]).dtype, dtypes.int32)
        self.assertEqual(Tensor([1.0, 2]).dtype, dtypes.float32)
        self.assertEqual(Tensor([True, False]).dtype, dtypes.bool)
        self.assertEqual(Tensor([[1, 2], [3, 4]]).tolist(), [[1, 2], [3, 4]])
        self.assertEqual(Tensor(2.5).item(), 2.5)
        self.assertEqual(Tensor(3).shape, ())
        self.assertEqual(Tensor([1, 2], dtype=dtypes.float16).tolist(), [1.0, 2.0])
        self.assertEqual(Tensor([1, 2], dtype="bfloat16").float().tolist(), [1.0, 2.0])
        with self.assertRaises(ValueError):
            Tensor([[1, 2], [3]])

    def test_from_numpy(self):
        for dt in ["float32", "float64", "int32", "int64", "uint8", "bool", "float16"]:
            a = (np.arange(12).reshape(3, 4) % 5).astype(dt)
            t = Tensor(a)
            self.assertEqual(t.dtype.name, dt)
            np.testing.assert_array_equal(t.numpy(), a)
        # Non-contiguous input.
        a = np.arange(12, dtype=np.float32).reshape(3, 4).T
        np.testing.assert_array_equal(Tensor(a).numpy(), a)

    def test_numpy_view(self):
        t = Tensor(seq(3, 4)).realize()
        a = t.numpy()
        self.assertFalse(a.flags.writeable)
        b = t.numpy()
        self.assertTrue(np.shares_memory(a, b))  # no copy for a buffer
        del t
        np.testing.assert_array_equal(a, seq(3, 4))  # the view keeps it alive
        self.assertEqual(Tensor(seq(2, 3)).T.numpy().shape, (3, 2))
        self.assertEqual(Tensor.zeros(0, 3).numpy().shape, (0, 3))

    def test_factories(self):
        close(self, Tensor.zeros(2, 3), np.zeros((2, 3)))
        close(self, Tensor.ones((2, 3)), np.ones((2, 3)))
        close(self, Tensor.full((2,), 7), np.full((2,), 7))
        self.assertEqual(Tensor.full((2,), 7).dtype, dtypes.int32)
        self.assertEqual(Tensor.arange(5).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(Tensor.arange(2, 11, 3).tolist(), [2, 5, 8])
        close(self, Tensor.arange(0, 1, 0.25), [0, 0.25, 0.5, 0.75])
        close(self, Tensor.eye(3), np.eye(3))
        close(self, Tensor.eye(2, 3), np.eye(2, 3))
        Tensor.manual_seed(1)
        a = Tensor.rand(1000).numpy()
        self.assertTrue(0 <= a.min() and a.max() < 1)
        b = Tensor.randn(2000).numpy()
        self.assertLess(abs(b.mean()), 0.1)
        self.assertLess(abs(b.std() - 1), 0.1)
        r = Tensor.randint(500, low=3, high=7).numpy()
        self.assertEqual((r.min(), r.max()), (3, 6))
        u = Tensor.uniform(3, 4, low=-2, high=-1)
        self.assertEqual(u.shape, (3, 4))
        self.assertTrue((u.numpy() < -1).all())


class TestOps(unittest.TestCase):
    def test_binary_and_broadcast(self):
        a, b = seq(3, 4), seq(4) + 2
        x, y = Tensor(a), Tensor(b)
        close(self, x + y, a + b)
        close(self, x - y, a - b)
        close(self, x * y, a * b)
        close(self, x / y, a / b)
        close(self, x.maximum(y), np.maximum(a, b))
        close(self, x.minimum(0.0), np.minimum(a, 0))
        close(self, 2 - x, 2 - a)
        close(self, 1 / y, 1 / b)
        close(self, x ** 2, a ** 2)
        close(self, y ** 0.5, b ** 0.5, 1e-4)
        close(self, 2 ** x, 2 ** a, 1e-4)
        close(self, -x, -a)
        np.testing.assert_array_equal((x < y).numpy(), a < b)
        np.testing.assert_array_equal((x == x).numpy(), np.ones((3, 4), bool))

    def test_type_rules(self):
        i = Tensor([1, 2, 3])
        self.assertEqual((i + 1).dtype, dtypes.int32)
        self.assertEqual((i + 1.5).dtype, dtypes.float32)
        self.assertEqual((i / 2).dtype, dtypes.float32)
        self.assertEqual((i // 2).tolist(), [0, 1, 1])
        self.assertEqual((i % 2).tolist(), [1, 0, 1])
        h = Tensor([1.0], dtype=dtypes.float16)
        self.assertEqual((h * 2.0).dtype, dtypes.float16)
        self.assertEqual((h + Tensor([1.0])).dtype, dtypes.float32)

    def test_unary(self):
        a = seq(2, 5) + 1.0
        x = Tensor(a)
        for name, f in [("exp", np.exp), ("log", np.log), ("sqrt", np.sqrt),
                        ("relu", lambda v: np.maximum(v, 0)), ("tanh", np.tanh),
                        ("sigmoid", lambda v: 1 / (1 + np.exp(-v))), ("abs", np.abs),
                        ("sin", np.sin), ("floor", np.floor), ("square", np.square)]:
            close(self, getattr(x, name)(), f(a), 1e-5)
        close(self, x.clip(0.8, 1.2), np.clip(a, 0.8, 1.2))
        close(self, (x - 1).leaky_relu(0.1), np.where(a > 1, a - 1, (a - 1) * 0.1))

    def test_where_cast(self):
        a = seq(3, 4)
        x = Tensor(a)
        close(self, (x > 0).where(x, 0.0), np.where(a > 0, a, 0))
        close(self, (x > 0).where(1.0, -1.0), np.where(a > 0, 1.0, -1.0))
        self.assertEqual(x.cast(dtypes.int32).dtype, dtypes.int32)
        self.assertEqual(x.half().dtype, dtypes.float16)

    def test_reductions(self):
        a = seq(2, 3, 4)
        x = Tensor(a)
        close(self, x.sum(), a.sum(), 1e-4)
        close(self, x.sum(axis=1), a.sum(axis=1))
        close(self, x.sum(axis=(0, 2), keepdim=True), a.sum(axis=(0, 2), keepdims=True))
        close(self, x.max(axis=-1), a.max(axis=-1))
        close(self, x.min(axis=0), a.min(axis=0))
        close(self, x.mean(axis=1), a.mean(axis=1))
        close(self, x.var(axis=2), a.var(axis=2, ddof=1), 1e-4)
        close(self, x.std(axis=2, correction=0), a.std(axis=2), 1e-4)
        self.assertEqual(x.argmax(axis=1).tolist(), a.argmax(axis=1).tolist())
        self.assertEqual(x.argmax().item(), int(a.argmax()))
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        close(self, x.softmax(), e / e.sum(axis=-1, keepdims=True))
        close(self, x.log_softmax(1).exp().sum(1), np.ones((2, 4)), 1e-5)
        close(self, x.logsumexp(1), np.log(np.exp(a).sum(1)), 1e-5)
        self.assertEqual(Tensor([[1, 2], [3, 4]]).sum().dtype, dtypes.int32)


class TestMovement(unittest.TestCase):
    def test_shape_ops(self):
        a = seq(2, 3, 4)
        x = Tensor(a)
        close(self, x.reshape(6, -1), a.reshape(6, -1))
        close(self, x.permute(2, 0, 1), a.transpose(2, 0, 1))
        close(self, x.transpose(0, 2), a.swapaxes(0, 2))
        close(self, x.flatten(1), a.reshape(2, 12))
        close(self, x.unsqueeze(1), a[:, None])
        close(self, x[:, :1].squeeze(1), a[:, 0])
        close(self, Tensor(seq(3, 1)).expand(3, 5), np.broadcast_to(seq(3, 1), (3, 5)))
        close(self, Tensor(seq(3)).expand(2, -1), np.broadcast_to(seq(3), (2, 3)))
        close(self, x.flip(1), a[:, ::-1])
        close(self, x.pad(((0, 0), (1, 2), None)), np.pad(a, ((0, 0), (1, 2), (0, 0))))
        close(self, x.pad((1, 1)), np.pad(a, ((0, 0), (0, 0), (1, 1))))
        close(self, x.pad((1, 0, 0, 2), value=5.0), np.pad(a, ((0, 0), (0, 2), (1, 0)), constant_values=5))
        close(self, x.shrink(((0, 1), None, (1, 3))), a[:1, :, 1:3])
        close(self, x.repeat(1, 2, 1), np.tile(a, (1, 2, 1)))
        close(self, x.cat(x, dim=1), np.concatenate([a, a], 1))
        close(self, Tensor.stack(x, x, dim=0), np.stack([a, a]))
        self.assertEqual([t.shape for t in x.split(2, dim=2)], [(2, 3, 2), (2, 3, 2)])
        self.assertEqual([t.shape for t in x.chunk(2, dim=1)], [(2, 2, 4), (2, 1, 4)])

    def test_getitem(self):
        a = seq(4, 5, 6)
        x = Tensor(a)
        for idx in [1, -1, (1, 2), (slice(None), 3), slice(1, 3), (Ellipsis, 2),
                    (slice(None, None, 2),), (slice(None), slice(None, None, -1)),
                    (slice(4, 0, -2), slice(1, None, 3)), (None, 1, None),
                    (Ellipsis, None), (2, Ellipsis, slice(1, 2)), (slice(None, None, 3),)]:
            close(self, x[idx], a[idx])
        with self.assertRaises(IndexError):
            x[4]

    def test_tensor_index(self):
        a = seq(10, 3)
        idx = np.array([3, 0, 9, 3], dtype=np.int32)
        close(self, Tensor(a)[Tensor(idx)], a[idx])
        close(self, Tensor(a)[Tensor(idx.reshape(2, 2))], a[idx.reshape(2, 2)])


class TestLayers(unittest.TestCase):
    def test_matmul(self):
        a, b = seq(5, 7), seq(7, 3)
        close(self, Tensor(a) @ Tensor(b), a @ b, 1e-5)
        close(self, Tensor(seq(2, 5, 7)) @ Tensor(b), seq(2, 5, 7) @ b, 1e-5)
        close(self, Tensor(a).dot(Tensor(seq(7))), a @ seq(7), 1e-5)

    def test_conv2d(self):
        x, w, bias = seq(2, 3, 7, 7), seq(4, 3, 3, 3), seq(4)
        got = Tensor(x).conv2d(Tensor(w), Tensor(bias), padding=1, stride=2)
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        want = np.zeros((2, 4, 4, 4), np.float32)
        for i in range(4):
            for j in range(4):
                win = xp[:, :, 2 * i : 2 * i + 3, 2 * j : 2 * j + 3]
                want[:, :, i, j] = np.einsum("bchw,ochw->bo", win, w)
        close(self, got, want + bias[None, :, None, None], 1e-4)

    def test_pooling(self):
        a = seq(1, 2, 4, 4)
        close(self, Tensor(a).max_pool2d(), a.reshape(1, 2, 2, 2, 2, 2).max(axis=(3, 5)))
        close(self, Tensor(a).avg_pool2d(), a.reshape(1, 2, 2, 2, 2, 2).mean(axis=(3, 5)))

    def test_modules(self):
        Tensor.manual_seed(0)
        lin = nn.Linear(4, 3)
        a = seq(2, 4)
        close(self, lin(Tensor(a)), a @ lin.weight.numpy().T + lin.bias.numpy(), 1e-5)
        ln = nn.LayerNorm(4)
        want = (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)
        close(self, ln(Tensor(a)), want, 1e-4)
        emb = nn.Embedding(5, 2)
        close(self, emb(Tensor([4, 0])), emb.weight.numpy()[[4, 0]])
        bn = nn.BatchNorm(3)
        x = seq(4, 3, 2, 2) + 1
        with Tensor.train():
            y = bn(Tensor(x)).numpy()
        m, v = x.mean(axis=(0, 2, 3)), x.var(axis=(0, 2, 3))
        close(self, y, (x - m[None, :, None, None]) / np.sqrt(v[None, :, None, None] + 1e-5), 1e-4)
        close(self, bn.running_mean, 0.1 * m, 1e-5)
        self.assertFalse(Tensor.training)

    def test_attention(self):
        q, k, v = seq(2, 3, 4), seq(2, 5, 4) * 2, seq(2, 5, 6)
        s = q @ k.transpose(0, 2, 1) / 2.0
        e = np.exp(s - s.max(-1, keepdims=True))
        want = e / e.sum(-1, keepdims=True) @ v
        close(self, Tensor(q).scaled_dot_product_attention(Tensor(k), Tensor(v)), want, 1e-5)

    def test_state(self):
        class Net:
            def __init__(self):
                self.l1 = nn.Linear(3, 4)
                self.layers = [nn.Linear(4, 2, bias=False)]

        net = Net()
        self.assertEqual(sorted(get_state_dict(net)), ["l1.bias", "l1.weight", "layers.0.weight"])
        self.assertEqual(len(get_parameters(net)), 3)
        sd = {"a": Tensor(seq(2, 3)), "b": Tensor([1, 2, 3], dtype=dtypes.int64),
              "c": Tensor(seq(4), dtype=dtypes.bfloat16)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.safetensors")
            safe_save(sd, path)
            back = safe_load(path)
        for k, t in sd.items():
            self.assertEqual(back[k].dtype, t.dtype)
            close(self, back[k].float(), t.float())


class TestGradients(unittest.TestCase):
    def test_backward(self):
        a, b = seq(3, 4), seq(4, 2) + 0.5
        x = Tensor(a, requires_grad=True)
        w = Tensor(b, requires_grad=True)
        loss = (x @ w).relu().sum()
        loss.backward()
        mask = (a @ b > 0).astype(np.float32)
        close(self, x.grad, mask @ b.T, 1e-5)
        close(self, w.grad, a.T @ mask, 1e-5)
        gx, = (x * x).sum().gradient(x)
        close(self, gx, 2 * a)
        x.grad = None
        self.assertIsNone(x.grad)

    def test_cross_entropy(self):
        logits = seq(4, 5)
        y = np.array([1, 0, 4, 2], np.int32)
        t = Tensor(logits, requires_grad=True)
        loss = t.sparse_categorical_crossentropy(Tensor(y))
        lsm = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        # backward first: realizing the loss would end its history.
        loss.backward()
        close(self, loss, -lsm[np.arange(4), y].mean(), 1e-5)
        p = np.exp(lsm)
        p[np.arange(4), y] -= 1
        close(self, t.grad, p / 4, 1e-5)

    def test_assign(self):
        t = Tensor([1.0, 2.0])
        u = t  # the same tensor
        old = t * 1
        t.assign(Tensor([5.0, 6.0]))
        self.assertEqual(u.tolist(), [5.0, 6.0])
        self.assertEqual(old.tolist(), [1.0, 2.0])
        with self.assertRaises(Exception):
            t.assign(Tensor([1.0, 2.0, 3.0]))


def _train(opt_cls, use_jit, **kw):
    Tensor.manual_seed(5)
    model = [nn.Linear(6, 8), Tensor.relu, nn.Linear(8, 3)]
    opt = opt_cls(get_parameters(model), **kw)

    def step(x, y):
        opt.zero_grad()
        loss = x.sequential(model).sparse_categorical_crossentropy(y).backward()
        return loss.realize(*opt.schedule_step())

    f = TinyJit(step) if use_jit else step
    losses = []
    for i in range(6):
        x = Tensor(seq(10, 6) * (1 + 0.1 * i)).contiguous().realize()
        y = Tensor(np.arange(10, dtype=np.int32) % 3)
        losses.append(f(x, y).item())
    return losses, [p.numpy().copy() for p in get_parameters(model)]


class TestTraining(unittest.TestCase):
    def test_adam_matches_numpy(self):
        w0 = seq(3, 2)
        w = Tensor(w0.copy())
        opt = nn.optim.Adam([w], lr=0.01)
        m = v = np.zeros_like(w0)
        ref = w0.copy()
        for t in range(1, 4):
            opt.zero_grad()
            (w * w).sum().backward()
            opt.step()
            g = 2 * ref
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            ref = ref - 0.01 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        close(self, w, ref, 1e-5)

    def test_jit_matches_eager(self):
        for opt_cls, kw in [(nn.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
                            (nn.optim.Adam, {"lr": 0.01}),
                            (nn.optim.AdamW, {"lr": 0.01})]:
            eager, pe = _train(opt_cls, False, **kw)
            jitted, pj = _train(opt_cls, True, **kw)
            np.testing.assert_allclose(jitted, eager, rtol=1e-5)
            for a, b in zip(pj, pe):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
            self.assertLess(eager[-1], eager[0])

    def test_jit_structure_and_errors(self):
        @TinyJit
        def f(x, scale=None):
            return {"bad": x} if scale is None else (x * 2, [x.sum()])

        x = Tensor(seq(3)).realize()
        with self.assertRaises(TypeError):
            f(x)
        for i in range(4):
            a, (s,) = f(Tensor(seq(3) + i).realize(), scale=1)
            close(self, a, (seq(3) + i) * 2)
            close(self, s, (seq(3) + i).sum(), 1e-5)
        with self.assertRaises(Exception):
            f(Tensor(seq(4)).realize(), scale=1)


if __name__ == "__main__":
    unittest.main()
