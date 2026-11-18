"""Optimizers, as in `tinygrad.nn.optim`. The update rules are those of
`minigrad/nn/optim.mojo`."""

from __future__ import annotations

from ..tensor import Tensor


class Optimizer:
    """Trains `params`: those not marked `requires_grad = False`, with
    unmarked ones turned on. Marked ones (batch normalization's running
    estimates) are realized with every step."""

    def __init__(self, params: list[Tensor], lr: float):
        for p in params:
            if p.requires_grad is None:
                p.requires_grad_(True)
        self.params = [p for p in params if p.requires_grad]
        self.buffers = [p for p in params if p.requires_grad is False]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, *also: Tensor):
        """Updates the parameters and realizes them together with
        `also` (the loss, say), in one schedule."""
        out = list(also) + self.schedule_step()
        Tensor.realize(*out)

    def schedule_step(self) -> list[Tensor]:
        """Assigns the updated parameters without realizing them; returns
        the tensors to realize (`loss.realize(*opt.schedule_step())`)."""
        return self.buffers + self._step()

    def _step(self) -> list[Tensor]:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight
    decay."""

    def __init__(self, params, lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
        self.b = [p.zeros_like().contiguous().realize() for p in self.params] if momentum else []

    def _step(self) -> list[Tensor]:
        out = []
        for i, t in enumerate(self.params):
            g = t.grad
            if g is None:
                continue
            if self.wd:
                g = g + t.detach() * self.wd
            if self.momentum:
                self.b[i].assign(self.b[i] * self.momentum + g)
                g = g + self.b[i] * self.momentum if self.nesterov else self.b[i]
                out.append(self.b[i])
            t.assign(t.detach() - g * self.lr)
            out.append(t)
        return out


class Adam(Optimizer):
    """Adam, and AdamW when `weight_decay` is nonzero (decoupled)."""

    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
        # Powers of b1 and b2 live in tensors, so every step's graph is
        # the same and its schedule is reused.
        self.b1_t = Tensor.full((), 1.0).contiguous().realize()
        self.b2_t = Tensor.full((), 1.0).contiguous().realize()
        self.m = [p.zeros_like().contiguous().realize() for p in self.params]
        self.v = [p.zeros_like().contiguous().realize() for p in self.params]

    def _step(self) -> list[Tensor]:
        self.b1_t.assign(self.b1_t * self.b1)
        self.b2_t.assign(self.b2_t * self.b2)
        out = [self.b1_t, self.b2_t]
        for i, t in enumerate(self.params):
            g = t.grad
            if g is None:
                continue
            self.m[i].assign(self.m[i] * self.b1 + g * (1.0 - self.b1))
            self.v[i].assign(self.v[i] * self.b2 + (g * g) * (1.0 - self.b2))
            m_hat = self.m[i] / (1.0 - self.b1_t)
            v_hat = self.v[i] / (1.0 - self.b2_t)
            up = m_hat / (v_hat.sqrt() + self.eps)
            if self.wd:
                up = up + t.detach() * self.wd
            t.assign(t.detach() - up * self.lr)
            out += [self.m[i], self.v[i], t]
        return out


def AdamW(params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01) -> Adam:
    return Adam(params, lr, b1, b2, eps, weight_decay)
