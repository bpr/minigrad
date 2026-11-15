"""Optimizers (`minigrad.nn.optim`).

An optimizer holds copies of the parameter tensors (copies of a `Tensor`
are the same tensor, so the model sees the updates): `step()` assigns
each parameter its new value and realizes all of them in one schedule,
which is also what runs the forward and backward passes. Pass the loss
to `step` to have it computed in the same schedule: `opt.step([loss])`.

Tensors in the parameter list that are marked `requires_grad_(False)`
are not trained but are realized by every step. `BatchNorm` lists its
running estimates this way.

The update rules are tinygrad's.
"""

from minigrad.tensor import Tensor


def _trainable(params: List[Tensor]) -> List[Tensor]:
    """The parameters to train: those not marked `requires_grad_(False)`,
    with the unmarked ones turned on."""
    var out = List[Tensor]()
    for p in params:
        if p.cell[].requires_grad == -1:
            _ = p.requires_grad_()
        if p.requires_grad():
            out.append(p)
    return out^


def _buffers(params: List[Tensor]) -> List[Tensor]:
    """The tensors marked `requires_grad_(False)`: state a model updates
    itself, such as batch normalization's running estimates. A step
    realizes them along with the parameters."""
    var out = List[Tensor]()
    for p in params:
        if p.cell[].requires_grad == 0:
            out.append(p)
    return out^


struct SGD(Movable):
    """Stochastic gradient descent with optional momentum and weight
    decay."""

    var params: List[Tensor]
    var lr: Float64
    var momentum: Float64
    var weight_decay: Float64
    var nesterov: Bool
    var b: List[Tensor]
    """The momentum buffer of each parameter (empty without momentum)."""
    var buffers: List[Tensor]
    """Untrained tensors each step realizes."""

    def __init__(
        out self,
        params: List[Tensor],
        lr: Float64 = 0.001,
        momentum: Float64 = 0.0,
        weight_decay: Float64 = 0.0,
        nesterov: Bool = False,
    ) raises:
        """Turns on gradients for `params` and allocates momentum
        buffers.

        Args:
            params: The tensors to train.
            lr: The learning rate.
            momentum: The momentum coefficient; 0 disables it.
            weight_decay: The L2 penalty coefficient.
            nesterov: Whether to use Nesterov momentum.
        """
        self.params = _trainable(params)
        self.buffers = _buffers(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.b = List[Tensor]()
        if momentum != 0.0:
            for p in self.params:
                self.b.append(p.zeros_like().contiguous().realize())

    def zero_grad(self):
        """Discards every parameter's gradient."""
        for p in self.params:
            p.zero_grad()

    def step(self) raises:
        """Updates the parameters from their gradients and realizes
        them."""
        self.step(List[Tensor]())

    def step(self, also: List[Tensor]) raises:
        """Updates the parameters and realizes them together with the
        tensors in `also` (the loss, typically)."""
        var out = also.copy()
        for b in self.buffers:
            out.append(b)
        for i in range(len(self.params)):
            var t = self.params[i]
            if not t.has_grad():
                continue
            var g = t.grad()
            if self.weight_decay != 0.0:
                g = g + t.detach() * self.weight_decay
            if self.momentum != 0.0:
                _ = self.b[i].assign(self.b[i] * self.momentum + g)
                g = (
                    g + self.b[i] * self.momentum
                ) if self.nesterov else self.b[i]
                out.append(self.b[i])
            _ = t.assign(t.detach() - g * self.lr)
            out.append(t)
        Tensor.realize_all(out)


struct Adam(Movable):
    """Adam, and AdamW when `weight_decay` is nonzero (the decay is
    decoupled from the gradient)."""

    var params: List[Tensor]
    var lr: Float64
    var b1: Float64
    var b2: Float64
    var eps: Float64
    var weight_decay: Float64
    var b1_t: Float64
    """`b1` to the power of the number of steps taken."""
    var b2_t: Float64
    var m: List[Tensor]
    """The running mean of each parameter's gradient."""
    var v: List[Tensor]
    """The running mean of each parameter's squared gradient."""
    var buffers: List[Tensor]
    """Untrained tensors each step realizes."""

    def __init__(
        out self,
        params: List[Tensor],
        lr: Float64 = 0.001,
        b1: Float64 = 0.9,
        b2: Float64 = 0.999,
        eps: Float64 = 1e-8,
        weight_decay: Float64 = 0.0,
    ) raises:
        """Turns on gradients for `params` and allocates moment buffers.

        Args:
            params: The tensors to train.
            lr: The learning rate.
            b1: The decay of the gradient mean.
            b2: The decay of the squared-gradient mean.
            eps: Added to the denominator for stability.
            weight_decay: The decoupled weight decay coefficient.
        """
        self.params = _trainable(params)
        self.buffers = _buffers(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.b1_t = 1.0
        self.b2_t = 1.0
        self.m = List[Tensor]()
        self.v = List[Tensor]()
        for p in self.params:
            self.m.append(p.zeros_like().contiguous().realize())
            self.v.append(p.zeros_like().contiguous().realize())

    def zero_grad(self):
        """Discards every parameter's gradient."""
        for p in self.params:
            p.zero_grad()

    def step(mut self) raises:
        """Updates the parameters from their gradients and realizes
        them."""
        self.step(List[Tensor]())

    def step(mut self, also: List[Tensor]) raises:
        """Updates the parameters and realizes them together with the
        tensors in `also` (the loss, typically)."""
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        var out = also.copy()
        for b in self.buffers:
            out.append(b)
        for i in range(len(self.params)):
            var t = self.params[i]
            if not t.has_grad():
                continue
            var g = t.grad()
            _ = self.m[i].assign(self.m[i] * self.b1 + g * (1.0 - self.b1))
            _ = self.v[i].assign(
                self.v[i] * self.b2 + (g * g) * (1.0 - self.b2)
            )
            var m_hat = self.m[i] * (1.0 / (1.0 - self.b1_t))
            var v_hat = self.v[i] * (1.0 / (1.0 - self.b2_t))
            var up = m_hat / (v_hat.sqrt() + self.eps)
            if self.weight_decay != 0.0:
                up = up + t.detach() * self.weight_decay
            _ = t.assign(t.detach() - up * self.lr)
            out.append(self.m[i])
            out.append(self.v[i])
            out.append(t)
        Tensor.realize_all(out)


def AdamW(
    params: List[Tensor],
    lr: Float64 = 0.001,
    b1: Float64 = 0.9,
    b2: Float64 = 0.999,
    eps: Float64 = 1e-8,
    weight_decay: Float64 = 0.01,
) raises -> Adam:
    """Adam with decoupled weight decay; see `Adam`."""
    return Adam(params, lr, b1, b2, eps, weight_decay)
