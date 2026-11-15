"""Reverse-mode differentiation as a graph transformation
(`minigrad.gradient`).

`compute_gradients` walks the graph below a root from the root down, and
for each node builds the nodes computing the gradient with respect to its
inputs from the gradient with respect to its output. Nothing is executed
and nothing is recorded during the forward pass: gradients are ordinary
lazy nodes, scheduled and fused like any others (and differentiable
again).

The rules are tinygrad's, restated for this operation set.
"""

from std.math import pi

from minigrad.dtype import DT
from minigrad.graph import (
    graph,
    OP_ADD,
    OP_CAST,
    OP_CMPEQ,
    OP_CMPLT,
    OP_CONTIGUOUS,
    OP_CONV,
    OP_DIV,
    OP_EXP,
    OP_EXPAND,
    OP_FLIP,
    OP_LOG,
    OP_MAX,
    OP_MUL,
    OP_NEG,
    OP_PAD,
    OP_PERMUTE,
    OP_POW,
    OP_RECIP,
    OP_REDUCE,
    OP_RESHAPE,
    OP_SHRINK,
    OP_SIN,
    OP_SQRT,
    OP_SUB,
    OP_WHERE,
    OP_WINDOW,
    REDUCE_SUM,
)
from minigrad.tensor import Tensor
from minigrad.uop import UOp


def _mul(a: UOp, b: UOp) raises -> UOp:
    return a.alu2(OP_MUL, b)


def _window_grad(ctx: UOp, ret: UOp) raises -> UOp:
    """The gradient with respect to the input of sliding windows: each
    input element collects the gradient of every window element that
    reads it. One term per offset within the window: that offset's slice
    of `ctx`, spread out by the stride and shifted into place by
    padding."""
    var arg = ret.arg()
    var nk = arg[0]
    var in_shape = ret.src(0).shape()
    var out_shape = ret.shape()
    var p = len(in_shape) - nk
    var total: Optional[UOp] = None
    var kappa = List[Int](length=nk, fill=0)
    while True:
        # The slice of ctx at window offset kappa, without the offset
        # dimensions.
        var sh = List[Int]()
        var cur = List[Int]()
        for i in range(p + nk):
            sh.append(0)
            sh.append(out_shape[i])
            cur.append(out_shape[i])
        for j in range(nk):
            sh.append(kappa[j])
            sh.append(kappa[j] + 1)
        var t = ctx.shrink(sh).reshape(cur)
        for j in range(nk):
            var s = arg[1 + nk + j]
            var d = arg[1 + 2 * nk + j]
            var o = out_shape[p + j]
            var length = (o - 1) * s + 1
            if s > 1:
                # Spread positions s apart: (o) -> (o, 1) -> pad to
                # (o, s) -> (o s) -> the first (o - 1) s + 1.
                var split = List[Int]()
                var pad = List[Int]()
                var keep = List[Int]()
                for i in range(len(cur)):
                    split.append(cur[i])
                    pad.append(0)
                    pad.append(0)
                    keep.append(0)
                    keep.append(length if i == p + j else cur[i])
                    if i == p + j:
                        split.append(1)
                        pad.append(0)
                        pad.append(s - 1)
                cur[p + j] = o * s
                t = t.reshape(split).pad(pad).reshape(cur).shrink(keep)
                cur[p + j] = length
            var before = kappa[j] * d
            var after = in_shape[p + j] - length - before
            if before != 0 or after != 0:
                var pad = List[Int](length=2 * len(cur), fill=0)
                pad[2 * (p + j)] = before
                pad[2 * (p + j) + 1] = after
                t = t.pad(pad)
                cur[p + j] = in_shape[p + j]
        if total:
            total = total.value().alu2(OP_ADD, t)
        else:
            total = t
        # Next offset.
        var j = nk - 1
        while j >= 0:
            kappa[j] += 1
            if kappa[j] < arg[1 + j]:
                break
            kappa[j] = 0
            j -= 1
        if j < 0:
            break
    return total.value()


def _conv_input_grad(ctx: UOp, ret: UOp) raises -> UOp:
    """The gradient of a convolution with respect to its input: the
    transposed convolution. The output gradient is spread out by the
    stride, padded by the kernel's extent, and convolved (stride 1) with
    the weight flipped and its channel dimensions swapped."""
    var x = Tensor(ret.src(1))
    var w = Tensor(ret.src(2))
    var arg = ret.arg()
    var groups = arg[0]
    var stride = arg[1]
    var dilation = arg[2]
    var padding = arg[3]
    var xs = x.shape()
    var ws = w.shape()
    var nd = len(ws) - 2
    var rcout = ws[0] // groups
    var cin = ws[1]
    var t = Tensor(ctx)
    var cur = t.shape()
    for j in range(nd):
        var dim = 2 + j
        var o = cur[dim]
        if stride > 1:
            # (o) -> (o, 1) -> pad to (o, stride) -> (o stride) -> the
            # first (o - 1) stride + 1.
            var split = cur.copy()
            split.insert(dim + 1, 1)
            var pad = List[Int](length=2 * len(split), fill=0)
            pad[2 * (dim + 1) + 1] = stride - 1
            cur[dim] = o * stride
            t = t.reshape(split).pad(pad).reshape(cur)
            cur[dim] = (o - 1) * stride + 1
            t = t.slice(dim, 0, cur[dim])
    var pad = List[Int](length=2 * (2 + nd), fill=0)
    for j in range(nd):
        var extent = dilation * (ws[2 + j] - 1)
        pad[2 * (2 + j)] = extent
        pad[2 * (2 + j) + 1] = extent
    # (cout, cin, k...) -> (groups cin, rcout, k...), kernel reversed.
    var grouped: List[Int] = [groups, rcout, cin]
    var swapped: List[Int] = [groups * cin, rcout]
    for j in range(nd):
        grouped.append(ws[2 + j])
        swapped.append(ws[2 + j])
    var wt = w.reshape(grouped).transpose(1, 2).reshape(swapped)
    var flags = List[Int](length=2 + nd, fill=0)
    for j in range(nd):
        flags[2 + j] = 1
    wt = Tensor(wt.uop().flip(flags))
    var gx = t.pad(pad).conv2d(wt, None, groups, 1, dilation, 0)
    # Input the forward pass never read (the stride left it over) gets
    # zero; then drop the forward pass's padding.
    for j in range(nd):
        var dim = 2 + j
        var have = gx.shape()[dim]
        var full = xs[dim] + 2 * padding
        if full > have:
            var tail = List[Int](length=2 * (2 + nd), fill=0)
            tail[2 * dim + 1] = full - have
            gx = gx.pad(tail)
        gx = gx.slice(dim, padding, padding + xs[dim])
    return gx.uop()


def _conv_weight_grad(ctx: UOp, ret: UOp) raises -> UOp:
    """The gradient of a convolution with respect to its weight: for
    each kernel offset, the windows of the input times the output
    gradient, summed over the batch and the window positions."""
    var x = Tensor(ret.src(1))
    var w = Tensor(ret.src(2))
    var arg = ret.arg()
    var groups = arg[0]
    var ws = w.shape()
    var nd = len(ws) - 2
    var bs = x.shape()[0]
    var rcout = ws[0] // groups
    var cin = ws[1]
    var kernel = List[Int]()
    var strides = List[Int]()
    var dilations = List[Int]()
    var pad = List[Int](length=4, fill=0)
    for j in range(nd):
        kernel.append(ws[2 + j])
        strides.append(arg[1])
        dilations.append(arg[2])
        pad.append(arg[3])
        pad.append(arg[3])
    # (bs, groups cin, o..., k...)
    var xw = x.pad(pad)._pool(kernel, strides, dilations)
    var o = List[Int]()
    for j in range(nd):
        o.append(xw.shape()[2 + j])
    # Both to (bs, groups, rcout, cin, o..., k...).
    var xr: List[Int] = [bs, groups, 1, cin]
    var cr: List[Int] = [bs, groups, rcout, 1]
    var full: List[Int] = [bs, groups, rcout, cin]
    var axes: List[Int] = [0]
    for j in range(nd):
        xr.append(o[j])
        cr.append(o[j])
        full.append(o[j])
        axes.append(4 + j)
    for j in range(nd):
        xr.append(kernel[j])
        cr.append(1)
        full.append(kernel[j])
    var prod = xw.reshape(xr).expand(full) * Tensor(ctx).reshape(cr).expand(
        full
    )
    return prod._reduce(REDUCE_SUM, axes, True).reshape(ws).uop()


def _input_grads(ctx: UOp, ret: UOp) raises -> List[Optional[UOp]]:
    """The gradients with respect to the inputs of `ret`, given the
    gradient `ctx` with respect to `ret`; `None` where there is none."""
    var out = List[Optional[UOp]]()
    var op = ret.op()
    if op == OP_RESHAPE:
        out.append(ctx.reshape(ret.src(0).shape()))
    elif op == OP_PERMUTE:
        var axes = ret.arg()
        var inv = List[Int](length=len(axes), fill=0)
        for i in range(len(axes)):
            inv[axes[i]] = i
        out.append(ctx.permute(inv))
    elif op == OP_EXPAND:
        var src_shape = ret.src(0).shape()
        var ret_shape = ret.shape()
        var axes = List[Int]()
        for d in range(len(src_shape)):
            if src_shape[d] != ret_shape[d]:
                axes.append(d)
        out.append(ctx.reduce(REDUCE_SUM, axes))
    elif op == OP_PAD:
        var arg = ret.arg()
        var src_shape = ret.src(0).shape()
        var sh = List[Int]()
        for d in range(len(src_shape)):
            sh.append(arg[2 * d])
            sh.append(arg[2 * d] + src_shape[d])
        out.append(ctx.shrink(sh))
    elif op == OP_SHRINK:
        var arg = ret.arg()
        var src_shape = ret.src(0).shape()
        var pd = List[Int]()
        for d in range(len(src_shape)):
            pd.append(arg[2 * d])
            pd.append(src_shape[d] - arg[2 * d + 1])
        out.append(ctx.pad(pd))
    elif op == OP_FLIP:
        out.append(ctx.flip(ret.arg()))
    elif op == OP_WINDOW:
        out.append(_window_grad(ctx, ret))
    elif op == OP_NEG:
        out.append(ctx.alu1(OP_NEG))
    elif op == OP_EXP:
        out.append(_mul(ctx, ret))
    elif op == OP_LOG:
        out.append(ctx.alu2(OP_DIV, ret.src(0)))
    elif op == OP_SIN:
        var x = ret.src(0)
        var cos = x.const_like(pi / 2).alu2(OP_SUB, x).alu1(OP_SIN)
        out.append(_mul(ctx, cos))
    elif op == OP_SQRT:
        out.append(ctx.alu2(OP_DIV, _mul(ret, ret.const_like(2.0))))
    elif op == OP_RECIP:
        out.append(_mul(_mul(ctx.alu1(OP_NEG), ret), ret))
    elif op == OP_CAST:
        var x = ret.src(0)
        if x.dtype().is_float():
            out.append(ctx.cast(x.dtype()))
        else:
            out.append(None)
    elif op == OP_ADD:
        out.append(ctx)
        out.append(ctx)
    elif op == OP_SUB:
        out.append(ctx)
        out.append(ctx.alu1(OP_NEG))
    elif op == OP_MUL:
        out.append(_mul(ctx, ret.src(1)))
        out.append(_mul(ctx, ret.src(0)))
    elif op == OP_DIV:
        var y = ret.src(1)
        out.append(ctx.alu2(OP_DIV, y))
        out.append(_mul(ctx.alu1(OP_NEG), ret).alu2(OP_DIV, y))
    elif op == OP_MAX:
        var x = ret.src(0)
        var y = ret.src(1)
        var zero = ctx.const_like(0.0)
        var half = _mul(ctx, ctx.const_like(0.5))
        var tie = x.alu2(OP_CMPEQ, y).where(half, zero)
        out.append(y.alu2(OP_CMPLT, x).where(ctx, tie))
        out.append(x.alu2(OP_CMPLT, y).where(ctx, tie))
    elif op == OP_POW:
        var b = ret.src(0)
        var e = ret.src(1)
        var zero = ctx.const_like(0.0)
        var one = ctx.const_like(1.0)
        # d/db b^e = e b^(e-1), and 0 when e is 0.
        var db = _mul(e, b.alu2(OP_POW, e.alu2(OP_SUB, one)))
        out.append(_mul(ctx, e.alu2(OP_CMPEQ, zero).where(zero, db)))
        # d/de b^e = b^e log(b); at b = 0 it is 0 for e >= 0, -inf below.
        var ninf = ctx.const_like(-1.0).alu2(OP_DIV, zero)
        var at_zero = e.alu2(OP_CMPLT, zero).where(ninf, zero)
        var de = _mul(ret, b.alu1(OP_LOG))
        out.append(_mul(ctx, b.alu2(OP_CMPEQ, zero).where(at_zero, de)))
    elif op == OP_WHERE:
        var c = ret.src(0)
        var zero = ctx.const_like(0.0)
        out.append(None)
        out.append(c.where(ctx, zero))
        out.append(c.where(zero, ctx))
    elif op == OP_REDUCE:
        var x = ret.src(0)
        var arg = ret.arg()
        var shape = x.shape()
        if arg[0] == REDUCE_SUM:
            out.append(ctx.expand(shape))
        else:
            # The gradient of a max is shared equally among the ties.
            var axes = List[Int]()
            for i in range(1, len(arg)):
                axes.append(arg[i])
            var mask = x.alu2(OP_CMPEQ, ret.expand(shape)).cast(ctx.dtype())
            var count = mask.reduce(REDUCE_SUM, axes).expand(shape)
            out.append(_mul(mask.alu2(OP_DIV, count), ctx.expand(shape)))
    elif op == OP_CONTIGUOUS:
        out.append(ctx)
    elif op == OP_CONV:
        # Not through s0, the operations that compute the convolution:
        # differentiating those costs a tensor the size of the input
        # times the kernel.
        out.append(None)
        out.append(_conv_input_grad(ctx, ret))
        out.append(_conv_weight_grad(ctx, ret))
    return out^


def compute_gradients(
    root: UOp, root_grad: UOp, targets: List[Int]
) raises -> Dict[Int, UOp]:
    """Differentiates `root` with respect to nodes below it.

    Args:
        root: The node to differentiate.
        root_grad: The gradient with respect to `root`, of its shape.
        targets: The ids of the nodes to differentiate with respect to.

    Returns:
        The gradient of each target that `root` depends on, by node id.
        Targets it does not depend on are absent.
    """
    var g = graph()
    var order = g[].toposort([root.id])
    var n = len(order)
    # need[i]: some target is at or below order[i].
    var need = List[Bool](length=n, fill=False)
    var index = Dict[Int, Int]()
    for i in range(n):
        index[order[i]] = i
    for t in targets:
        var p = index.get(t, -1)
        if p >= 0:
            need[p] = True
    for i in range(n):
        if need[i]:
            continue
        var node = order[i]
        for j in range(3):
            var s = g[].nodes[node].s0
            if j == 1:
                s = g[].nodes[node].s1
            elif j == 2:
                s = g[].nodes[node].s2
            if s >= 0 and need[index.get(s, 0)]:
                need[i] = True

    var grads = Dict[Int, UOp]()
    grads[root.id] = root_grad
    for ii in range(n):
        var i = n - 1 - ii
        var node = order[i]
        if not need[i] or node not in grads:
            continue
        var ctx = grads[node]
        var ret = UOp(id=node)
        var gs = _input_grads(ctx, ret)
        for j in range(len(gs)):
            if not gs[j]:
                continue
            var s = ret.src(j)
            if not need[index.get(s.id, 0)]:
                continue
            if s.id in grads:
                grads[s.id] = grads[s.id].alu2(OP_ADD, gs[j].value())
            else:
                grads[s.id] = gs[j].value()
    var result = Dict[Int, UOp]()
    for t in targets:
        if t in grads:
            result[t] = grads[t]
    return result^
