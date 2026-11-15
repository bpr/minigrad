"""Strided views of a buffer (`minigrad.view`).

A `View` says how an n-dimensional index maps to an element of a flat
buffer: `offset + sum(index[d] * strides[d])`, valid only where every
`index[d]` lies inside the optional `mask`. Elements outside the mask read
as zero, which is how padding is represented without copying.

Movement operations (reshape, permute, expand, pad, shrink, flip) never
touch data: each maps a `View` to another `View`. All of them always
succeed except `reshape`, which returns `None` when the result is not
expressible as one strided view; the scheduler then materializes the
source and reshapes the copy.

The algebra follows tinygrad's original `View`.
"""


def prod(xs: List[Int]) -> Int:
    """The product of the entries of `xs` (1 when empty)."""
    var p = 1
    for x in xs:
        p *= x
    return p


def strides_for_shape(shape: List[Int]) -> List[Int]:
    """Row-major strides of a contiguous array, with 0 for size-1 dims.

    Args:
        shape: The array shape.

    Returns:
        One stride per dimension, in elements.
    """
    var n = len(shape)
    var strides = List[Int](length=n, fill=0)
    var acc = 1
    for i in range(n):
        var d = n - 1 - i
        strides[d] = acc if shape[d] != 1 else 0
        acc *= shape[d]
    return strides^


struct View(Copyable, Equatable, Hashable, Writable):
    """One strided, optionally masked, view of a flat buffer."""

    var shape: List[Int]
    """The size of each dimension."""
    var strides: List[Int]
    """Elements to step in the buffer per unit step of each index."""
    var offset: Int
    """The buffer element at index zero."""
    var mask: List[Int]
    """Empty, or `(lo, hi)` per dimension flattened to `2 * ndim` entries:
    an index is valid when `lo <= index[d] < hi` for every `d`."""
    var contiguous: Bool
    """Whether this is the identity view of a row-major array."""

    def __init__(
        out self,
        var shape: List[Int],
        var strides: List[Int],
        offset: Int,
        var mask: List[Int],
    ):
        """Builds a view in canonical form.

        Size-1 dimensions get stride 0, a mask covering everything is
        dropped, and a dimension whose mask keeps a single index gets
        stride 0 with the offset adjusted.

        Args:
            shape: The size of each dimension.
            strides: The stride of each dimension.
            offset: The buffer element at index zero.
            mask: Empty, or `2 * ndim` entries of `(lo, hi)` pairs.
        """
        var n = len(shape)
        var off = offset
        for d in range(n):
            if shape[d] == 1:
                strides[d] = 0
        var has_zero = False
        for d in range(n):
            if shape[d] == 0:
                has_zero = True
        if has_zero:
            for d in range(n):
                strides[d] = 0
            off = 0
            mask.clear()
        if len(mask) > 0:
            var full = True
            for d in range(n):
                if mask[2 * d] != 0 or mask[2 * d + 1] != shape[d]:
                    full = False
            if full:
                mask.clear()
        if len(mask) > 0:
            var empty = False
            var any_elim = False
            for d in range(n):
                if not (mask[2 * d] < mask[2 * d + 1]):
                    empty = True
                if not (mask[2 * d] + 1 < mask[2 * d + 1]):
                    any_elim = True
            if empty:
                for d in range(n):
                    strides[d] = 0
                    mask[2 * d] = 0
                    mask[2 * d + 1] = 0
                off = 0
            elif any_elim:
                for d in range(n):
                    if not (mask[2 * d] + 1 < mask[2 * d + 1]):
                        off += strides[d] * mask[2 * d]
                        strides[d] = 0
        var contiguous = off == 0 and len(mask) == 0
        if contiguous:
            var acc = 1
            for i in range(n):
                var d = n - 1 - i
                if strides[d] != (acc if shape[d] != 1 else 0):
                    contiguous = False
                acc *= shape[d]
        self.shape = shape^
        self.strides = strides^
        self.offset = off
        self.mask = mask^
        self.contiguous = contiguous

    @staticmethod
    def identity(shape: List[Int]) -> View:
        """The contiguous view of a row-major array of `shape`."""
        return View(shape.copy(), strides_for_shape(shape), 0, List[Int]())

    def ndim(self) -> Int:
        """The number of dimensions."""
        return len(self.shape)

    def size(self) -> Int:
        """The number of elements the view addresses."""
        return prod(self.shape)

    def has_mask(self) -> Bool:
        """Whether some indices are masked out (read as zero)."""
        return len(self.mask) > 0

    def permute(self, axes: List[Int]) -> View:
        """Reorders dimensions: dimension `i` of the result is `axes[i]`."""
        var n = len(axes)
        var shape = List[Int](capacity=n)
        var strides = List[Int](capacity=n)
        var mask = List[Int]()
        for i in range(n):
            shape.append(self.shape[axes[i]])
            strides.append(self.strides[axes[i]])
        if len(self.mask) > 0:
            for i in range(n):
                mask.append(self.mask[2 * axes[i]])
                mask.append(self.mask[2 * axes[i] + 1])
        return View(shape^, strides^, self.offset, mask^)

    def expand(self, new_shape: List[Int]) -> View:
        """Broadcasts size-1 dimensions to the sizes in `new_shape`."""
        var n = len(new_shape)
        for d in range(n):
            if self.shape[d] == 0:
                return View.identity(new_shape)
        var mask = List[Int]()
        if len(self.mask) > 0:
            for d in range(n):
                var lo = self.mask[2 * d]
                var hi = self.mask[2 * d + 1]
                if self.shape[d] != new_shape[d]:
                    if lo == 0 and hi == 1:
                        hi = new_shape[d]
                    else:
                        lo = 0
                        hi = 0
                mask.append(lo)
                mask.append(hi)
        return View(new_shape.copy(), self.strides.copy(), self.offset, mask^)

    def _resize(self, arg: List[Int], var mask: List[Int]) -> View:
        """Restricts or extends each dimension to `[arg[2d], arg[2d+1])`,
        intersecting the moved old mask with `mask` (if any)."""
        var n = len(self.shape)
        var offset = self.offset
        for d in range(n):
            offset += self.strides[d] * arg[2 * d]
        if len(self.mask) > 0:
            var nmask = List[Int](capacity=2 * n)
            for d in range(n):
                var ax = arg[2 * d]
                var ay = arg[2 * d + 1]
                nmask.append(max(0, min(self.mask[2 * d] - ax, ay - ax)))
                nmask.append(max(0, min(self.mask[2 * d + 1] - ax, ay - ax)))
            if len(mask) > 0:
                for d in range(n):
                    mask[2 * d] = max(mask[2 * d], nmask[2 * d])
                    mask[2 * d + 1] = min(mask[2 * d + 1], nmask[2 * d + 1])
            else:
                mask = nmask^
        var shape = List[Int](capacity=n)
        for d in range(n):
            shape.append(arg[2 * d + 1] - arg[2 * d])
        return View(shape^, self.strides.copy(), offset, mask^)

    def pad(self, arg: List[Int]) -> View:
        """Pads with zeros: `arg[2d]` before and `arg[2d+1]` after
        dimension `d`."""
        var n = len(self.shape)
        var nonzero = False
        for x in arg:
            if x != 0:
                nonzero = True
        if not nonzero:
            return self.copy()
        var zvarg = List[Int](capacity=2 * n)
        var mask = List[Int](capacity=2 * n)
        for d in range(n):
            zvarg.append(-arg[2 * d])
            zvarg.append(self.shape[d] + arg[2 * d + 1])
            mask.append(arg[2 * d])
            mask.append(self.shape[d] + arg[2 * d])
        return self._resize(zvarg, mask^)

    def shrink(self, arg: List[Int]) -> View:
        """Keeps indices `[arg[2d], arg[2d+1])` of each dimension `d`."""
        return self._resize(arg, List[Int]())

    def flip(self, axes: List[Int]) -> View:
        """Reverses each dimension `d` with `axes[d] != 0`."""
        var n = len(self.shape)
        var strides = List[Int](capacity=n)
        var offset = self.offset
        var mask = List[Int]()
        for d in range(n):
            if axes[d] != 0:
                strides.append(-self.strides[d])
                offset += (self.shape[d] - 1) * self.strides[d]
            else:
                strides.append(self.strides[d])
        if len(self.mask) > 0:
            for d in range(n):
                if axes[d] != 0:
                    mask.append(self.shape[d] - self.mask[2 * d + 1])
                    mask.append(self.shape[d] - self.mask[2 * d])
                else:
                    mask.append(self.mask[2 * d])
                    mask.append(self.mask[2 * d + 1])
        return View(self.shape.copy(), strides^, offset, mask^)

    def window(
        self, kernel: List[Int], stride: List[Int], dilation: List[Int]
    ) -> Optional[View]:
        """Sliding windows over the last `len(kernel)` dimensions.

        Each of those dimensions, of size `i`, becomes two: the window
        position, of size `o = (i - d (k - 1) - 1) // s + 1`, and the
        offset within the window, of size `k`. The result's shape is the
        leading dimensions, then every `o`, then every `k`; the element
        at position `p` and offset `q` is the input's at `p * s + q * d`.

        Args:
            kernel: The window size `k` per dimension.
            stride: The step `s` between windows per dimension.
            dilation: The step `d` between window elements per dimension.

        Returns:
            The view, or `None` if this view is masked: whether a window
            element is valid then depends on position and offset
            together, which a per-dimension mask cannot say.
        """
        if self.has_mask():
            return None
        var nk = len(kernel)
        var p = len(self.shape) - nk
        var shape = List[Int]()
        var strides = List[Int]()
        for i in range(p):
            shape.append(self.shape[i])
            strides.append(self.strides[i])
        for j in range(nk):
            var size = self.shape[p + j]
            shape.append(
                (size - dilation[j] * (kernel[j] - 1) - 1) // stride[j] + 1
            )
            strides.append(self.strides[p + j] * stride[j])
        for j in range(nk):
            shape.append(kernel[j])
            strides.append(self.strides[p + j] * dilation[j])
        return View(shape^, strides^, self.offset, List[Int]())

    def reshape(self, new_shape: List[Int]) -> Optional[View]:
        """Reinterprets the view's elements, in row-major order, as an
        array of `new_shape`.

        Args:
            new_shape: The new shape, with the same number of elements.

        Returns:
            The reshaped view, or `None` when it is not one strided view
            of the buffer (for example, flattening a transposed matrix).
        """
        if self.shape == new_shape:
            return self.copy()
        for s in self.shape:
            if s == 0:
                return View.identity(new_shape)
        if self.contiguous:
            return View.identity(new_shape)

        var merged = _merge_dims(self.shape, self.strides, self.mask)
        var nnew = len(new_shape)
        var rstrides = List[Int](capacity=nnew)
        var r = nnew - 1
        var ok = True
        for i in range(len(merged) // 3):
            var m = len(merged) // 3 - 1 - i
            var merged_dim = merged[3 * m]
            var new_stride = merged[3 * m + 1]
            var real_dim = merged[3 * m + 2]
            var acc = 1
            while acc < merged_dim and r >= 0:
                var new_dim = new_shape[r]
                r -= 1
                rstrides.append(new_stride)
                if new_dim != 1:
                    acc *= new_dim
                    new_stride *= new_dim if acc < real_dim else 0
            if acc != merged_dim:
                ok = False
                break
        if not ok:
            return None
        while len(rstrides) < nnew:
            rstrides.append(0)
        var new_mask = _reshape_mask(self.mask, self.shape, new_shape)
        if not new_mask:
            return None
        ref nm = new_mask.value()
        var new_strides = List[Int](capacity=nnew)
        for d in range(nnew):
            var st = rstrides[nnew - 1 - d]
            new_strides.append(0 if nm[2 * d + 1] - nm[2 * d] == 1 else st)
        var extra = 0
        if len(self.mask) > 0:
            for d in range(len(self.shape)):
                extra += self.mask[2 * d] * self.strides[d]
        for d in range(nnew):
            extra -= nm[2 * d] * new_strides[d]
        return View(
            new_shape.copy(), new_strides^, self.offset + extra, nm.copy()
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "View(shape=",
            self.shape,
            ", strides=",
            self.strides,
            ", offset=",
            self.offset,
            ", mask=",
            self.mask,
            ")",
        )


def _merge_dims(
    shape: List[Int], strides: List[Int], mask: List[Int]
) -> List[Int]:
    """Merges adjacent dimensions that step through the buffer as one.

    Returns triples `(merged size, stride, merged size without the
    zero-strided part)`, flattened.
    """
    var ret = List[Int]()
    var n = len(shape)
    if n == 0:
        return ret^
    var has_mask = len(mask) > 0
    ret.append(shape[0])
    ret.append(strides[0])
    ret.append(shape[0] if strides[0] != 0 else 0)
    var merging = (mask[1] - mask[0] == 1) if has_mask else shape[0] == 1
    for i in range(1, n):
        var s = shape[i]
        var st = strides[i]
        if s == 1:
            continue
        var k = len(ret) - 3
        var last_s = ret[k]
        var last_st = ret[k + 1]
        var last_pre = ret[k + 2]
        if merging or last_st == s * st:
            ret[k] = last_s * s
            ret[k + 1] = st
            if st != 0:
                ret[k + 2] = s if merging else last_pre * s
            else:
                ret[k + 2] = 0
        else:
            ret.append(s)
            ret.append(st)
            ret.append(s if st != 0 else 0)
        merging = (mask[2 * i + 1] - mask[2 * i] == 1) if has_mask else s == 1
    return ret^


def _reshape_mask(
    mask: List[Int], old_shape: List[Int], new_shape: List[Int]
) -> Optional[List[Int]]:
    """The mask of a reshaped view, or `None` when the reshape cuts across
    the valid region."""
    var nnew = len(new_shape)
    var nold = len(old_shape)
    var out = List[Int](length=2 * nnew, fill=0)
    if len(mask) == 0:
        for d in range(nnew):
            out[2 * d + 1] = new_shape[d]
        return out^
    for d in range(nold):
        if mask[2 * d + 1] - mask[2 * d] < 1:
            return out^  # nothing is valid

    # Walk both shapes from the last dimension.
    var ro = nold - 1  # next old dimension to take
    var rn = nnew - 1  # next new dimension to take
    var curr_stride = 1
    var old_dim = 1
    var l = 0
    var r = 1
    if ro >= 0:
        old_dim = old_shape[ro]
        l = mask[2 * ro]
        r = mask[2 * ro + 1]
        ro -= 1
    var new_dim = 1
    if rn >= 0:
        new_dim = new_shape[rn]
        rn -= 1
    var filled = 0  # new mask entries written, from the last dimension
    while filled < nnew:
        var next_stride = new_dim * curr_stride
        if old_dim >= next_stride:
            if old_dim == next_stride:
                out[2 * (nnew - 1 - filled)] = l // curr_stride
                out[2 * (nnew - 1 - filled) + 1] = (r - 1) // curr_stride + 1
                filled += 1
                curr_stride = 1
                old_dim = 1
                l = 0
                r = 1
                if ro >= 0:
                    old_dim = old_shape[ro]
                    l = mask[2 * ro]
                    r = mask[2 * ro + 1]
                    ro -= 1
                new_dim = 1
                if rn >= 0:
                    new_dim = new_shape[rn]
                    rn -= 1
            else:
                if (
                    (l % next_stride != 0 or r % next_stride != 0)
                    and l // next_stride != (r - 1) // next_stride
                ) or old_dim % next_stride != 0:
                    return None
                out[2 * (nnew - 1 - filled)] = l % next_stride // curr_stride
                out[2 * (nnew - 1 - filled) + 1] = (
                    r - 1
                ) % next_stride // curr_stride + 1
                filled += 1
                curr_stride = next_stride
                new_dim = 1
                if rn >= 0:
                    new_dim = new_shape[rn]
                    rn -= 1
        else:
            var nl = 0
            var nr = 1
            var nd = 1
            if ro >= 0:
                nl = mask[2 * ro]
                nr = mask[2 * ro + 1]
                nd = old_shape[ro]
                ro -= 1
            if not (l == 0 and r == old_dim) and nr - nl != 1:
                return None
            l = nl * old_dim + l
            r = (nr - 1) * old_dim + r
            old_dim = old_dim * nd
    # Leading old dimensions left over must be unmasked size-1 dims.
    while ro >= 0:
        if mask[2 * ro] != 0 or mask[2 * ro + 1] != 1:
            for d in range(nnew):
                out[2 * d] = 0
                out[2 * d + 1] = 0
            return out^
        ro -= 1
    return out^
