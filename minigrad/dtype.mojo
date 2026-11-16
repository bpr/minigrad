"""Element data types (`minigrad.dtype`).

`DT` is a small integer code naming one of the element types a tensor can
hold. The codes are ordered by promotion priority, as tinygrad's dtypes
are, so `promote` is a lattice join computed from a table.
"""


struct DT(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """An element data type."""

    var code: Int
    """The type's index; see the `comptime` members."""

    comptime bool = DT(0)
    comptime int8 = DT(1)
    comptime uint8 = DT(2)
    comptime int16 = DT(3)
    comptime uint16 = DT(4)
    comptime int32 = DT(5)
    comptime uint32 = DT(6)
    comptime int64 = DT(7)
    comptime uint64 = DT(8)
    comptime float16 = DT(9)
    comptime bfloat16 = DT(10)
    comptime float32 = DT(11)
    comptime float64 = DT(12)

    comptime COUNT = 13
    """The number of data types."""

    comptime default_float = DT(11)
    """The type of a float literal: `float32`."""
    comptime default_int = DT(5)
    """The type of an integer literal: `int32`."""

    def __init__(out self, code: Int):
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code

    def itemsize(self) -> Int:
        """The size of one element in bytes."""
        if self.code <= 2:
            return 1
        if self.code <= 4 or self.code == 9 or self.code == 10:
            return 2
        if self.code <= 6 or self.code == 11:
            return 4
        return 8

    def is_float(self) -> Bool:
        """Whether this is a floating-point type."""
        return self.code >= 9

    def is_half(self) -> Bool:
        """Whether this is a 16-bit float: `float16` or `bfloat16`. They
        are for storage: arithmetic on them is done in `float32`."""
        return self.code == 9 or self.code == 10

    def is_bool(self) -> Bool:
        """Whether this is `bool`."""
        return self.code == 0

    def is_int(self) -> Bool:
        """Whether this is a signed or unsigned integer type (not `bool`)."""
        return self.code >= 1 and self.code <= 8

    def is_unsigned(self) -> Bool:
        """Whether this is an unsigned integer type."""
        return (
            self.code == 2
            or self.code == 4
            or self.code == 6
            or (self.code == 8)
        )

    def name(self) -> String:
        """The type's name, as in NumPy."""
        if self.code == 0:
            return "bool"
        if self.code == 1:
            return "int8"
        if self.code == 2:
            return "uint8"
        if self.code == 3:
            return "int16"
        if self.code == 4:
            return "uint16"
        if self.code == 5:
            return "int32"
        if self.code == 6:
            return "uint32"
        if self.code == 7:
            return "int64"
        if self.code == 8:
            return "uint64"
        if self.code == 9:
            return "float16"
        if self.code == 10:
            return "bfloat16"
        if self.code == 11:
            return "float32"
        return "float64"

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.name())

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("DT.", self.name())


def parse_dtype(name: String) raises -> DT:
    """The type named `name` (`"float32"`, `"bfloat16"`, ...).

    Raises:
        If there is no such type.
    """
    for c in range(DT.COUNT):
        if DT(c).name() == name:
            return DT(c)
    raise Error(t"unknown element type '{name}'")


def promote(a: DT, b: DT) -> DT:
    """The smallest type both `a` and `b` convert to without loss of kind.

    This is tinygrad's `least_upper_dtype` on its promotion lattice: a
    signed and an unsigned integer of the same width join at the next
    signed width, `int64` and `uint64` join at `float32`, `float16` and
    `bfloat16` join at `float32`, and any float absorbs any integer.

    Args:
        a: A data type.
        b: A data type.

    Returns:
        The join of `a` and `b`.
    """
    if a == b:
        return a
    if a.is_half() and b.is_half():
        return DT.float32  # neither holds the other's values
    if a.is_float() or b.is_float():
        return DT(max(a.code, b.code))
    if a.is_bool():
        return b
    if b.is_bool():
        return a
    # Both are integers of different types.
    if a.is_unsigned() == b.is_unsigned():
        return DT(max(a.code, b.code))
    var s = b if a.is_unsigned() else a
    var u = a if a.is_unsigned() else b
    if s.itemsize() > u.itemsize():
        return s
    # The unsigned type is at least as wide: the next signed width.
    if u == DT.uint8:
        return DT.int16
    if u == DT.uint16:
        return DT.int32
    if u == DT.uint32:
        return DT.int64
    return DT.float32


def sum_acc_dtype(dt: DT) -> DT:
    """The type a sum of `dt` elements accumulates in.

    As in tinygrad: `bool` and narrow integers widen to at least 32 bits,
    keeping their signedness, and 16-bit floats to `float32`; other types
    are unchanged.

    Args:
        dt: The element type being summed.

    Returns:
        The accumulator type.
    """
    if dt.is_bool():
        return DT.int32
    if dt.is_int() and dt.itemsize() < 4:
        return DT.uint32 if dt.is_unsigned() else DT.int32
    if dt.is_half():
        return DT.float32
    return dt
