"""Element types, with tinygrad's names (`dtypes.float32`, ...)."""

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class DType:
    """An element type. `code` is the engine's number for it (`DT` in
    `minigrad/dtype.mojo`); `fmt` is its `struct` format character, or
    None for bfloat16, which Python has no format for."""

    code: int
    name: str
    itemsize: int
    fmt: str | None

    def __repr__(self) -> str:
        return f"dtypes.{self.name}"

    @property
    def is_float(self) -> bool:
        return self.code >= 9

    @property
    def is_int(self) -> bool:
        return 1 <= self.code <= 8


class dtypes:
    """The element types, as attributes (`dtypes.float32`)."""

    bool = DType(0, "bool", 1, "?")
    int8 = DType(1, "int8", 1, "b")
    uint8 = DType(2, "uint8", 1, "B")
    int16 = DType(3, "int16", 2, "h")
    uint16 = DType(4, "uint16", 2, "H")
    int32 = DType(5, "int32", 4, "i")
    uint32 = DType(6, "uint32", 4, "I")
    int64 = DType(7, "int64", 8, "q")
    uint64 = DType(8, "uint64", 8, "Q")
    float16 = DType(9, "float16", 2, "e")
    bfloat16 = DType(10, "bfloat16", 2, None)
    float32 = DType(11, "float32", 4, "f")
    float64 = DType(12, "float64", 8, "d")

    half = float16
    float = float32
    double = float64
    int = int32
    long = int64
    uchar = uint8
    char = int8

    default_float = float32
    default_int = int32

    ALL: tuple = ()

    @staticmethod
    def from_code(code: int) -> DType:
        return dtypes.ALL[code]

    @staticmethod
    def from_name(name: str) -> DType:
        for d in dtypes.ALL:
            if d.name == name:
                return d
        raise ValueError(f"no element type named {name!r}")

    @staticmethod
    def is_float(d: DType) -> bool:
        return d.is_float

    @staticmethod
    def is_int(d: DType) -> bool:
        return d.is_int


dtypes.ALL = tuple(
    getattr(dtypes, n)
    for n in (
        "bool int8 uint8 int16 uint16 int32 uint32 int64 uint64"
        " float16 bfloat16 float32 float64"
    ).split()
)


def to_dtype(d) -> DType:
    """A `DType` from a `DType`, a name, or a NumPy dtype."""
    if isinstance(d, DType):
        return d
    if isinstance(d, str):
        return dtypes.from_name(d)
    name = getattr(d, "name", None) or getattr(d, "__name__", None)
    if name is not None:
        return dtypes.from_name(str(name))
    raise TypeError(f"not an element type: {d!r}")
