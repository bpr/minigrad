"""Reading safetensors files (`minigrad.safetensors`).

The format: an 8-byte little-endian header length, a JSON header mapping
each tensor's name to its element type, shape, and byte range, then the
data. Each tensor is read straight from the file into a new buffer.
"""

from minigrad.dtype import DT
from minigrad.json import JsonCursor
from minigrad.tensor import Tensor


def _dtype(name: String) raises -> DT:
    if name == "F32":
        return DT.float32
    if name == "F64":
        return DT.float64
    if name == "I64":
        return DT.int64
    if name == "I32":
        return DT.int32
    if name == "I16":
        return DT.int16
    if name == "I8":
        return DT.int8
    if name == "U8":
        return DT.uint8
    if name == "BOOL":
        return DT.bool
    raise Error(t"safetensors: unsupported element type {name}")


def safe_load(
    path: String, skip_suffix: String = ""
) raises -> Dict[String, Tensor]:
    """Reads every tensor in a safetensors file.

    Args:
        path: The file.
        skip_suffix: Tensors whose names end with this are not read
            (empty: read all).

    Returns:
        Realized tensors by name.

    Raises:
        If the file is malformed, or holds an element type minigrad does
        not have (such as `F16`).
    """
    var out = Dict[String, Tensor]()
    with open(path, "r") as f:
        var head = f.read_bytes(8)
        if len(head) != 8:
            raise Error(t"{path}: not a safetensors file")
        var n = 0
        for i in range(8):
            n |= Int(head[i]) << (8 * i)
        var cur = JsonCursor(f.read_bytes(n))
        var base = 8 + n
        cur.expect("{")
        if cur.accept("}"):
            return out^
        while True:
            var name = cur.string()
            cur.expect(":")
            if name == "__metadata__" or (
                skip_suffix.byte_length() > 0 and name.endswith(skip_suffix)
            ):
                cur.skip_value()
            else:
                var dtype_name = String("")
                var shape = List[Int]()
                var offsets = List[Int]()
                cur.expect("{")
                while True:
                    var field = cur.string()
                    cur.expect(":")
                    if field == "dtype":
                        dtype_name = cur.string()
                    elif field == "shape":
                        shape = cur.int_list()
                    elif field == "data_offsets":
                        offsets = cur.int_list()
                    else:
                        cur.skip_value()
                    if cur.accept("}"):
                        break
                    cur.expect(",")
                var dt = _dtype(dtype_name)
                var count = 1
                for s in shape:
                    count *= s
                var nbytes = count * dt.itemsize()
                if len(offsets) != 2 or offsets[1] - offsets[0] != nbytes:
                    raise Error(t"{path}: bad byte range for {name}")
                var t = Tensor.empty(shape, dt)
                var p = t.data_ptr()
                _ = f.seek(base + offsets[0])
                var done = 0
                while done < nbytes:
                    var got = f.read(
                        Span[Byte, MutUntrackedOrigin](
                            unsafe_ptr=p.unsafe_offset(done),
                            length=nbytes - done,
                        )
                    )
                    if got <= 0:
                        raise Error(t"{path}: truncated data for {name}")
                    done += got
                out[name] = t
            if cur.accept("}"):
                break
            cur.expect(",")
    return out^
