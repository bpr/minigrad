"""Parameters of a model, and safetensors files, as in
`tinygrad.nn.state`."""

from __future__ import annotations

import ctypes
import json
import struct
from collections import OrderedDict

from .. import _ext
from ..dtype import dtypes
from ..tensor import Tensor


def get_state_dict(obj, prefix: str = "") -> dict[str, Tensor]:
    """Every tensor reachable from `obj` through attributes, lists and
    dicts, by dotted name."""
    if isinstance(obj, Tensor):
        return {prefix.strip("."): obj}
    if hasattr(obj, "_asdict"):
        return get_state_dict(obj._asdict(), prefix)
    if isinstance(obj, OrderedDict):
        return get_state_dict(dict(obj), prefix)
    if hasattr(obj, "__dict__"):
        return get_state_dict(obj.__dict__, prefix)
    out: dict[str, Tensor] = {}
    if isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            out.update(get_state_dict(x, f"{prefix}{i}."))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(get_state_dict(v, f"{prefix}{k}."))
    return out


def get_parameters(obj) -> list[Tensor]:
    """The tensors of `obj`, each once."""
    seen, out = set(), []
    for t in get_state_dict(obj).values():
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def load_state_dict(model, state_dict: dict[str, Tensor], strict: bool = True, verbose: bool = False):
    """Assigns each of the model's tensors the value of the same name."""
    mine = get_state_dict(model)
    if strict:
        missing = sorted(set(mine) - set(state_dict))
        if missing:
            raise KeyError(f"missing from the state dict: {missing}")
    for k, t in mine.items():
        if k not in state_dict:
            continue
        v = state_dict[k]
        if v.shape != t.shape:
            raise ValueError(f"shape mismatch for {k}: {v.shape} vs {t.shape}")
        t.assign(v.cast(t.dtype).contiguous()).realize()


_SAFETENSORS = {
    "BOOL": dtypes.bool, "I8": dtypes.int8, "U8": dtypes.uint8,
    "I16": dtypes.int16, "U16": dtypes.uint16, "I32": dtypes.int32,
    "U32": dtypes.uint32, "I64": dtypes.int64, "U64": dtypes.uint64,
    "F16": dtypes.float16, "BF16": dtypes.bfloat16, "F32": dtypes.float32,
    "F64": dtypes.float64,
}


def safe_load(fn: str) -> dict[str, Tensor]:
    """The tensors of a safetensors file, copied into the engine."""
    with open(fn, "rb") as f:
        data = bytearray(f.read())
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + n])
    base = ctypes.addressof((ctypes.c_char * len(data)).from_buffer(data)) + 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = _SAFETENSORS[info["dtype"]]
        start, _end = info["data_offsets"]
        out[name] = Tensor._new(_ext.from_memory(base + start, tuple(info["shape"]), dt.code))
    return out


def safe_save(tensors: dict[str, Tensor], fn: str):
    """Writes `tensors` as a safetensors file."""
    names = {v: k for k, v in _SAFETENSORS.items()}
    header, blobs, offset = {}, [], 0
    for k, t in tensors.items():
        b = bytes(t._memory().cast("B")) if t.dtype != dtypes.bfloat16 else _bf16_bytes(t)
        header[k] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(fn, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def _bf16_bytes(t: Tensor) -> bytes:
    h, address = _ext.memory(t._h)
    return ctypes.string_at(address, t.nbytes())
