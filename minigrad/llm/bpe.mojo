"""GPT-2's byte-level BPE tokenizer (`minigrad.llm.bpe`).

Encoding splits text into pieces with GPT-2's pattern

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

then turns each piece's bytes into tokens by applying merges in order of
rank. Everything after loading works on token ids: each byte starts as
its own token, and a merge maps a pair of ids to a rank and a new id.

The pattern is matched by hand. Letter, number, and whitespace classes
are exact for ASCII and Latin-1; elsewhere, code points in Unicode's
punctuation and symbol blocks count as neither letters nor numbers, and
everything else as a letter. That agrees with tiktoken on English text
and most Latin-script text, and may split rare scripts differently.
"""

from minigrad.json import JsonCursor, _append_utf8

comptime _OTHER = 0
comptime _LETTER = 1
comptime _NUMBER = 2
comptime _SPACE = 3


def _byte_to_codepoint() -> List[Int]:
    """GPT-2's map from bytes to printable code points: printable
    Latin-1 bytes map to themselves, the rest to 256, 257, ..."""
    var out = List[Int](length=256, fill=-1)
    for b in range(256):
        if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174:
            out[b] = b
    var n = 0
    for b in range(256):
        if out[b] < 0:
            out[b] = 256 + n
            n += 1
    return out^


def _class(cp: Int) -> Int:
    """The class of a code point in GPT-2's pattern."""
    if cp < 128:
        if (cp >= 97 and cp <= 122) or (cp >= 65 and cp <= 90):
            return _LETTER
        if cp >= 48 and cp <= 57:
            return _NUMBER
        if cp == 32 or (cp >= 9 and cp <= 13):
            return _SPACE
        return _OTHER
    # Whitespace.
    if cp == 0x85 or cp == 0xA0 or cp == 0x1680 or cp == 0x2028:
        return _SPACE
    if (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2029 or cp == 0x202F:
        return _SPACE
    if cp == 0x205F or cp == 0x3000:
        return _SPACE
    # Latin-1: punctuation and symbols, except a few letters and numbers.
    if cp < 0xC0:
        if cp == 0xAA or cp == 0xB5 or cp == 0xBA:
            return _LETTER
        if (
            cp == 0xB2
            or cp == 0xB3
            or cp == 0xB9
            or (cp >= 0xBC and cp <= 0xBE)
        ):
            return _NUMBER
        return _OTHER
    if cp == 0xD7 or cp == 0xF7:
        return _OTHER
    # Punctuation and symbol blocks.
    if cp >= 0x2010 and cp <= 0x2BFF:  # general punctuation ... arrows
        if cp >= 0x2070 and cp <= 0x209F:  # super- and subscripts
            return _NUMBER
        if cp >= 0x2150 and cp <= 0x218F:  # number forms
            return _NUMBER
        if cp >= 0x2460 and cp <= 0x24FF:  # enclosed numbers
            return _NUMBER
        return _OTHER
    if cp >= 0x3001 and cp <= 0x303F:  # CJK punctuation
        return _OTHER
    if cp >= 0xFF01 and cp <= 0xFF0F:  # full-width punctuation
        return _OTHER
    if cp >= 0xFF10 and cp <= 0xFF19:  # full-width digits
        return _NUMBER
    if cp >= 0x1F000 and cp <= 0x1FAFF:  # emoji and pictographs
        return _OTHER
    if cp >= 0xFE00 and cp <= 0xFE0F:  # variation selectors
        return _OTHER
    if cp >= 0x0300 and cp <= 0x036F:  # combining marks
        return _OTHER
    if cp >= 0x0660 and cp <= 0x0669:  # Arabic-Indic digits
        return _NUMBER
    if cp >= 0x0966 and cp <= 0x096F:  # Devanagari digits
        return _NUMBER
    return _LETTER


def _decode_utf8(data: List[UInt8], pos: Int) -> Tuple[Int, Int]:
    """The code point at `pos` and its length in bytes. A malformed
    byte counts as a one-byte code point of that value."""
    var b0 = Int(data[pos])
    var n = len(data)
    if b0 < 0x80:
        return (b0, 1)
    if b0 >= 0xC0 and b0 < 0xE0 and pos + 1 < n:
        return (((b0 & 0x1F) << 6) | (Int(data[pos + 1]) & 0x3F), 2)
    if b0 >= 0xE0 and b0 < 0xF0 and pos + 2 < n:
        return (
            ((b0 & 0x0F) << 12)
            | ((Int(data[pos + 1]) & 0x3F) << 6)
            | (Int(data[pos + 2]) & 0x3F),
            3,
        )
    if b0 >= 0xF0 and pos + 3 < n:
        return (
            ((b0 & 0x07) << 18)
            | ((Int(data[pos + 1]) & 0x3F) << 12)
            | ((Int(data[pos + 2]) & 0x3F) << 6)
            | (Int(data[pos + 3]) & 0x3F),
            4,
        )
    return (b0, 1)


struct BPETokenizer(Movable):
    """GPT-2's tokenizer, read from Hugging Face's `vocab.json` and
    `merges.txt`."""

    var byte_token: List[Int]
    """The token of each single byte."""
    var token_bytes: List[List[UInt8]]
    """The bytes of each token."""
    var merge_rank: Dict[Int, Int]
    """For a pair of tokens `a * 65536 + b`, its merge's rank."""
    var merge_token: Dict[Int, Int]
    """For a pair of tokens `a * 65536 + b`, the token they merge into."""
    var specials: List[String]
    """Special tokens recognized in text: `<|endoftext|>`."""
    var special_ids: List[Int]
    var cache: Dict[String, List[Int]]
    """Tokens of pieces seen before."""

    comptime EOT = 50256
    """The id of `<|endoftext|>`."""

    def __init__(out self, vocab_path: String, merges_path: String) raises:
        """Reads the vocabulary and the merges.

        Args:
            vocab_path: `vocab.json`: a JSON object from token text (in
                GPT-2's printable byte encoding) to id.
            merges_path: `merges.txt`: one merge per line, `a b`, in rank
                order, after a `#version` line.
        """
        var to_cp = _byte_to_codepoint()
        var from_cp = Dict[Int, Int]()
        for b in range(256):
            from_cp[to_cp[b]] = b

        var vocab = Dict[String, Int]()
        var data: List[UInt8]
        with open(vocab_path, "r") as f:
            data = f.read_bytes()
        var cur = JsonCursor(data^)
        cur.expect("{")
        var max_id = -1
        while not cur.accept("}"):
            var key = cur.string()
            cur.expect(":")
            var id = cur.int()
            vocab[key] = id
            max_id = max(max_id, id)
            _ = cur.accept(",")

        self.token_bytes = List[List[UInt8]](
            length=max_id + 1, fill=List[UInt8]()
        )
        for entry in vocab.items():
            var bytes = List[UInt8]()
            var kb = entry.key.as_bytes()
            var kl = List[UInt8](capacity=len(kb))
            for b in kb:
                kl.append(b)
            var p = 0
            while p < len(kl):
                var cp_len = _decode_utf8(kl, p)
                var b = from_cp.get(cp_len[0], -1)
                if b < 0:
                    # Not in the byte encoding (cannot happen for GPT-2):
                    # keep the code point's own bytes.
                    for i in range(cp_len[1]):
                        bytes.append(kl[p + i])
                else:
                    bytes.append(UInt8(b))
                p += cp_len[1]
            self.token_bytes[entry.value] = bytes^

        self.byte_token = List[Int](length=256, fill=-1)
        for b in range(256):
            var s = List[UInt8]()
            _append_utf8(s, to_cp[b])
            var key = String(unsafe_from_utf8=Span(s))
            self.byte_token[b] = vocab.get(key, -1)
            if self.byte_token[b] < 0:
                raise Error(t"{vocab_path}: no token for byte {b}")

        self.merge_rank = Dict[Int, Int]()
        self.merge_token = Dict[Int, Int]()
        var text: String
        with open(merges_path, "r") as f:
            text = f.read()
        var rank = 0
        for line in text.splitlines():
            if line.startswith("#") or line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            if len(parts) != 2:
                continue
            var a = String(parts[0])
            var b = String(parts[1])
            var ia = vocab.get(a, -1)
            var ib = vocab.get(b, -1)
            var im = vocab.get(a + b, -1)
            if ia < 0 or ib < 0 or im < 0:
                raise Error(
                    t"{merges_path}: merge '{a} {b}' not in the vocabulary"
                )
            var key = ia * 65536 + ib
            if key not in self.merge_rank:
                self.merge_rank[key] = rank
                self.merge_token[key] = im
            rank += 1

        self.specials = ["<|endoftext|>"]
        self.special_ids = [vocab.get("<|endoftext|>", Self.EOT)]
        self.cache = Dict[String, List[Int]]()

    def vocab_size(self) -> Int:
        """The number of tokens."""
        return len(self.token_bytes)

    def _bpe(self, bytes: List[UInt8], lo: Int, hi: Int) -> List[Int]:
        """The tokens of `bytes[lo:hi]`."""
        var ids = List[Int](capacity=hi - lo)
        for i in range(lo, hi):
            ids.append(self.byte_token[Int(bytes[i])])
        while len(ids) > 1:
            # The pair with the lowest rank.
            var best = -1
            var best_rank = 1 << 62
            for i in range(len(ids) - 1):
                var r = self.merge_rank.get(ids[i] * 65536 + ids[i + 1], -1)
                if r >= 0 and r < best_rank:
                    best_rank = r
                    best = ids[i] * 65536 + ids[i + 1]
            if best < 0:
                break
            var merged = self.merge_token.get(best, 0)
            var out = List[Int](capacity=len(ids))
            var i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] * 65536 + ids[i + 1] == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out^
        return ids^

    def _piece(
        mut self, bytes: List[UInt8], lo: Int, hi: Int, mut out: List[Int]
    ):
        """Appends the tokens of the piece `bytes[lo:hi]`, through the
        cache."""
        var key = String(
            unsafe_from_utf8=Span[Byte, origin_of(bytes)](
                unsafe_ptr=bytes.unsafe_ptr().unsafe_offset(lo), length=hi - lo
            )
        )
        var found = self.cache.get(key)
        if found:
            for t in found.value():
                out.append(t)
            return
        var ids = self._bpe(bytes, lo, hi)
        for t in ids:
            out.append(t)
        if len(self.cache) < 100_000:
            self.cache[key] = ids^

    def _encode_ordinary(
        mut self, bytes: List[UInt8], lo: Int, hi: Int, mut out: List[Int]
    ):
        """Appends the tokens of `bytes[lo:hi]`, with no special tokens."""
        # Code points: their starting byte and class.
        var starts = List[Int]()
        var classes = List[Int]()
        var cps = List[Int]()
        var p = lo
        while p < hi:
            var cl = _decode_utf8(bytes, p)
            starts.append(p)
            cps.append(cl[0])
            classes.append(_class(cl[0]))
            p += cl[1]
        starts.append(hi)
        var n = len(cps)
        var i = 0
        while i < n:
            var c = cps[i]
            var end: Int
            # 's 't 're 've 'm 'll 'd
            if c == 39 and i + 1 < n:
                var c1 = cps[i + 1]
                var c2 = cps[i + 2] if i + 2 < n else 0
                if c1 == 115 or c1 == 116 or c1 == 109 or c1 == 100:
                    self._piece(bytes, starts[i], starts[i + 2], out)
                    i += 2
                    continue
                if (
                    (c1 == 114 and c2 == 101)
                    or (c1 == 118 and c2 == 101)
                    or (c1 == 108 and c2 == 108)
                ):
                    self._piece(bytes, starts[i], starts[i + 3], out)
                    i += 3
                    continue
            # An optional space, then a run of letters, numbers, or others.
            var j = i
            if c == 32 and i + 1 < n and classes[i + 1] != _SPACE:
                j = i + 1
            var cls = classes[j]
            if cls != _SPACE:
                end = j + 1
                while end < n and classes[end] == cls:
                    end += 1
                self._piece(bytes, starts[i], starts[end], out)
                i = end
                continue
            # Whitespace: the run, less its last character when that is
            # followed by something else.
            end = i + 1
            while end < n and classes[end] == _SPACE:
                end += 1
            if end < n and end - i > 1:
                end -= 1
            self._piece(bytes, starts[i], starts[end], out)
            i = end

    def encode(mut self, text: String) -> List[Int]:
        """The tokens of `text`; `<|endoftext|>` is the special token."""
        var bytes = List[UInt8]()
        for b in text.as_bytes():
            bytes.append(b)
        var out = List[Int]()
        var lo = 0
        var p = 0
        var n = len(bytes)
        while p < n:
            var matched = -1
            if bytes[p] == 60:  # <
                for s in range(len(self.specials)):
                    var sb = self.specials[s].as_bytes()
                    var ok = p + len(sb) <= n
                    if ok:
                        for k in range(len(sb)):
                            if bytes[p + k] != sb[k]:
                                ok = False
                                break
                    if ok:
                        matched = s
                        break
            if matched >= 0:
                self._encode_ordinary(bytes, lo, p, out)
                out.append(self.special_ids[matched])
                p += self.specials[matched].byte_length()
                lo = p
            else:
                p += 1
        self._encode_ordinary(bytes, lo, n, out)
        return out^

    def decode_bytes(self, ids: List[Int]) -> List[UInt8]:
        """The bytes of a sequence of tokens."""
        var out = List[UInt8]()
        for t in ids:
            if t >= 0 and t < len(self.token_bytes):
                for b in self.token_bytes[t]:
                    out.append(b)
        return out^

    def decode(self, ids: List[Int]) -> String:
        """The text of a sequence of tokens; a sequence that ends inside
        a multibyte character decodes with replacement characters."""
        var bytes = self.decode_bytes(ids)
        return String(from_utf8_lossy=Span(bytes))
