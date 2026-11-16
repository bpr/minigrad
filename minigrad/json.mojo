"""A minimal JSON reader (`minigrad.json`).

Enough JSON for the files minigrad reads: safetensors headers and BPE
vocabularies. `JsonCursor` walks a byte buffer; callers ask for the value
they expect (`string()`, `int()`, `expect("{")`) or skip one
(`skip_value()`). There is no document tree.
"""


struct JsonCursor(Movable):
    """A position in a JSON text."""

    var data: List[UInt8]
    var pos: Int

    def __init__(out self, var data: List[UInt8]):
        """Starts at the beginning of `data`."""
        self.data = data^
        self.pos = 0

    def _fail(self, what: String) raises:
        raise Error(t"JSON: {what} at byte {self.pos}")

    def ws(mut self):
        """Skips whitespace."""
        while self.pos < len(self.data):
            var c = self.data[self.pos]
            if c == 32 or c == 9 or c == 10 or c == 13:
                self.pos += 1
            else:
                return

    def peek(mut self) -> UInt8:
        """The next byte after whitespace, or 0 at the end."""
        self.ws()
        if self.pos < len(self.data):
            return self.data[self.pos]
        return 0

    def accept(mut self, c: StaticString) -> Bool:
        """Consumes the one-byte token `c` if it comes next."""
        if self.peek() == c.as_bytes()[0]:
            self.pos += 1
            return True
        return False

    def expect(mut self, c: StaticString) raises:
        """Consumes the one-byte token `c`, which must come next."""
        if not self.accept(c):
            self._fail(String(t"expected '{c}'"))

    def string(mut self) raises -> String:
        """Reads a string, decoding escapes into UTF-8."""
        self.expect('"')
        var out = List[UInt8]()
        while True:
            if self.pos >= len(self.data):
                self._fail("unterminated string")
            var c = self.data[self.pos]
            self.pos += 1
            if c == 34:  # "
                break
            if c != 92:  # not a backslash
                out.append(c)
                continue
            if self.pos >= len(self.data):
                self._fail("unterminated escape")
            var e = self.data[self.pos]
            self.pos += 1
            if e == 117:  # u
                var cp = self._hex4()
                if cp >= 0xD800 and cp < 0xDC00:
                    # A surrogate pair.
                    if (
                        self.pos + 1 < len(self.data)
                        and self.data[self.pos] == 92
                        and self.data[self.pos + 1] == 117
                    ):
                        self.pos += 2
                        var lo = self._hex4()
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                _append_utf8(out, cp)
            elif e == 110:
                out.append(10)
            elif e == 116:
                out.append(9)
            elif e == 114:
                out.append(13)
            elif e == 98:
                out.append(8)
            elif e == 102:
                out.append(12)
            else:  # \" \\ \/
                out.append(e)
        return String(unsafe_from_utf8=Span(out))

    def _hex4(mut self) raises -> Int:
        if self.pos + 4 > len(self.data):
            self._fail("truncated \\u escape")
        var v = 0
        for _ in range(4):
            var c = Int(self.data[self.pos])
            self.pos += 1
            var d: Int
            if c >= 48 and c <= 57:
                d = c - 48
            elif c >= 97 and c <= 102:
                d = c - 87
            elif c >= 65 and c <= 70:
                d = c - 55
            else:
                d = 0
                self._fail("bad hex digit")
            v = v * 16 + d
        return v

    def int(mut self) raises -> Int:
        """Reads an integer."""
        self.ws()
        var neg = False
        if self.pos < len(self.data) and self.data[self.pos] == 45:
            neg = True
            self.pos += 1
        var start = self.pos
        var v = 0
        while self.pos < len(self.data):
            var c = Int(self.data[self.pos])
            if c < 48 or c > 57:
                break
            v = v * 10 + (c - 48)
            self.pos += 1
        if self.pos == start:
            self._fail("expected an integer")
        return -v if neg else v

    def int_list(mut self) raises -> List[Int]:
        """Reads an array of integers."""
        var out = List[Int]()
        self.expect("[")
        if self.accept("]"):
            return out^
        while True:
            out.append(self.int())
            if self.accept("]"):
                return out^
            self.expect(",")

    def skip_value(mut self) raises:
        """Skips one value of any kind."""
        var c = self.peek()
        if c == 34:
            _ = self.string()
        elif c == 123 or c == 91:  # { [
            var close = UInt8(125) if c == 123 else UInt8(93)
            self.pos += 1
            if self.peek() == close:
                self.pos += 1
                return
            while True:
                if c == 123:
                    _ = self.string()
                    self.expect(":")
                self.skip_value()
                if self.peek() == close:
                    self.pos += 1
                    return
                self.expect(",")
        else:
            # A number, true, false, or null.
            var start = self.pos
            while self.pos < len(self.data):
                var d = self.data[self.pos]
                if d == 44 or d == 125 or d == 93 or d == 32 or d == 10:
                    break
                self.pos += 1
            if self.pos == start:
                self._fail("expected a value")


def _append_utf8(mut out: List[UInt8], cp: Int):
    """Appends the UTF-8 encoding of code point `cp`."""
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
