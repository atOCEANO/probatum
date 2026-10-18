"""Arrow IPC files written and read in the package rather than through an Arrow library,
which the core does not depend on (ADR 0044): a table of named columns, each UTC
timestamps in nanoseconds, float64 numbers, int32 or int64 integers, booleans, or UTF-8
text with 32-bit or 64-bit offsets, any of them with missing values, and string metadata
on its schema. Any Arrow reader opens what this writes.

The format's metadata is flatbuffers. They are written here front to back, each table's
vtable before it and everything it points at after it, which the format allows since a
table finds its vtable by a signed offset and every other offset points forward.
"""

import struct

import numpy as np

from ._errors import ProbatumValueError

MAGIC        = b"ARROW1"
CONTINUATION = b"\xff\xff\xff\xff"
# the format's numbers for its metadata version, its message kinds and the types a
# column here can be
V5 = 4
SCHEMA, RECORD_BATCH = 1, 3
INT, FLOAT, TEXT, BOOL, TIME, LARGE_TEXT = 2, 3, 5, 6, 10, 20
# each kind of column: its type's number, the fields of its type's table, and the
# format of its values, None for booleans, laid out as bits, and for text
KINDS = {
    "int":        (INT, [(0, "i", 32), (1, "?", True)], "<i4"),
    "int64":      (INT, [(0, "i", 64), (1, "?", True)], "<i8"),
    "float":      (FLOAT, [(0, "h", 2)], "<f8"),
    "time":       (TIME, [(0, "h", 3), (1, "ref", ("string", "UTC"))], "<i8"),
    "bool":       (BOOL, [], None),
    "text":       (TEXT, [], None),
    "large_text": (LARGE_TEXT, [], None),
}
BLOCK = "<qi4xq"


class Refused(ValueError):
    """A file this reader does not read, and what about it."""


def write(path, columns, metadata):
    """Write ``columns`` to the file ``path`` as one record batch: each column is
    (name, kind, values, present), and a fifth, whether the column can hold a null,
    where it cannot. Kind ``"time"`` takes int64 nanoseconds, ``"float"`` float64
    numbers, ``"int"`` and ``"int64"`` integers of those widths, ``"bool"`` booleans,
    ``"text"`` strings, and ``"large_text"`` the offset of each row into its UTF-8
    bytes and one past the last, with the bytes; present is a boolean mask, or None
    where every value is there; and ``metadata`` maps strings to strings on the
    schema."""
    rows = _rows(columns[0]) if columns else 0
    body, size, nodes, buffers = [], 0, [], []

    for name, kind, values, present, *_ in columns:
        missing = 0 if present is None else int(np.count_nonzero(~present))
        nodes.append((rows, missing))
        parts = [np.packbits(present, bitorder="little") if missing else b""]
        if kind == "large_text":
            ends, data = values
            parts += [np.ascontiguousarray(ends, dtype="<i8").view(np.uint8),
                      np.ascontiguousarray(data, dtype=np.uint8)]
        elif kind == "bool":
            parts.append(np.packbits(np.asarray(values, dtype=bool), bitorder="little"))
        elif kind == "text":
            data = [b"" if value is None else value.encode() for value in values]
            ends = np.cumsum([0] + [len(item) for item in data], dtype=np.int64)
            if ends[-1] > np.iinfo(np.int32).max:
                raise ProbatumValueError(f"{name}: {ends[-1]:,} bytes of text, past "
                                         f"the 2 GiB an Arrow text column holds; write "
                                         f"fewer rows")
            parts += [ends.astype("<i4").view(np.uint8), b"".join(data)]
        else:
            values = np.ascontiguousarray(values, dtype=KINDS[kind][2])
            parts.append(values.view(np.uint8))

        for part in parts:
            buffers.append((size, len(part)))
            body += [part, bytes(-len(part) % 8)]
            size += len(part) + (-len(part) % 8)

    schema = _schema(columns, metadata)
    batch  = ("table", [(0, "q", rows), (1, "ref", ("structs", "<qq", nodes)),
                        (2, "ref", ("structs", "<qq", buffers))])
    start  = MAGIC + bytes(2) + _framed(_message(SCHEMA, schema, 0))
    framed = _framed(_message(RECORD_BATCH, batch, size))
    footer = _flatbuffer(("table", [(0, "h", V5), (1, "ref", schema),
                                    (2, "ref", ("structs", BLOCK, [])),
                                    (3, "ref", ("structs", BLOCK,
                                                [(len(start), len(framed), size)]))]))

    with open(path, "wb") as file:
        # the record batch, then the end of the stream, before the footer
        file.write(start + framed)
        for part in body:
            file.write(part)
        file.write(CONTINUATION + bytes(4) + footer + struct.pack("<i", len(footer))
                   + MAGIC)


def read(path):
    """The columns of the Arrow IPC file ``path``, as a dict of name to (kind, values,
    present, nullable), values an array of the column's kind, a list of strings for
    text, and for large text its offsets from 0 and its bytes, present a boolean mask,
    and nullable whether its field says it can hold a null, and the metadata on its
    schema. It reads the types ``write`` writes, little-endian, uncompressed and on the
    8-byte boundaries the format puts every buffer on, and raises `Refused` for any
    other file."""
    with open(path, "rb") as file:
        raw = memoryview(file.read())

    try:
        return _read(raw)
    except Refused:
        raise
    except (struct.error, TypeError, IndexError, ValueError, StopIteration):
        raise Refused("is not an Arrow IPC file this reader can read") from None


def _read(raw):
    if len(raw) < 18 or raw[:6] != MAGIC or raw[-6:] != MAGIC:
        raise Refused("is not an Arrow IPC file")

    size   = struct.unpack_from("<i", raw, len(raw) - 10)[0]
    footer = _Table.root(raw[len(raw) - 10 - size:len(raw) - 10])
    schema = footer.table(1)
    if schema.scalar(0, "h") != 0:
        raise Refused("is big-endian, which this reader does not read")

    fields = []
    for field in schema.tables(1):
        name, kind = field.string(0), _kind(field.scalar(2, "B"), field.table(3))
        if field.ref(4) is not None or kind is None:
            raise Refused(f"holds a column {name!r} of a type this reader does not "
                          f"read")
        fields.append((name, kind, bool(field.scalar(1, "?"))))

    metadata = {pair.string(0): pair.string(1) for pair in schema.tables(2)}
    out      = {name: (kind, [], []) for name, kind, _ in fields}

    for offset, framed, length in footer.structs(3, BLOCK):
        message = _Table.root(raw[offset + 8:offset + framed])
        batch   = message.table(2) if message.scalar(1, "B") == RECORD_BATCH else None
        if batch is None or batch.ref(3) is not None:
            raise Refused("holds a compressed or unknown message")

        places = batch.structs(2, "<qq")
        if (offset + framed) % 8 or any(at % 8 for at, _ in places):
            raise Refused("holds a buffer off the 8-byte boundary the format puts it on")

        body    = raw[offset + framed:offset + framed + length]
        buffers = iter(places)
        for (name, kind, _), (count, missing) in zip(fields, batch.structs(1, "<qq")):
            at, size = next(buffers)
            present = (np.unpackbits(np.frombuffer(body, np.uint8, size, at),
                                     count=count, bitorder="little").astype(bool)
                       if missing and size else np.ones(count, dtype=bool))

            if kind == "large_text":
                (ends_at, _), (data_at, data_size) = next(buffers), next(buffers)
                ends = np.frombuffer(body, "<i8", count + 1, ends_at)
                data = np.frombuffer(body, np.uint8, data_size, data_at)
                if not 0 <= ends[0] <= ends[-1] <= data_size:
                    raise Refused(f"holds text in {name!r} whose offsets run past its "
                                  f"bytes")
                values = (ends - ends[0], data[ends[0]:ends[-1]])
            elif kind == "bool":
                at, size = next(buffers)
                values = np.unpackbits(np.frombuffer(body, np.uint8, size, at),
                                       count=count, bitorder="little").astype(bool)
            elif kind == "text":
                (ends_at, _), (data_at, data_size) = next(buffers), next(buffers)
                ends   = np.frombuffer(body, "<i4", count + 1, ends_at).tolist()
                data   = bytes(body[data_at:data_at + data_size])
                values = [data[a:b].decode() if there else None
                          for a, b, there in zip(ends, ends[1:], present.tolist())]
            else:
                at, _ = next(buffers)
                values = np.frombuffer(body, KINDS[kind][2], count, at)

            out[name][1].append(values)
            out[name][2].append(present)

    nullable = {name: can for name, _, can in fields}
    for name, (kind, parts, present) in out.items():
        if kind == "text":
            joined = [value for part in parts for value in part]
        elif kind == "large_text" and len(parts) == 1:
            joined = parts[0]
        elif kind == "large_text":
            ends, data, size = [np.zeros(1, dtype=np.int64)], [], 0
            for part_ends, part_data in parts:
                ends.append(part_ends[1:] + size)
                data.append(part_data)
                size += len(part_data)
            joined = (np.concatenate(ends),
                      np.concatenate(data) if data else np.zeros(0, dtype=np.uint8))
        elif len(parts) == 1:
            joined = parts[0]
        else:
            joined = (np.concatenate(parts) if parts
                      else np.zeros(0, dtype=KINDS[kind][2] or bool))

        out[name] = (kind, joined, present[0] if len(present) == 1
                     else np.concatenate(present) if present
                     else np.zeros(0, dtype=bool), nullable[name])

    return out, metadata


def _rows(column):
    """The rows of ``column``, as ``write`` takes it."""
    if column[1] == "large_text":
        return len(column[2][0]) - 1
    return len(column[2])


def _kind(number, detail):
    """The kind of a column of the type ``number`` with the type table ``detail``, or
    None where it is none this reads: a double, a signed integer of 32 or 64 bits, a
    boolean, text, or a time in nanoseconds."""
    if number == FLOAT:
        return "float" if detail.scalar(0, "h") == 2 else None

    if number == INT:
        bits = detail.scalar(0, "i") if detail.scalar(1, "?") else None
        return {32: "int", 64: "int64"}.get(bits)

    if number == TIME:
        return "time" if detail.scalar(0, "h") == 3 else None

    return {BOOL: "bool", TEXT: "text", LARGE_TEXT: "large_text"}.get(number)


def _schema(columns, metadata):
    fields = []
    for name, kind, _, _, *nullable in columns:
        number, detail, _ = KINDS[kind]
        fields.append(("table", [(0, "ref", ("string", name)),
                                 (1, "?", nullable[0] if nullable else True),
                                 (2, "B", number), (3, "ref", ("table", detail)),
                                 (5, "ref", ("tables", []))]))

    pairs = [("table", [(0, "ref", ("string", key)), (1, "ref", ("string", value))])
             for key, value in metadata.items()]
    return ("table", [(1, "ref", ("tables", fields)), (2, "ref", ("tables", pairs))])


def _message(kind, header, body):
    return _flatbuffer(("table", [(0, "h", V5), (1, "B", kind), (2, "ref", header),
                                  (3, "q", body)]))


def _framed(flatbuffer):
    """A message's metadata as a file holds it: a marker, its length, then itself."""
    return CONTINUATION + struct.pack("<i", len(flatbuffer)) + flatbuffer


def _flatbuffer(root):
    """``root`` and all it points at as a flatbuffer, padded to 8 bytes. An object is
    ("table", [(slot, format, value)]), its format a struct code, or "ref" for a value
    that is another object; ("string", text); ("tables", objects); or ("structs",
    format, rows), each row a tuple of an 8-byte aligned struct."""
    out = bytearray(4)
    _point(out, 0, _place(out, root))
    return bytes(out + bytes(-len(out) % 8))


def _place(out, item):
    """Write ``item`` at the end of ``out``, then what it points at, and return where
    it starts."""
    if item[0] == "string":
        _align(out, 4)
        at, data = len(out), item[1].encode()
        out += struct.pack("<I", len(data)) + data + b"\0"
        return at

    if item[0] == "structs":
        _align(out, 8, 4)
        at = len(out)
        out += struct.pack("<I", len(item[2]))
        for row in item[2]:
            out += struct.pack(item[1], *row)
        return at

    if item[0] == "tables":
        _align(out, 4)
        at = len(out)
        out += struct.pack("<I", len(item[1])) + bytes(4 * len(item[1]))
        for k, child in enumerate(item[1]):
            _point(out, at + 4 + 4 * k, _place(out, child))
        return at

    # a table: the widest fields first, so each sits aligned once the first does
    fields = sorted(item[1], key=lambda field: -_width(field[1]))
    offsets, end = {}, 4
    for slot, form, _ in fields:
        offsets[slot] = end
        end += _width(form)

    end += -end % 4
    slots = max(offsets, default=-1) + 1

    _align(out, 2)
    vtable = len(out)
    out += struct.pack(f"<HH{slots}H", 4 + 2 * slots, end,
                       *(offsets.get(slot, 0) for slot in range(slots)))

    _align(out, 8, 4)
    at = len(out)
    out += struct.pack("<i", at - vtable) + bytes(end - 4)

    for slot, form, value in fields:
        if form != "ref":
            struct.pack_into("<" + form, out, at + offsets[slot], value)

    for slot, form, value in fields:
        if form == "ref":
            _point(out, at + offsets[slot], _place(out, value))

    return at


def _width(form):
    return 4 if form == "ref" else struct.calcsize("<" + form)


def _align(out, size, past=0):
    """Pad ``out`` until its end sits ``past`` bytes beyond a multiple of ``size``."""
    out += bytes((past - len(out)) % size)


def _point(out, at, target):
    struct.pack_into("<I", out, at, target - at)


class _Table:
    """A flatbuffer table being read: its fields found through its vtable."""


    def __init__(self, data, at):
        self.data, self.at = data, at
        self.vtable = at - struct.unpack_from("<i", data, at)[0]
        self.size   = struct.unpack_from("<H", data, self.vtable)[0]


    @classmethod
    def root(cls, data):
        return cls(data, struct.unpack_from("<I", data, 0)[0])


    def field(self, slot):
        entry = 4 + 2 * slot
        if entry >= self.size:
            return None

        offset = struct.unpack_from("<H", self.data, self.vtable + entry)[0]
        return self.at + offset if offset else None


    def scalar(self, slot, form, default=0):
        at = self.field(slot)
        if at is None:
            return default
        return struct.unpack_from("<" + form, self.data, at)[0]


    def ref(self, slot):
        at = self.field(slot)
        return None if at is None else at + struct.unpack_from("<I", self.data, at)[0]


    def table(self, slot):
        return _Table(self.data, self.ref(slot))


    def string(self, slot):
        at   = self.ref(slot)
        size = struct.unpack_from("<I", self.data, at)[0]
        return bytes(self.data[at + 4:at + 4 + size]).decode()


    def tables(self, slot):
        at = self.ref(slot)
        if at is None:
            return []

        count = struct.unpack_from("<I", self.data, at)[0]
        return [_Table(self.data, at + 4 + 4 * k
                       + struct.unpack_from("<I", self.data, at + 4 + 4 * k)[0])
                for k in range(count)]


    def structs(self, slot, form):
        at = self.ref(slot)
        if at is None:
            return []

        count, size = struct.unpack_from("<I", self.data, at)[0], struct.calcsize(form)
        return [struct.unpack_from(form, self.data, at + 4 + size * k)
                for k in range(count)]
