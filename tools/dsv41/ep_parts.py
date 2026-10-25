#!/usr/bin/env python3
"""The expert-parallel parts pack (unit U3 of the dsv41 EP design, results/dsv41-perf-20261025/ep-design.md 4.4).

The worker PC needs the n_ff slice [k, 9) of every routed expert (a CPU lane that reads it in place) and the whole
experts of the hot set (the 3080's pool).  The pack stores an expert as three ggml matrices (w1 gate, w3 up, w2 down);
this tool cuts them at 256-value blocks and writes the cut pieces as contiguous "parts":

  gate, up   n_ff rows x dim: the slice [a, b) of n_ff is rows [256 a, 256 b), already contiguous
  down       dim rows x n_ff: the slice is blocks [a, b) of every row, strided in the pack; a part holds the
             dim rows, each cut to its blocks [a, b), packed back to back (so the CPU lane streams it)

A parts set is a file `ep-parts-<a>-<b>.bin`: a header page (EPPARTS1, the run stamp, the file's number and range),
then for each listed expert, in heat order, gate | up | down, every matrix aligned to 4096.  One index,
`ep-parts.txt`, describes all the sets written together (text, one line a part, each line after the lines it needs:
geometry and the files before the parts):

  ep-parts 1
  pack_hash <16 hex>        fnv1a of tensors.txt + ds4.txt as ds4_stage.hpp pack_hash computes it
  stamp <16 hex>            fnv1a of every other line of this index; also in each file's header page: an index and
                            files from different runs (a re-cut with another profile, a run that stopped half way)
                            do not open together
  nonce <16 hex>            random, one per run: it is a line of the index, so the stamp differs even when a re-cut
                            changes only the weights' bytes (same tensors.txt and ds4.txt, same heat, same layout)
  geometry <dim> <n_ff> <layers> <experts> <unit> <align>
  covered <layer>...        the layers that have parts (a subset of the pack's: other layers have no view)
  file <i> <name> <a> <b> <bytes>
  heat <layer> <expert> <count>     every listed expert once, hottest first (the order of the files)
  part <layer> <expert> <kind> <type> <file> <a> <b> <rows> <row_bytes> <offset> <bytes>
       kind gate|up|down, type the matrix's ggml type id; (rows x row_bytes) = bytes; a/b = the n_ff blocks held

So the worker's [6, 9) of every expert and the pool's [0, 6) of the hot experts are two sets of one index, and a whole
expert is its parts in both (ds4_epparts.hpp assembles it back, byte for byte).  Read-only on the pack.

  ep_parts.py PACK OUT --dry-run            sizes per k and the pool, nothing written
  ep_parts.py PACK OUT                      write the sets and the index
  ep_parts.py PACK OUT --verify             re-read the index and the pack: every part, every whole expert, byte-exact
  ep_parts.py PACK OUT --down-only 0:6      one set of the down matrices alone, their [0, 6) of every expert (the main's
                                            CPU lane: DS4_IQ_DOWN_PARTS=OUT reads its down slice there, rows contiguous)
"""
import argparse
import glob
import os
import secrets
import sys
import time

ALIGN = 4096
UNIT = 256            # n_ff is cut in blocks of 256 values (a superblock of every low-bit type below)
KINDS = ("gate", "up", "down")
ROLES = ("w1", "w3", "w2")
INDEX_NAME = "ep-parts.txt"
MAGIC = b"EPPARTS1"    # a part file starts with a header page: MAGIC, the run stamp (u64), file index, a, b (u32 each)
HEADER = ALIGN
GB = 1e9

# ggml type name -> (type id, values a block, bytes a block); the ids are ggml's (ds4_iq.cuh)
TYPES = {
    "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210),
    "IQ2_XXS": (16, 256, 66), "IQ2_XS": (17, 256, 74), "IQ3_XXS": (18, 256, 98), "IQ1_S": (19, 256, 50),
    "IQ4_NL": (20, 32, 18), "IQ3_S": (21, 256, 110), "IQ2_S": (22, 256, 82), "IQ4_XS": (23, 256, 136),
    "IQ1_M": (29, 256, 56), "MXFP4": (39, 32, 17),
}
TYPE_NAME = {v[0]: k for k, v in TYPES.items()}


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def set_file_name(a, b):
    return "ep-parts-%d-%d.bin" % (a, b)


# ---------------------------------------------------------------------------------------------- the pack (read-only)

def fnv1a(data, h):
    for c in data:
        h ^= c
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def pack_hash(pack_dir):
    """ds4_stage.hpp's pack_hash: each file's name, then its bytes line by line; ds4.txt's model_dir line and every
    '\\r' left out; 0 when neither file could be read."""
    h = 0
    for f in ("/tensors.txt", "/ds4.txt"):
        try:
            with open(pack_dir + f, "rb") as fh:
                body = fh.read()
        except OSError:
            continue
        h = fnv1a(f.encode(), h if h != 0 else 1469598103934665603)
        geometry = f == "/ds4.txt"
        chunks = body.split(b"\n")     # (not splitlines: it also breaks at a lone '\r', which the C++ side keeps inside its line)
        for line in [c + b"\n" for c in chunks[:-1]] + ([chunks[-1]] if chunks[-1] else []):
            if geometry and line.startswith(b"model_dir "):
                continue
            h = fnv1a(line.replace(b"\r", b""), h)
    return h


class Matrix:
    """One role of one layer's experts: its type and the row geometry (the same for every expert of the layer)."""

    def __init__(self, layer, kind, dtype, rows, k, nbytes):
        if dtype not in TYPES:
            raise ValueError("layer %d %s: ggml type %s is not one the parts pack knows" % (layer, KINDS[kind], dtype))
        self.layer, self.kind, self.dtype = layer, kind, dtype
        self.type_id, values, blk = TYPES[dtype]
        if k % UNIT != 0 or UNIT % values != 0:
            raise ValueError("layer %d %s: %d values a row is not a multiple of %d" % (layer, KINDS[kind], k, UNIT))
        self.rows = rows
        self.row_bytes = k // values * blk
        self.unit_bytes = UNIT // values * blk     # bytes of 256 values of a row
        self.nbytes = rows * self.row_bytes
        if nbytes != self.nbytes:
            raise ValueError("layer %d %s: tensors.txt says %d bytes, %s %dx%d is %d" %
                             (layer, KINDS[kind], nbytes, dtype, rows, k, self.nbytes))

    def part_shape(self, a, b):
        """(rows, row_bytes) of the part holding n_ff blocks [a, b)."""
        if self.kind < 2:
            return (b - a) * UNIT, self.row_bytes
        return self.rows, (b - a) * self.unit_bytes


class Pack:
    def __init__(self, pack_dir, model_dir=None):
        self.dir = pack_dir
        self.kv = {}
        with open(os.path.join(pack_dir, "ds4.txt"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if " " in line:
                    k, v = line.split(" ", 1)
                    self.kv[k] = v
        self.model_dir = model_dir or self.kv.get("model_dir", "")
        self.files = {}
        self.tensors = {}
        self.shapes = {}                           # name -> the dims tensors.txt lists after the size
        with open(os.path.join(pack_dir, "tensors.txt"), "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if p[0] == "file" and len(p) >= 3:
                    self.files[int(p[1])] = p[2]
                elif p[0] == "t" and len(p) >= 6:
                    self.tensors[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]))
                    self.shapes[p[1]] = tuple(int(x) for x in p[6:])
        self.dim = int(self.kv["hidden_size"])
        self.ff = int(self.kv["moe_intermediate_size"])
        self.nl = int(self.kv["num_hidden_layers"])
        self.ne = int(self.kv["n_routed_experts"])
        if self.ff % UNIT != 0:
            raise ValueError("n_ff %d is not a multiple of %d" % (self.ff, UNIT))
        self.nb = self.ff // UNIT                  # blocks of n_ff (9)
        self._fh = {}
        self._mats = {}

    def tensor(self, layer, e, kind):
        return self.tensors["layers.%d.ffn.experts.%d.%s.weight" % (layer, e, ROLES[kind])]

    def matrix(self, layer, kind):
        """The layer's Matrix for a role; every expert of the layer must agree with expert 0 on type and size."""
        key = (layer, kind)
        m = self._mats.get(key)
        if m is None:
            dtype, _, _, nbytes = self.tensor(layer, 0, kind)
            rows, k = (self.ff, self.dim) if kind < 2 else (self.dim, self.ff)
            m = Matrix(layer, kind, dtype, rows, k, nbytes)
            for e in range(self.ne):
                t = self.tensor(layer, e, kind)
                if t[0] != dtype or t[3] != nbytes:
                    raise ValueError("layer %d %s: expert %d is %s %d bytes, expert 0 is %s %d" %
                                     (layer, KINDS[kind], e, t[0], t[3], dtype, nbytes))
                # (the size alone passes a transposed matrix: [out][in] is what the cut assumes)
                shape = self.shapes["layers.%d.ffn.experts.%d.%s.weight" % (layer, e, ROLES[kind])]
                if shape != (rows, k):
                    raise ValueError("layer %d %s: expert %d is shaped %s in tensors.txt, the cut needs [%d][%d]" %
                                     (layer, KINDS[kind], e, "x".join(map(str, shape)) or "(none)", rows, k))
            self._mats[key] = m
        return m

    def expert_bytes(self, layer):
        return sum(self.matrix(layer, k).nbytes for k in range(3))

    def read(self, file, off, n):
        fh = self._fh.get(file)
        if fh is None:
            fh = self._fh[file] = open(os.path.join(self.model_dir, self.files[file]), "rb")
        fh.seek(off)
        buf = fh.read(n)
        if len(buf) != n:
            raise IOError("%s: short read at %d (%d of %d bytes)" % (self.files[file], off, len(buf), n))
        return buf

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh = {}


# ----------------------------------------------------------------------------------------------- cutting the matrices

def cut(m, buf, a, b):
    """The part of a whole matrix `buf` holding n_ff blocks [a, b)."""
    if m.kind < 2:
        return bytes(memoryview(buf)[a * UNIT * m.row_bytes:b * UNIT * m.row_bytes])
    w, lo, rb = (b - a) * m.unit_bytes, a * m.unit_bytes, m.row_bytes
    out = bytearray(m.rows * w)
    mv = memoryview(buf)
    for r in range(m.rows):
        out[r * w:(r + 1) * w] = mv[r * rb + lo:r * rb + lo + w]
    return bytes(out)


def read_part(pack, layer, e, kind, a, b):
    """Part [a, b) of an expert's matrix, from the pack."""
    m = pack.matrix(layer, kind)
    _, file, off, _ = pack.tensor(layer, e, kind)
    if kind < 2:                                   # contiguous rows: read only them
        n = (b - a) * UNIT * m.row_bytes
        return pack.read(file, off + a * UNIT * m.row_bytes, n)
    return cut(m, pack.read(file, off, m.nbytes), a, b)


# --------------------------------------------------------------------------------------------------- the heat order

def read_profile(path, nl, ne):
    """uint32 counts [layer][expert] (the engine's --expert-profile file: it may carry the MTP layers after the model's);
    None when there is no usable file."""
    if not path or path == "none":
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read(nl * ne * 4)
    except OSError:
        return None
    if len(raw) < nl * ne * 4:
        return None
    from array import array
    a = array("I")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def heat_order(pack, profile, layers, rank="per-byte"):
    """[(layer, expert, count)] hottest first.  per-byte: the share of a layer's picks (count / (layer total / 6),
    the pick rate of the expert) over the expert's bytes - what the design's pool tables rank by (epmodel.py top_set);
    count: the raw count.  Ties, and a pack with no profile, fall back to (layer, expert) order."""
    rows = []
    for l in layers:
        base = l * pack.ne
        counts = [profile[base + e] for e in range(pack.ne)] if profile is not None else [0] * pack.ne
        tot = sum(counts)
        eb = pack.expert_bytes(l)
        for e in range(pack.ne):
            q = counts[e] / (tot / 6.0) if tot > 0 else 0.0
            key = q / eb if rank == "per-byte" else counts[e]
            rows.append((-key, l, e, counts[e]))
    rows.sort()
    return [(l, e, c) for _, l, e, c in rows]


# --------------------------------------------------------------------------------------------------- planning a set

class Entry:
    __slots__ = ("layer", "expert", "kind", "type_id", "file", "a", "b", "rows", "row_bytes", "offset", "nbytes")

    def __init__(self, layer, expert, kind, type_id, file, a, b, rows, row_bytes, offset):
        self.layer, self.expert, self.kind, self.type_id, self.file = layer, expert, kind, type_id, file
        self.a, self.b, self.rows, self.row_bytes, self.offset = a, b, rows, row_bytes, offset
        self.nbytes = rows * row_bytes


def plan_set(pack, file, a, b, experts, kinds=(0, 1, 2)):
    """Where each part of the set [a, b) lands: ([Entry], file bytes).  The first page of a file is its header; the
    writer and --dry-run share this.  kinds: the matrices a set holds of each expert (--down-only: the down alone)."""
    if not (0 <= a < b <= pack.nb):
        raise ValueError("a set [%d, %d) is not inside [0, %d)" % (a, b, pack.nb))
    out, pos = [], HEADER
    for l, e in experts:
        for kind in kinds:
            m = pack.matrix(l, kind)
            rows, rb = m.part_shape(a, b)
            pos = align_up(pos)
            out.append(Entry(l, e, kind, m.type_id, file, a, b, rows, rb, pos))
            pos += rows * rb
    return out, align_up(pos)


def choose_pool(pack, order, a, b, n_experts=None, gb=None):
    """The heat-order prefix whose [a, b) parts fit `gb` (decimal GB, header and padding counted), or its first
    `n_experts`."""
    chosen, pos = [], HEADER
    for l, e, _ in order:
        if n_experts is not None and len(chosen) >= n_experts:
            break
        end = pos
        for kind in range(3):
            m = pack.matrix(l, kind)
            rows, rb = m.part_shape(a, b)
            end = align_up(end) + rows * rb
        if gb is not None and align_up(end) > gb * GB:
            break
        chosen.append((l, e))
        pos = end
    return chosen


def check_complementary(tail, pool, nb):
    """The two sets of one parts pack tile [0, nb): the pool [0, k) and the worker's [k, nb).  Anything else leaves an
    expert that can never be whole (or a double part), so the writer refuses it.  With no worker set the pool alone
    has to be [0, nb)."""
    if pool is None:
        return
    if tail is None:
        if pool != (0, nb):
            raise ValueError("with no worker set the pool %d:%d is all there is, and no expert can be whole unless it is "
                             "0:%d" % (pool + (nb,)))
        return
    if pool[0] != 0 or pool[1] != tail[0] or tail[1] != nb:
        raise ValueError("the pool %d:%d and the worker's set %d:%d do not tile 0:%d (the pool must be 0:k and the "
                         "worker's k:%d)" % (pool + tail + (nb, nb)))


# ---------------------------------------------------------------------------------------------------- writing

def index_lines(pack, phash, sets, entries, order, nonce=0):
    """The index as lines, without the stamp.  The stamp (a hash of every other line) is also in each part file's header
    page, so an index and files from different runs never go together; the nonce makes every run's stamp its own."""
    lines = ["ep-parts 1", "pack_hash %016x" % phash, "nonce %016x" % nonce,
             "geometry %d %d %d %d %d %d" % (pack.dim, pack.ff, pack.nl, pack.ne, UNIT, ALIGN),
             "covered " + " ".join(str(l) for l in sorted({en.layer for en in entries}))]
    for i, (name, a, b, size) in enumerate(sets):
        lines.append("file %d %s %d %d %d" % (i, name, a, b, size))
    for l, e, c in order:
        lines.append("heat %d %d %d" % (l, e, c))
    for en in entries:
        lines.append("part %d %d %s %d %d %d %d %d %d %d %d" %
                     (en.layer, en.expert, KINDS[en.kind], en.type_id, en.file, en.a, en.b, en.rows, en.row_bytes,
                      en.offset, en.nbytes))
    return lines


def stamp_of(lines):
    """FNV-1a over the index's lines (each with its '\\n', no '\\r', the stamp line itself left out)."""
    h = 1469598103934665603
    for line in lines:
        if not line.startswith("stamp "):
            h = fnv1a(line.replace("\r", "").encode("utf-8") + b"\n", h)
    return h


def file_header(stamp, file, a, b):
    return (MAGIC + stamp.to_bytes(8, "little") + file.to_bytes(4, "little") + a.to_bytes(4, "little") +
            b.to_bytes(4, "little")).ljust(HEADER, b"\0")


def fsync_dir(path):
    """Make the renames in `path` durable (a no-op where a directory cannot be opened, as on Windows)."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def leftover_parts(out_dir, names):
    """The part files in `out_dir` that an index naming `names` does not use (an earlier run's other range)."""
    return sorted(os.path.basename(x) for x in glob.glob(os.path.join(out_dir, "ep-parts-*.bin"))
                  if os.path.basename(x) not in names)


def write_parts(pack, out_dir, sets, order, log=None, nonce=None):
    """sets: [(a, b, [(layer, expert)...])] (each in heat order), or (a, b, experts, kinds) for a set of some matrices
    only (--down-only: (2,)).  Writes the files and the index; returns the entries.
    Everything goes to .tmp names first (flushed to disk) and is renamed at the end, the index last: a reader that has a
    file mapped keeps the old bytes, and a run that stops half way leaves no index that matches the new files (the
    stamp says so, and the nonce keeps two runs' stamps apart whatever their weights).  Part files of an earlier run
    that this index does not name are left where they are and reported through `log`.  `nonce` is for tests."""
    os.makedirs(out_dir, exist_ok=True)
    phash = pack_hash(pack.dir)
    names = [set_file_name(s[0], s[1]) for s in sets]
    if len(set(names)) != len(names):
        raise ValueError("two sets with the same range")
    all_entries, file_info, plans = [], [], []
    for fi, s in enumerate(sets):
        a, b, experts = s[:3]
        entries, size = plan_set(pack, fi, a, b, experts, s[3] if len(s) > 3 else (0, 1, 2))
        plans.append((a, b, entries, size))
        all_entries.extend(entries)
        file_info.append((names[fi], a, b, size))
    held = {x for s in sets for x in s[2]}
    listed = [(l, e, c) for l, e, c in order if (l, e) in held]     # the heat lines: every expert of any set
    lines = index_lines(pack, phash, file_info, all_entries, listed, secrets.randbits(64) if nonce is None else nonce)
    stamp = stamp_of(lines)
    lines.insert(2, "stamp %016x" % stamp)
    tmps = []
    try:
        t0, done, total = time.time(), 0, sum(len(x[2]) for x in sets)
        for fi, (a, b, entries, size) in enumerate(plans):
            tmp = os.path.join(out_dir, names[fi] + ".tmp")
            tmps.append(tmp)
            with open(tmp, "wb") as f:
                f.write(file_header(stamp, fi, a, b))
                pos = HEADER
                for en in entries:
                    f.write(b"\0" * (en.offset - pos))
                    buf = read_part(pack, en.layer, en.expert, en.kind, a, b)
                    if len(buf) != en.nbytes:
                        raise RuntimeError("part size %d, planned %d" % (len(buf), en.nbytes))
                    f.write(buf)
                    pos = en.offset + en.nbytes
                    if en.kind == 2:
                        done += 1
                        if log and done % 500 == 0:
                            log("  %d / %d experts, %.0f s" % (done, total, time.time() - t0))
                f.write(b"\0" * (size - pos))
                f.flush()
                os.fsync(f.fileno())
        index_tmp = os.path.join(out_dir, INDEX_NAME + ".tmp")
        tmps.append(index_tmp)
        with open(index_tmp, "w", newline="\n", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        for fi, tmp in enumerate(tmps[:-1]):
            os.replace(tmp, os.path.join(out_dir, names[fi]))
        os.replace(index_tmp, os.path.join(out_dir, INDEX_NAME))
        tmps = []
        fsync_dir(out_dir)
    finally:
        for t in tmps:
            if os.path.exists(t):
                os.remove(t)
    old = leftover_parts(out_dir, names)
    if old and log:
        log("warning: %s holds part files this index does not name (an earlier run's; not deleted): %s" %
            (out_dir, " ".join(old)))
    return all_entries


# ---------------------------------------------------------------------------------------------------- reading it back

class PartsIndex:
    """The index and the part files, for the verifier and the tests (the engine's reader is ds4_epparts.hpp)."""

    def __init__(self, out_dir):
        self.dir = out_dir
        self.files, self.heat, self.parts = [], [], {}
        self.pack_hash = 0
        self.stamp = self.nonce = self.covered = None
        with open(os.path.join(out_dir, INDEX_NAME), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        if not lines or lines[0] != "ep-parts 1":
            raise ValueError("%s: not an ep-parts 1 index" % INDEX_NAME)
        for line in lines[1:]:
            p = line.split()
            if not p:
                continue
            if p[0] == "pack_hash":
                self.pack_hash = int(p[1], 16)
            elif p[0] == "stamp":
                self.stamp = int(p[1], 16)
            elif p[0] == "nonce":
                self.nonce = int(p[1], 16)
            elif p[0] == "covered":
                self.covered = {int(x) for x in p[1:]}
            elif p[0] == "geometry":
                self.dim, self.ff, self.nl, self.ne, self.unit, self.align = map(int, p[1:7])
            elif p[0] == "file":
                self.files.append((p[2], int(p[3]), int(p[4]), int(p[5])))
            elif p[0] == "heat":
                self.heat.append((int(p[1]), int(p[2]), int(p[3])))
            elif p[0] == "part":
                en = Entry(int(p[1]), int(p[2]), KINDS.index(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7]),
                           int(p[8]), int(p[9]), int(p[10]))
                if en.nbytes != int(p[11]):
                    raise ValueError("part %s: rows x row_bytes != bytes" % line)
                self.parts.setdefault((en.layer, en.expert, en.kind), []).append(en)
            else:
                raise ValueError("unknown index line: %s" % line)
        for v in self.parts.values():
            v.sort(key=lambda x: x.a)
        self._fh = {}
        if self.stamp is None or self.nonce is None or self.covered is None:
            raise ValueError("%s: an index of the old format (no stamp, nonce or covered line): write the parts again" % INDEX_NAME)
        if stamp_of([x.rstrip("\r") for x in lines if x]) != self.stamp:
            raise ValueError("%s: its stamp does not match its own lines (edited or damaged)" % INDEX_NAME)
        for i, (name, a, b, size) in enumerate(self.files):
            with open(os.path.join(out_dir, name), "rb") as f:
                head = f.read(HEADER)
            if len(head) != HEADER or os.path.getsize(os.path.join(out_dir, name)) != size:
                raise ValueError("%s: not the %d bytes the index lists" % (name, size))
            if head != file_header(self.stamp, i, a, b):
                raise ValueError("%s: its header is from another run than %s (stale index or files)" % (name, INDEX_NAME))

    def read(self, en):
        fh = self._fh.get(en.file)
        if fh is None:
            fh = self._fh[en.file] = open(os.path.join(self.dir, self.files[en.file][0]), "rb")
        fh.seek(en.offset)
        buf = fh.read(en.nbytes)
        if len(buf) != en.nbytes:
            raise IOError("short part read")
        return buf

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh = {}

    def nb(self):
        return self.ff // self.unit

    def whole(self, layer, e):
        """True when the parts of each matrix tile [0, nb) without a gap or an overlap."""
        for kind in range(3):
            at = 0
            for en in self.parts.get((layer, e, kind), []):
                if en.a != at:
                    return False
                at = en.b
            if at != self.nb():
                return False
        return True

    def assemble(self, layer, e):
        """(gate, up, down) bytes of the whole expert, as the pack holds them."""
        out = []
        for kind in range(3):
            ps = self.parts[(layer, e, kind)]
            if kind < 2:
                out.append(b"".join(self.read(p) for p in ps))
                continue
            full = sum(p.row_bytes for p in ps)
            buf = bytearray(ps[0].rows * full)
            at = 0
            for p in ps:
                w, data = p.row_bytes, self.read(p)
                for r in range(p.rows):
                    buf[r * full + at:r * full + at + w] = data[r * w:(r + 1) * w]
                at += w
            out.append(bytes(buf))
        return out


def verify(pack, out_dir, layers, log=print):
    """Index against the pack: its hash, then every expert of `layers` that has parts - each part equal to the pack's
    own slice, and a whole expert's assembled matrices equal to the pack's bytes.  Returns the problem count."""
    try:
        ix = PartsIndex(out_dir)
    except (ValueError, OSError) as ex:
        log("FAIL %s" % ex)
        return 1
    bad, n_parts, n_whole = 0, 0, 0
    try:
        if ix.pack_hash != pack_hash(pack.dir):
            log("FAIL pack_hash: index %016x, pack %016x" % (ix.pack_hash, pack_hash(pack.dir)))
            bad += 1
        if (ix.dim, ix.ff, ix.nl, ix.ne) != (pack.dim, pack.ff, pack.nl, pack.ne):
            log("FAIL geometry differs from the pack's")
            bad += 1
        for (l, e) in sorted({(k[0], k[1]) for k in ix.parts}):
            if l not in layers:
                continue
            for kind in range(3):
                for p in ix.parts.get((l, e, kind), []):   # (a --down-only set has no gate / up parts)
                    n_parts += 1
                    if ix.read(p) != read_part(pack, l, e, kind, p.a, p.b):
                        log("FAIL part layer %d expert %d %s [%d,%d)" % (l, e, KINDS[kind], p.a, p.b))
                        bad += 1
            if ix.whole(l, e):
                n_whole += 1
                got = ix.assemble(l, e)
                for kind in range(3):
                    _, file, off, nbytes = pack.tensor(l, e, kind)
                    if got[kind] != pack.read(file, off, nbytes):
                        log("FAIL whole layer %d expert %d %s" % (l, e, KINDS[kind]))
                        bad += 1
        log("verify: %d parts compared, %d whole experts assembled, %d problems" % (n_parts, n_whole, bad))
    finally:
        ix.close()
    return bad


# ---------------------------------------------------------------------------------------------------------- dry run

def dry_run(pack, order, ks, tail_b, pool, pool_n, pool_gb, ship_whole, out=print):
    layers = sorted({l for l, _, _ in order})
    allx = [(l, e) for l, e, _ in order]
    out("pack %s: %d layers x %d experts, dim %d, n_ff %d (%d blocks of %d)" % (pack.dir, len(layers), pack.ne, pack.dim,
                                                                             pack.ff, pack.nb, UNIT))
    kinds = {}
    for l in layers:
        t = tuple(pack.matrix(l, k).dtype for k in range(3))
        kinds[t] = kinds.get(t, 0) + 1
    out("layer type triples (gate/up/down): " + ", ".join("%s/%s/%s x%d layers" % (t + (n,)) for t, n in sorted(kinds.items())))
    out("")
    out("| worker slice [k,%d) of all %d experts | stored with a = k | bytes (GB) | per expert mean (MB) |" % (tail_b, len(allx)))
    out("|---|---:|---:|---:|")
    for k in ks:
        _, size = plan_set(pack, 0, k, tail_b, allx)
        out("| k = %d | ep-parts-%d-%d.bin | %.2f | %.2f |" % (k, k, tail_b, size / GB, size / len(allx) / 1e6))
    a0 = min(ks)
    full_size = plan_set(pack, 0, a0, tail_b, allx)[1]
    out("")
    out("The worker stores one file [%d,%d) and reads [k,%d) of it in place for any k >= %d: %.2f GB on disk." %
        (a0, tail_b, tail_b, a0, full_size / GB))
    out("| k | bytes the CPU lane reads (GB) | of the file |")
    out("|---|---:|---:|")
    for k in ks:
        used = 0
        for l, e in allx:
            for kind in range(3):
                m = pack.matrix(l, kind)
                rows, rb = m.part_shape(k, tail_b)
                used += rows * rb
        out("| k = %d | %.2f | %.0f%% |" % (k, used / GB, 100.0 * used / full_size))
    pa, pb = pool
    chosen = choose_pool(pack, order, pa, pb, pool_n, pool_gb)
    _, psize = plan_set(pack, 0, pa, pb, chosen)
    out("")
    if chosen:
        out("pool [%d,%d) of the %d hottest experts (budget %s): %.2f GB, heat share of those experts %.1f%% of all picks" %
            (pa, pb, len(chosen), ("%d experts" % pool_n) if pool_n is not None else ("%.1f GB" % pool_gb), psize / GB,
             100.0 * _heat_share(order, chosen)))
    else:
        out("pool [%d,%d): empty" % (pa, pb))
    if ship_whole:
        top = [(l, e) for l, e, _ in order[:ship_whole]]
        whole = sum(pack.expert_bytes(l) for l, _ in top)
        out("whole experts (both sets) of the hottest %d: %.2f GB of raw expert bytes (%.2f GB with the parts' padding)" %
            (ship_whole, whole / GB,
             (plan_set(pack, 0, 0, pack.nb, top)[1]) / GB))
    return 0


def _heat_share(order, chosen):
    tot = sum(c for _, _, c in order)
    got = {x for x in chosen}
    return sum(c for l, e, c in order if (l, e) in got) / tot if tot else 0.0


# ------------------------------------------------------------------------------------------------------------- main

def parse_range(s, nb):
    a, b = s.split(":")
    a, b = int(a), int(b)
    if not (0 <= a < b <= nb):
        raise argparse.ArgumentTypeError("%s is not inside 0:%d" % (s, nb))
    return a, b


def parse_layers(s, nl):
    if s in (None, "all"):
        return list(range(nl))
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if any(not 0 <= l < nl for l in out):
        raise argparse.ArgumentTypeError("layers must be in 0..%d" % (nl - 1))
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("pack", help="the pack folder (ds4.txt, tensors.txt, expert-profile.bin)")
    ap.add_argument("out", nargs="?", help="where the parts and the index go (not needed with --dry-run)")
    ap.add_argument("--model-dir", help="the pack's weight files, when not ds4.txt's model_dir")
    ap.add_argument("--tail", default="6:9", help="the worker's set A:B in n_ff blocks, all experts ('none': skip; default 6:9)")
    ap.add_argument("--pool", default="0:6", help="the pool's set A:B, the hottest experts only (default 0:6)")
    ap.add_argument("--pool-experts", type=int, help="the pool's expert count")
    ap.add_argument("--pool-gb", type=float, default=44.5, help="the pool's size in GB (default 44.5; --pool-experts wins)")
    ap.add_argument("--down-only", metavar="A:B",
                    help="one set instead of --tail / --pool: the down matrices' blocks [A, B) of every expert of --layers, "
                         "rows back to back (the main's CPU lane reads its [0, k) of the down contiguous: in place in the "
                         "pack each row's [0, k) is strided, and the prefetchers read the rest of the row as well)")
    ap.add_argument("--layers", default="all", help="layers to write, e.g. 3 or 0-9,12 (default all)")
    ap.add_argument("--profile", help="expert profile (default PACK/expert-profile.bin; 'none': index order)")
    ap.add_argument("--rank", choices=("per-byte", "count"), default="per-byte", help="heat order key (default per-byte)")
    ap.add_argument("--ks", default="6,7,8", help="--dry-run: the worker's k values")
    ap.add_argument("--ship-whole", type=int, default=2500, help="--dry-run: also size the hottest N whole experts")
    ap.add_argument("--dry-run", action="store_true", help="print the sizes, write nothing")
    ap.add_argument("--verify", action="store_true", help="compare OUT's parts and wholes with the pack, byte for byte")
    args = ap.parse_args(argv)
    pack = Pack(args.pack, args.model_dir)
    try:
        return run(ap, args, pack)
    finally:
        pack.close()


def run(ap, args, pack):
    try:
        layers = parse_layers(args.layers, pack.nl)
        tail = None if args.tail == "none" else parse_range(args.tail, pack.nb)
        pool = parse_range(args.pool, pack.nb)
        down = parse_range(args.down_only, pack.nb) if args.down_only else None
        if down is None:
            check_complementary(tail, pool, pack.nb)
    except (argparse.ArgumentTypeError, ValueError) as ex:
        ap.error(str(ex))
    profile_path = args.profile if args.profile else os.path.join(args.pack, "expert-profile.bin")
    if args.verify:
        if not args.out:
            ap.error("--verify needs OUT")
        return 1 if verify(pack, args.out, set(layers)) else 0
    profile = read_profile(profile_path, pack.nl, pack.ne)
    if profile is None:
        print("warning: no expert profile at %s: the heat order is (layer, expert)" % profile_path, file=sys.stderr)
    order = heat_order(pack, profile, layers, args.rank)
    if down is not None:
        allx = [(l, e) for l, e, _ in order]
        size = plan_set(pack, 0, down[0], down[1], allx, (2,))[1]
        if args.dry_run:
            print("down-only [%d,%d) of all %d experts of %d layers: %s, %.2f GB (%.2f MB an expert)" %
                  (down[0], down[1], len(allx), len(layers), set_file_name(*down), size / GB, size / len(allx) / 1e6))
            return 0
        if not args.out:
            ap.error("OUT is needed (or --dry-run)")
        entries = write_parts(pack, args.out, [(down[0], down[1], allx, (2,))], order, log=lambda s: print(s, file=sys.stderr))
        print("wrote %d down parts [%d,%d) to %s" % (len(entries), down[0], down[1], args.out))
        return 0
    pool_n = args.pool_experts
    pool_gb = None if pool_n is not None else args.pool_gb
    if args.dry_run:
        ks = [int(x) for x in args.ks.split(",")]
        for k in ks:
            if not 0 <= k < pack.nb:
                ap.error("k %d is not inside 0..%d" % (k, pack.nb - 1))
        return dry_run(pack, order, ks, tail[1] if tail else pack.nb, pool, pool_n, pool_gb, args.ship_whole)
    if not args.out:
        ap.error("OUT is needed (or --dry-run)")
    sets = []
    if tail:
        sets.append((tail[0], tail[1], [(l, e) for l, e, _ in order]))
    chosen = choose_pool(pack, order, pool[0], pool[1], pool_n, pool_gb)
    if chosen:
        sets.append((pool[0], pool[1], chosen))
    entries = write_parts(pack, args.out, sets, order, log=lambda s: print(s, file=sys.stderr))
    print("wrote %d parts in %d sets to %s" % (len(entries), len(sets), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
