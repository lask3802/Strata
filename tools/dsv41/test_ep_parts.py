"""tools/dsv41/ep_parts.py: the parts pack writer, its index and its heat order, on a synthetic pack (no weights, no GPU).

    python -m unittest tools.dsv41.test_ep_parts

EP_PARTS_CPP_TEST=<path to ds4_epparts_test> also runs the C++ reader (src/dsv41/ds4_epparts.hpp) over the parts this
writer made: parts the Python wrote, read by the C++ one, against the pack.
"""
import io
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ep_parts as ep  # noqa: E402

DIM, FF, NL, NE = 512, 2304, 4, 6
# per layer: gate / up / down ggml types (a mix, so every type and a mixed layer are in the pack)
TYPES = (("IQ1_M", "IQ1_M", "IQ2_XXS"), ("IQ2_XXS", "IQ2_XXS", "IQ2_XS"), ("IQ2_XS", "IQ2_XS", "IQ1_M"),
         ("MXFP4", "MXFP4", "MXFP4"))


def make_pack(root, seed=1, crlf=False):
    """A pack folder (ds4.txt, tensors.txt, expert-profile.bin) and its weights file; returns (pack dir, originals)
    where originals[(layer, expert, kind)] is the matrix's bytes."""
    rng = random.Random(seed)
    pack, model = os.path.join(root, "pack"), os.path.join(root, "model")
    os.makedirs(pack)
    os.makedirs(model)
    nl = "\r\n" if crlf else "\n"
    with open(os.path.join(pack, "ds4.txt"), "w", newline="") as f:
        for k, v in (("hidden_size", DIM), ("num_hidden_layers", NL), ("n_routed_experts", NE),
                     ("moe_intermediate_size", FF), ("format", "gguf"), ("model_dir", model)):
            f.write("%s %s%s" % (k, v, nl))
    orig, lines, pos = {}, ["file 0 weights.bin"], 0
    with open(os.path.join(model, "weights.bin"), "wb") as w:
        w.write(b"HEAD")                            # (the experts do not start at 0)
        pos = 4
        for l in range(NL):
            for kind in (0, 2, 1):                  # (the real pack lists w1, w2, w3: per matrix, all experts in a row)
                m = ep.Matrix(l, kind, TYPES[l][kind], FF if kind < 2 else DIM, DIM if kind < 2 else FF,
                              (FF if kind < 2 else DIM) * ((DIM if kind < 2 else FF) // ep.TYPES[TYPES[l][kind]][1]) * ep.TYPES[TYPES[l][kind]][2])
                for e in range(NE):
                    buf = rng.randbytes(m.nbytes)
                    orig[(l, e, kind)] = buf
                    lines.append("t layers.%d.ffn.experts.%d.%s.weight %s 0 %d %d %d %d" %
                                 (l, e, ep.ROLES[kind], m.dtype, pos, m.nbytes,
                                  FF if kind < 2 else DIM, DIM if kind < 2 else FF))
                    w.write(buf)
                    pos += m.nbytes
    with open(os.path.join(pack, "tensors.txt"), "w", newline="") as f:
        f.write(nl.join(lines) + nl)
    counts = [rng.randrange(0, 1000) for _ in range(NL * NE)]
    with open(os.path.join(pack, "expert-profile.bin"), "wb") as f:
        for c in counts:
            f.write(c.to_bytes(4, "little"))
    return pack, orig, counts


def naive_cut(m, whole, a, b):
    """The part [a, b) of a matrix, written the long way: blocks of a row listed, then picked."""
    if m.kind < 2:
        rows = [whole[r * m.row_bytes:(r + 1) * m.row_bytes] for r in range(m.rows)]
        return b"".join(rows[a * 256:b * 256])
    out = []
    for r in range(m.rows):
        row = whole[r * m.row_bytes:(r + 1) * m.row_bytes]
        blocks = [row[i:i + m.unit_bytes] for i in range(0, m.row_bytes, m.unit_bytes)]
        out.append(b"".join(blocks[a:b]))
    return b"".join(out)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pack_dir, self.orig, self.counts = make_pack(self.tmp.name)
        self.pack = ep.Pack(self.pack_dir)
        self.addCleanup(self.pack.close)
        self.out = os.path.join(self.tmp.name, "out")

    def order(self, rank="per-byte"):
        prof = ep.read_profile(os.path.join(self.pack_dir, "expert-profile.bin"), NL, NE)
        return ep.heat_order(self.pack, prof, list(range(NL)), rank)


class Cutting(Base):
    def test_cut_matches_the_long_way_for_every_type_and_range(self):
        for l in range(NL):
            for kind in range(3):
                m = self.pack.matrix(l, kind)
                whole = self.orig[(l, 1, kind)]
                for a in range(0, 9):
                    for b in range(a + 1, 10):
                        self.assertEqual(ep.cut(m, whole, a, b), naive_cut(m, whole, a, b), (l, kind, a, b))

    def test_read_part_reads_the_pack_not_a_cache(self):
        for (l, e, kind), whole in list(self.orig.items())[:20]:
            m = self.pack.matrix(l, kind)
            self.assertEqual(ep.read_part(self.pack, l, e, kind, 6, 9), naive_cut(m, whole, 6, 9))

    def test_part_shape_and_down_row_bytes(self):
        m = self.pack.matrix(0, 2)       # IQ2_XXS down: 9 units of 66 bytes a row
        self.assertEqual((m.row_bytes, m.unit_bytes), (594, 66))
        self.assertEqual(m.part_shape(6, 9), (DIM, 198))
        g = self.pack.matrix(0, 0)       # IQ1_M gate: 512 values = 2 units of 56 bytes a row
        self.assertEqual(g.part_shape(6, 9), (768, 112))

    def test_a_size_that_disagrees_with_the_type_is_refused(self):
        with self.assertRaises(ValueError):
            ep.Matrix(0, 0, "IQ1_M", FF, DIM, 5)
        with self.assertRaises(ValueError):
            ep.Matrix(0, 0, "Q8_0", FF, DIM, 5)


class RoundTrip(Base):
    def test_parts_to_whole_is_byte_exact_for_every_k(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        for k in range(1, 9):
            out = os.path.join(self.out, "k%d" % k)
            ep.write_parts(self.pack, out, [(k, 9, allx), (0, k, allx[::2])], order)
            ix = ep.PartsIndex(out)
            try:
                for (l, e) in allx:
                    self.assertEqual(ix.whole(l, e), (l, e) in allx[::2], (k, l, e))
                    if not ix.whole(l, e):
                        continue
                    got = ix.assemble(l, e)
                    for kind in range(3):
                        self.assertEqual(got[kind], self.orig[(l, e, kind)], (k, l, e, kind))
            finally:
                ix.close()

    def test_default_sets_and_the_verifier(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        pool = ep.choose_pool(self.pack, order, 0, 6, n_experts=7)
        ep.write_parts(self.pack, self.out, [(6, 9, allx), (0, 6, pool)], order)
        msgs = []
        self.assertEqual(ep.verify(self.pack, self.out, set(range(NL)), log=msgs.append), 0, msgs)
        self.assertIn("7 whole experts", msgs[-1])

    def test_verifier_catches_a_flipped_byte(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        ep.write_parts(self.pack, self.out, [(0, 9, allx[:4])], order)
        path = os.path.join(self.out, "ep-parts-0-9.bin")
        with open(path, "r+b") as f:
            ix = ep.PartsIndex(self.out)
            en = ix.parts[(allx[2][0], allx[2][1], 2)][0]
            ix.close()
            f.seek(en.offset + en.nbytes // 2)
            b = f.read(1)
            f.seek(en.offset + en.nbytes // 2)
            f.write(bytes([b[0] ^ 1]))
        msgs = []
        self.assertGreater(ep.verify(self.pack, self.out, set(range(NL)), log=msgs.append), 0)
        self.assertTrue(any("FAIL" in m for m in msgs))

    def test_verifier_catches_a_pack_that_changed(self):
        order = self.order()
        ep.write_parts(self.pack, self.out, [(6, 9, [(0, 0)])], order)
        with open(os.path.join(self.pack_dir, "tensors.txt"), "a") as f:
            f.write("t extra.weight F32 0 0 4 1\n")
        msgs = []
        changed = ep.Pack(self.pack_dir)
        self.addCleanup(changed.close)
        self.assertGreater(ep.verify(changed, self.out, {0}, log=msgs.append), 0)
        self.assertTrue(any("pack_hash" in m for m in msgs))


class Index(Base):
    def test_index_lines(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        ep.write_parts(self.pack, self.out, [(6, 9, allx), (0, 6, allx[:5])], order)
        with open(os.path.join(self.out, "ep-parts.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")
        self.assertEqual(lines[0], "ep-parts 1")
        self.assertEqual(lines[1], "pack_hash %016x" % ep.pack_hash(self.pack_dir))
        self.assertEqual(lines[2], "geometry %d %d %d %d 256 4096" % (DIM, FF, NL, NE))
        files = [x.split() for x in lines if x.startswith("file ")]
        self.assertEqual([f[2] for f in files], ["ep-parts-6-9.bin", "ep-parts-0-6.bin"])
        for f in files:
            self.assertEqual(os.path.getsize(os.path.join(self.out, f[2])), int(f[5]))
            self.assertEqual(int(f[5]) % 4096, 0)
        parts = [x.split() for x in lines if x.startswith("part ")]
        self.assertEqual(len(parts), NL * NE * 3 + 5 * 3)
        for p in parts:
            self.assertEqual(int(p[10]) % 4096, 0)                       # (every matrix 4096-aligned)
            self.assertEqual(int(p[8]) * int(p[9]), int(p[11]))          # rows x row_bytes = bytes
            l, kind = int(p[1]), ep.KINDS.index(p[3])
            self.assertEqual(int(p[4]), ep.TYPES[TYPES[l][kind]][0])     # the ggml type id
        heat = [tuple(map(int, x.split()[1:])) for x in lines if x.startswith("heat ")]
        self.assertEqual(heat, order)                                    # every expert once, in heat order
        # a file's parts follow the heat order, offsets rising
        for fi in (0, 1):
            offs = [int(p[10]) for p in parts if int(p[5]) == fi]
            self.assertEqual(offs, sorted(offs))
        first = [(int(p[1]), int(p[2])) for p in parts if int(p[5]) == 0 and p[3] == "gate"]
        self.assertEqual(first, allx)

    def test_parts_index_reads_back_what_was_written(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        entries = ep.write_parts(self.pack, self.out, [(6, 9, allx)], order)
        ix = ep.PartsIndex(self.out)
        self.addCleanup(ix.close)
        self.assertEqual(ix.pack_hash, ep.pack_hash(self.pack_dir))
        self.assertEqual(sum(len(v) for v in ix.parts.values()), len(entries))
        self.assertEqual(ix.heat, order)
        for en in entries:
            back = ix.parts[(en.layer, en.expert, en.kind)][0]
            self.assertEqual((back.offset, back.nbytes, back.a, back.b, back.type_id), (en.offset, en.nbytes, 6, 9, en.type_id))

    def test_a_file_that_is_not_an_index_is_refused(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "ep-parts.txt"), "w") as f:
            f.write("something else\n")
        with self.assertRaises(ValueError):
            ep.PartsIndex(self.out)


class HeatOrder(Base):
    def test_per_byte_ranks_by_pick_rate_over_bytes(self):
        order = self.order()
        keys = []
        for l, e, c in order:
            tot = sum(self.counts[l * NE:(l + 1) * NE])
            keys.append((c / (tot / 6.0)) / self.pack.expert_bytes(l))
            self.assertEqual(c, self.counts[l * NE + e])
        self.assertEqual(keys, sorted(keys, reverse=True))
        self.assertEqual(len(order), NL * NE)
        self.assertEqual(len(set((l, e) for l, e, _ in order)), NL * NE)

    def test_count_ranks_by_raw_count(self):
        order = self.order("count")
        cs = [c for _, _, c in order]
        self.assertEqual(cs, sorted(cs, reverse=True))

    def test_ties_and_a_missing_profile_fall_back_to_index_order(self):
        order = ep.heat_order(self.pack, None, list(range(NL)))
        self.assertEqual([(l, e) for l, e, _ in order], [(l, e) for l in range(NL) for e in range(NE)])
        self.assertEqual(ep.read_profile("/no/such/file", NL, NE), None)
        self.assertEqual(ep.read_profile("none", NL, NE), None)

    def test_a_profile_with_the_mtp_layers_after_the_model_is_read(self):
        path = os.path.join(self.tmp.name, "long.bin")
        with open(os.path.join(self.pack_dir, "expert-profile.bin"), "rb") as src, open(path, "wb") as f:
            f.write(src.read() + bytes(4 * NE))
        self.assertEqual(list(ep.read_profile(path, NL, NE)), self.counts)

    def test_a_layer_subset_keeps_its_own_ranking(self):
        prof = ep.read_profile(os.path.join(self.pack_dir, "expert-profile.bin"), NL, NE)
        sub = ep.heat_order(self.pack, prof, [1], "count")
        self.assertEqual({l for l, _, _ in sub}, {1})
        self.assertEqual(len(sub), NE)


class Sizes(Base):
    def test_choose_pool_is_a_prefix_inside_the_budget(self):
        order = self.order()
        one = ep.plan_set(self.pack, 0, 0, 6, [(order[0][0], order[0][1])])[1]
        self.assertEqual(len(ep.choose_pool(self.pack, order, 0, 6, n_experts=4)), 4)
        got = ep.choose_pool(self.pack, order, 0, 6, gb=2.5 * one / ep.GB)
        self.assertEqual(got, [(l, e) for l, e, _ in order[:len(got)]])
        self.assertIn(len(got), (1, 2))
        self.assertLessEqual(ep.plan_set(self.pack, 0, 0, 6, got)[1], 2.5 * one)
        self.assertEqual(ep.choose_pool(self.pack, order, 0, 6, gb=0.0), [])

    def test_the_dry_run_sizes_are_the_sizes_written(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        for k in (6, 7, 8):
            written = os.path.join(self.out, "w%d" % k)
            ep.write_parts(self.pack, written, [(k, 9, allx)], order)
            self.assertEqual(ep.plan_set(self.pack, 0, k, 9, allx)[1], os.path.getsize(os.path.join(written, "ep-parts-%d-9.bin" % k)))
        lines = []
        ep.dry_run(self.pack, order, [6, 7, 8], 9, (0, 6), 5, None, 3, out=lines.append)
        text = "\n".join(lines)
        for k in (6, 7, 8):
            gb = ep.plan_set(self.pack, 0, k, 9, allx)[1] / ep.GB
            self.assertIn("| k = %d | ep-parts-%d-9.bin | %.2f |" % (k, k, gb), text)
        self.assertIn("pool [0,6) of the 5 hottest experts", text)

    def test_k_slices_shrink_with_k(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        sizes = [ep.plan_set(self.pack, 0, k, 9, allx)[1] for k in (6, 7, 8)]
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[1], sizes[2])


class PackHash(unittest.TestCase):
    def test_crlf_and_model_dir_do_not_change_it_and_a_tensor_does(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            a, _, _ = make_pack(t1, crlf=False)
            b, _, _ = make_pack(t2, crlf=True)
            self.assertNotEqual(ep.pack_hash(a), 0)
            self.assertEqual(ep.pack_hash(a), ep.pack_hash(b))     # (CRLF; and another model_dir line)
            with open(os.path.join(b, "tensors.txt"), "a") as f:
                f.write("t x.weight F32 0 0 4 1\n")
            self.assertNotEqual(ep.pack_hash(a), ep.pack_hash(b))
            self.assertEqual(ep.pack_hash(os.path.join(t1, "nowhere")), 0)


class Cli(Base):
    def test_dry_run_and_write_and_verify(self):
        out = io.StringIO()
        old, sys.stdout = sys.stdout, out
        try:
            self.assertEqual(ep.main([self.pack_dir, "--dry-run", "--pool-experts", "6", "--ship-whole", "4"]), 0)
            self.assertEqual(ep.main([self.pack_dir, self.out, "--layers", "1-2", "--pool-experts", "3"]), 0)
            self.assertEqual(ep.main([self.pack_dir, self.out, "--verify", "--layers", "1-2"]), 0)
        finally:
            sys.stdout = old
        text = out.getvalue()
        self.assertIn("| k = 6 |", text)
        self.assertIn("wrote 45 parts in 2 sets", text)       # (12 experts + 3 pool experts) x 3 matrices
        self.assertIn("3 whole experts assembled, 0 problems", text)
        ix = ep.PartsIndex(self.out)
        self.addCleanup(ix.close)
        self.assertEqual({k[0] for k in ix.parts}, {1, 2})

    def test_bad_arguments(self):
        with self.assertRaises(SystemExit):
            ep.main([self.pack_dir, "--tail", "6:12", "--dry-run"])
        with self.assertRaises(SystemExit):
            ep.main([self.pack_dir])


@unittest.skipUnless(os.environ.get("EP_PARTS_CPP_TEST"), "needs EP_PARTS_CPP_TEST=<ds4_epparts_test>")
class CppReader(Base):
    def test_cpp_reader_agrees_with_the_writer(self):
        order = self.order()
        allx = [(l, e) for l, e, _ in order]
        ep.write_parts(self.pack, self.out, [(6, 9, allx), (0, 6, ep.choose_pool(self.pack, order, 0, 6, n_experts=9))], order)
        r = subprocess.run([os.environ["EP_PARTS_CPP_TEST"], self.pack_dir, self.out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("9 whole experts assembled", r.stdout)


if __name__ == "__main__":
    unittest.main()
