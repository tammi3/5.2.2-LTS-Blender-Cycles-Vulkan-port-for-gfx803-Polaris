#!/usr/bin/env python3
"""
inspect_bvh.py -- decode the .bin files written by cycles_bvh_dump and sanity-check them.

usage:  python inspect_bvh.py C:/dev/bvh_out

WHAT IT CHECKS (all from reading Cycles' kernel/bvh/nodes.h and traversal.h):

  bvh_nodes  = array of float4. One INNER node = 4 consecutive float4:
      [n+0]  x,y = visibility flags of child 0 / child 1   (read as uint)
             z,w = address of child 0 / child 1             (read as int)
      [n+1]  x,y = child0.min.x, child1.min.x   z,w = child0.max.x, child1.max.x
      [n+2]  same for y
      [n+3]  same for z
    A child address >= 0 is another inner node (index into bvh_nodes, in float4 units).
    A child address <  0 is a LEAF: leaf index = -addr - 1 into bvh_leaf_nodes.

  bvh_leaf_nodes = array of float4:
      x = first prim slot (int; negative would mean "instance")
      y = one-past-last prim slot (int)
      w = primitive type bits (uint)
      z = not used by traversal.h; printed as raw bits

  prim slot p  ->  prim_index[p] = triangle number  ->  tri_vindex[tri] = 3 vertex numbers
                                                    ->  tri_verts[v]    = vertex position

SELF-CHECKS:
  1. every triangle appears in exactly one leaf
  2. every leaf's triangles lie inside the box that the parent stored for that leaf
If both pass, my reading of the layout is right.
"""
import struct
import sys
from pathlib import Path

SENTINEL = 0x76543210  # ENTRYPOINT_SENTINEL guess; only used for printing, never for logic


def read_words(path, fmt_char):
    data = Path(path).read_bytes()
    n = len(data) // 4
    return list(struct.unpack("<%d%s" % (n, fmt_char), data[: n * 4]))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])

    nodes_f = read_words(d / "bvh_nodes.bin", "f")
    nodes_i = read_words(d / "bvh_nodes.bin", "i")
    nodes_u = read_words(d / "bvh_nodes.bin", "I")
    leaf_i = read_words(d / "bvh_leaf_nodes.bin", "i")
    leaf_u = read_words(d / "bvh_leaf_nodes.bin", "I")
    prim_index = read_words(d / "prim_index.bin", "I")
    prim_type = read_words(d / "prim_type.bin", "I")
    prim_vis = read_words(d / "prim_visibility.bin", "I")
    tri_verts = read_words(d / "tri_verts.bin", "f")
    tri_vindex = read_words(d / "tri_vindex.bin", "I")

    n_nodes4 = len(nodes_f) // 4
    n_leaves = len(leaf_i) // 4
    n_tris = len(tri_vindex) // 3
    verts = [tuple(tri_verts[i * 3 : i * 3 + 3]) for i in range(len(tri_verts) // 3)]
    tris = [tuple(tri_vindex[i * 3 : i * 3 + 3]) for i in range(n_tris)]

    kb = (d / "kernel_bvh.bin").read_bytes()
    print("kernel_bvh raw (%d bytes) as ints:" % len(kb), list(struct.unpack("<%di" % (len(kb) // 4), kb)))
    print("  (first int should be the root address; layout is checked below)\n")

    print("bvh_nodes: %d float4 -> %d inner node(s)   leaves: %d   prims: %d   triangles: %d\n"
          % (n_nodes4, n_nodes4 // 4, n_leaves, len(prim_index), n_tris))

    def f4(i):
        return nodes_f[i * 4 : i * 4 + 4]

    def i4(i):
        return nodes_i[i * 4 : i * 4 + 4]

    def u4(i):
        return nodes_u[i * 4 : i * 4 + 4]

    # --- print inner nodes -------------------------------------------------
    leaf_box = {}  # leaf index -> (min3, max3) as stored by the parent
    for n in range(0, n_nodes4, 4):
        flags = u4(n)
        addr = i4(n)
        bx, by, bz = f4(n + 1), f4(n + 2), f4(n + 3)
        print("inner node @ float4 %d" % n)
        for c in (0, 1):
            lo = (bx[c], by[c], bz[c])
            hi = (bx[c + 2], by[c + 2], bz[c + 2])
            a = addr[2 + c]
            kind = ("inner node @ %d" % a) if a >= 0 else ("LEAF %d" % (-a - 1))
            if a == SENTINEL:
                kind = "SENTINEL"
            print("  child %d: vis=0x%x  addr=%d (%s)" % (c, flags[c], a, kind))
            print("           box min=(%.4f, %.4f, %.4f) max=(%.4f, %.4f, %.4f)" % (lo + hi))
            if a < 0:
                leaf_box[-a - 1] = (lo, hi)
    print()

    # --- print leaves and check them --------------------------------------
    seen = [0] * n_tris
    problems = 0
    for L in range(n_leaves):
        first, last = leaf_i[L * 4], leaf_i[L * 4 + 1]
        ltype = leaf_u[L * 4 + 3]
        z_raw = leaf_u[L * 4 + 2]
        print("leaf %d: prim slots [%d, %d)  type=0x%x  z_raw=0x%x" % (L, first, last, ltype, z_raw))
        if first < 0:
            print("   (negative start = instance leaf; not checked here)")
            continue
        for p in range(first, last):
            if p >= len(prim_index):
                print("   slot %d: OUT OF RANGE of prim_index" % p)
                problems += 1
                continue
            t = prim_index[p]
            if t >= n_tris:
                print("   slot %d: prim_index=%d is outside tri_vindex (%d tris)" % (p, t, n_tris))
                problems += 1
                continue
            seen[t] += 1
            vs = [verts[v] for v in tris[t]]
            print("   slot %d -> tri %d  verts %s  type=0x%x vis=0x%x" % (p, t, tris[t], prim_type[p], prim_vis[p]))
            if L in leaf_box:
                lo, hi = leaf_box[L]
                eps = 1e-4
                ok = all(lo[a] - eps <= v[a] <= hi[a] + eps for v in vs for a in range(3))
                if not ok:
                    print("      !!! triangle is OUTSIDE the leaf box its parent stored")
                    problems += 1

    missing = [t for t in range(n_tris) if seen[t] == 0]
    dup = [t for t in range(n_tris) if seen[t] > 1]
    print()
    print("triangles missing from every leaf: %s" % (missing or "none"))
    print("triangles in more than one leaf:   %s" % (dup or "none"))
    ok = not missing and not dup and problems == 0
    print("\nRESULT: %s" % ("LAYOUT UNDERSTOOD (all checks pass)" if ok else "CHECKS FAILED -- send me this output"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
