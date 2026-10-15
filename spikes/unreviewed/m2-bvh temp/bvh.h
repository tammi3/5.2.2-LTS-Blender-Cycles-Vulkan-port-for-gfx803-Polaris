// bvh.h -- CPU BVH builder using the *binned* Surface Area Heuristic (SAH).
//
// WHAT A BVH IS (see docs/how-a-path-tracer-works.md, section 5):
//   A binary tree of axis-aligned boxes (AABBs). Every node IS one box.
//     - interior node: its box contains its two children's boxes
//     - leaf node:     its box contains a few triangles (we keep them <= 4)
//   A ray that misses a box can skip everything inside it.
//
// THE TWO THINGS TO UNDERSTAND IN THIS FILE
//   1. BVHNode: a 32-byte struct. The GPU shader will read exactly these bytes
//      later, so its layout is pinned down with static_asserts (like `Node`
//      in your M1 pointer-chase test).
//   2. SahBuilder::build_node(): how we decide where to cut a group of
//      triangles in two. That is the SAH part.
//
// NOTE ON "POINTERS": nodes refer to each other with plain integer INDICES
// into a std::vector, not raw pointers. On the GPU an index is just
// (base address + index * 32), the same memory access a pointer would do, so
// it is not slower. It is also safer: an index can be range-checked on the
// CPU before upload, a raw pointer cannot be checked by anybody.
#pragma once

#include "obj_loader.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

// ===========================================================================
// SECTION 1: tiny vector helpers
// ===========================================================================
static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(const Vec3& a, float s)       { return {a.x * s, a.y * s, a.z * s}; }
static inline float dot(const Vec3& a, const Vec3& b)      { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
static inline Vec3 normalize(const Vec3& a) { return a * (1.0f / length(a)); }
static inline Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
static inline Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }

// Pick x, y or z by number (0, 1, 2). Lets us write one loop over the 3 axes.
static inline float axis_of(const Vec3& v, int axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }

// ===========================================================================
// SECTION 2: axis-aligned bounding box
// ===========================================================================
struct AABB {
    // Starts "inverted" (lo > hi) = EMPTY. Growing it by anything fixes that,
    // and growing by an empty box changes nothing.
    Vec3 lo{ FLT_MAX,  FLT_MAX,  FLT_MAX};
    Vec3 hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};

    void grow(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void grow(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }

    // Surface area of the box: 2(xy + yz + zx). This is the number SAH is built on.
    // An empty box has area 0.
    float area() const {
        if (lo.x > hi.x) return 0.0f;
        const Vec3 d = hi - lo;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

static inline AABB triangle_bounds(const Triangle& t)
{
    AABB b;
    for (int i = 0; i < 3; ++i) b.grow(t.vertices[i].position);
    return b;
}

// ===========================================================================
// SECTION 3: the node layout (this is what the GPU will read)
// ===========================================================================
struct BVHNode {
    float    lo[3];       // box minimum corner            (offset  0, 12 bytes)
    uint32_t left_first;  // see below                     (offset 12,  4 bytes)
    float    hi[3];       // box maximum corner            (offset 16, 12 bytes)
    uint32_t count;       // 0 = interior, >0 = leaf       (offset 28,  4 bytes)
};
// How to read the two uint32 fields:
//   count == 0  -> INTERIOR node. left_first = index of the left child in the
//                  node array. The right child is ALWAYS the very next slot
//                  (left_first + 1), so we only need to store one index.
//   count  > 0  -> LEAF node. It holds `count` triangles, starting at slot
//                  left_first of BVH::tri_order.
// Packing the "is it a leaf?" flag into `count` is why a node fits in 32 bytes,
// a nice size: two nodes fill one 64-byte cache line.
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes");
static_assert(offsetof(BVHNode, left_first) == 12, "left_first must be at offset 12");
static_assert(offsetof(BVHNode, hi) == 16, "hi must be at offset 16");
static_assert(offsetof(BVHNode, count) == 28, "count must be at offset 28");

struct BVH {
    std::vector<BVHNode>  nodes;      // nodes[0] is the root
    std::vector<uint32_t> tri_order;  // leaves point into this; each entry is an
                                      // index into the ORIGINAL triangle array.
                                      // (Triangles are not copied or moved, we
                                      // only record the order the leaves want.)
    uint32_t max_depth = 0;
};

// ===========================================================================
// SECTION 4: tuning knobs
// ===========================================================================
namespace bvh_cfg {
constexpr int      kBins          = 16;    // buckets per axis in the binned SAH
constexpr uint32_t kMaxLeafTris   = 4;     // a leaf bigger than this MUST be split
// Relative costs used by SAH. Only their RATIO matters. "1 and 1" is a neutral
// guess. Once we have timings from the RX 570 we can tune them (a box test on a
// GPU without RT hardware also costs a memory fetch, so traversal is probably
// NOT cheaper than a triangle test).
constexpr float    kCostTraversal = 1.0f;  // cost of visiting a node (box test)
constexpr float    kCostIntersect = 1.0f;  // cost of one ray-triangle test
}  // namespace bvh_cfg

// ===========================================================================
// SECTION 5: the builder
// ===========================================================================
//
// BINNED SAH IN ONE PARAGRAPH
// We have a group of triangles and want to cut it into a left and right group.
// A "good" cut gives two boxes that are SMALL and hold FEW triangles, because a
// random ray is more likely to hit a big box (probability ~ surface area) and
// then has to test every triangle inside it. So we score a cut with:
//
//   cost = C_trav + C_isect * ( area(left)/area(parent)  * count(left)
//                             + area(right)/area(parent) * count(right) )
//
// and compare with the cost of NOT cutting (a leaf): C_isect * count.
// The "exact" SAH would try a cut between every pair of triangles. "Binned"
// SAH instead drops each triangle's CENTER into one of 16 buckets along an axis
// (like a histogram that remembers a box and a count per bucket) and only tries
// the 15 cut lines between buckets. Slightly worse cuts, MUCH cheaper to build.
struct SahBuilder {
    // One entry per triangle while building. We shuffle these around in place;
    // when done, their order becomes BVH::tri_order.
    struct Prim {
        AABB     box;       // the triangle's own bounding box
        Vec3     centroid;  // center of that box (what we sort into bins)
        uint32_t tri;       // index into the original triangle array
    };

    struct SplitChoice {
        bool  valid = false;      // false = the centers can't be separated at all
        int   axis = 0;           // 0 = x, 1 = y, 2 = z
        int   last_left_bin = 0;  // bins 0..last_left_bin go left, the rest go right
        float cost = FLT_MAX;
        float cmin = 0.0f;        // bin geometry, kept so the partition step bins
        float scale = 0.0f;       // triangles EXACTLY like the scoring step did
    };

    std::vector<Prim> prims;
    BVH* out = nullptr;

    static int bin_index(float c, float cmin, float scale)
    {
        int b = (int)((c - cmin) * scale);
        if (b < 0) b = 0;
        if (b >= bvh_cfg::kBins) b = bvh_cfg::kBins - 1;  // a center exactly on the max edge
        return b;
    }

    static void set_bounds(BVHNode& n, const AABB& b)
    {
        n.lo[0] = b.lo.x; n.lo[1] = b.lo.y; n.lo[2] = b.lo.z;
        n.hi[0] = b.hi.x; n.hi[1] = b.hi.y; n.hi[2] = b.hi.z;
    }

    // Try all 3 axes x 15 cut lines and return the cheapest.
    SplitChoice find_best_split(uint32_t first, uint32_t count, const AABB& centroid_box, float parent_area) const
    {
        using namespace bvh_cfg;
        SplitChoice best;
        const float inv_parent = parent_area > 0.0f ? 1.0f / parent_area : 0.0f;

        for (int axis = 0; axis < 3; ++axis) {
            const float cmin = axis_of(centroid_box.lo, axis);
            const float extent = axis_of(centroid_box.hi, axis) - cmin;
            if (extent <= 0.0f) continue;  // every center has the same coordinate here: no cut possible
            const float scale = (float)kBins / extent;

            // ---- Step 1: fill the histogram. Each bin remembers a box and a count.
            struct Bin { AABB box; uint32_t n = 0; } bins[kBins];
            for (uint32_t i = first; i < first + count; ++i) {
                const int b = bin_index(axis_of(prims[i].centroid, axis), cmin, scale);
                bins[b].n++;
                bins[b].box.grow(prims[i].box);
            }

            // ---- Step 2: sweep left->right. After bin i, how big is the LEFT side?
            // left_*[i] describes "everything in bins 0..i".
            float    left_area[kBins - 1];
            uint32_t left_n[kBins - 1];
            AABB acc;
            uint32_t n = 0;
            for (int i = 0; i < kBins - 1; ++i) {
                acc.grow(bins[i].box);
                n += bins[i].n;
                left_area[i] = acc.area();
                left_n[i] = n;
            }

            // ---- Step 3: sweep right->left, and score each cut line as we go.
            // At bin i, "acc" is everything in bins i..kBins-1 = the RIGHT side of
            // the cut that sits between bin i-1 and bin i.
            acc = AABB();
            n = 0;
            for (int i = kBins - 1; i >= 1; --i) {
                acc.grow(bins[i].box);
                n += bins[i].n;
                const uint32_t ln = left_n[i - 1];
                const uint32_t rn = n;
                if (ln == 0 || rn == 0) continue;  // a cut with an empty side is no cut
                const float cost = kCostTraversal +
                    kCostIntersect * (left_area[i - 1] * (float)ln + acc.area() * (float)rn) * inv_parent;
                if (cost < best.cost) {
                    best.valid = true;
                    best.axis = axis;
                    best.last_left_bin = i - 1;
                    best.cost = cost;
                    best.cmin = cmin;
                    best.scale = scale;
                }
            }
        }
        return best;
    }

    // Build the node at nodes[ni], covering prims[first .. first+count).
    // (Recursive. Depth is ~log2(N) for sane meshes; main.cpp checks the final
    // depth because the traversal stack on the CPU/GPU has a fixed size.)
    void build_node(uint32_t ni, uint32_t first, uint32_t count, uint32_t depth)
    {
        using namespace bvh_cfg;
        out->max_depth = std::max(out->max_depth, depth);

        // 1) The box around everything in this node, and a second box around
        //    just the triangle CENTERS (that one decides where cuts can happen).
        AABB box, centroid_box;
        for (uint32_t i = first; i < first + count; ++i) {
            box.grow(prims[i].box);
            centroid_box.grow(prims[i].centroid);
        }
        set_bounds(out->nodes[ni], box);

        // 2) Decide: leaf, or cut? And if cut, where?
        bool     split = false;
        uint32_t mid = 0;  // prims[first..mid) go left, prims[mid..first+count) go right
        if (count > 1) {
            const float leaf_cost = kCostIntersect * (float)count;
            const SplitChoice s = find_best_split(first, count, centroid_box, box.area());
            if (s.valid) {
                // Cut if it is cheaper than a leaf. ALSO cut if the leaf would be
                // bigger than kMaxLeafTris, even when SAH says it's not worth it:
                // we never allow huge leaves.
                split = (s.cost < leaf_cost) || (count > kMaxLeafTris);
                if (split) {
                    auto it = std::partition(prims.begin() + first, prims.begin() + first + count,
                        [&](const Prim& p) {
                            return bin_index(axis_of(p.centroid, s.axis), s.cmin, s.scale) <= s.last_left_bin;
                        });
                    mid = (uint32_t)(it - prims.begin());
                }
            } else {
                // All centers are identical (e.g. duplicated triangles). No spatial
                // cut exists. Only split if the leaf would be too big, and then
                // just halve the list: the tree stays valid, only less efficient.
                split = count > kMaxLeafTris;
                if (split) mid = first + count / 2;
            }
            // Safety net: a cut must leave something on both sides.
            if (split && (mid == first || mid == first + count)) mid = first + count / 2;
        }

        // 3a) Leaf: remember which slice of tri_order belongs to it.
        if (!split) {
            out->nodes[ni].left_first = first;
            out->nodes[ni].count = count;
            return;
        }

        // 3b) Interior: reserve two ADJACENT slots for the children (this is what
        //     lets us store only one child index), then build each child.
        //     Careful: emplace_back may move the whole vector in memory, so we
        //     never keep a reference to a node across these calls, only indices.
        const uint32_t left = (uint32_t)out->nodes.size();
        out->nodes.emplace_back();
        out->nodes.emplace_back();
        out->nodes[ni].left_first = left;
        out->nodes[ni].count = 0;
        build_node(left,     first, mid - first,          depth + 1);
        build_node(left + 1, mid,   first + count - mid,  depth + 1);
    }
};

// Build a BVH over `tris`. The triangle array itself is not modified.
inline BVH build_bvh_sah(const std::vector<Triangle>& tris)
{
    BVH bvh;
    if (tris.empty()) return bvh;

    SahBuilder builder;
    builder.out = &bvh;
    builder.prims.resize(tris.size());
    for (uint32_t i = 0; i < (uint32_t)tris.size(); ++i) {
        const AABB b = triangle_bounds(tris[i]);
        builder.prims[i].box = b;
        builder.prims[i].centroid = (b.lo + b.hi) * 0.5f;
        builder.prims[i].tri = i;
    }

    bvh.nodes.emplace_back();  // slot 0 = root
    builder.build_node(0, 0, (uint32_t)tris.size(), 0);

    bvh.tri_order.resize(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) bvh.tri_order[i] = builder.prims[i].tri;
    return bvh;
}

// ===========================================================================
// SECTION 6: statistics (so we can SEE what the builder did)
// ===========================================================================
struct BVHStats {
    uint32_t nodes = 0, leaves = 0, interior = 0, max_depth = 0;
    uint32_t min_leaf = 0, max_leaf = 0;
    float    avg_leaf = 0.0f;
    float    sah_cost = 0.0f;  // expected cost of a random ray, in units of C_isect
};

static inline float node_area(const BVHNode& n)
{
    const float dx = n.hi[0] - n.lo[0], dy = n.hi[1] - n.lo[1], dz = n.hi[2] - n.lo[2];
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

inline BVHStats compute_stats(const BVH& bvh)
{
    BVHStats s;
    if (bvh.nodes.empty()) return s;
    s.nodes = (uint32_t)bvh.nodes.size();
    s.max_depth = bvh.max_depth;
    s.min_leaf = UINT32_MAX;
    const float root_area = node_area(bvh.nodes[0]);
    uint64_t tris_in_leaves = 0;
    float cost = 0.0f;
    for (const BVHNode& n : bvh.nodes) {
        const float rel = root_area > 0.0f ? node_area(n) / root_area : 0.0f;
        if (n.count == 0) {
            s.interior++;
            cost += bvh_cfg::kCostTraversal * rel;                         // visiting this box
        } else {
            s.leaves++;
            tris_in_leaves += n.count;
            s.min_leaf = std::min(s.min_leaf, n.count);
            s.max_leaf = std::max(s.max_leaf, n.count);
            cost += bvh_cfg::kCostIntersect * (float)n.count * rel;        // testing its triangles
        }
    }
    s.avg_leaf = s.leaves ? (float)tris_in_leaves / (float)s.leaves : 0.0f;
    s.sah_cost = cost;
    return s;
}

// ===========================================================================
// SECTION 7: validation (the "is this tree even legal?" check)
// ===========================================================================
// Checks, in order:
//   - tri_order lists every triangle exactly once
//   - every child index is in range, comes after its parent, and is reached once
//   - every child box sits inside its parent box
//   - every leaf's triangles sit inside the leaf's box
// If any of these fails, traversal could silently skip triangles, so we want to
// find out on the CPU, long before a shader runs. Returns false and prints why.
inline bool validate_bvh(const BVH& bvh, const std::vector<Triangle>& tris)
{
    const uint32_t N = (uint32_t)tris.size();
    auto bad = [](const char* msg, uint32_t node) {
        std::fprintf(stderr, "BVH INVALID at node %u: %s\n", node, msg);
        return false;
    };

    if (bvh.nodes.empty()) return bad("no nodes", 0);
    if (bvh.tri_order.size() != N) return bad("tri_order size != triangle count", 0);

    std::vector<uint8_t> seen(N, 0);
    for (uint32_t t : bvh.tri_order) {
        if (t >= N) return bad("tri_order entry out of range", 0);
        if (seen[t]++) return bad("a triangle is listed twice in tri_order", 0);
    }

    std::vector<uint8_t> visited(bvh.nodes.size(), 0);
    std::vector<uint32_t> stack;
    stack.push_back(0);
    uint32_t tris_in_leaves = 0;

    while (!stack.empty()) {
        const uint32_t ni = stack.back();
        stack.pop_back();
        if (ni >= bvh.nodes.size()) return bad("child index out of range", ni);
        if (visited[ni]++) return bad("node reached twice (two parents?)", ni);
        const BVHNode& n = bvh.nodes[ni];

        if (n.count == 0) {
            const uint32_t l = n.left_first;
            if (l <= ni || l + 1 >= bvh.nodes.size()) return bad("child indices invalid", ni);
            for (uint32_t c : {l, l + 1}) {
                const BVHNode& ch = bvh.nodes[c];
                for (int a = 0; a < 3; ++a)
                    if (ch.lo[a] < n.lo[a] || ch.hi[a] > n.hi[a])
                        return bad("child box sticks out of its parent box", ni);
                stack.push_back(c);
            }
        } else {
            if ((uint64_t)n.left_first + n.count > N) return bad("leaf range runs past tri_order", ni);
            for (uint32_t i = 0; i < n.count; ++i) {
                const Triangle& t = tris[bvh.tri_order[n.left_first + i]];
                for (int v = 0; v < 3; ++v) {
                    const Vec3& p = t.vertices[v].position;
                    if (p.x < n.lo[0] || p.x > n.hi[0] || p.y < n.lo[1] || p.y > n.hi[1] ||
                        p.z < n.lo[2] || p.z > n.hi[2])
                        return bad("a triangle sticks out of its leaf box", ni);
                }
            }
            tris_in_leaves += n.count;
        }
    }
    if (tris_in_leaves != N) return bad("leaves do not cover every triangle exactly once", 0);
    return true;
}
