// cpu_ref.h -- CPU "reference" ray tracing, used to CHECK everything else.
//
// Two ways to ask "what is the closest triangle this ray hits?":
//
//   trace_brute_force()  test the ray against EVERY triangle. Slow, but so
//                        simple that it is almost certainly right.
//   trace_bvh()          walk the BVH, skipping boxes the ray misses.
//
// If the BVH is correct, both must agree on every ray. That comparison is the
// test in main.cpp. Later, the Slang shader's output gets compared to these
// same functions (that is plan item 4 for M2).
#pragma once

#include "bvh.h"

struct Ray {
    Vec3 o;  // origin
    Vec3 d;  // direction (does not need to be normalized)
};

struct Hit {
    float    t = FLT_MAX;          // distance along the ray: point = o + t * d
    uint32_t tri = UINT32_MAX;     // which triangle (UINT32_MAX = nothing hit)
    bool valid() const { return tri != UINT32_MAX; }
};

// We count the work done, so we can SEE how much the BVH saves.
struct TraceCounters {
    uint64_t box_tests = 0;
    uint64_t tri_tests = 0;
};

constexpr float    kTMin = 1e-6f;   // ignore hits "behind" or exactly at the origin
constexpr int      kMaxStack = 128; // traversal stack size (max tree depth must stay below this)

// ---------------------------------------------------------------------------
// Ray vs triangle: the Moller-Trumbore algorithm.
// Idea: write the hit point as  a + u*(b-a) + v*(c-a)  (u, v are coordinates
// INSIDE the triangle) and solve for (t, u, v) so it equals  o + t*d.
// The hit is real if u >= 0, v >= 0 and u + v <= 1. Two-sided: no backface culling.
// Only hits with kTMin < t < t_max count; t_max shrinks as closer hits are found.
// ---------------------------------------------------------------------------
inline bool intersect_triangle(const Ray& r, const Triangle& tri, float t_max, float& t_out)
{
    const Vec3& a = tri.vertices[0].position;
    const Vec3& b = tri.vertices[1].position;
    const Vec3& c = tri.vertices[2].position;
    const Vec3 e1 = b - a;
    const Vec3 e2 = c - a;
    const Vec3 p = cross(r.d, e2);
    const float det = dot(e1, p);
    if (std::fabs(det) < 1e-12f) return false;   // ray parallel to the triangle (or triangle degenerate)
    const float inv = 1.0f / det;
    const Vec3 s = r.o - a;
    const float u = dot(s, p) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    const Vec3 q = cross(s, e1);
    const float v = dot(r.d, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = dot(e2, q) * inv;
    if (t <= kTMin || t >= t_max) return false;
    t_out = t;
    return true;
}

// ---------------------------------------------------------------------------
// Ray vs box: the "slab" test.
// A box is the overlap of three slabs (the space between two parallel planes
// on x, on y and on z). The ray is inside each slab during some t-interval;
// it hits the box only if all three intervals overlap. `inv_d` is 1/direction,
// precomputed once per ray because dividing is slow.
//
// Returns true if the ray enters the box before t_max, and gives the entry
// distance in t_entry (0 if the origin is already inside).
//
// Two robustness details:
//   - If a direction component is 0, inv_d is infinity, and (0 * infinity) is
//     NaN when the origin lies exactly on a slab plane. The `?:` comparisons
//     below are written so a NaN never wins, i.e. it is ignored.
//   - We stretch the far distance by a hair (1.0000004). Rounding errors could
//     otherwise make a ray that grazes a box edge "miss" the box while still
//     hitting a triangle inside it. A BVH may say "maybe" too often (costs a
//     little time) but must never say "no" wrongly (would lose triangles).
// ---------------------------------------------------------------------------
inline bool aabb_entry(const BVHNode& n, const Vec3& o, const Vec3& inv_d, float t_max, float& t_entry)
{
    float tn = 0.0f;
    float tf = t_max;
    for (int a = 0; a < 3; ++a) {
        float t0 = (n.lo[a] - axis_of(o, a)) * axis_of(inv_d, a);
        float t1 = (n.hi[a] - axis_of(o, a)) * axis_of(inv_d, a);
        if (t0 > t1) std::swap(t0, t1);
        tn = (t0 > tn) ? t0 : tn;
        tf = (t1 * 1.0000004f < tf) ? t1 * 1.0000004f : tf;
    }
    t_entry = tn;
    return tn <= tf;
}

// ---------------------------------------------------------------------------
// Brute force: every ray against every triangle.
// ---------------------------------------------------------------------------
inline Hit trace_brute_force(const std::vector<Triangle>& tris, const Ray& ray, TraceCounters& counters)
{
    Hit best;
    for (uint32_t i = 0; i < (uint32_t)tris.size(); ++i) {
        counters.tri_tests++;
        float t;
        if (intersect_triangle(ray, tris[i], best.t, t)) {
            best.t = t;
            best.tri = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// BVH traversal, closest hit.
//
// Keep a small stack of "boxes still to visit", each with the distance at which
// the ray enters it. Loop: take a box off the stack; if it is a leaf, test its
// triangles; if it is interior, test BOTH child boxes and push the ones the ray
// hits, pushing the FARTHER one first so the NEARER one is popped next.
//
// This is where the "BVH shrinks as the ray goes" effect happens:
//   - best.t starts at infinity and gets smaller every time a hit is found.
//   - Any box whose entry distance is >= best.t cannot contain a closer hit,
//     so it is thrown away, even if it was pushed earlier.
// Visiting near boxes first makes best.t shrink early, so more boxes get
// thrown away. (The tree itself is built ONCE and never changes per ray.)
// ---------------------------------------------------------------------------
inline Hit trace_bvh(const BVH& bvh, const std::vector<Triangle>& tris, const Ray& ray, TraceCounters& counters)
{
    Hit best;
    const Vec3 inv_d{1.0f / ray.d.x, 1.0f / ray.d.y, 1.0f / ray.d.z};

    uint32_t stack_node[kMaxStack];
    float    stack_t[kMaxStack];
    int sp = 0;

    float t_root;
    counters.box_tests++;
    if (!aabb_entry(bvh.nodes[0], ray.o, inv_d, best.t, t_root)) return best;
    stack_node[sp] = 0;
    stack_t[sp] = t_root;
    ++sp;

    while (sp > 0) {
        --sp;
        if (stack_t[sp] >= best.t) continue;  // a closer hit was found after this box was pushed
        const BVHNode& n = bvh.nodes[stack_node[sp]];

        if (n.count > 0) {
            // Leaf: brute-force just the few triangles in here.
            for (uint32_t i = 0; i < n.count; ++i) {
                const uint32_t tri_index = bvh.tri_order[n.left_first + i];
                counters.tri_tests++;
                float t;
                if (intersect_triangle(ray, tris[tri_index], best.t, t)) {
                    best.t = t;
                    best.tri = tri_index;
                }
            }
        } else {
            const uint32_t l = n.left_first;
            const uint32_t r = l + 1;
            float tl, tr;
            counters.box_tests += 2;
            const bool hl = aabb_entry(bvh.nodes[l], ray.o, inv_d, best.t, tl);
            const bool hr = aabb_entry(bvh.nodes[r], ray.o, inv_d, best.t, tr);
            if (hl && hr) {
                // Push far first, near last (the stack pops the last one pushed).
                if (tl <= tr) {
                    stack_node[sp] = r; stack_t[sp] = tr; ++sp;
                    stack_node[sp] = l; stack_t[sp] = tl; ++sp;
                } else {
                    stack_node[sp] = l; stack_t[sp] = tl; ++sp;
                    stack_node[sp] = r; stack_t[sp] = tr; ++sp;
                }
            } else if (hl) {
                stack_node[sp] = l; stack_t[sp] = tl; ++sp;
            } else if (hr) {
                stack_node[sp] = r; stack_t[sp] = tr; ++sp;
            }
        }
    }
    return best;
}
