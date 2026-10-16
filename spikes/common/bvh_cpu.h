// bvh_cpu.h : a plain-C++ copy of Cycles' BVH2 traversal, line for line.
//
// WHY THIS EXISTS
// The GPU shader (bvh_traverse.slang) is a port of Cycles' kernel/bvh/traversal.h.
// If the GPU result is wrong we need to know WHICH part is wrong:
//     brute force  vs  this file   -> is my understanding of the BVH data right?
//     this file    vs  the GPU     -> did the Slang translation go wrong?
// So this file is written to be as close to the Slang version as possible.
// Keep the two in sync: any change here must be made in the .slang too.
//
// WHAT IS PORTED (Cycles v5.2.0): kernel/bvh/traversal.h (the aligned-node,
// triangle-only path), kernel/bvh/nodes.h (bvh_aligned_node_intersect),
// kernel/geom/triangle_intersect.h, util/math_intersect.h (ray_triangle_intersect,
// the non-SSE branches), kernel/geom/object.h (bvh_clamp_direction).
//
// WHAT IS NOT: unaligned nodes (hair only), motion blur, curves, points, instances.
// If the data contains an instance leaf we return kErrInstance instead of guessing.
//
// Every array access is bounds-checked against the counts, because on the GPU the same
// access would be a raw pointer read with no checking at all.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "bvh_dump.h"

namespace cyc {

constexpr int kEntrySentinel = 0x76543210;  // ENTRYPOINT_SENTINEL (kernel/bvh/types.h:17)
constexpr int kStackSize = 192;             // BVH_STACK_SIZE      (kernel/bvh/types.h:20)
constexpr uint32_t kMaxIterations = 1000000;  // loop guard: a bad BVH must not hang the GPU

// Error codes. They are written into Hit::id, so they must never collide with a real
// triangle number (those are small) or with kNoHitId (0xFFFFFFFF).
constexpr uint32_t kErrLoop = 0xFFFFFFFBu;
constexpr uint32_t kErrStack = 0xFFFFFFFCu;
constexpr uint32_t kErrBadIndex = 0xFFFFFFFDu;
constexpr uint32_t kErrInstance = 0xFFFFFFFEu;
constexpr uint32_t kNoPrim = 0xFFFFFFFFu;

inline bool is_error_id(uint32_t id) { return id >= kErrLoop && id != kNoPrim; }

// PRIMITIVE_TRIANGLE. I believe it is (1 << 0); main.cpp checks the dump agrees.
constexpr uint32_t kPrimTriangle = 1u;

struct V3 {
  float x, y, z;
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross3(V3 a, V3 b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline int as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline uint32_t as_uint(float f) { uint32_t i; std::memcpy(&i, &f, 4); return i; }
inline float as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Read-only view of the arrays (the GPU gets the same thing as raw pointers).
struct SceneView {
  const float *nodes;
  const float *leaves;
  const uint32_t *prim_index, *prim_visibility, *prim_object, *tri_vindex;
  const float *tri_verts;
  const uint32_t *objects;
  int32_t root;
  uint32_t node_count, leaf_count, prim_count, tri_count, vert_count, object_count;
};

inline SceneView make_view(const BvhDump &d)
{
  SceneView v{};
  v.nodes = d.nodes.data();
  v.leaves = d.leaves.data();
  v.prim_index = d.prim_index.data();
  v.prim_visibility = d.prim_visibility.data();
  v.prim_object = d.prim_object.data();
  v.tri_vindex = d.tri_vindex.data();
  v.tri_verts = d.tri_verts.data();
  v.objects = d.objects.data();
  v.root = d.root();
  v.node_count = d.node_count();
  v.leaf_count = d.leaf_count();
  v.prim_count = d.prim_count();
  v.tri_count = d.tri_count();
  v.vert_count = d.vert_count();
  v.object_count = d.object_count();
  return v;
}

// bvh_clamp_direction (kernel/geom/object.h:449): keep direction components away from 0
// so that 1/dir stays finite.
inline float clamp_component(float c)
{
  const float ooeps = 8.271806E-25f;
  if (std::fabs(c) > ooeps) return c;
  return std::copysign(ooeps, c);
}

// ray_triangle_intersect (util/math_intersect.h), non-SSE branches: Plucker-style test
// that matches Embree. Plain a*b+c*d arithmetic, 1/x for the reciprocal.
inline bool ray_triangle_intersect(V3 P, V3 D, float tmin, float tmax, V3 a, V3 b, V3 c,
                                   float &ou, float &ov, float &ot)
{
  const V3 v0 = a - P;
  const V3 v1 = b - P;
  const V3 v2 = c - P;

  const V3 e0 = v2 - v0;
  const V3 e1 = v0 - v1;
  const V3 e2 = v1 - v2;

  const float U = dot3(cross3(e0, v2 + v0), D);
  const float V = dot3(cross3(e1, v0 + v1), D);
  const float W = dot3(cross3(e2, v1 + v2), D);

  const float UVW = U + V + W;
  const float eps = 1.1920929e-07f * std::fabs(UVW);  // FLT_EPSILON * |UVW|
  const float minUVW = std::fmin(U, std::fmin(V, W));
  const float maxUVW = std::fmax(U, std::fmax(V, W));

  if (!(minUVW >= -eps || maxUVW <= eps)) return false;

  const V3 Ng1 = cross3(e1, e0);
  const V3 Ng = Ng1 + Ng1;
  const float den = dot3(Ng, D);
  if (den == 0.0f) return false;

  const float T = dot3(v0, Ng);
  const float t = T / den;
  if (!(t >= tmin && t <= tmax)) return false;

  const float rcp_uvw = (std::fabs(UVW) < 1e-18f) ? 0.0f : 1.0f / UVW;
  ou = std::fmin(U * rcp_uvw, 1.0f);
  ov = std::fmin(V * rcp_uvw, 1.0f);
  ot = t;
  return true;
}

// Fetch float4 number `i` out of a float array.
struct F4 {
  float x, y, z, w;
};
inline F4 fetch4(const float *base, uint32_t i)
{
  return {base[i * 4 + 0], base[i * 4 + 1], base[i * 4 + 2], base[i * 4 + 3]};
}

// The traversal. Returns 0 on success, or one of the kErr* codes.
// On success hit_prim is the triangle number of the closest hit, or kNoPrim.
inline uint32_t traverse(const SceneView &sc, V3 P, V3 Din, float tmin, float tmax,
                         uint32_t visibility, float &hit_t, uint32_t &hit_prim)
{
  int stack[kStackSize];
  stack[0] = kEntrySentinel;
  int sp = 0;
  int node_addr = sc.root;

  const V3 dir = {clamp_component(Din.x), clamp_component(Din.y), clamp_component(Din.z)};
  const V3 idir = {1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z};

  hit_t = tmax;
  hit_prim = kNoPrim;
  uint32_t iterations = 0;

  do {
    // ---- walk down through inner nodes --------------------------------------
    while (node_addr >= 0 && node_addr != kEntrySentinel) {
      if (++iterations > kMaxIterations) return kErrLoop;
      if ((uint32_t)node_addr + 3u >= sc.node_count) return kErrBadIndex;

      const F4 cnodes = fetch4(sc.nodes, (uint32_t)node_addr + 0);
      const F4 node0 = fetch4(sc.nodes, (uint32_t)node_addr + 1);
      const F4 node1 = fetch4(sc.nodes, (uint32_t)node_addr + 2);
      const F4 node2 = fetch4(sc.nodes, (uint32_t)node_addr + 3);

      // bvh_aligned_node_intersect (nodes.h:24): slab test against both children.
      const float c0lox = (node0.x - P.x) * idir.x;
      const float c0hix = (node0.z - P.x) * idir.x;
      const float c0loy = (node1.x - P.y) * idir.y;
      const float c0hiy = (node1.z - P.y) * idir.y;
      const float c0loz = (node2.x - P.z) * idir.z;
      const float c0hiz = (node2.z - P.z) * idir.z;
      const float c0min = std::fmax(std::fmax(tmin, std::fmin(c0lox, c0hix)),
                                    std::fmax(std::fmin(c0loy, c0hiy), std::fmin(c0loz, c0hiz)));
      const float c0max = std::fmin(std::fmin(hit_t, std::fmax(c0lox, c0hix)),
                                    std::fmin(std::fmax(c0loy, c0hiy), std::fmax(c0loz, c0hiz)));

      const float c1lox = (node0.y - P.x) * idir.x;
      const float c1hix = (node0.w - P.x) * idir.x;
      const float c1loy = (node1.y - P.y) * idir.y;
      const float c1hiy = (node1.w - P.y) * idir.y;
      const float c1loz = (node2.y - P.z) * idir.z;
      const float c1hiz = (node2.w - P.z) * idir.z;
      const float c1min = std::fmax(std::fmax(tmin, std::fmin(c1lox, c1hix)),
                                    std::fmax(std::fmin(c1loy, c1hiy), std::fmin(c1loz, c1hiz)));
      const float c1max = std::fmin(std::fmin(hit_t, std::fmax(c1lox, c1hix)),
                                    std::fmin(std::fmax(c1loy, c1hiy), std::fmax(c1loz, c1hiz)));

      const float dist0 = c0min;
      const float dist1 = c1min;

      // __VISIBILITY_FLAG__ is defined, so each child also has to be visible to this ray.
      const int traverse_mask = (((c0max >= c0min) && (as_uint(cnodes.x) & visibility)) ? 1 : 0) |
                                (((c1max >= c1min) && (as_uint(cnodes.y) & visibility)) ? 2 : 0);

      node_addr = as_int(cnodes.z);
      int node_addr_child1 = as_int(cnodes.w);

      if (traverse_mask == 3) {
        // Both children hit: go into the nearer one, push the farther one.
        if (dist1 < dist0) {
          const int tmp = node_addr;
          node_addr = node_addr_child1;
          node_addr_child1 = tmp;
        }
        ++sp;
        if (sp >= kStackSize) return kErrStack;
        stack[sp] = node_addr_child1;
      }
      else {
        if (traverse_mask == 2) {
          node_addr = node_addr_child1;
        }
        else if (traverse_mask == 0) {
          // Neither child hit: pop.
          if (sp < 0) return kErrStack;
          node_addr = stack[sp];
          --sp;
        }
      }
    }

    // ---- leaf: test its triangles ---------------------------------------------
    if (node_addr < 0) {
      const int leaf_index = -node_addr - 1;
      if ((uint32_t)leaf_index >= sc.leaf_count) return kErrBadIndex;
      const F4 leaf = fetch4(sc.leaves, (uint32_t)leaf_index);
      int prim_addr = as_int(leaf.x);

      if (prim_addr >= 0) {
        const int prim_addr2 = as_int(leaf.y);
        const uint32_t type = as_uint(leaf.w);

        // pop
        if (sp < 0) return kErrStack;
        node_addr = stack[sp];
        --sp;

        if (prim_addr2 < prim_addr || (uint32_t)prim_addr2 > sc.prim_count) return kErrBadIndex;

        for (; prim_addr < prim_addr2; prim_addr++) {
          const uint32_t pobj = sc.prim_object[prim_addr];
          const uint32_t prim = sc.prim_index[prim_addr];

          if (type & kPrimTriangle) {
            if (pobj >= sc.object_count || prim >= sc.tri_count) return kErrBadIndex;
            const uint32_t pos_off = sc.objects[(size_t)pobj * kObjectWords + kPositionOffsetWord];
            const uint32_t ia = sc.tri_vindex[prim * 3 + 0] + pos_off;
            const uint32_t ib = sc.tri_vindex[prim * 3 + 1] + pos_off;
            const uint32_t ic = sc.tri_vindex[prim * 3 + 2] + pos_off;
            if (ia >= sc.vert_count || ib >= sc.vert_count || ic >= sc.vert_count) return kErrBadIndex;

            const V3 a = {sc.tri_verts[ia * 3 + 0], sc.tri_verts[ia * 3 + 1], sc.tri_verts[ia * 3 + 2]};
            const V3 b = {sc.tri_verts[ib * 3 + 0], sc.tri_verts[ib * 3 + 1], sc.tri_verts[ib * 3 + 2]};
            const V3 c = {sc.tri_verts[ic * 3 + 0], sc.tri_verts[ic * 3 + 1], sc.tri_verts[ic * 3 + 2]};

            float u, v, t;
            if (ray_triangle_intersect(P, dir, tmin, hit_t, a, b, c, u, v, t)) {
              if (sc.prim_visibility[prim_addr] & visibility) {
                hit_t = t;
                hit_prim = prim;
              }
            }
          }
          else {
            return kErrInstance;  // not a plain triangle: motion/curve/point, unsupported here
          }
        }
      }
      else {
        return kErrInstance;  // negative start = instance leaf (needs object transforms)
      }
    }
  } while (node_addr != kEntrySentinel);

  return 0;
}

}  // namespace cyc
