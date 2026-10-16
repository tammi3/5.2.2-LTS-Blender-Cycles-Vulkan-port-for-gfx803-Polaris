// M2c: trace rays through the BVH that CYCLES built, and check the answer three ways.
//
// Usage:  m2c_bvh <bvh_traverse.spv | -> <dump_dir> [width] [height] [visibility_mask_hex] [cpu_threads]
//   cpu_threads = how many threads the CPU passes use (default: all hardware threads).
//   spv = "-"  runs everything except the GPU (useful to debug the CPU side alone).
//
// ###########################################################################
// ##  RAW GPU POINTERS HAVE NO BOUNDS CHECKING.                            ##
// ##  The shader checks every index against the counts we pass in, the      ##
// ##  output buffer has canary words after it, and every struct shared      ##
// ##  with the shader is locked with static_assert. Keep it that way.       ##
// ###########################################################################
//
// What happens, in order:
//   1. load the dump written by cycles_bvh_dump and validate it
//   2. build a brute-force reference from the SAME triangles (id = Cycles' triangle number)
//   3. CPU A: brute force            (Moller-Trumbore, no BVH)          <- the answer key (multi-threaded)
//   4. CPU B: Cycles' traversal in C++ (common/bvh_cpu.h)               <- proves we read the BVH right
//             run twice: 1 thread (the baseline, feeds the comparisons) and N threads (speed
//             reference; must give bit-identical results to the 1-thread run)
//   5. GPU:   Cycles' traversal in Slang                                <- proves the port is right
//   6. compare A-B, B-GPU, A-GPU, write images

#include <algorithm>
#include <array>
#include <atomic>   // std::atomic: the shared "next row to work on" counter
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>   // std::thread: the CPU worker threads
#include <vector>

#include "bvh_cpu.h"
#include "bvh_dump.h"
#include "ppm.h"
#include "raytri_cpu.h"
#include "vk_util.h"

// ---------------------------------------------------------------------------
// Structs shared with the shader. MUST match bvh_traverse.slang offset by offset.
// ---------------------------------------------------------------------------
struct SceneDataGpu {
  uint64_t bvh_nodes;        // 0
  uint64_t bvh_leaf_nodes;   // 8
  uint64_t prim_index;       // 16
  uint64_t prim_visibility;  // 24
  uint64_t prim_object;      // 32
  uint64_t tri_vindex;       // 40
  uint64_t tri_verts;        // 48
  uint64_t objects;          // 56
  int32_t root;              // 64
  uint32_t node_count;       // 68
  uint32_t leaf_count;       // 72
  uint32_t prim_count;       // 76
  uint32_t tri_count;        // 80
  uint32_t vert_count;       // 84
  uint32_t object_count;     // 88
  uint32_t pad;              // 92
};
static_assert(offsetof(SceneDataGpu, bvh_nodes) == 0, "layout");
static_assert(offsetof(SceneDataGpu, bvh_leaf_nodes) == 8, "layout");
static_assert(offsetof(SceneDataGpu, prim_index) == 16, "layout");
static_assert(offsetof(SceneDataGpu, prim_visibility) == 24, "layout");
static_assert(offsetof(SceneDataGpu, prim_object) == 32, "layout");
static_assert(offsetof(SceneDataGpu, tri_vindex) == 40, "layout");
static_assert(offsetof(SceneDataGpu, tri_verts) == 48, "layout");
static_assert(offsetof(SceneDataGpu, objects) == 56, "layout");
static_assert(offsetof(SceneDataGpu, root) == 64, "layout");
static_assert(offsetof(SceneDataGpu, node_count) == 68, "layout");
static_assert(offsetof(SceneDataGpu, leaf_count) == 72, "layout");
static_assert(offsetof(SceneDataGpu, prim_count) == 76, "layout");
static_assert(offsetof(SceneDataGpu, tri_count) == 80, "layout");
static_assert(offsetof(SceneDataGpu, vert_count) == 84, "layout");
static_assert(offsetof(SceneDataGpu, object_count) == 88, "layout");
static_assert(offsetof(SceneDataGpu, pad) == 92, "layout");
static_assert(sizeof(SceneDataGpu) == 96, "layout");

struct TraversePush {
  uint64_t scene;       // 0
  uint64_t out;         // 8
  uint32_t width;       // 16
  uint32_t height;      // 20
  uint32_t visibility;  // 24
  float o[3];           // 28
  float l[3];           // 40
  float u[3];           // 52
  float v[3];           // 64
};
static_assert(offsetof(TraversePush, scene) == 0, "layout");
static_assert(offsetof(TraversePush, out) == 8, "layout");
static_assert(offsetof(TraversePush, width) == 16, "layout");
static_assert(offsetof(TraversePush, height) == 20, "layout");
static_assert(offsetof(TraversePush, visibility) == 24, "layout");
static_assert(offsetof(TraversePush, o) == 28, "layout");
static_assert(offsetof(TraversePush, l) == 40, "layout");
static_assert(offsetof(TraversePush, u) == 52, "layout");
static_assert(offsetof(TraversePush, v) == 64, "layout");
static_assert(sizeof(TraversePush) <= 128, "push constants are limited to 128 bytes on this GPU");

static constexpr uint32_t kCanaryHits = 16;
static constexpr uint32_t kCanaryWord = 0xCAFEBABEu;
static constexpr uint32_t kUnwrittenWord = 0xDEADBEEFu;

static uint32_t float_bits(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

// ---------------------------------------------------------------------------
// Compare two hit lists. `ref` is the one we trust more.
//
// Two different ray/triangle tests (Moller-Trumbore vs Cycles' Plucker test, or the same
// test on a CPU vs a GPU that fuses multiplies and adds) can legitimately disagree for a
// ray that passes within floating-point rounding of a triangle EDGE. Such a ray may hit
// one triangle in one test and the neighbour (or nothing) in the other. That is not a bug.
// So every disagreement is checked in double precision: does this pixel's ray pass within
// kEdgeTol (barycentric units) of the boundary of ANY triangle?
//   yes -> "edge-grazing" (expected, counted separately, details printed)
//   no  -> "MISMATCH"     (a real disagreement: this is what fails the run)
// ---------------------------------------------------------------------------
constexpr double kEdgeTol = 1e-5;  // about 100x float epsilon; my judgment, not a Cycles number

struct EdgeProbe {
  double margin = 1e30;  // smallest |barycentric| found among triangles the ray crosses
  int tri = -1;
};

// Double-precision look at the ray against every triangle: does it graze an edge?
static EdgeProbe probe_edge(const float *tris, uint32_t tri_count, const float *o, const float *d)
{
  EdgeProbe best;
  for (uint32_t i = 0; i < tri_count; i++) {
    const float *t = tris + (size_t)i * 9;
    double e1[3], e2[3], tv[3], dd[3];
    for (int a = 0; a < 3; a++) {
      e1[a] = (double)t[3 + a] - t[a];
      e2[a] = (double)t[6 + a] - t[a];
      tv[a] = (double)o[a] - t[a];
      dd[a] = d[a];
    }
    auto cross = [](const double *x, const double *y, double *r) {
      r[0] = x[1] * y[2] - x[2] * y[1];
      r[1] = x[2] * y[0] - x[0] * y[2];
      r[2] = x[0] * y[1] - x[1] * y[0];
    };
    auto dot = [](const double *x, const double *y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    double p[3], q[3];
    cross(dd, e2, p);
    const double det = dot(e1, p);
    if (std::fabs(det) < 1e-300) continue;
    const double u = dot(tv, p) / det;
    cross(tv, e1, q);
    const double v = dot(dd, q) / det;
    const double tt = dot(e2, q) / det;
    if (tt <= 0.0) continue;  // behind the ray origin
    const double w = 1.0 - u - v;
    const double mn = std::min(u, std::min(v, w));
    if (mn < -kEdgeTol) continue;  // clearly outside this triangle
    if (mn < best.margin) {        // inside, or within tolerance of the boundary
      best.margin = std::fabs(mn);
      best.tri = (int)i;
    }
  }
  return best;
}

struct Cmp {
  size_t same = 0;      // same triangle, same distance
  size_t miss = 0;      // both missed
  size_t ties = 0;      // different triangle at (almost) the same distance: a shared edge
  size_t grazing = 0;   // disagreement, but the ray grazes a triangle edge (expected)
  size_t bad = 0;       // disagreement away from any edge: a real problem
  size_t errors = 0;    // an error code on either side
  std::vector<size_t> examples;  // first few disagreeing pixels, for printing
};

struct CmpContext {
  const float *tris;
  uint32_t tri_count;
  const Camera *cam;
  uint32_t W, H;
};

static Cmp compare_hits(const std::vector<Hit> &ref, const std::vector<Hit> &test, const CmpContext &cx)
{
  Cmp c;
  for (size_t i = 0; i < ref.size(); i++) {
    const Hit &r = ref[i], &g = test[i];
    if (cyc::is_error_id(r.id) || cyc::is_error_id(g.id)) {
      c.errors++;
      continue;
    }
    bool disagree = false;
    if (r.id == g.id) {
      if (r.id == kNoHitId)
        c.miss++;
      else if (std::fabs(g.t - r.t) <= 1e-4f * std::max(1.0f, r.t))
        c.same++;
      else
        disagree = true;
    }
    else if (r.id != kNoHitId && g.id != kNoHitId &&
             std::fabs(g.t - r.t) <= 1e-3f * std::max(1.0f, r.t)) {
      c.ties++;
    }
    else {
      disagree = true;  // different triangles at different distances, or hit vs miss
    }
    if (!disagree)
      continue;

    float d[3];
    primary_ray_dir(*cx.cam, (uint32_t)(i % cx.W), (uint32_t)(i / cx.W), cx.W, cx.H, d);
    const EdgeProbe e = probe_edge(cx.tris, cx.tri_count, cx.cam->o, d);
    if (e.margin <= kEdgeTol)
      c.grazing++;
    else
      c.bad++;
    if (c.examples.size() < 5)
      c.examples.push_back(i);
  }
  return c;
}

// ---------------------------------------------------------------------------
// print_tri_diag: for one pixel's ray and one triangle id, print what the EXACT answer would be
// (double precision) and how fragile that answer is in float.
//
// WHY: when two renderers disagree about the distance t to the SAME triangle, the question is
// "who is right, and was the ray hard to get right in the first place?". The exact plane distance
// answers the first. The second is the grazing angle: if the ray runs almost parallel to the
// triangle's plane, the hit point slides a long way along the plane for a tiny change in the ray
// direction (or a tiny rounding error in the maths), so float t is inherently noisy there.
//
//   sin(graz)  = |cos(angle between ray and triangle normal)|; 1 = head-on, 0 = skimming the plane
//   t_per_ulp  = roughly how much t moves if the ray direction is off by 1e-7 (about one float
//                rounding step): t * 1e-7 / sin(graz). If this is as big as the disagreement you
//                see, the disagreement is just float noise amplified by a grazing ray, not a bug.
// ---------------------------------------------------------------------------
static void print_tri_diag(const char *who, const float *tris, uint32_t tri_count, uint32_t id, const float *o,
                           const float *d)
{
  if (id >= tri_count) {  // a miss (4294967295) or an error code: there is no triangle to analyse
    std::printf("        %s: id %u is not a triangle (miss or error code)\n", who, id);
    return;
  }
  const float *t = tris + (size_t)id * 9;
  double e1[3], e2[3], v0o[3], dd[3];
  for (int a = 0; a < 3; a++) {
    e1[a] = (double)t[3 + a] - t[a];       // edge v0->v1
    e2[a] = (double)t[6 + a] - t[a];       // edge v0->v2
    v0o[a] = (double)t[a] - (double)o[a];  // from the ray origin to vertex 0
    dd[a] = d[a];
  }
  // Triangle normal n = e1 x e2 (not normalised: its length is twice the triangle's area).
  const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  const double nlen = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  const double dlen = std::sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]);
  const double dn = n[0] * dd[0] + n[1] * dd[1] + n[2] * dd[2];
  if (nlen == 0.0 || dn == 0.0) {
    std::printf("        %s: tri %u is degenerate or exactly parallel to the ray\n", who, id);
    return;
  }
  // Ray/plane distance: solve dot(n, o + t*d - v0) = 0  ->  t = dot(n, v0 - o) / dot(n, d).
  const double t_exact = (n[0] * v0o[0] + n[1] * v0o[1] + n[2] * v0o[2]) / dn;
  const double sin_graz = std::fabs(dn) / (nlen * dlen);
  std::printf("        %s: tri %u  exact plane distance %.6f  sin(grazing angle) %.2e  t moves ~%.2e per 1e-7 direction error\n",
              who, id, t_exact, sin_graz, std::fabs(t_exact) * 1e-7 / sin_graz);

  // Does the ray actually pass THROUGH this triangle, or only through its plane? Compute the
  // barycentric coordinates of the plane hit in double precision (Moller-Trumbore):
  //   u, v = how far along edges v0->v1 and v0->v2 the hit point lies, w = 1 - u - v.
  // All three >= 0 means inside. The smallest one is the distance to the nearest edge in barycentric
  // units: a hit the float code gets wrong should have this close to 0 (within about 1e-5); if a
  // renderer reports a hit with this clearly negative, or misses with it clearly positive, that is a
  // real error and not rounding.
  const double tv[3] = {-v0o[0], -v0o[1], -v0o[2]};  // from vertex 0 to the ray origin
  const double p[3] = {dd[1] * e2[2] - dd[2] * e2[1], dd[2] * e2[0] - dd[0] * e2[2], dd[0] * e2[1] - dd[1] * e2[0]};
  const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (det == 0.0)
    return;
  const double q[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]};
  const double bu = (tv[0] * p[0] + tv[1] * p[1] + tv[2] * p[2]) / det;
  const double bv = (dd[0] * q[0] + dd[1] * q[1] + dd[2] * q[2]) / det;
  const double bw = 1.0 - bu - bv;
  const double bmin = std::min(bu, std::min(bv, bw));
  std::printf("        %s: barycentric (u,v,w) = (%.6f, %.6f, %.6f), smallest %.3e -> ray passes %s this triangle\n", who,
              bu, bv, bw, bmin, bmin >= 0.0 ? "INSIDE" : "OUTSIDE");
}

static void print_cmp(const char *title, const Cmp &c, const std::vector<Hit> &ref, const std::vector<Hit> &test,
                      const CmpContext &cx)
{
  const size_t n = ref.size();
  std::printf("%s (%zu pixels)\n", title, n);
  std::printf("  both missed:                     %zu\n", c.miss);
  std::printf("  same triangle, same distance:    %zu\n", c.same);
  std::printf("  edge ties (acceptable):          %zu\n", c.ties);
  std::printf("  edge-grazing disagreements:      %zu\n", c.grazing);
  std::printf("  MISMATCHES (not near an edge):   %zu\n", c.bad);
  std::printf("  error codes:                     %zu\n", c.errors);
  for (size_t i : c.examples) {
    float d[3];
    primary_ray_dir(*cx.cam, (uint32_t)(i % cx.W), (uint32_t)(i / cx.W), cx.W, cx.H, d);
    const EdgeProbe e = probe_edge(cx.tris, cx.tri_count, cx.cam->o, d);
    std::printf("    pixel (%zu,%zu): ref id=%u t=%.6f | test id=%u t=%.6f | nearest triangle edge: tri %d, margin %.2e\n",
                i % cx.W, i / cx.W, ref[i].id, ref[i].t, test[i].id, test[i].t, e.tri, e.margin);
    // Extra diagnosis: exact distance and grazing angle for the triangle each side reported.
    print_tri_diag("ref ", cx.tris, cx.tri_count, ref[i].id, cx.cam->o, d);
    print_tri_diag("test", cx.tris, cx.tri_count, test[i].id, cx.cam->o, d);
  }
}

// Edge-grazing pixels are expected to be extremely rare. This limit (0.002% of pixels, at
// least 8) is my judgment: a real bug would disagree on far more pixels than that.
static bool cmp_ok(const Cmp &c, size_t n)
{
  const size_t grazing_limit = std::max<size_t>(8, n / 50000);
  return c.bad == 0 && c.errors == 0 && c.grazing <= grazing_limit;
}

static Buffer upload(const Context &ctx, const void *data, size_t bytes)
{
  Buffer b = create_buffer(ctx, (VkDeviceSize)std::max<size_t>(bytes, 16));
  std::memset(b.mapped, 0, (size_t)b.size);
  if (bytes)
    std::memcpy(b.mapped, data, bytes);
  return b;
}

// ---------------------------------------------------------------------------
// parallel_rows: call body(y) once for every image row y in [0, rows), spread over n_threads threads.
//
// WHY ROWS, AND WHY "DYNAMIC":
//   Every pixel's ray is completely independent of every other pixel's ray, and each one writes only
//   its OWN slot of the output vector. So there is nothing to lock: threads never touch the same
//   memory. (This is the same property that lets the GPU run one thread per pixel.)
//   We hand out work one row at a time from a shared atomic counter instead of giving each thread a
//   fixed block of rows. Rows are NOT equal cost (a row crossing a mesh needs far more traversal
//   steps than a row that only sees empty space), so with fixed blocks the threads that got the
//   cheap rows would finish early and sit idle. With a counter, a thread that finishes a row simply
//   grabs the next free one, so all threads stay busy until the very end.
//
// THREAD-SAFETY RULES for the code called inside body(y):
//   * it may READ shared data (the BVH, the triangles, the camera): nobody writes them.
//   * it may WRITE only to data that belongs to row y (hits[y * W + x]).
//   * anything shared and written (like an error counter) must be a std::atomic.
//   * it must not use static/global mutable state. If cyc::traverse or trace_cpu ever gain a
//     "static" scratch buffer, this stops being safe: look for that first if results ever differ.
//   * it must not throw or exit(): an exception escaping a std::thread calls std::terminate.
//
// With n_threads == 1 no thread is created: body runs on the calling thread, exactly like the
// plain loop it replaces, so the 1-thread baseline is still a fair "original code" measurement.
// ---------------------------------------------------------------------------
template <class Body>
static void parallel_rows(uint32_t rows, unsigned n_threads, const Body &body)
{
  // Never use more threads than rows (the extras would have nothing to do) and never fewer than 1.
  n_threads = std::max(1u, std::min<unsigned>(n_threads, rows));

  // The shared counter. fetch_add returns the OLD value and adds 1 in one indivisible step, so two
  // threads can never receive the same row. "relaxed" ordering is enough: the counter only hands out
  // numbers, it does not publish any data between threads (join() below does that).
  std::atomic<uint32_t> next_row{0};

  auto worker = [&]() {
    for (;;) {
      const uint32_t y = next_row.fetch_add(1, std::memory_order_relaxed);
      if (y >= rows)
        break;  // every row has been claimed: this worker is done
      body(y);
    }
  };

  // The calling thread works too, so we only spawn n_threads - 1 extra ones.
  std::vector<std::thread> pool;
  pool.reserve(n_threads - 1);
  for (unsigned i = 1; i < n_threads; i++)
    pool.emplace_back(worker);
  worker();
  // join() waits for a thread to finish AND guarantees everything it wrote is visible to us.
  for (std::thread &t : pool)
    t.join();
}

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <bvh_traverse.spv | -> <dump_dir> [width] [height] [visibility_hex] [cpu_threads]\n", argv[0]);
    return 1;
  }
  const char *spv_path = argv[1];
  const std::string dump_dir = argv[2];
  const uint32_t W = (argc >= 4) ? (uint32_t)std::atoi(argv[3]) : 256;
  const uint32_t H = (argc >= 5) ? (uint32_t)std::atoi(argv[4]) : 256;
  const uint32_t visibility = (argc >= 6) ? (uint32_t)std::strtoul(argv[5], nullptr, 16) : 0xFFFFFFFFu;
  // How many threads the multi-threaded CPU passes use. hardware_concurrency() can return 0 when the
  // OS will not say, so fall back to 1 in that case.
  const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
  const unsigned n_threads = (argc >= 7) ? (unsigned)std::max(1, std::atoi(argv[6])) : hw_threads;
  const bool gpu_wanted = std::strcmp(spv_path, "-") != 0;
  if (W < 1 || H < 1 || W > 4096 || H > 4096)
    fail("width and height must be between 1 and 4096");

  // ===== 1. load + validate the dump =======================================
  BvhDump dump;
  std::string err;
  if (!load_bvh_dump(dump_dir, dump, err)) {
    std::fprintf(stderr, "FAIL loading dump: %s\n", err.c_str());
    return 1;
  }
  std::printf("Dump: %u inner-node float4s (%u nodes), %u leaves, %u prims, %u triangles, %u vertices, %u objects\n",
              dump.node_count(), dump.node_count() / 4, dump.leaf_count(), dump.prim_count(), dump.tri_count(),
              dump.vert_count(), dump.object_count());
  std::printf("Root address: %d\n", dump.root());

  bool assumptions_ok = true;

  // Assumption: every leaf and every primitive is a plain triangle (type bit 0).
  {
    std::set<uint32_t> leaf_types, prim_types;
    for (uint32_t i = 0; i < dump.leaf_count(); i++) {
      uint32_t t;
      std::memcpy(&t, &dump.leaves[(size_t)i * 4 + 3], 4);
      leaf_types.insert(t);
    }
    for (uint32_t t : dump.prim_type) prim_types.insert(t);
    std::printf("Leaf type words:");
    for (uint32_t t : leaf_types) std::printf(" 0x%x", t);
    std::printf("\nPrimitive type words:");
    for (uint32_t t : prim_types) std::printf(" 0x%x", t);
    std::printf("\n");
    for (uint32_t t : leaf_types)
      if (!(t & cyc::kPrimTriangle)) {
        std::printf("!!! a leaf type word has no triangle bit; my PRIMITIVE_TRIANGLE guess may be wrong. Send me the line above.\n");
        assumptions_ok = false;
      }
  }

  // Assumption: position_offset is the int at byte 168 of KernelObject.
  // Only objects that OWN triangles matter. The scene also contains objects with no
  // triangles (a point light is an object too), and their fields can hold anything.
  {
    std::vector<uint32_t> prims_of(dump.object_count(), 0);
    for (uint32_t s = 0; s < dump.prim_count(); s++)
      if (dump.prim_object[s] < dump.object_count())
        prims_of[dump.prim_object[s]]++;
      else {
        std::printf("!!! primitive slot %u names object %u but there are only %u objects\n", s, dump.prim_object[s],
                    dump.object_count());
        assumptions_ok = false;
      }
    for (uint32_t o = 0; o < dump.object_count(); o++) {
      if (prims_of[o] == 0) {
        std::printf("Object %u: owns no triangles (probably a light); its fields are not used here.\n", o);
        continue;
      }
      std::printf("Object %u: owns %u prim slots | numverts (word %zu) = %u, numprims (word %zu) = %u, position_offset (word %zu) = %u\n",
                  o, prims_of[o], kNumVertsWord, dump.object_numverts(o), kNumPrimsWord, dump.object_numprims(o),
                  kPositionOffsetWord, dump.position_offset(o));
      // Soft confirmation of my hand-counted layout: numprims should equal the triangles
      // this object owns, and numverts should fit inside the vertex array.
      if (dump.object_numprims(o) != prims_of[o])
        std::printf("   note: numprims != prim slots owned. Either my layout guess is off, or the mesh is shared. "
                    "Send me this line.\n");
      if (dump.position_offset(o) + dump.object_numverts(o) > dump.vert_count()) {
        std::printf("!!! position_offset + numverts runs past the vertex array: my guess for these fields is wrong.\n");
        assumptions_ok = false;
      }
    }
  }
  if (!assumptions_ok)
    return 3;

  // ===== 2. reference triangles, indexed by Cycles' triangle number ===========
  // Slot order inside the BVH is shuffled; `prim_index[slot]` says which triangle a slot is.
  const uint32_t tri_count = dump.tri_count();
  std::vector<float> ref_tris((size_t)tri_count * 9, 0.0f);  // zero = degenerate = never hit
  std::vector<int> tri_seen(tri_count, 0);
  for (uint32_t s = 0; s < dump.prim_count(); s++) {
    const uint32_t prim = dump.prim_index[s];
    const uint32_t obj = dump.prim_object[s];
    if (prim >= tri_count || obj >= dump.object_count()) {
      std::fprintf(stderr, "FAIL: slot %u points outside the arrays (prim %u, object %u)\n", s, prim, obj);
      return 3;
    }
    tri_seen[prim]++;
    const uint32_t off = dump.position_offset(obj);
    for (int k = 0; k < 3; k++) {
      const uint32_t v = dump.tri_vindex[(size_t)prim * 3 + k] + off;
      if (v >= dump.vert_count()) {
        std::fprintf(stderr, "FAIL: triangle %u uses vertex %u, but there are only %u vertices\n", prim, v, dump.vert_count());
        return 3;
      }
      for (int a = 0; a < 3; a++)
        ref_tris[(size_t)prim * 9 + k * 3 + a] = dump.tri_verts[(size_t)v * 3 + a];
    }
  }
  size_t covered = 0, repeated = 0;
  for (int n : tri_seen) {
    covered += (n >= 1);
    repeated += (n > 1);
  }
  std::printf("Triangles referenced by the BVH: %zu of %u (repeated: %zu)\n", covered, tri_count, repeated);

  // Per-object bounding boxes of the triangles we just resolved. The exporter bakes each
  // object's world matrix into its vertices, so these boxes must match where the objects
  // are in Blender. If position_offset were read wrongly, a second object would show the
  // FIRST object's box here.
  {
    std::vector<std::array<float, 6>> boxes(dump.object_count(), {kNoHitT, kNoHitT, kNoHitT, -kNoHitT, -kNoHitT, -kNoHitT});
    for (uint32_t s = 0; s < dump.prim_count(); s++) {
      const uint32_t prim = dump.prim_index[s], obj = dump.prim_object[s];
      for (int p = 0; p < 3; p++)
        for (int a = 0; a < 3; a++) {
          const float v = ref_tris[(size_t)prim * 9 + p * 3 + a];
          boxes[obj][a] = std::min(boxes[obj][a], v);
          boxes[obj][3 + a] = std::max(boxes[obj][3 + a], v);
        }
    }
    for (uint32_t o = 0; o < dump.object_count(); o++)
      if (boxes[o][0] <= boxes[o][3])
        std::printf("Object %u bounding box: min (%.4f, %.4f, %.4f)  max (%.4f, %.4f, %.4f)\n", o, boxes[o][0],
                    boxes[o][1], boxes[o][2], boxes[o][3], boxes[o][4], boxes[o][5]);
  }

  float bmin[3] = {kNoHitT, kNoHitT, kNoHitT}, bmax[3] = {-kNoHitT, -kNoHitT, -kNoHitT};
  for (uint32_t i = 0; i < tri_count; i++) {
    if (!tri_seen[i]) continue;
    for (int p = 0; p < 3; p++)
      for (int a = 0; a < 3; a++) {
        bmin[a] = std::min(bmin[a], ref_tris[(size_t)i * 9 + p * 3 + a]);
        bmax[a] = std::max(bmax[a], ref_tris[(size_t)i * 9 + p * 3 + a]);
      }
  }
  const Camera cam = make_camera(bmin, bmax, 40.0f, (float)W / (float)H);
  const size_t pixel_count = (size_t)W * H;

  // ===== 3. CPU A: brute force ==============================================
  // Multi-threaded: brute force is rays x triangles, by far the slowest step, and every row is independent.
  // Each iteration writes only hits_brute[y * W + x] for its own row y, so no locking is needed.
  std::vector<Hit> hits_brute(pixel_count);
  parallel_rows(H, n_threads, [&](uint32_t y) {
    for (uint32_t x = 0; x < W; x++) {
      float d[3];
      primary_ray_dir(cam, x, y, W, H, d);
      hits_brute[(size_t)y * W + x] = trace_cpu(ref_tris.data(), tri_count, cam.o, d);
    }
  });
  std::printf("\nCPU A (brute force, no BVH) done.\n");

  // ===== 4. CPU B: Cycles' traversal, in C++ ================================
  const cyc::SceneView view = cyc::make_view(dump);
  // One BVH pass over the whole image with `threads` threads. Writes into `out`, stores the number of
  // error codes in `errors`, returns the elapsed milliseconds. Used twice below, so that the 1-thread
  // and N-thread runs are guaranteed to execute the exact same code.
  auto run_bvh_pass = [&](std::vector<Hit> &out, unsigned threads, size_t &errors) -> double {
    // Shared by all threads and written by several of them -> must be atomic. (Errors are rare, so
    // the cost of the atomic increment does not matter.)
    std::atomic<size_t> err_count{0};
    const auto start = std::chrono::steady_clock::now();
    parallel_rows(H, threads, [&](uint32_t y) {
      for (uint32_t x = 0; x < W; x++) {
        float d[3];
        primary_ray_dir(cam, x, y, W, H, d);
        float t;
        uint32_t prim;
        const uint32_t status = cyc::traverse(view, {cam.o[0], cam.o[1], cam.o[2]}, {d[0], d[1], d[2]}, kRayTMin,
                                              kNoHitT, visibility, t, prim);
        Hit &h = out[(size_t)y * W + x];  // this pixel's own slot: no other thread writes it
        if (status != 0) {
          h = {-1.0f, status};
          err_count.fetch_add(1, std::memory_order_relaxed);
        }
        else {
          h = {t, prim};
        }
      }
    });
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    errors = err_count.load();
    return ms;
  };

  // --- B1: single thread. This is the original baseline; every comparison below uses these results.
  std::vector<Hit> hits_cpu_bvh(pixel_count);
  size_t cpu_err = 0;
  const double cpu_ms = run_bvh_pass(hits_cpu_bvh, 1, cpu_err);
  std::printf("CPU B (Cycles traversal in C++, 1 thread) done: %.1f ms, %.2f Mrays/s, error codes: %zu\n", cpu_ms,
              (double)pixel_count / cpu_ms / 1000.0, cpu_err);

  // --- B2: same traversal, n_threads threads. Purely a speed reference for the real alternative a
  // user has (their multi-core CPU). It must reproduce B1 exactly: same code, same inputs, and the
  // pixels are independent, so ANY difference means a threading bug (a data race), not a rounding issue.
  std::vector<Hit> hits_cpu_bvh_mt(pixel_count);
  size_t cpu_mt_err = 0;
  const double cpu_mt_ms = run_bvh_pass(hits_cpu_bvh_mt, n_threads, cpu_mt_err);
  size_t mt_diffs = 0;
  for (size_t i = 0; i < pixel_count; i++)
    if (float_bits(hits_cpu_bvh[i].t) != float_bits(hits_cpu_bvh_mt[i].t) || hits_cpu_bvh[i].id != hits_cpu_bvh_mt[i].id)
      mt_diffs++;
  const bool mt_ok = (mt_diffs == 0) && (cpu_mt_err == cpu_err);
  std::printf("CPU B (same traversal, %u threads) done: %.1f ms, %.2f Mrays/s (%.1fx the 1-thread run), "
              "results identical to 1 thread: %s (%zu pixel differences)\n",
              n_threads, cpu_mt_ms, (double)pixel_count / cpu_mt_ms / 1000.0, cpu_ms / cpu_mt_ms,
              mt_ok ? "YES" : "NO !!!", mt_diffs);

  const CmpContext cx{ref_tris.data(), tri_count, &cam, W, H};
  const Cmp ab = compare_hits(hits_brute, hits_cpu_bvh, cx);
  std::printf("\n");
  print_cmp("A (brute force) vs B (Cycles BVH traversal, C++)", ab, hits_brute, hits_cpu_bvh, cx);
  const bool ab_ok = cmp_ok(ab, pixel_count);

  write_ppm("depth_cpu_brute.ppm", W, H, hits_to_depth_rgb(hits_brute));
  write_ppm("depth_cpu_bvh.ppm", W, H, hits_to_depth_rgb(hits_cpu_bvh));

  if (!gpu_wanted) {
    std::printf("\nGPU skipped (spv = \"-\").\n%s\n", (ab_ok && mt_ok) ? "CPU-ONLY PASS" : "CPU-ONLY FAIL: see the numbers above");
    return (ab_ok && mt_ok) ? 0 : 2;
  }

  // ===== 5. GPU =============================================================
  Context ctx = create_context();
  Kernel kernel = create_kernel(ctx, spv_path, (uint32_t)sizeof(TraversePush));
  print_kernel_stats(ctx, kernel);

  Buffer b_nodes = upload(ctx, dump.nodes.data(), dump.nodes.size() * 4);
  Buffer b_leaves = upload(ctx, dump.leaves.data(), dump.leaves.size() * 4);
  Buffer b_pindex = upload(ctx, dump.prim_index.data(), dump.prim_index.size() * 4);
  Buffer b_pvis = upload(ctx, dump.prim_visibility.data(), dump.prim_visibility.size() * 4);
  Buffer b_pobj = upload(ctx, dump.prim_object.data(), dump.prim_object.size() * 4);
  Buffer b_vindex = upload(ctx, dump.tri_vindex.data(), dump.tri_vindex.size() * 4);
  Buffer b_verts = upload(ctx, dump.tri_verts.data(), dump.tri_verts.size() * 4);
  Buffer b_objects = upload(ctx, dump.objects.data(), dump.objects.size() * 4);

  SceneDataGpu sd{};
  sd.bvh_nodes = b_nodes.address;
  sd.bvh_leaf_nodes = b_leaves.address;
  sd.prim_index = b_pindex.address;
  sd.prim_visibility = b_pvis.address;
  sd.prim_object = b_pobj.address;
  sd.tri_vindex = b_vindex.address;
  sd.tri_verts = b_verts.address;
  sd.objects = b_objects.address;
  sd.root = dump.root();
  sd.node_count = dump.node_count();
  sd.leaf_count = dump.leaf_count();
  sd.prim_count = dump.prim_count();
  sd.tri_count = dump.tri_count();
  sd.vert_count = dump.vert_count();
  sd.object_count = dump.object_count();
  Buffer b_scene = upload(ctx, &sd, sizeof(sd));

  const VkDeviceSize out_bytes = (VkDeviceSize)(pixel_count + kCanaryHits) * sizeof(Hit);
  Buffer out_buf = create_buffer(ctx, out_bytes);
  {
    uint32_t *words = static_cast<uint32_t *>(out_buf.mapped);
    const size_t real_words = pixel_count * 2;
    for (size_t i = 0; i < real_words; i++) words[i] = kUnwrittenWord;
    for (size_t i = 0; i < (size_t)kCanaryHits * 2; i++) words[real_words + i] = kCanaryWord;
  }
  std::printf("\nout buffer: %llu bytes @ 0x%llx (%zu real pixels + %u canary slots)\n", (unsigned long long)out_bytes,
              (unsigned long long)out_buf.address, pixel_count, kCanaryHits);

  TraversePush pc{};
  pc.scene = b_scene.address;
  pc.out = out_buf.address;
  pc.width = W;
  pc.height = H;
  pc.visibility = visibility;
  for (int i = 0; i < 3; i++) {
    pc.o[i] = cam.o[i];
    pc.l[i] = cam.l[i];
    pc.u[i] = cam.u[i];
    pc.v[i] = cam.v[i];
  }

  const uint32_t groups_x = (W + 7) / 8, groups_y = (H + 7) / 8;
  const double ms = run_kernel(ctx, kernel, &pc, groups_x, groups_y);
  std::printf("GPU finished: %ux%u groups of 8x8 threads, %.3f ms, %.2f Mrays/s\n", groups_x, groups_y, ms,
              (double)pixel_count / ms / 1000.0);

  // ===== safety checks + comparisons =========================================
  const Hit *gpu_raw = static_cast<const Hit *>(out_buf.mapped);
  bool canaries_ok = true;
  for (uint32_t i = 0; i < kCanaryHits; i++) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&gpu_raw[pixel_count + i]);
    if (w[0] != kCanaryWord || w[1] != kCanaryWord) canaries_ok = false;
  }
  std::printf("\nCanaries after the output buffer: %s\n",
              canaries_ok ? "INTACT" : "!!! OVERWRITTEN: the shader wrote out of bounds !!!");

  std::vector<Hit> hits_gpu(gpu_raw, gpu_raw + pixel_count);
  size_t unwritten = 0, gpu_err = 0;
  for (const Hit &h : hits_gpu) {
    if (float_bits(h.t) == kUnwrittenWord && h.id == kUnwrittenWord) unwritten++;
    else if (cyc::is_error_id(h.id)) gpu_err++;
  }
  std::printf("Pixels the shader never wrote: %zu\n", unwritten);
  std::printf("Pixels where the shader returned an error code: %zu\n", gpu_err);
  if (gpu_err) {
    for (uint32_t code = cyc::kErrLoop; code <= cyc::kErrInstance; code++) {
      size_t n = 0;
      for (const Hit &h : hits_gpu) n += (h.id == code);
      if (n) std::printf("    0x%X: %zu pixels (FB=loop cap, FC=stack, FD=bad index, FE=instance leaf)\n", code, n);
    }
  }

  std::printf("\n");
  const Cmp bg = compare_hits(hits_cpu_bvh, hits_gpu, cx);
  print_cmp("B (Cycles traversal, C++) vs GPU (Cycles traversal, Slang)", bg, hits_cpu_bvh, hits_gpu, cx);
  std::printf("\n");
  const Cmp ag = compare_hits(hits_brute, hits_gpu, cx);
  print_cmp("A (brute force) vs GPU", ag, hits_brute, hits_gpu, cx);

  std::printf("\nSpeed (same rays, same BVH):  CPU 1 thread %.2f Mrays/s   CPU %u threads %.2f Mrays/s   GPU %.2f Mrays/s\n",
              (double)pixel_count / cpu_ms / 1000.0, n_threads, (double)pixel_count / cpu_mt_ms / 1000.0,
              (double)pixel_count / ms / 1000.0);
  std::printf("GPU vs CPU %u threads: %.1fx   (GPU time includes fixed launch overhead, see notes)\n", n_threads,
              cpu_mt_ms / ms);

  write_ppm("depth_gpu_bvh.ppm", W, H, hits_to_depth_rgb(hits_gpu));
  write_ppm("ids_gpu_bvh.ppm", W, H, hits_to_id_rgb(hits_gpu));
  std::printf("Wrote depth_cpu_brute.ppm, depth_cpu_bvh.ppm, depth_gpu_bvh.ppm, ids_gpu_bvh.ppm\n");

  const bool ok = ab_ok && mt_ok && canaries_ok && unwritten == 0 && gpu_err == 0 && cmp_ok(bg, pixel_count) &&
                  cmp_ok(ag, pixel_count);
  std::printf("%s\n", ok ? "PASS" : "FAIL: see the numbers above");

  destroy_buffer(ctx, out_buf);
  destroy_buffer(ctx, b_scene);
  destroy_buffer(ctx, b_objects);
  destroy_buffer(ctx, b_verts);
  destroy_buffer(ctx, b_vindex);
  destroy_buffer(ctx, b_pobj);
  destroy_buffer(ctx, b_pvis);
  destroy_buffer(ctx, b_pindex);
  destroy_buffer(ctx, b_leaves);
  destroy_buffer(ctx, b_nodes);
  destroy_kernel(ctx, kernel);
  destroy_context(ctx);
  return ok ? 0 : 2;
}
