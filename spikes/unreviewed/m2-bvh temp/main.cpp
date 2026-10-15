// M2 spike: build a BVH on the CPU with binned SAH, then PROVE it is correct
// by comparing it against brute force on thousands of rays.
//
// Usage:  bvh_test [path-to-model.obj]
//
// Read the files in this order:
//   obj_loader.h  load triangles from an .obj  (your parser)
//   bvh.h         the node layout + the SAH builder  <- the heart of M2
//   cpu_ref.h     ray-triangle, ray-box, BVH traversal, brute force
//   main.cpp      this file: glue + the test
//
// There is no GPU code here yet. That is on purpose: if the CPU version is
// proven right first, then any disagreement later is a GPU/shader bug.
#include "bvh.h"
#include "cpu_ref.h"
#include "obj_loader.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

[[noreturn]] static void fail(const char* msg)
{
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

// ---------------------------------------------------------------------------
// Ray generation
// ---------------------------------------------------------------------------

// One ray per pixel of a W x H "photo" taken from a camera that looks at the
// middle of the model. This is the same idea as the shader will use in M2.
static std::vector<Ray> make_camera_rays(const AABB& scene, int W, int H)
{
    const Vec3 center = (scene.lo + scene.hi) * 0.5f;
    const float radius = length(scene.hi - scene.lo) * 0.5f;

    const Vec3 cam = center + normalize({0.6f, 0.5f, 1.0f}) * (radius * 3.0f);
    const Vec3 fwd = normalize(center - cam);
    const Vec3 right = normalize(cross(fwd, {0.0f, 1.0f, 0.0f}));
    const Vec3 up = cross(right, fwd);
    const float half = 0.4663f;  // tan(25 degrees): the model's bounding sphere fits the view

    std::vector<Ray> rays;
    rays.reserve((size_t)W * H);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            const float px = (((float)i + 0.5f) / (float)W * 2.0f - 1.0f) * half;
            const float py = (1.0f - ((float)j + 0.5f) / (float)H * 2.0f) * half;
            rays.push_back({cam, normalize(fwd + right * px + up * py)});
        }
    }
    return rays;
}

// Rays from random places around the model in random directions. Many of them
// graze edges, start inside boxes, or miss everything: good for catching bugs
// the tidy camera rays would never trigger.
static std::vector<Ray> make_random_rays(const AABB& scene, int count, unsigned seed)
{
    const Vec3 center = (scene.lo + scene.hi) * 0.5f;
    const float radius = length(scene.hi - scene.lo) * 0.5f;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);  // normalized gaussians = uniform directions

    std::vector<Ray> rays;
    rays.reserve(count);
    for (int k = 0; k < count; ++k) {
        const Vec3 o = center + Vec3{uni(rng), uni(rng), uni(rng)} * (radius * 1.5f);
        Vec3 d{gauss(rng), gauss(rng), gauss(rng)};
        if (length(d) < 1e-6f) d = {0.0f, 0.0f, 1.0f};
        rays.push_back({o, normalize(d)});
    }
    return rays;
}

// Rays travelling exactly along +x, -x, +y, -y, +z or -z, starting on a grid
// whose lines include the model's bounding box edges. These are the nasty cases
// for the box test: a direction component of exactly 0 means 1/0 = infinity, and
// an origin sitting exactly on a box face makes (0 * infinity) = NaN. The
// camera and random rays practically never produce either.
static std::vector<Ray> make_axis_aligned_rays(const AABB& scene, int steps)
{
    const float radius = length(scene.hi - scene.lo) * 0.5f;
    std::vector<Ray> rays;
    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3, v = (axis + 2) % 3;  // the two axes across the face
        for (int sign = -1; sign <= 1; sign += 2) {
            for (int gi = 0; gi <= steps; ++gi) {
                for (int gj = 0; gj <= steps; ++gj) {
                    float o[3], d[3] = {0.0f, 0.0f, 0.0f};
                    const float fu = (float)gi / (float)steps, fv = (float)gj / (float)steps;
                    o[u] = axis_of(scene.lo, u) + (axis_of(scene.hi, u) - axis_of(scene.lo, u)) * fu;
                    o[v] = axis_of(scene.lo, v) + (axis_of(scene.hi, v) - axis_of(scene.lo, v)) * fv;
                    // Start outside the model, on the side the ray is travelling away from.
                    o[axis] = (sign > 0) ? axis_of(scene.lo, axis) - radius : axis_of(scene.hi, axis) + radius;
                    d[axis] = (float)sign;
                    rays.push_back({{o[0], o[1], o[2]}, {d[0], d[1], d[2]}});
                }
            }
        }
    }
    return rays;
}

// ---------------------------------------------------------------------------
// The test: BVH vs brute force
// ---------------------------------------------------------------------------
struct CheckResult {
    uint32_t rays = 0, hits = 0, mismatches = 0;
    TraceCounters bvh, brute;
};

static CheckResult compare_bvh_to_brute_force(const BVH& bvh, const std::vector<Triangle>& tris,
                                              const std::vector<Ray>& rays, std::vector<Hit>* bvh_hits_out)
{
    CheckResult r;
    r.rays = (uint32_t)rays.size();
    if (bvh_hits_out) bvh_hits_out->clear();

    for (const Ray& ray : rays) {
        const Hit hb = trace_brute_force(tris, ray, r.brute);
        const Hit ha = trace_bvh(bvh, tris, ray, r.bvh);
        if (bvh_hits_out) bvh_hits_out->push_back(ha);
        if (hb.valid()) r.hits++;

        // They must agree on hit/miss and on the DISTANCE. We deliberately do not
        // compare triangle ids: when a ray hits exactly the shared edge of two
        // triangles, either one is a correct answer (same t).
        bool ok = (hb.valid() == ha.valid());
        if (ok && hb.valid()) ok = std::fabs(hb.t - ha.t) <= 1e-4f * std::max(1.0f, hb.t);
        if (!ok) {
            if (r.mismatches < 5)
                std::fprintf(stderr, "  MISMATCH: brute %s t=%g tri=%u | bvh %s t=%g tri=%u\n",
                             hb.valid() ? "hit" : "miss", hb.t, hb.tri,
                             ha.valid() ? "hit" : "miss", ha.t, ha.tri);
            r.mismatches++;
        }
    }
    return r;
}

static void print_check(const char* name, const CheckResult& r, uint32_t tri_count)
{
    std::printf("%s: %u rays, %u hit the model, %u mismatches\n", name, r.rays, r.hits, r.mismatches);
    const double n = (double)r.rays;
    std::printf("  triangle tests per ray: brute force %.1f, BVH %.2f  | box tests per ray (BVH): %.2f\n",
                (double)r.brute.tri_tests / n, (double)r.bvh.tri_tests / n, (double)r.bvh.box_tests / n);
    (void)tri_count;
}

// A tiny text "picture" of the camera rays, shaded by distance (near = dense).
// It is only a sanity check; the real image output comes later in M2.
static void print_ascii_depth(const std::vector<Hit>& hits, int W, int H)
{
    float tmin = FLT_MAX, tmax = 0.0f;
    for (const Hit& h : hits)
        if (h.valid()) { tmin = std::min(tmin, h.t); tmax = std::max(tmax, h.t); }
    if (tmin > tmax) { std::printf("(camera sees nothing)\n"); return; }

    const char* ramp = "@%#*+=-:. ";  // dense -> sparse
    std::printf("\nDepth preview from the camera (near = @, far = ., background = blank):\n");
    for (int j = 0; j < H; j += 2) {  // every 2nd row: text characters are ~2x taller than wide
        for (int i = 0; i < W; ++i) {
            const Hit& h = hits[(size_t)j * W + i];
            if (!h.valid()) { std::putchar(' '); continue; }
            const float f = (tmax > tmin) ? (h.t - tmin) / (tmax - tmin) : 0.0f;
            std::putchar(ramp[std::min(8, (int)(f * 9.0f))]);
        }
        std::putchar('\n');
    }
}

int main(int argc, char** argv)
{
    // Default path = the one used in the parser spike. Pass another as argv[1].
    const std::string path = (argc >= 2) ? argv[1] : "c:/dev/cycles-vulkan/assets/cube.obj";

    // 1) Load.
    std::vector<Triangle> tris;
    if (!load_obj(path, tris)) fail("could not load the model");
    if (tris.empty()) fail("the model has no triangles");

    // 2) Build.
    const auto t0 = std::chrono::steady_clock::now();
    BVH bvh = build_bvh_sah(tris);
    const auto t1 = std::chrono::steady_clock::now();
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const BVHStats st = compute_stats(bvh);
    std::printf("\nBVH built in %.3f ms (binned SAH, %d bins, max %u triangles per leaf)\n",
                build_ms, bvh_cfg::kBins, bvh_cfg::kMaxLeafTris);
    std::printf("  nodes: %u  (%u interior + %u leaves)   [%zu bytes]\n",
                st.nodes, st.interior, st.leaves, bvh.nodes.size() * sizeof(BVHNode));
    std::printf("  depth: %u   triangles per leaf: min %u, avg %.2f, max %u\n",
                st.max_depth, st.min_leaf, st.avg_leaf, st.max_leaf);
    std::printf("  SAH cost of the whole tree: %.2f   (a single big leaf would cost %u)\n",
                st.sah_cost, (uint32_t)tris.size());

    if (st.max_depth + 2 >= (uint32_t)kMaxStack) fail("tree is deeper than the traversal stack");

    // 3) Is the tree legal?
    if (!validate_bvh(bvh, tris)) fail("BVH structure check failed");
    std::printf("Structure check: PASS (every triangle in exactly one leaf, all boxes nest properly)\n\n");

    // 4) Is it CORRECT? Compare with brute force.
    AABB scene;
    for (const Triangle& t : tris) scene.grow(triangle_bounds(t));

    const int W = 64, H = 64;
    std::vector<Hit> camera_hits;
    const CheckResult cam = compare_bvh_to_brute_force(bvh, tris, make_camera_rays(scene, W, H), &camera_hits);
    const CheckResult rnd = compare_bvh_to_brute_force(bvh, tris, make_random_rays(scene, 20000, 42), nullptr);
    const CheckResult axs = compare_bvh_to_brute_force(bvh, tris, make_axis_aligned_rays(scene, 16), nullptr);
    print_check("Camera rays", cam, (uint32_t)tris.size());
    print_check("Random rays", rnd, (uint32_t)tris.size());
    print_check("Axis-aligned rays", axs, (uint32_t)tris.size());

    print_ascii_depth(camera_hits, W, H);

    const bool ok = (cam.mismatches == 0 && rnd.mismatches == 0 && axs.mismatches == 0);
    std::printf("\n%s\n", ok ? "PASS" : "MISMATCH: the BVH disagrees with brute force");
    return ok ? 0 : 2;
}
