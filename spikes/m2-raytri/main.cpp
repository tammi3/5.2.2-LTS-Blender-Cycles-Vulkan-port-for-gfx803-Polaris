// M2a: trace rays against a triangle mesh on the GPU, compare with the CPU, save images.
//
// Usage:  m2_raytri <raytri.spv> <mesh.obj> [width] [height]
//
// ###########################################################################
// ##  RAW GPU POINTERS HAVE NO BOUNDS CHECKING.                            ##
// ##  The validation layer and spirv-val CANNOT see inside them at runtime. ##
// ##  A wrong size or offset means reading/writing memory you do not own.   ##
// ##  Every struct below is locked with static_assert. Every buffer that    ##
// ##  the shader WRITES has canary words after it. Keep it that way.        ##
// ###########################################################################
//
// What happens, in order:
//   1. load the mesh and place a camera
//   2. CPU: trace every pixel (the "answer key")
//   3. GPU: upload triangles, run the shader (one thread per pixel)
//   4. safety checks (canaries, unwritten pixels)
//   5. compare GPU to CPU, write .ppm images

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "obj_loader.h"
#include "ppm.h"
#include "raytri_cpu.h"
#include "vk_util.h"

// ---------------------------------------------------------------------------
// The push-constant block. MUST match `struct Params` in raytri.slang, offset by offset.
// (After compiling, check the "OpMemberDecorate %Params_std430 N Offset X" lines in the
//  disassembly: they must say the same numbers as the asserts below.)
// ---------------------------------------------------------------------------
struct RayTriPush {
  uint64_t tris;      // offset  0
  uint64_t out;       // offset  8
  uint32_t tri_count; // offset 16
  uint32_t width;     // offset 20
  uint32_t height;    // offset 24
  float o[3];         // offset 28
  float l[3];         // offset 40
  float u[3];         // offset 52
  float v[3];         // offset 64
};
static_assert(offsetof(RayTriPush, tris) == 0, "layout");
static_assert(offsetof(RayTriPush, out) == 8, "layout");
static_assert(offsetof(RayTriPush, tri_count) == 16, "layout");
static_assert(offsetof(RayTriPush, width) == 20, "layout");
static_assert(offsetof(RayTriPush, height) == 24, "layout");
static_assert(offsetof(RayTriPush, o) == 28, "layout");
static_assert(offsetof(RayTriPush, l) == 40, "layout");
static_assert(offsetof(RayTriPush, u) == 52, "layout");
static_assert(offsetof(RayTriPush, v) == 64, "layout");
static_assert(sizeof(RayTriPush) <= 128, "push constants are limited to 128 bytes on this GPU");

// Safety markers
static constexpr uint32_t kCanaryHits = 16;          // extra Hit slots after the real output
static constexpr uint32_t kCanaryWord = 0xCAFEBABEu; // pattern in the canary slots
static constexpr uint32_t kUnwrittenWord = 0xDEADBEEFu; // pattern in slots the shader must overwrite

static uint32_t float_bits(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <raytri.spv> <mesh.obj> [width] [height]\n", argv[0]);
    return 1;
  }
  const char *spv_path = argv[1];
  const char *obj_path = argv[2];
  const uint32_t W = (argc >= 4) ? (uint32_t)std::atoi(argv[3]) : 256;
  const uint32_t H = (argc >= 5) ? (uint32_t)std::atoi(argv[4]) : 256;
  if (W < 1 || H < 1 || W > 4096 || H > 4096)
    fail("width and height must be between 1 and 4096");

  // ===== 1. mesh + camera ==================================================
  TriangleMesh mesh;
  std::string err;
  if (!load_obj_positions(obj_path, mesh, err)) {
    std::fprintf(stderr, "FAIL loading %s: %s\n", obj_path, err.c_str());
    return 1;
  }
  const uint32_t tri_count = (uint32_t)mesh.triangle_count();
  std::printf("Loaded %s: %u triangles\n", obj_path, tri_count);

  float bmin[3] = {kNoHitT, kNoHitT, kNoHitT}, bmax[3] = {-kNoHitT, -kNoHitT, -kNoHitT};
  for (size_t i = 0; i < mesh.positions.size(); i += 3)
    for (int a = 0; a < 3; a++) {
      bmin[a] = std::min(bmin[a], mesh.positions[i + a]);
      bmax[a] = std::max(bmax[a], mesh.positions[i + a]);
    }
  const Camera cam = make_camera(bmin, bmax, 40.0f, (float)W / (float)H);

  // ===== 2. CPU answer key =================================================
  std::vector<Hit> cpu_hits((size_t)W * H);
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      float d[3];
      primary_ray_dir(cam, x, y, W, H, d);
      cpu_hits[(size_t)y * W + x] = trace_cpu(mesh.positions.data(), tri_count, cam.o, d);
    }
  std::printf("CPU reference done (%ux%u).\n", W, H);

  // ===== 3. GPU ============================================================
  Context ctx = create_context();
  Kernel kernel = create_kernel(ctx, spv_path, (uint32_t)sizeof(RayTriPush));
  print_kernel_stats(ctx, kernel);

  const VkDeviceSize tris_bytes = (VkDeviceSize)tri_count * 9 * sizeof(float);
  const size_t pixel_count = (size_t)W * H;
  const VkDeviceSize out_bytes = (VkDeviceSize)(pixel_count + kCanaryHits) * sizeof(Hit);

  Buffer tris_buf = create_buffer(ctx, tris_bytes);
  Buffer out_buf = create_buffer(ctx, out_bytes);
  std::printf("tris buffer: %llu bytes @ 0x%llx\n", (unsigned long long)tris_bytes,
              (unsigned long long)tris_buf.address);
  std::printf("out  buffer: %llu bytes @ 0x%llx  (%u real pixels + %u canary slots)\n",
              (unsigned long long)out_bytes, (unsigned long long)out_buf.address, (unsigned)pixel_count,
              kCanaryHits);

  // Upload triangles.
  std::memcpy(tris_buf.mapped, mesh.positions.data(), (size_t)tris_bytes);

  // Prepare the output: real slots = "unwritten" marker, extra slots = canaries.
  {
    uint32_t *words = static_cast<uint32_t *>(out_buf.mapped);
    const size_t real_words = pixel_count * 2;
    for (size_t i = 0; i < real_words; i++)
      words[i] = kUnwrittenWord;
    for (size_t i = 0; i < (size_t)kCanaryHits * 2; i++)
      words[real_words + i] = kCanaryWord;
  }

  RayTriPush pc{};
  pc.tris = tris_buf.address;
  pc.out = out_buf.address;
  pc.tri_count = tri_count;
  pc.width = W;
  pc.height = H;
  for (int i = 0; i < 3; i++) {
    pc.o[i] = cam.o[i];
    pc.l[i] = cam.l[i];
    pc.u[i] = cam.u[i];
    pc.v[i] = cam.v[i];
  }

  // The shader's workgroups are 8x8 threads, so round the grid UP to cover the image.
  // (Threads that fall outside the image return early inside the shader.)
  const uint32_t groups_x = (W + 7) / 8;
  const uint32_t groups_y = (H + 7) / 8;
  const double ms = run_kernel(ctx, kernel, &pc, groups_x, groups_y);
  std::printf("GPU finished: %ux%u groups of 8x8 threads, %.3f ms\n", groups_x, groups_y, ms);

  // ===== 4. safety checks ==================================================
  const Hit *gpu_raw = static_cast<const Hit *>(out_buf.mapped);

  bool canaries_ok = true;
  for (uint32_t i = 0; i < kCanaryHits; i++) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&gpu_raw[pixel_count + i]);
    if (w[0] != kCanaryWord || w[1] != kCanaryWord)
      canaries_ok = false;
  }
  std::printf("\nCanaries after the output buffer: %s\n",
              canaries_ok ? "INTACT" : "!!! OVERWRITTEN: the shader wrote out of bounds !!!");

  std::vector<Hit> gpu_hits(gpu_raw, gpu_raw + pixel_count);
  size_t unwritten = 0;
  for (const Hit &h : gpu_hits)
    if (float_bits(h.t) == kUnwrittenWord && h.id == kUnwrittenWord)
      unwritten++;
  std::printf("Pixels the shader never wrote: %zu\n", unwritten);

  // ===== 5. compare GPU vs CPU ============================================
  size_t exact = 0, miss_both = 0, ties = 0, bad = 0;
  for (size_t i = 0; i < pixel_count; i++) {
    const Hit &g = gpu_hits[i], &c = cpu_hits[i];
    if (float_bits(g.t) == kUnwrittenWord && g.id == kUnwrittenWord)
      continue; // already counted as unwritten
    if (g.id == c.id) {
      if (c.id == kNoHitId)
        miss_both++;
      else if (std::fabs(g.t - c.t) <= 1e-4f * std::max(1.0f, c.t))
        exact++;
      else
        bad++; // same triangle but different distance: a real discrepancy
    }
    else if (g.id != kNoHitId && c.id != kNoHitId &&
             std::fabs(g.t - c.t) <= 1e-3f * std::max(1.0f, c.t)) {
      ties++; // different triangle at (almost) the same distance: a pixel on a shared edge
    }
    else {
      bad++;
    }
  }
  std::printf("\nGPU vs CPU over %zu pixels:\n", pixel_count);
  std::printf("  both missed:                     %zu\n", miss_both);
  std::printf("  same triangle, same distance:    %zu\n", exact);
  std::printf("  edge ties (acceptable):          %zu\n", ties);
  std::printf("  MISMATCHES:                      %zu\n", bad);

  // ===== images ============================================================
  write_ppm("depth_gpu.ppm", W, H, hits_to_depth_rgb(gpu_hits));
  write_ppm("ids_gpu.ppm", W, H, hits_to_id_rgb(gpu_hits));
  write_ppm("depth_cpu.ppm", W, H, hits_to_depth_rgb(cpu_hits));
  std::printf("\nWrote depth_gpu.ppm, ids_gpu.ppm, depth_cpu.ppm in the current folder.\n");

  const bool ok = canaries_ok && unwritten == 0 && bad == 0;
  std::printf("%s\n", ok ? "PASS" : "FAIL: see the numbers above");

  destroy_buffer(ctx, out_buf);
  destroy_buffer(ctx, tris_buf);
  destroy_kernel(ctx, kernel);
  destroy_context(ctx);
  return ok ? 0 : 2;
}
