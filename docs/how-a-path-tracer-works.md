# How a path tracer works (and where our project fits)

Written for the Cycles-on-Vulkan port for the RX 570 (gfx803 / Polaris).
Everything about Cycles' structure below was checked against the `release/5.2` source.

---

## 1. The one-sentence idea

A path tracer makes an image by simulating light **backwards**: for every pixel,
shoot a ray from the camera into the scene, see what it hits, bounce it somewhere
sensible, repeat, and add up the light it picks up along the way. One path gives a
very noisy answer; averaging hundreds of paths per pixel gives a clean one.

## 2. What one path does

For a single pixel, one sample:

1. **Camera ray.** Pick a random point inside the pixel, make a ray from the camera through it.
2. **Intersect.** Find the closest surface the ray hits. (No hit: add the sky/world color, stop.)
3. **Shade.** Look at the material at that point. A material answers two questions:
   *how much light does this surface emit?* and *how does it scatter light that arrives?*
   (That scattering rule is called the **BSDF**; "Principled BSDF" is Blender's all-purpose one.)
4. **Direct light.** Pick a light, shoot a **shadow ray** toward it. If nothing blocks it,
   add that light's contribution. (Cycles calls this *next event estimation*, `shade_light_nee` in the source.)
5. **Bounce.** Pick a random direction according to the BSDF, shoot a new ray, go back to step 2.
6. **Stop** after a maximum number of bounces, or randomly ("Russian roulette") when the path
   would contribute little.
7. **Write.** Add the collected color into that pixel's accumulator in the film buffer.

Repeat for N samples per pixel, divide by N.

## 3. Why this suits a GPU

Every pixel's path is independent of every other pixel's. That is exactly the shape of
work GPUs are built for: **one small program, run on a huge number of independent items.**

If you know PySpark: a shader is like the function you pass to `withColumn`.
You write it once; the engine runs it on every row in parallel; each copy only knows its own row.
(Here, "its own row" is its thread ID, which says which pixel or path it handles.)

## 4. Why it is still hard on a GPU

- **Divergence.** Threads run in groups of 64 (a *wave* on your card) in lockstep. If one path hits
  glass and its neighbor hits a matte wall, they want to run different code. The wave must do both,
  one after the other, and the idle threads wait. Paths also die at different depths.
- **Memory is the bottleneck, not math.** Following a ray through a scene means chasing pointers
  through large data structures, which is slow and unpredictable.
- **Huge, branchy shader code.** Material evaluation is thousands of lines. Compilers allocate
  many registers for it, which limits how many waves fit at once (see section 7).

### How Cycles deals with it: the wavefront design

Instead of one giant program that does the whole path, Cycles splits the work into **stages** and
keeps the state of every path in big arrays. Each stage is its own kernel that only processes the
paths that are currently waiting for that stage. From `src/kernel/device/gpu/kernel.h`, the main ones:

| Kernel | Job |
|---|---|
| `integrator_init_from_camera` | create the camera ray for a pixel |
| `integrator_intersect_closest` | find the nearest hit |
| `integrator_intersect_shadow` | is this shadow ray blocked? |
| `integrator_shade_surface` | evaluate the material, pick the bounce |
| `integrator_shade_light_nee` | add direct light |
| `integrator_shade_background` | the ray escaped; add sky |
| `integrator_shade_volume` | fog/smoke (not in our v1) |
| `..._sorted_paths_array`, `..._compact_*` | regroup paths so similar ones run together |

There are 51 GPU kernel entry points in total. Sorting and compacting paths between stages is how
Cycles keeps a wave's threads doing similar work.

## 5. The BVH: why rays are not slow

A scene has millions of triangles. Testing a ray against all of them would take forever.
A **BVH** (bounding volume hierarchy) is a tree of boxes: each box contains smaller boxes, and leaves
contain a few triangles. A ray tests the big box first; if it misses, everything inside is skipped.

- Modern RTX/RDNA GPUs have dedicated hardware that walks this tree.
- **Your RX 570 has no such hardware.** Traversal runs as ordinary shader code on the ALUs. This is
  the "software ray tracing" path, and it is slower. That is expected and accepted for this project.
- **Verified in the 5.2 source:** the HIP backend (a non-RT GPU backend) reports `BVH_LAYOUT_BVH2`,
  and the default `Device::build_bvh` builds that tree on the **CPU** with Cycles' own builder.
  The GPU only traverses it. Our Vulkan backend can do the same: no GPU BVH building needed.

## 6. From Slang code to a pixel on screen

```
.slang source ──slangc──► SPIR-V ──Vulkan driver──► GCN machine code ──► your RX 570
(we write this)       (portable bytecode)     (AMD's compiler,           runs it on thousands
                                                vkCreateComputePipelines)   of threads at once
```

1. We write kernels in Slang. `slangc` compiles them to SPIR-V (we read this in `chase.txt`).
2. At runtime, `vkCreateComputePipelines` hands the SPIR-V to AMD's driver, which compiles it to
   real GPU instructions. (This is also when the driver can report register counts.)
3. We **dispatch**: "run this kernel on N workgroups of 64 threads." Each thread uses its ID to find
   which path or pixel is its job.
4. Threads read scene data through raw GPU pointers (what `chase.slang` proved works) and
   accumulate light into the film buffer, one float per color channel per pixel.
   Many threads can land on the same pixel, which is why we needed **atomic float add**.
5. Blender reads the film buffer, applies the denoiser, and displays it.

The "image" is nothing more than a big array of floats that the kernels add light into.

## 7. What our first tests measured on the RX 570

| Test | Result | What it taught us |
|---|---|---|
| `chase.spv` | correct sum; 5 VGPRs, 8 SGPRs, 0 scratch | pointers work on this driver |
| `atomic_add.spv` | exact 65536.0; 4 VGPRs, 10 SGPRs, 0 scratch | CAS emulation is correct under contention |
| atomic_add timing | ~840 ms for 65,536 threads on ONE float | worst case: everyone fights over one address |

Reading the numbers:

- **VGPR** = per-thread vector register. The fewer a kernel uses, the more waves fit on a compute
  unit at once. As far as I know, on GCN a kernel using up to about 24 VGPRs can run 10 waves per
  SIMD; 64 VGPRs gives 4; 128 gives 2. Our toy shaders are tiny. Cycles' surface shading will not be.
- **Scratch = 0** means no registers were spilled to slow memory. Watch this number as kernels grow.
- The 840 ms is expected to be a *worst case*. When all 64 lanes of a wave compare-and-swap the
  same address, only one wins per round, so a wave may need up to ~64 retries. Real film writes
  go to different pixels, so contention will be much lower. We should still measure that
  (a test where each thread adds to its own address) before trusting this explanation.

## 8. Where each milestone fits

| Milestone | What it builds | Path-tracer step |
|---|---|---|
| M0/M1 (done) | pointers, atomics, running code on the GPU | the foundation |
| M2 | load triangles, CPU-built BVH, traverse on GPU | step 2: intersect |
| M3 | Cycles `Device` skeleton, `DEVICE_VULKAN` | plumbing into Cycles |
| M4 | camera rays, background, film write | steps 1 and 7: first image |
| M5-M6 | diffuse, Principled BSDF, lights, textures, denoise | steps 3, 4, 5 |
| M7 | Blender Preferences integration | shows up next to CUDA/HIP |

## 9. Check yourself

Try answering these in your own words before moving on:

1. Why is every pixel's path "independent," and why does that matter for a GPU?
2. What does a BVH let a ray skip, and why can't we skip it on a GPU without ray-tracing hardware?
3. Why does Cycles split the path into many small kernels instead of one big one?
4. Why did we need an atomic float add, and why can't plain "read, add, write" do it?
5. What does "scratch memory = 0" tell you about a compiled kernel?

## 10. Words to know

- **Ray**: an origin and a direction.
- **Sample**: one path for one pixel. More samples, less noise.
- **BSDF**: the rule for how a surface scatters light.
- **NEE (next event estimation)**: deliberately aiming a shadow ray at a light instead of hoping a bounce finds it.
- **Wave / wavefront**: a group of 64 threads on your GPU running in lockstep.
- **Workgroup**: a block of threads launched together (`numthreads` in the shader).
- **Film buffer**: the array of floats that becomes the image.
- **VGPR / SGPR**: per-thread vs per-wave registers.
