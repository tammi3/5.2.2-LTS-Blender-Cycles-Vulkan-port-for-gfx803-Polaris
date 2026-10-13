// raytri.slang : for every pixel, fire a ray and find the closest triangle.
//
// This is the GPU twin of trace_cpu() in common/raytri_cpu.h. The math is the same
// on purpose, so we can compare the two outputs pixel by pixel.
//
// !!! RAW POINTERS: nothing below is bounds-checked by Vulkan, spirv-val, or the
// !!! validation layer. We check indices OURSELVES (see the `if` guards).

// What one pixel's ray found. Must match `struct Hit` in raytri_cpu.h (8 bytes).
struct Hit
{
    float t;
    uint id;
};

// Everything the shader needs arrives through the push constants.
// Every field is a plain 4- or 8-byte scalar ON PURPOSE: no float3 here, because
// shading languages and C++ disagree about how float3 is padded. The matching C++
// struct (RayTriPush in main.cpp) has static_asserts on every offset.
struct Params
{
    float* tris;       // offset  0 : 9 floats per triangle (v0.xyz v1.xyz v2.xyz)
    Hit* out;          // offset  8 : one Hit per pixel
    uint tri_count;    // offset 16
    uint width;        // offset 20
    uint height;       // offset 24
    float ox, oy, oz;  // offset 28 : camera position
    float lx, ly, lz;  // offset 40 : image plane lower-left corner
    float ux, uy, uz;  // offset 52 : vector across the whole image width
    float vx, vy, vz;  // offset 64 : vector up the whole image height
};

[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x;
    uint y = tid.y;

    // The grid is rounded UP to whole 8x8 groups, so some threads fall outside the
    // image. They MUST stop here, or they would write past the end of the buffer.
    if (x >= params.width || y >= params.height)
        return;

    // ---- build this pixel's ray (same formula as primary_ray_dir on the CPU) ----
    float s = ((float)x + 0.5f) / (float)params.width;
    float t_img = 1.0f - ((float)y + 0.5f) / (float)params.height;
    float3 o = float3(params.ox, params.oy, params.oz);
    float3 L = float3(params.lx, params.ly, params.lz);
    float3 U = float3(params.ux, params.uy, params.uz);
    float3 V = float3(params.vx, params.vy, params.vz);
    float3 d = normalize(L + s * U + t_img * V - o);

    float best_t = 1e30f;        // kNoHitT
    uint best_id = 0xFFFFFFFFu;  // kNoHitId

    // ---- test every triangle (Moller-Trumbore), keep the closest ----
    for (uint i = 0; i < params.tri_count; i++)
    {
        float* tri = params.tris + i * 9;
        float3 v0 = float3(tri[0], tri[1], tri[2]);
        float3 v1 = float3(tri[3], tri[4], tri[5]);
        float3 v2 = float3(tri[6], tri[7], tri[8]);

        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 p = cross(d, e2);
        float det = dot(e1, p);
        if (abs(det) < 1e-8f)
            continue; // ray parallel to the triangle's plane

        float inv = 1.0f / det;
        float3 tv = o - v0;
        float u = dot(tv, p) * inv;
        if (u < 0.0f || u > 1.0f)
            continue;
        float3 q = cross(tv, e1);
        float v = dot(d, q) * inv;
        if (v < 0.0f || u + v > 1.0f)
            continue;
        float t = dot(e2, q) * inv;
        if (t > 1e-4f && t < best_t) // kRayTMin; strict '<' so ties keep the lower index
        {
            best_t = t;
            best_id = i;
        }
    }

    // ---- write the result: exactly one Hit, at this pixel's own slot ----
    Hit* slot = params.out + (y * params.width + x);
    slot->t = best_t;
    slot->id = best_id;
}
