// bvh_traverse.slang : walk the BVH that Cycles built, on the GPU.
//
// This is a Slang port of Cycles' kernel/bvh/traversal.h (v5.2.0), aligned nodes and
// triangles only. It must stay line-for-line equivalent to common/bvh_cpu.h, which is
// the same algorithm in plain C++ (that file is how we debug THIS one).
//
// Original code:
//   SPDX-FileCopyrightText: 2009-2010 NVIDIA Corporation
//   SPDX-FileCopyrightText: 2009-2012 Intel Corporation
//   SPDX-FileCopyrightText: 2011-2022 Blender Foundation
//   SPDX-License-Identifier: Apache-2.0
// This port is a derivative work and keeps those notices.
//
// ###########################################################################
// ##  RAW GPU POINTERS HAVE NO BOUNDS CHECKING.                            ##
// ##  So every array index below is checked against its count BEFORE it is  ##
// ##  used, and the traversal loop has an iteration cap (a hung shader on   ##
// ##  Windows means a driver reset). On any violation the pixel gets an     ##
// ##  error code in `Hit::id` instead of a triangle number.                 ##
// ###########################################################################

// What one pixel's ray found. Same 8-byte layout as `struct Hit` in raytri_cpu.h.
struct Hit
{
    float t;
    uint id;
};

// All the BVH arrays, as raw GPU pointers. Cycles calls this idea "KernelGlobals".
// Layout is locked by static_asserts in main.cpp (SceneData there). Size: 96 bytes.
struct SceneData
{
    float4* bvh_nodes;        // offset  0
    float4* bvh_leaf_nodes;   // offset  8
    uint* prim_index;         // offset 16
    uint* prim_visibility;    // offset 24
    uint* prim_object;        // offset 32
    uint* tri_vindex;         // offset 40 : 3 uints per triangle
    float* tri_verts;         // offset 48 : 3 floats per vertex
    uint* objects;            // offset 56 : 64 uints per KernelObject
    int root;                 // offset 64
    uint node_count;          // offset 68 : in float4s
    uint leaf_count;          // offset 72 : in float4s
    uint prim_count;          // offset 76
    uint tri_count;           // offset 80
    uint vert_count;          // offset 84
    uint object_count;        // offset 88
    uint pad;                 // offset 92
};

// Push constants: one pointer to SceneData plus the per-dispatch values. Scalars only
// (no float3) so the C++ and Slang layouts cannot disagree about padding.
struct Params
{
    SceneData* scene;  // offset  0
    Hit* out;          // offset  8
    uint width;        // offset 16
    uint height;       // offset 20
    uint visibility;   // offset 24 : ray visibility mask
    float ox, oy, oz;  // offset 28 : camera position
    float lx, ly, lz;  // offset 40 : image plane lower-left corner
    float ux, uy, uz;  // offset 52 : vector across the image width
    float vx, vy, vz;  // offset 64 : vector up the image height
};

[[vk::push_constant]] ConstantBuffer<Params> params;

static const int ENTRYPOINT_SENTINEL = 0x76543210;  // kernel/bvh/types.h:17
static const int BVH_STACK_SIZE = 192;              // kernel/bvh/types.h:20
static const uint MAX_ITERATIONS = 1000000u;
static const uint OBJECT_WORDS = 64u;               // sizeof(KernelObject) / 4
static const uint POSITION_OFFSET_WORD = 42u;       // offsetof(position_offset) / 4
static const uint PRIMITIVE_TRIANGLE = 1u;

static const uint NO_PRIM = 0xFFFFFFFFu;
static const uint ERR_LOOP = 0xFFFFFFFBu;
static const uint ERR_STACK = 0xFFFFFFFCu;
static const uint ERR_BAD_INDEX = 0xFFFFFFFDu;
static const uint ERR_INSTANCE = 0xFFFFFFFEu;

// Copy of the sign of `s` onto the magnitude `m`. Written with bit operations because
// I am not sure Slang has copysign for floats.
float copysign_bits(float m, float s)
{
    return asfloat((asuint(m) & 0x7FFFFFFFu) | (asuint(s) & 0x80000000u));
}

// bvh_clamp_direction (kernel/geom/object.h:449)
float clamp_component(float c)
{
    const float ooeps = 8.271806E-25f;
    return (abs(c) > ooeps) ? c : copysign_bits(ooeps, c);
}

float3 rt_cross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float rt_dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ray_triangle_intersect (util/math_intersect.h), non-SSE branches.
bool ray_triangle_intersect(float3 P, float3 D, float tmin, float tmax,
                            float3 a, float3 b, float3 c,
                            out float ou, out float ov, out float ot)
{
    ou = 0.0f;
    ov = 0.0f;
    ot = 0.0f;

    float3 v0 = a - P;
    float3 v1 = b - P;
    float3 v2 = c - P;

    float3 e0 = v2 - v0;
    float3 e1 = v0 - v1;
    float3 e2 = v1 - v2;

    float U = rt_dot(rt_cross(e0, v2 + v0), D);
    float V = rt_dot(rt_cross(e1, v0 + v1), D);
    float W = rt_dot(rt_cross(e2, v1 + v2), D);

    float UVW = U + V + W;
    float eps = 1.1920929e-07f * abs(UVW);  // FLT_EPSILON * |UVW|
    float minUVW = min(U, min(V, W));
    float maxUVW = max(U, max(V, W));

    if (!(minUVW >= -eps || maxUVW <= eps))
        return false;

    float3 Ng1 = rt_cross(e1, e0);
    float3 Ng = Ng1 + Ng1;
    float den = rt_dot(Ng, D);
    if (den == 0.0f)
        return false;

    float T = rt_dot(v0, Ng);
    float t = T / den;
    if (!(t >= tmin && t <= tmax))
        return false;

    float rcp_uvw = (abs(UVW) < 1e-18f) ? 0.0f : 1.0f / UVW;
    ou = min(U * rcp_uvw, 1.0f);
    ov = min(V * rcp_uvw, 1.0f);
    ot = t;
    return true;
}

// The traversal. Returns 0 on success or an ERR_* code. Mirrors cyc::traverse().
uint traverse(SceneData* sc, float3 P, float3 Din, float tmin, float tmax, uint visibility,
              out float hit_t, out uint hit_prim)
{
    int stack[BVH_STACK_SIZE];
    stack[0] = ENTRYPOINT_SENTINEL;
    int sp = 0;
    int node_addr = sc->root;

    float3 dir = float3(clamp_component(Din.x), clamp_component(Din.y), clamp_component(Din.z));
    float3 idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    hit_t = tmax;
    hit_prim = NO_PRIM;
    uint iterations = 0;

    do
    {
        // ---- walk down through inner nodes --------------------------------------
        while (node_addr >= 0 && node_addr != ENTRYPOINT_SENTINEL)
        {
            iterations++;
            if (iterations > MAX_ITERATIONS)
                return ERR_LOOP;
            if ((uint)node_addr + 3u >= sc->node_count)
                return ERR_BAD_INDEX;

            float4* np = sc->bvh_nodes + (uint)node_addr;
            float4 cnodes = np[0];
            float4 node0 = np[1];
            float4 node1 = np[2];
            float4 node2 = np[3];

            // bvh_aligned_node_intersect (nodes.h:24)
            float c0lox = (node0.x - P.x) * idir.x;
            float c0hix = (node0.z - P.x) * idir.x;
            float c0loy = (node1.x - P.y) * idir.y;
            float c0hiy = (node1.z - P.y) * idir.y;
            float c0loz = (node2.x - P.z) * idir.z;
            float c0hiz = (node2.z - P.z) * idir.z;
            float c0min = max(max(tmin, min(c0lox, c0hix)), max(min(c0loy, c0hiy), min(c0loz, c0hiz)));
            float c0max = min(min(hit_t, max(c0lox, c0hix)), min(max(c0loy, c0hiy), max(c0loz, c0hiz)));

            float c1lox = (node0.y - P.x) * idir.x;
            float c1hix = (node0.w - P.x) * idir.x;
            float c1loy = (node1.y - P.y) * idir.y;
            float c1hiy = (node1.w - P.y) * idir.y;
            float c1loz = (node2.y - P.z) * idir.z;
            float c1hiz = (node2.w - P.z) * idir.z;
            float c1min = max(max(tmin, min(c1lox, c1hix)), max(min(c1loy, c1hiy), min(c1loz, c1hiz)));
            float c1max = min(min(hit_t, max(c1lox, c1hix)), min(max(c1loy, c1hiy), max(c1loz, c1hiz)));

            float dist0 = c0min;
            float dist1 = c1min;

            int traverse_mask = 0;
            if ((c0max >= c0min) && ((asuint(cnodes.x) & visibility) != 0u))
                traverse_mask |= 1;
            if ((c1max >= c1min) && ((asuint(cnodes.y) & visibility) != 0u))
                traverse_mask |= 2;

            node_addr = asint(cnodes.z);
            int node_addr_child1 = asint(cnodes.w);

            if (traverse_mask == 3)
            {
                // Both children hit: go into the nearer one, push the farther one.
                if (dist1 < dist0)
                {
                    int tmp = node_addr;
                    node_addr = node_addr_child1;
                    node_addr_child1 = tmp;
                }
                sp++;
                if (sp >= BVH_STACK_SIZE)
                    return ERR_STACK;
                stack[sp] = node_addr_child1;
            }
            else
            {
                if (traverse_mask == 2)
                {
                    node_addr = node_addr_child1;
                }
                else if (traverse_mask == 0)
                {
                    // Neither child hit: pop.
                    if (sp < 0)
                        return ERR_STACK;
                    node_addr = stack[sp];
                    sp--;
                }
            }
        }

        // ---- leaf: test its triangles ---------------------------------------------
        if (node_addr < 0)
        {
            int leaf_index = -node_addr - 1;
            if ((uint)leaf_index >= sc->leaf_count)
                return ERR_BAD_INDEX;
            float4* lp = sc->bvh_leaf_nodes + (uint)leaf_index;
            float4 leaf = lp[0];
            int prim_addr = asint(leaf.x);

            if (prim_addr >= 0)
            {
                int prim_addr2 = asint(leaf.y);
                uint type = asuint(leaf.w);

                // pop
                if (sp < 0)
                    return ERR_STACK;
                node_addr = stack[sp];
                sp--;

                if (prim_addr2 < prim_addr || (uint)prim_addr2 > sc->prim_count)
                    return ERR_BAD_INDEX;

                for (; prim_addr < prim_addr2; prim_addr++)
                {
                    uint pobj = (sc->prim_object + (uint)prim_addr)[0];
                    uint prim = (sc->prim_index + (uint)prim_addr)[0];

                    if ((type & PRIMITIVE_TRIANGLE) != 0u)
                    {
                        if (pobj >= sc->object_count || prim >= sc->tri_count)
                            return ERR_BAD_INDEX;

                        uint pos_off = (sc->objects + (pobj * OBJECT_WORDS + POSITION_OFFSET_WORD))[0];
                        uint* vi = sc->tri_vindex + prim * 3u;
                        uint ia = vi[0] + pos_off;
                        uint ib = vi[1] + pos_off;
                        uint ic = vi[2] + pos_off;
                        if (ia >= sc->vert_count || ib >= sc->vert_count || ic >= sc->vert_count)
                            return ERR_BAD_INDEX;

                        float* pa = sc->tri_verts + ia * 3u;
                        float* pb = sc->tri_verts + ib * 3u;
                        float* pc = sc->tri_verts + ic * 3u;
                        float3 a = float3(pa[0], pa[1], pa[2]);
                        float3 b = float3(pb[0], pb[1], pb[2]);
                        float3 c = float3(pc[0], pc[1], pc[2]);

                        float u;
                        float v;
                        float t;
                        if (ray_triangle_intersect(P, dir, tmin, hit_t, a, b, c, u, v, t))
                        {
                            uint pvis = (sc->prim_visibility + (uint)prim_addr)[0];
                            if ((pvis & visibility) != 0u)
                            {
                                hit_t = t;
                                hit_prim = prim;
                            }
                        }
                    }
                    else
                    {
                        return ERR_INSTANCE;  // motion / curve / point: not supported yet
                    }
                }
            }
            else
            {
                return ERR_INSTANCE;  // negative start = instance leaf: not supported yet
            }
        }
    } while (node_addr != ENTRYPOINT_SENTINEL);

    return 0u;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x;
    uint y = tid.y;

    // The grid is rounded UP to whole 8x8 groups; threads outside the image MUST stop.
    if (x >= params.width || y >= params.height)
        return;

    // Same ray construction as primary_ray_dir() in raytri_cpu.h.
    float s = ((float)x + 0.5f) / (float)params.width;
    float t_img = 1.0f - ((float)y + 0.5f) / (float)params.height;
    float3 o = float3(params.ox, params.oy, params.oz);
    float3 L = float3(params.lx, params.ly, params.lz);
    float3 U = float3(params.ux, params.uy, params.uz);
    float3 V = float3(params.vx, params.vy, params.vz);
    float3 d = normalize(L + s * U + t_img * V - o);

    float hit_t;
    uint hit_prim;
    uint status = traverse(params.scene, o, d, 1e-4f, 1e30f, params.visibility, hit_t, hit_prim);

    Hit* slot = params.out + (y * params.width + x);
    if (status != 0u)
    {
        slot->t = -1.0f;
        slot->id = status;  // an error code, see ERR_* above
    }
    else
    {
        slot->t = hit_t;
        slot->id = hit_prim;
    }
}
