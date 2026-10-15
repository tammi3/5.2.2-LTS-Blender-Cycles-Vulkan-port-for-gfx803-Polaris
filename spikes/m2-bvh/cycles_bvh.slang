// cycles_bvh.slang
// Bare-minimum Cycles BVH2 traversal for Vulkan Compute (Software Ray Tracing)
// Matches the data layout dumped by cycles_bvh_dump.cpp

// ============================================================================
// 1. Constants (Matching Cycles kernel/types.h and kernel/bvh/types.h)
// ============================================================================
static const int PRIM_NONE = -1;
static const int OBJECT_NONE = -1;
static const int ENTRYPOINT_SENTINEL = 0x76543210;
static const int BVH_STACK_SIZE = 192; // 64 object + 64 mesh + 64 object node splitting

static const int PRIMITIVE_NONE = 0;
static const int PRIMITIVE_TRIANGLE = 1;          // (1 << 0)
static const int PRIMITIVE_MOTION = 128;          // (1 << 7)
static const int PRIMITIVE_ALL = 255;             // Mask for all base types + motion

// ============================================================================
// 2. Minimal Data Structures
// ============================================================================

struct Ray {
    float3 P;           // Origin
    float3 D;           // Direction
    float tmin;         // Start distance (e.g., 1e-4f to avoid self-intersection)
    float tmax;         // End distance
    float time;         // Keep for motion blur later (0.0f for now)
    int self_prim;      // Primitive ID to avoid self-intersection
    int self_object;    // Object ID to avoid self-intersection
};

struct Intersection {
    float t;            // Distance along ray
    float u, v;         // Barycentric coordinates
    int prim;           // Primitive index
    int object;         // Object index
    int type;           // Primitive type (e.g., PRIMITIVE_TRIANGLE)
    int shader;         // Shader ID
};

// Helper to safely convert float bits to int (matches Cycles __float_as_int)
int float_as_int(float f) {
    return asint(f);
}

// ============================================================================
// 3. Global Buffer Bindings (Matching cycles_bvh_dump.cpp output)
// ============================================================================

// NOTE: We use ByteAddressBuffer for tri_verts to guarantee 12-byte packed_float3 
// layout. Slang/HLSL StructuredBuffer<float3> will pad to 16 bytes (std430), 
// which would misalign with Cycles' tightly packed CPU dump.
StructuredBuffer<float4> bvh_nodes;
StructuredBuffer<float4> bvh_leaf_nodes;
StructuredBuffer<int> prim_type;
StructuredBuffer<int> prim_visibility;
StructuredBuffer<int> prim_index;
StructuredBuffer<int> prim_object;
StructuredBuffer<int> object_node;
ByteAddressBuffer tri_verts; 

// Helper to fetch a tightly packed float3 (12 bytes) from tri_verts
float3 get_tri_vert(int index) {
    uint offset = (uint)index * 12;
    uint x = tri_verts.Load(offset);
    uint y = tri_verts.Load(offset + 4);
    uint z = tri_verts.Load(offset + 8);
    return float3(asfloat(x), asfloat(y), asfloat(z));
}

// ============================================================================
// 4. Minimal BVH Traversal (Software Ray Tracing)
// ============================================================================
// Stripped of motion, hair, and point cloud features for the bare minimum.

bool bvh_intersect(
    int bvh_root,
    Ray ray,
    out Intersection isect,
    uint visibility)
{
    int traversal_stack[BVH_STACK_SIZE];
    traversal_stack[0] = ENTRYPOINT_SENTINEL;
    
    int stack_ptr = 0;
    int node_addr = bvh_root;
    
    float3 P = ray.P;
    float3 dir = ray.D;
    
    // Safe inverse direction (matches Cycles bvh_inverse_direction)
    float3 idir = 1.0f / dir;
    idir = (dir == 0.0f) ? 1e30f : idir;
    
    float tmin = ray.tmin;
    int object = OBJECT_NONE;
    
    isect.t = ray.tmax;
    isect.u = 0.0f;
    isect.v = 0.0f;
    isect.prim = PRIM_NONE;
    isect.object = OBJECT_NONE;
    
    do {
        // Traverse internal nodes
        while (node_addr >= 0 && node_addr != ENTRYPOINT_SENTINEL) {
            // Fetch node data (4 x float4 = 16 floats, 64 bytes)
            // Layout verified from intern/cycles/kernel/bvh/nodes.h
            float4 cnodes = bvh_nodes[node_addr + 0];
            float4 node0  = bvh_nodes[node_addr + 1];
            float4 node1  = bvh_nodes[node_addr + 2];
            float4 node2  = bvh_nodes[node_addr + 3];
            
            int child0_addr = float_as_int(cnodes.z);
            int child1_addr = float_as_int(cnodes.w);
            
            // Child 0 AABB intersection (Slab method)
            float c0lox = (node0.x - P.x) * idir.x;
            float c0hix = (node0.z - P.x) * idir.x;
            float c0loy = (node1.x - P.y) * idir.y;
            float c0hiy = (node1.z - P.y) * idir.y;
            float c0loz = (node2.x - P.z) * idir.z;
            float c0hiz = (node2.z - P.z) * idir.z;
            
            float c0min_val = max(tmin, min(min(c0lox, c0hix), min(c0loy, c0hiy), min(c0loz, c0hiz)));
            float c0max_val = min(ray.tmax, max(max(c0lox, c0hix), max(c0loy, c0hiy), max(c0loz, c0hiz)));
            
            // Child 1 AABB intersection
            float c1lox = (node0.y - P.x) * idir.x;
            float c1hix = (node0.w - P.x) * idir.x;
            float c1loy = (node1.y - P.y) * idir.y;
            float c1hiy = (node1.w - P.y) * idir.y;
            float c1loz = (node2.y - P.z) * idir.z;
            float c1hiz = (node2.w - P.z) * idir.z;
            
            float c1min_val = max(tmin, min(min(c1lox, c1hix), min(c1loy, c1hiy), min(c1loz, c1hiz)));
            float c1max_val = min(ray.tmax, max(max(c1lox, c1hix), max(c1loy, c1hiy), max(c1loz, c1hiz)));
            
            int traverse_mask = 0;
            if (c0max_val >= c0min_val) traverse_mask |= 1;
            if (c1max_val >= c1min_val) traverse_mask |= 2;
            
            if (traverse_mask == 3) {
                // Both children intersected, push the farther one
                bool is_closest_child1 = (c1min_val < c0min_val);
                int near_addr = is_closest_child1 ? child1_addr : child0_addr;
                int far_addr  = is_closest_child1 ? child0_addr : child1_addr;
                
                node_addr = near_addr;
                stack_ptr++;
                traversal_stack[stack_ptr] = far_addr;
            }
            else if (traverse_mask == 2) {
                node_addr = child1_addr;
            }
            else if (traverse_mask == 1) {
                node_addr = child0_addr;
            }
            else {
                // Neither intersected, pop stack
                node_addr = traversal_stack[stack_ptr];
                stack_ptr--;
            }
        }
        
        // If node is leaf, fetch primitive list
        if (node_addr < 0 && node_addr != ENTRYPOINT_SENTINEL) {
            int leaf_index = -node_addr - 1;
            float4 leaf = bvh_leaf_nodes[leaf_index];
            
            int prim_addr = float_as_int(leaf.x);
            
            if (prim_addr >= 0) {
                // Primitive leaf
                int prim_addr2 = float_as_int(leaf.y);
                uint type = (uint)float_as_int(leaf.w);
                
                // Pop stack
                node_addr = traversal_stack[stack_ptr];
                stack_ptr--;
                
                // Primitive intersection loop
                for (int p = prim_addr; p < prim_addr2; p++) {
                    int p_object = (object == OBJECT_NONE) ? prim_object[p] : object;
                    int p_prim = prim_index[p];
                    
                    // Skip self-intersection (simplified)
                    if (ray.self_prim >= 0 && p_prim == ray.self_prim && p_object == ray.self_object) {
                        continue;
                    }
                    
                    // For bare minimum, we only handle PRIMITIVE_TRIANGLE
                    if ((type & PRIMITIVE_ALL) == PRIMITIVE_TRIANGLE) {
                        // TODO: Call triangle_intersect here (Möller–Trumbore)
                        // We will implement this in the next step.
                        // When a hit occurs and t < isect.t, update isect and 
                        // return true early if (visibility & PATH_RAY_VISIBILITY_SHADOW_OPAQUE).
                    }
                }
            }
            else {
                // Instance leaf (simplified: we skip instance transform pushing for bare minimum)
                // object = prim_object[-prim_addr - 1];
                node_addr = traversal_stack[stack_ptr];
                stack_ptr--;
            }
        }
    } while (node_addr != ENTRYPOINT_SENTINEL);
    
    return (isect.prim != PRIM_NONE);
}