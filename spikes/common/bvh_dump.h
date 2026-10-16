// bvh_dump.h : read the .bin files written by tools/cycles_bvh_dump.
//
// The dump tool wrote each Cycles array as raw bytes. Here we read them back into
// std::vectors and CHECK that the sizes make sense, because everything after this
// (CPU traversal, GPU traversal) trusts these numbers.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// KernelObject (kernel/types.h:1412) is 256 bytes. Counting its fields by hand gives
// `int position_offset` at byte 168 = 32-bit word 42. M2c double-checks this guess
// on the real data before trusting it.
constexpr size_t kObjectWords = 64;          // 256 bytes / 4
constexpr size_t kPositionOffsetWord = 42;   // byte 168 / 4
// Two more fields from the same hand count, used to CONFIRM the layout: for an object that
// is a plain mesh, numverts and numprims must equal the mesh's vertex and triangle counts.
constexpr size_t kNumVertsWord = 38;         // byte 152 / 4
constexpr size_t kNumPrimsWord = 39;         // byte 156 / 4

struct BvhDump {
  std::vector<float> nodes;            // bvh_nodes: float4s, 4 float4 per inner node
  std::vector<float> leaves;           // bvh_leaf_nodes: float4s
  std::vector<uint32_t> prim_type;     // one per primitive slot
  std::vector<uint32_t> prim_visibility;
  std::vector<uint32_t> prim_index;    // slot -> triangle number
  std::vector<uint32_t> prim_object;   // slot -> object number
  std::vector<uint32_t> tri_vindex;    // 3 vertex numbers per triangle
  std::vector<float> tri_verts;        // 3 floats per vertex
  std::vector<uint32_t> objects;       // kObjectWords words per KernelObject
  std::vector<int32_t> kernel_bvh;     // KernelBVH as ints; [0] is the root address

  uint32_t node_count() const { return (uint32_t)(nodes.size() / 4); }   // in float4s
  uint32_t leaf_count() const { return (uint32_t)(leaves.size() / 4); }  // in float4s
  uint32_t prim_count() const { return (uint32_t)prim_index.size(); }
  uint32_t tri_count() const { return (uint32_t)(tri_vindex.size() / 3); }
  uint32_t vert_count() const { return (uint32_t)(tri_verts.size() / 3); }
  uint32_t object_count() const { return (uint32_t)(objects.size() / kObjectWords); }
  int32_t root() const { return kernel_bvh.empty() ? 0 : kernel_bvh[0]; }
  uint32_t position_offset(uint32_t obj) const { return objects[obj * kObjectWords + kPositionOffsetWord]; }
  uint32_t object_numverts(uint32_t obj) const { return objects[obj * kObjectWords + kNumVertsWord]; }
  uint32_t object_numprims(uint32_t obj) const { return objects[obj * kObjectWords + kNumPrimsWord]; }
};

template<typename T>
inline bool read_bin(const std::string &dir, const char *name, std::vector<T> &out, std::string &err)
{
  const std::string path = dir + "/" + name + ".bin";
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    err = "cannot open " + path + " (was the dump run, and is this the right folder?)";
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes < 0 || bytes % (long)sizeof(T) != 0) {
    std::fclose(f);
    err = std::string(name) + ".bin has a size that is not a multiple of " + std::to_string(sizeof(T));
    return false;
  }
  out.resize((size_t)bytes / sizeof(T));
  if (bytes > 0 && std::fread(out.data(), 1, (size_t)bytes, f) != (size_t)bytes) {
    std::fclose(f);
    err = std::string("short read on ") + path;
    return false;
  }
  std::fclose(f);
  return true;
}

inline bool load_bvh_dump(const std::string &dir, BvhDump &d, std::string &err)
{
  if (!read_bin(dir, "bvh_nodes", d.nodes, err)) return false;
  if (!read_bin(dir, "bvh_leaf_nodes", d.leaves, err)) return false;
  if (!read_bin(dir, "prim_type", d.prim_type, err)) return false;
  if (!read_bin(dir, "prim_visibility", d.prim_visibility, err)) return false;
  if (!read_bin(dir, "prim_index", d.prim_index, err)) return false;
  if (!read_bin(dir, "prim_object", d.prim_object, err)) return false;
  if (!read_bin(dir, "tri_vindex", d.tri_vindex, err)) return false;
  if (!read_bin(dir, "tri_verts", d.tri_verts, err)) return false;
  if (!read_bin(dir, "objects", d.objects, err)) return false;
  if (!read_bin(dir, "kernel_bvh", d.kernel_bvh, err)) return false;

  // Structural checks: these catch a wrong guess about the layout before it can
  // turn into a wild read.
  if (d.nodes.size() % 16 != 0) { err = "bvh_nodes is not a whole number of inner nodes (4 float4 each)"; return false; }
  if (d.leaves.size() % 4 != 0) { err = "bvh_leaf_nodes is not a whole number of float4"; return false; }
  const size_t n = d.prim_index.size();
  if (d.prim_type.size() != n || d.prim_visibility.size() != n || d.prim_object.size() != n) {
    err = "prim_type / prim_visibility / prim_index / prim_object have different lengths";
    return false;
  }
  if (d.tri_vindex.size() % 3 != 0) { err = "tri_vindex is not a whole number of triples"; return false; }
  if (d.tri_verts.size() % 3 != 0) { err = "tri_verts is not a whole number of vertices"; return false; }
  if (d.objects.size() % kObjectWords != 0) {
    err = "objects.bin is not a whole number of 256-byte KernelObject structs (did sizeof(KernelObject) change?)";
    return false;
  }
  if (d.kernel_bvh.size() < 1) { err = "kernel_bvh.bin is empty"; return false; }
  return true;
}
