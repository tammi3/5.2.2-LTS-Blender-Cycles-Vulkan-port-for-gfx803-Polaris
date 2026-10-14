// obj_loader.h : reads triangle positions from a Wavefront .obj file.
//
// Based on the m2-parser spike. Changes from that version:
//   * Bad input is an ERROR with a line number, never a silent (0,0,0) vertex.
//   * Negative (relative) indices are supported, as the OBJ format allows.
//   * Output is a flat float array in a layout WE chose for the GPU:
//        9 floats per triangle = v0.xyz, v1.xyz, v2.xyz   (36 bytes, no padding)
//     Normals and UVs are ignored for now (we only need positions to trace rays).
//   * Quads and bigger polygons are split into a fan of triangles. That is only
//     correct for CONVEX polygons, which is what Blender exports for simple meshes.
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct TriangleMesh {
  std::vector<float> positions; // 9 floats per triangle
  size_t triangle_count() const { return positions.size() / 9; }
};

namespace obj_detail {

// Reads the leading integer of a face token such as "7", "7/2/1", "7//1".
inline bool parse_leading_int(const std::string &token, long &out)
{
  char *end = nullptr;
  errno = 0;
  const long v = std::strtol(token.c_str(), &end, 10);
  if (end == token.c_str() || errno != 0)
    return false;
  if (*end != '\0' && *end != '/') // allow only "N", "N/..." forms
    return false;
  out = v;
  return true;
}

} // namespace obj_detail

// Returns true on success. On failure, `error` says what went wrong and where.
inline bool load_obj_positions(const std::string &path, TriangleMesh &mesh, std::string &error)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    error = "could not open file: " + path;
    return false;
  }

  std::vector<float> verts; // x,y,z per vertex
  // Each face = list of ABSOLUTE zero-based vertex indices.
  std::vector<std::vector<long>> faces;
  std::vector<size_t> face_line; // line number of each face, for error messages

  std::string line;
  size_t line_no = 0;
  while (std::getline(file, line)) {
    line_no++;
    std::istringstream iss(line);
    std::string prefix;
    if (!(iss >> prefix) || prefix[0] == '#')
      continue;

    if (prefix == "v") {
      float x, y, z;
      if (!(iss >> x >> y >> z)) {
        error = "line " + std::to_string(line_no) + ": malformed vertex";
        return false;
      }
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        error = "line " + std::to_string(line_no) + ": vertex has NaN/inf";
        return false;
      }
      verts.push_back(x);
      verts.push_back(y);
      verts.push_back(z);
    }
    else if (prefix == "f") {
      std::vector<long> idx;
      std::string token;
      const long vertex_count_so_far = (long)(verts.size() / 3);
      while (iss >> token) {
        long v = 0;
        if (!obj_detail::parse_leading_int(token, v) || v == 0) {
          error = "line " + std::to_string(line_no) + ": bad face index '" + token + "'";
          return false;
        }
        // OBJ: positive = 1-based from the start; negative = relative to the
        // vertices defined SO FAR (-1 is the most recent one).
        idx.push_back(v > 0 ? v - 1 : vertex_count_so_far + v);
      }
      if (idx.size() < 3) {
        error = "line " + std::to_string(line_no) + ": face with fewer than 3 vertices";
        return false;
      }
      faces.push_back(std::move(idx));
      face_line.push_back(line_no);
    }
    // everything else (vn, vt, o, g, s, usemtl, ...) is ignored on purpose
  }

  const long vertex_total = (long)(verts.size() / 3);
  mesh.positions.clear();
  for (size_t f = 0; f < faces.size(); f++) {
    for (long i : faces[f]) {
      if (i < 0 || i >= vertex_total) {
        error = "line " + std::to_string(face_line[f]) + ": face refers to vertex " +
                std::to_string(i + 1) + " but the file only has " +
                std::to_string(vertex_total) + " vertices";
        return false;
      }
    }
    // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
    for (size_t k = 1; k + 1 < faces[f].size(); k++) {
      const long tri[3] = {faces[f][0], faces[f][k], faces[f][k + 1]};
      for (int c = 0; c < 3; c++)
        for (int axis = 0; axis < 3; axis++)
          mesh.positions.push_back(verts[(size_t)tri[c] * 3 + axis]);
    }
  }

  if (mesh.positions.empty()) {
    error = "file contains no triangles";
    return false;
  }
  return true;
}
