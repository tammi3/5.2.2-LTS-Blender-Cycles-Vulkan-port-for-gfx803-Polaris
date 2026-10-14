// raytri_cpu.h : the CPU "answer key" for ray-triangle tracing, plus the camera.
//
// The GPU shader (raytri.slang) does the SAME math. We run both and compare.
// If they disagree, one of them has a bug, and the CPU one is far easier to debug.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

// What one ray found. Both CPU and GPU write exactly this 8-byte layout.
struct Hit {
  float t;      // distance along the ray to the closest hit (kNoHitT if none)
  uint32_t id;  // index of the triangle that was hit (kNoHitId if none)
};
static_assert(sizeof(Hit) == 8, "Hit must be 8 bytes (the shader assumes it)");

constexpr float kRayTMin = 1e-4f;      // ignore hits closer than this (self-hit guard)
constexpr float kNoHitT = 1e30f;       // "infinitely far"
constexpr uint32_t kNoHitId = 0xFFFFFFFFu;

// ---- tiny vector helpers on float[3] -------------------------------------
inline void v_sub(const float *a, const float *b, float *o) { for (int i = 0; i < 3; i++) o[i] = a[i] - b[i]; }
inline float v_dot(const float *a, const float *b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline void v_cross(const float *a, const float *b, float *o)
{
  o[0] = a[1]*b[2] - a[2]*b[1];
  o[1] = a[2]*b[0] - a[0]*b[2];
  o[2] = a[0]*b[1] - a[1]*b[0];
}
inline void v_normalize(float *a)
{
  const float len = std::sqrt(v_dot(a, a));
  for (int i = 0; i < 3; i++) a[i] /= len;
}

// ---- Camera ---------------------------------------------------------------
// A pinhole camera described by: its position o, the image plane's lower-left
// corner l, and the two vectors u (across the full width) and v (up the full height).
// The point on the image plane for pixel fractions (s,t in 0..1) is  l + s*u + t*v,
// and the ray goes from o through that point.
struct Camera {
  float o[3], l[3], u[3], v[3];
};

// Places a camera so the whole bounding box is in view.
inline Camera make_camera(const float bmin[3], const float bmax[3], float fov_deg, float aspect)
{
  float center[3], diag[3];
  for (int i = 0; i < 3; i++) {
    center[i] = 0.5f * (bmin[i] + bmax[i]);
    diag[i] = bmax[i] - bmin[i];
  }
  const float radius = 0.5f * std::sqrt(v_dot(diag, diag));
  const float half_fov = 0.5f * fov_deg * 3.14159265358979f / 180.0f;
  const float dist = 1.1f * radius / std::sin(half_fov);

  float w[3] = {0.8f, 0.6f, 1.0f}; // direction from the object toward the camera
  v_normalize(w);
  const float up[3] = {0.0f, 1.0f, 0.0f}; // OBJ files from Blender are Y-up

  Camera cam{};
  for (int i = 0; i < 3; i++) cam.o[i] = center[i] + w[i] * dist;

  float right[3], true_up[3];
  v_cross(up, w, right);
  v_normalize(right);
  v_cross(w, right, true_up);

  const float half_h = std::tan(half_fov);
  const float half_w = half_h * aspect;
  for (int i = 0; i < 3; i++) {
    // image plane sits 1 unit in front of the camera (along -w)
    cam.l[i] = cam.o[i] - w[i] - half_w * right[i] - half_h * true_up[i];
    cam.u[i] = 2.0f * half_w * right[i];
    cam.v[i] = 2.0f * half_h * true_up[i];
  }
  return cam;
}

// Direction of the ray through pixel (x,y). y = 0 is the TOP row of the image.
inline void primary_ray_dir(const Camera &cam, uint32_t x, uint32_t y, uint32_t w, uint32_t h, float *dir)
{
  const float s = ((float)x + 0.5f) / (float)w;
  const float t = 1.0f - ((float)y + 0.5f) / (float)h;
  for (int i = 0; i < 3; i++) dir[i] = cam.l[i] + s * cam.u[i] + t * cam.v[i] - cam.o[i];
  v_normalize(dir);
}

// ---- Moller-Trumbore ray/triangle test, looping over every triangle ---------
// `tris` holds 9 floats per triangle. Returns the closest hit.
inline Hit trace_cpu(const float *tris, uint32_t tri_count, const float *o, const float *d)
{
  Hit best{kNoHitT, kNoHitId};
  for (uint32_t i = 0; i < tri_count; i++) {
    const float *v0 = tris + (size_t)i * 9;
    const float *v1 = v0 + 3;
    const float *v2 = v0 + 6;

    float e1[3], e2[3], p[3], tv[3], q[3];
    v_sub(v1, v0, e1);
    v_sub(v2, v0, e2);
    v_cross(d, e2, p);
    const float det = v_dot(e1, p);
    if (std::fabs(det) < 1e-8f) continue; // ray parallel to the triangle's plane

    const float inv = 1.0f / det;
    v_sub(o, v0, tv);
    const float u = v_dot(tv, p) * inv;
    if (u < 0.0f || u > 1.0f) continue;
    v_cross(tv, e1, q);
    const float v = v_dot(d, q) * inv;
    if (v < 0.0f || u + v > 1.0f) continue;
    const float t = v_dot(e2, q) * inv;
    if (t > kRayTMin && t < best.t) { // strict '<': on a tie the lower index wins
      best.t = t;
      best.id = i;
    }
  }
  return best;
}
