// ppm.h : turn hit results into pictures, and save them as .ppm files.
//
// PPM is the simplest image format there is: a short text header, then raw RGB bytes.
// Many viewers open it (GIMP, IrfanView, XnView, some VS Code extensions).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "raytri_cpu.h"

inline bool write_ppm(const std::string &path, uint32_t w, uint32_t h, const std::vector<uint8_t> &rgb)
{
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (!f)
    return false;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  std::fwrite(rgb.data(), 1, rgb.size(), f);
  std::fclose(f);
  return true;
}

// Brightness by distance: near = bright, far = dark, miss = black.
inline std::vector<uint8_t> hits_to_depth_rgb(const std::vector<Hit> &hits)
{
  float tmin = kNoHitT, tmax = 0.0f;
  for (const Hit &h : hits) {
    if (h.id == kNoHitId) continue;
    tmin = std::min(tmin, h.t);
    tmax = std::max(tmax, h.t);
  }
  std::vector<uint8_t> rgb(hits.size() * 3, 0);
  for (size_t i = 0; i < hits.size(); i++) {
    if (hits[i].id == kNoHitId) continue;
    const float n = (tmax > tmin) ? (hits[i].t - tmin) / (tmax - tmin) : 0.0f;
    const uint8_t v = (uint8_t)(40.0f + 215.0f * (1.0f - std::min(std::max(n, 0.0f), 1.0f)));
    rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
  }
  return rgb;
}

// A different color per triangle, so you can see which triangle each pixel hit.
inline std::vector<uint8_t> hits_to_id_rgb(const std::vector<Hit> &hits)
{
  std::vector<uint8_t> rgb(hits.size() * 3);
  for (size_t i = 0; i < hits.size(); i++) {
    if (hits[i].id == kNoHitId) {
      rgb[i * 3 + 0] = 30; rgb[i * 3 + 1] = 30; rgb[i * 3 + 2] = 40;
      continue;
    }
    const uint32_t id = hits[i].id;
    rgb[i * 3 + 0] = (uint8_t)(70 + (id * 97) % 160);
    rgb[i * 3 + 1] = (uint8_t)(70 + (id * 53 + 40) % 160);
    rgb[i * 3 + 2] = (uint8_t)(70 + (id * 29 + 90) % 160);
  }
  return rgb;
}
