/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

// cuda header
struct cudaArray;
typedef struct cudaArray *cudaArray_t;
typedef unsigned long long cudaTextureObject_t;

namespace wp {
namespace volume {

struct DenseVolumeCuda {
  cudaArray_t array;
  cudaTextureObject_t texture;
  vec3 position;
  quat rotation;
  vec3 scale;
};

struct DenseVolumeCPU {
  void *data;
  int shape[3];
  vec3 position;
  quat rotation;
  vec3 scale;
};

} // namespace volume

CUDA_CALLABLE inline vec3 dense_volume_index_to_world(uint64_t id, vec3 uvw) {
#ifdef __CUDA_ARCH__
  auto &buffer = *(volume::DenseVolumeCuda *)id;
#else
  auto &buffer = *(volume::DenseVolumeCPU *)id;
#endif
  return quat_rotate(buffer.rotation,
                     cw_mul(uvw, buffer.scale) + buffer.position);
}

CUDA_CALLABLE inline vec3 dense_volume_world_to_index(uint64_t id, vec3 xyz) {
#ifdef __CUDA_ARCH__
  auto &buffer = *(volume::DenseVolumeCuda *)id;
#else
  auto &buffer = *(volume::DenseVolumeCPU *)id;
#endif
  return cw_div(quat_rotate_inv(buffer.rotation, xyz) - buffer.position,
                buffer.scale);
}

#ifdef __CUDA_ARCH__

template <typename T> CUDA_CALLABLE inline T convert_float4(float4 data);
template <> CUDA_CALLABLE inline float convert_float4<float>(float4 data) {
  return data.x;
}
template <> CUDA_CALLABLE inline vec2 convert_float4<vec2>(float4 data) {
  return vec2(data.x, data.y);
}
template <> CUDA_CALLABLE inline vec3 convert_float4<vec3>(float4 data) {
  return vec3(data.x, data.y, data.z);
}
template <> CUDA_CALLABLE inline vec4 convert_float4<vec4>(float4 data) {
  return vec4(data.x, data.y, data.z, data.w);
}

#endif

inline int get_index(int u, int v, int w, int dim2, int dim1) {
  return dim2 * (dim1 * u + v) + w;
}

template <typename T>
inline T bilinear(float tx, float ty, T c00, T c10, T c01, T c11) {
  T c0 = c00 * (1 - tx) + c10 * tx;
  T c1 = c01 * (1 - tx) + c11 * tx;
  return c0 * (1 - ty) + c1 * ty;
}

template <typename T>
CUDA_CALLABLE inline T dense_volume_sample(uint64_t id, vec3 xyz) {
  vec3 uvw = dense_volume_world_to_index(id, xyz);
#ifdef __CUDA_ARCH__
  auto &buffer = *(volume::DenseVolumeCuda *)id;

  float4 result = tex3D<float4>(buffer.texture, uvw.x, uvw.y, uvw.z);
  return convert_float4<T>(result);

#else
  auto &buffer = *(volume::DenseVolumeCPU *)id;
  auto data = (T *)buffer.data;

  uvw.x -= 0.5f;
  uvw.y -= 0.5f;
  uvw.z -= 0.5f;

  uvw.x = min(max(uvw.x, 0.f), (float)(buffer.shape[0] - 1));
  uvw.y = min(max(uvw.y, 0.f), (float)(buffer.shape[1] - 1));
  uvw.z = min(max(uvw.z, 0.f), (float)(buffer.shape[2] - 1));

  int u = min((int)floorf(uvw.x), buffer.shape[0] - 2);
  int v = min((int)floorf(uvw.y), buffer.shape[1] - 2);
  int w = min((int)floorf(uvw.z), buffer.shape[2] - 2);

  float frac_u = uvw.x - u;
  float frac_v = uvw.y - v;
  float frac_w = uvw.z - w;

  float wx[2]{1 - frac_u, frac_u};
  float wy[2]{1 - frac_v, frac_v};
  float wz[2]{1 - frac_w, frac_w};

  int dim2 = buffer.shape[2];
  int dim1 = buffer.shape[1];
  T c000 = data[get_index(u, v, w, dim2, dim1)];
  T c100 = data[get_index(u + 1, v, w, dim2, dim1)];
  T c010 = data[get_index(u, v + 1, w, dim2, dim1)];
  T c110 = data[get_index(u + 1, v + 1, w, dim2, dim1)];
  T c001 = data[get_index(u, v, w + 1, dim2, dim1)];
  T c101 = data[get_index(u + 1, v, w + 1, dim2, dim1)];
  T c011 = data[get_index(u, v + 1, w + 1, dim2, dim1)];
  T c111 = data[get_index(u + 1, v + 1, w + 1, dim2, dim1)];

  T c0 = bilinear(frac_u, frac_v, c000, c100, c010, c110);
  T c1 = bilinear(frac_u, frac_v, c001, c101, c011, c111);
  return c0 * (1 - frac_w) + c1 * (1 - frac_w);

#endif
}

CUDA_CALLABLE inline float dense_volume_sample_f(uint64_t id, vec3 xyz) {
  return dense_volume_sample<float>(id, xyz);
}
CUDA_CALLABLE inline void adj_dense_volume_sample_f(uint64_t id, vec3 xyz,
                                                    uint64_t &adj_id,
                                                    vec3 &adj_xyz,
                                                    float &adj_ret) {}

CUDA_CALLABLE inline vec4 dense_volume_sample_vec4(uint64_t id, vec3 xyz) {
  return dense_volume_sample<vec4>(id, xyz);
}
CUDA_CALLABLE inline void adj_dense_volume_sample_vec4(uint64_t id, vec3 xyz,
                                                       uint64_t &adj_id,
                                                       vec3 &adj_xyz,
                                                       vec4 &adj_ret) {}

} // namespace wp
