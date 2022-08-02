#include "dense_volume.h"
#include "warp.h"

uint64_t dense_volume_create_host(float *buf, uint64_t x, uint64_t y,
                                  uint64_t z, int channels,
                                  const wp::vec3 *position,
                                  const wp::quat *rotation,
                                  const wp::vec3 *scale) {
  size_t size = x * y * z * channels * sizeof(float);
  auto buffer = (wp::volume::DenseVolumeCPU *)alloc_host(
      sizeof(wp::volume::DenseVolumeCPU));
  buffer->shape[0] = x;
  buffer->shape[1] = y;
  buffer->shape[2] = z;
  buffer->position = *position;
  buffer->rotation = *rotation;
  buffer->scale = *scale;

  buffer->data = alloc_host(size);
  memcpy_h2h(buffer->data, buf, size);

  return (uint64_t)buffer;
}

void dense_volume_destroy_host(uint64_t id) {
  auto buffer = (wp::volume::DenseVolumeCPU *)id;
  free_host(buffer->data);
  free_host(buffer);
}
