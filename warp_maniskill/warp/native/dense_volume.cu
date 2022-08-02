/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "dense_volume.h"

#include <cuda.h>
#include <cuda_runtime_api.h>


uint64_t dense_volume_create_device(float *buf, uint64_t x, uint64_t y,
                                    uint64_t z, int channels, const wp::vec3 *position,
                                    const wp::quat *rotation, const wp::vec3 *scale) {
  cudaArray_t cuArray = 0;
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32,
                            channels >= 2 ? 32 : 0,
                            channels >= 3 ? 32 : 0,
                            channels >= 4 ? 32 : 0,
                            cudaChannelFormatKindFloat);

  check_cuda(
      cudaMalloc3DArray(&cuArray, &channelDesc, make_cudaExtent(x, y, z)));

  // copy data
  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(params));
  params.srcPtr = make_cudaPitchedPtr(buf, x * 4 * channels, x, y);
  params.dstArray = cuArray;
  params.extent = make_cudaExtent(x, y, z);
  params.kind = cudaMemcpyHostToDevice;
  check_cuda(cudaMemcpy3D(&params));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  check_cuda(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

  wp::volume::DenseVolumeCuda volume;
  volume.array = cuArray;
  volume.texture = texObj;
  volume.position = *position;
  volume.scale = *scale;
  volume.rotation = *rotation;

  wp::volume::DenseVolumeCuda *result =
      (wp::volume::DenseVolumeCuda *)wp::alloc<wp::Device::CUDA>(
          sizeof(wp::volume::DenseVolumeCuda));
  memcpy_h2d(result, &volume, sizeof(wp::volume::DenseVolumeCuda));

  return (uint64_t)result;
}

void dense_volume_destroy_device(uint64_t id) {
  if (!id) {
    return;
  }
  wp::volume::DenseVolumeCuda *src = (wp::volume::DenseVolumeCuda *)id;
  wp::volume::DenseVolumeCuda volume;
  memcpy_d2h(&volume, src, sizeof(wp::volume::DenseVolumeCuda));
  cudaDestroyTextureObject(volume.texture);
  cudaFreeArray(volume.array);
  wp::free<wp::Device::CUDA>(src);
}
