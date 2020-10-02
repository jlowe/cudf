/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "batched_device_copy.hpp"

namespace {

__global__ void batched_copy(cudf::jni::buffer_copy_descr const* descrs) {
  int buffer_idx = blockIdx.x;
  auto dest = static_cast<std::uint8_t*>(descrs[buffer_idx].dest_addr);
  auto src = static_cast<std::uint8_t const*>(descrs[buffer_idx].src_addr);
  auto size = descrs[buffer_idx].size;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    dest[i] = src[i];
  }
}

} // anonymous namespace

namespace cudf {
namespace jni {

cudaError_t batched_memcpy_async(rmm::device_uvector<buffer_copy_descr> const &descrs,
                                 cudaStream_t stream) {
  dim3 const grid(descrs.size());

  // copying entire buffer per thread block, so maximize
  // the number of threads per block
  constexpr int block_size = 1024;
  dim3 const block(block_size);

  batched_copy<<<grid, block, 0, stream>>>(descrs.data());
  return cudaGetLastError();
}

} // namespace jni
} // namespace cudf
