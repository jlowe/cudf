/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "decimal_utils.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/table/table_view.hpp>

#include <cstddef>

namespace {

// Holds the 64-bit chunks of a 256-bit value
struct chunked256 {
  // sign-extend a 128-bit value into a chunked 256-bit value
  inline __device__ chunked256(__int128_t x) {
    chunks[0] = static_cast<uint64_t>(x);
    __int128_t x_shifted = x >> 64;
    chunks[1] = static_cast<uint64_t>(x_shifted);
    chunks[2] = static_cast<uint64_t>(x_shifted >> 64);
    chunks[3] = chunks[2];
  }

  inline __device__ uint64_t operator[](int i) const { return chunks[i]; }
  inline __device__ uint64_t &operator[](int i) { return chunks[i]; }

private:
  uint64_t chunks[4];
};

// Functor to multiply two DECIMAL128 columns with rounding and overflow detection.
struct dec128_multiplier : public thrust::unary_function<cudf::size_type, __int128_t> {
  dec128_multiplier(bool *overflows, cudf::column_mutable_view const &product_view,
                    cudf::column_view const &a_col, cudf::column_view const &b_col,
                    cudf::rounding_mode round_mode)
      : overflows(overflows), a_data(a_col.data<__int128_t>()), b_data(b_col.data<__int128_t>()),
        a_scale(a_col.type().scale()), b_scale(b_col.type().scale()),
        product_scale(product_view.type().scale()), round_mode(round_mode) {}

  __device__ __int128_t operator()(cudf::size_type i) const {
    chunked256 const a(a_data[i]);
    chunked256 const b(b_data[i]);

    chunked256 mul = multiply(a_chunks, b_chunks);

  }

private:
  // Perform a 256-bit multiply in 64-bit chunks
  static __device__ chunked256
  multiply(chunked256 const &a, chunked256 const &b) {
    chunked256 r;
    __uint128_t mul;
    uint64_t carry = 0;
    for (int a_idx = 0; a_idx < 4; ++a_idx) {
      mul = static_cast<__uint128_t>(a[a_idx]) * b[0] + carry;
      r[a_idx] = static_cast<uint64_t>(mul);
      carry = static_cast<uint64_t>(mul >> 64);
    }
    for (int b_idx = 1; b_idx < 4; ++b_idx) {
      carry = 0;
      for (int a = 0; a_idx < 4 - b_idx; ++a_idx) {
        int r_idx = a_idx + b_idx;
        mul = static_cast<__uint128_t>(a[a_idx]) * b[b_idx] + r[r_idx] + carry;
        r[r_idx] = static_cast<uint64_t>(mul);
        carry = static_cast<uint64_t>(mul >> 64);
      }
    }
    return r;
  }

  static __device__ std::pair<chunked256, chunked256>
  divide(chunked256 const &a, chunked256 const &b) {
  }

  // output column for overflow detected
  bool *const overflows;

  // input data for multiply
  __int128_t const * const a_data;
  __int128_t const * const b_data;
  int32_t const a_scale;
  int32_t const b_scale;
  int32_t const product_scale;
  cudf::rounding_mode const round_mode;
};

} // anonymous namespace

namespace cudf::jni {

std::unique_ptr<cudf::table>
multiply_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t product_scale,
                    cudf::rounding_method round_mode, rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] = cudf::detail::bitmask_and(cudf::table_view{{a, b}}, stream);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8}, num_rows,
                                                  cudf::detail::copy_bitmask(result_null_mask.data(), 0, num_rows, stream),
                                                  null_count, stream));
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, product_scale}, num_rows, result_null_mask, null_count, stream));
  auto overflows_view = columns[0]->mutable_view();
  auto product_view = columns[1]->mutable_view();
  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    product_view.begin<__int128_t>(),
                    dec128_multiplier(overflows_view.begin<bool>(), product_view, a, b, round_mode));
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace cudf::jni
