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

// Functor to multiply two DECIMAL128 columns with rounding and overflow detection.
class dec128_multiplier : public thrust::unary_function<cudf::size_type, __int128_t> {
public:
  dec128_multiplier(bool *overflows, cudf::column_mutable_view const &product_view,
                    cudf::column_view const &a, cudf::column_view const &b,
                    cudf::rounding_mode round_mode)
      : overflows(overflows), a_data(a.data<__int128_t>()), b_data(b.data<__int128_t>()),
        a_scale(a.type().scale()), b_scale(b.type().scale()),
        product_scale(product_view.type().scale()), round_mode(round_mode) {}

  __device__ __int128_t operator()(cudf::size_type i) const {
    
  }

private:
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
