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

#include "decimal128_utils.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_DecimalUtils_multiply128(JNIEnv *env, jclass,
                                                                          jlong j_view_a,
                                                                          jlong j_view_b,
                                                                          jint j_product_scale,
                                                                          jint j_round_mode) {
  JNI_NULL_CHECK(env, j_view_a, "column is null", 0);
  JNI_NULL_CHECK(env, j_view_b, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto view_a = reinterpret_cast<cudf::column_view const *>(j_view_a);
    auto view_b = reinterpret_cast<cudf::column_view const *>(j_view_b);
    auto scale = static_cast<int>(j_product_scale);
    auto round_method = static_cast<cudf::rounding_method>(j_round_mode);
    return cudf::jni::convert_table_for_return(env, cudf::jni::multiply_decimal128(view_a, view_b,
                                                                                   scale,
                                                                                   round_method));
  }
  CATCH_STD(env, 0);
}

} // extern "C"
