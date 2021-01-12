/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"

namespace {

#define CONTIGUOUS_TABLE_CLASS "ai/rapids/cudf/ContiguousTable"
#define CONTIGUOUS_TABLE_FACTORY_SIG(param_sig) "(" param_sig ")L" CONTIGUOUS_TABLE_CLASS ";"

jclass Contiguous_table_jclass;
jmethodID From_packed_table_method;

} // anonymous namespace

namespace cudf {
namespace jni {

bool cache_contiguous_table_jni(JNIEnv *env) {
  jclass cls = env->FindClass(CONTIGUOUS_TABLE_CLASS);
  if (cls == nullptr) {
    return false;
  }

  From_packed_table_method =
      env->GetStaticMethodID(cls, "fromPackedTable", CONTIGUOUS_TABLE_FACTORY_SIG("JJJJ"));
  if (From_packed_table_method == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  Contiguous_table_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Contiguous_table_jclass == nullptr) {
    return false;
  }
  return true;
}

void release_contiguous_table_jni(JNIEnv *env) {
  if (Contiguous_table_jclass != nullptr) {
    env->DeleteGlobalRef(Contiguous_table_jclass);
    Contiguous_table_jclass = nullptr;
  }
}

jobject contiguous_table_from(JNIEnv *env, cudf::packed_columns &split) {
  jlong metadata_address = reinterpret_cast<jlong>(split.metadata.get());
  jlong data_address = reinterpret_cast<jlong>(split.gpu_data->data());
  jlong data_size = static_cast<jlong>(split.gpu_data->size());
  jlong rmm_buffer_address = reinterpret_cast<jlong>(split.gpu_data.get());

  jobject contig_table_obj =
      env->CallStaticObjectMethod(Contiguous_table_jclass, From_packed_table_method,
                                  metadata_address, data_address, data_size, rmm_buffer_address);

  if (contig_table_obj != nullptr) {
    split.metadata.release();
    split.gpu_data.release();
  }

  return contig_table_obj;
}

native_jobjectArray<jobject> contiguous_table_array(JNIEnv *env, jsize length) {
  return native_jobjectArray<jobject>(
      env, env->NewObjectArray(length, Contiguous_table_jclass, nullptr));
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_ContiguousTable_createMetadataDirectBuffer(
    JNIEnv *env, jclass, jlong j_metadata_ptr) {
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", nullptr);
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t> *>(j_metadata_ptr);
    return env->NewDirectByteBuffer(metadata->data(), metadata->size());
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ContiguousTable_closeMetadata(JNIEnv *env, jclass,
                                                                         jlong j_metadata_ptr) {
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", );
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t> *>(j_metadata_ptr);
    delete metadata;
  }
  CATCH_STD(env, );
}

} // extern "C"
