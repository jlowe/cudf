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

package ai.rapids.cudf;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class DeviceMemoryBufferTest {
  @Test
  void testBatchedCopyAsync() {
    int[] bufferSizes = new int[] {1, 7, 129, 51234, 12345678 };
    int numBuffers = bufferSizes.length;
    int totalSize = Arrays.stream(bufferSizes).sum();
    byte[] data = new byte[totalSize];
    for (int i = 0; i < totalSize; ++i) {
      data[i] = (byte) i;
    }
    HostMemoryBuffer[] expected = new HostMemoryBuffer[numBuffers];
    DeviceMemoryBuffer[] srcBuffers = new DeviceMemoryBuffer[numBuffers];
    DeviceMemoryBuffer[] destBuffers = new DeviceMemoryBuffer[numBuffers];
    try {
      // initialize expected buffers and place on device
      int dataOffset = 0;
      for (int bufidx = 0; bufidx < numBuffers; ++bufidx) {
        int bufferSize = bufferSizes[bufidx];
        expected[bufidx] = HostMemoryBuffer.allocate(bufferSize);
        expected[bufidx].setBytes(0, data, dataOffset, bufferSize);
        dataOffset += bufferSize;
        srcBuffers[bufidx] = DeviceMemoryBuffer.allocate(bufferSize);
        srcBuffers[bufidx].copyFromHostBuffer(expected[bufidx]);
      }

      // allocate destination buffers in reverse order
      for (int bufidx = numBuffers - 1; bufidx >= 0; --bufidx) {
        destBuffers[bufidx] = DeviceMemoryBuffer.allocate(bufferSizes[bufidx]);
      }

      // perform the batched copy
      DeviceMemoryBuffer.batchedCopyAsync(destBuffers, srcBuffers, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();

      // copy back to host and compare
      for (int bufidx = 0; bufidx < numBuffers; ++bufidx) {
        int bufferSize = bufferSizes[bufidx];
        try (HostMemoryBuffer actual = HostMemoryBuffer.allocate(bufferSize)) {
          actual.copyFromDeviceBuffer(destBuffers[bufidx]);
          byte[] expectedData = new byte[bufferSize];
          expected[bufidx].getBytes(expectedData, 0, 0, bufferSize);
          byte[] actualData = new byte[bufferSize];
          actual.getBytes(actualData, 0, 0, bufferSize);
          Assertions.assertArrayEquals(expectedData, actualData, "mismatch on buffer " + bufidx);
        }
      }
    } finally {
      closeBuffers(expected);
      closeBuffers(srcBuffers);
      closeBuffers(destBuffers);
    }
  }

  private void closeBuffers(MemoryBuffer[] buffers) {
    for (MemoryBuffer buffer : buffers) {
      if (buffer != null) {
        buffer.close();
      }
    }
  }
}
