/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.CloseableArray;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.MemoryBuffer;

import java.util.Arrays;

/** Multi-buffer LZ4 compressor */
public class BatchedLZ4Compressor {
  static final long MIN_CHUNK_SIZE = 32768;
  static final long MAX_CHUNK_SIZE = 16777216;
  // each chunk has a 64-bit integer value as metadata containing the compressed size
  static final long METADATA_BYTES_PER_CHUNK = 8;

  private final int chunkSize;
  private final long targetIntermediateBufferSize;
  private final long maxOutputChunkSize;

  /**
   * Construct a batched LZ4 compressor instance
   * @param chunkSize maximum amount of uncompressed data to compress as a single chunk. Inputs
   *                  larger than this will be compressed in multiple chunks.
   * @param targetIntermediateBufferSize desired maximum size of intermediate device buffers
   *                                     used during compression.
   */
  public BatchedLZ4Compressor(int chunkSize, long targetIntermediateBufferSize) {
    validateChunkSize(chunkSize);
    this.chunkSize = chunkSize;
    this.maxOutputChunkSize = NvcompJni.batchedLZ4CompressGetMaxOutputChunkSize(chunkSize);
    assert maxOutputChunkSize < Integer.MAX_VALUE;
    this.targetIntermediateBufferSize = Math.max(targetIntermediateBufferSize, maxOutputChunkSize);
  }

  /**
   * Compress a batch of buffers with LZ4. The input buffers will be closed.
   * @param origInputs    buffers to compress
   * @param stream    CUDA stream to use
   * @return compressed buffers corresponding to the input buffers
   */
  public DeviceMemoryBuffer[] compress(BaseDeviceMemoryBuffer[] origInputs, Cuda.Stream stream) {
    try (CloseableArray<BaseDeviceMemoryBuffer> inputs =
             CloseableArray.wrap(Arrays.copyOf(origInputs, origInputs.length))) {
      if (chunkSize <= 0) {
        throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
      }
      final int numInputs = inputs.size();
      if (numInputs == 0) {
        return new DeviceMemoryBuffer[0];
      }

      int[] chunksPerInput = new int[numInputs];
      int numChunks = 0;
      for (int i = 0; i < numInputs; i++) {
        BaseDeviceMemoryBuffer buffer = inputs.get(i);
        int numBufferChunks = getNumChunksInBuffer(buffer);
        chunksPerInput[i] = numBufferChunks;
        numChunks += numBufferChunks;
      }

      try (CloseableArray<DeviceMemoryBuffer> compressedBuffers =
               allocCompressedBuffers(numChunks, stream);
           DeviceMemoryBuffer compressedChunkSizes =
               DeviceMemoryBuffer.allocate(numChunks * 8L, stream)) {
        long[] inputChunkAddrs = new long[numChunks];
        int[] inputChunkSizes = new int[numChunks];
        long[] outputChunkAddrs = new long[numChunks];
        buildAddrsAndSizes(inputs, inputChunkAddrs, inputChunkSizes,
            compressedBuffers, outputChunkAddrs);

        int[] outputChunkSizes;
        final long tempBufferSize = NvcompJni.batchedLZ4CompressGetTempSize(numChunks, chunkSize);
        try (DeviceMemoryBuffer addrsAndSizes =
                 putAddrsAndSizesOnDevice(inputChunkAddrs, inputChunkSizes, outputChunkAddrs, stream);
             DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempBufferSize, stream)) {
          final long devOutputAddrsPtr = addrsAndSizes.getAddress() + numChunks * 8L;
          final long devInputSizesPtr = devOutputAddrsPtr + numChunks * 8L;
          NvcompJni.batchedLZ4CompressAsync(
              addrsAndSizes.getAddress(),
              devInputSizesPtr,
              chunkSize,
              numChunks,
              tempBuffer.getAddress(),
              tempBufferSize,
              devOutputAddrsPtr,
              compressedChunkSizes.getAddress(),
              stream.getStream());
        }

        outputChunkSizes = getOutputChunkSizes(compressedChunkSizes, stream);

        // inputs are no longer needed at this point, so free them early
        inputs.close();

        return stitchOutput(chunksPerInput, compressedChunkSizes, outputChunkAddrs,
            outputChunkSizes, stream);
      }
    }
  }

  static void validateChunkSize(int chunkSize) {
    if (chunkSize < MIN_CHUNK_SIZE || chunkSize > MAX_CHUNK_SIZE) {
      throw new IllegalArgumentException("Chunk size must be between " + MIN_CHUNK_SIZE +
          " and " + MAX_CHUNK_SIZE);
    }
  }

  private static long ceilingDivide(long x, long y) {
    return (x + y - 1) / y;
  }

  private int getNumChunksInBuffer(MemoryBuffer buffer) {
    return (int) ceilingDivide(buffer.getLength(), chunkSize);
  }

  private CloseableArray<DeviceMemoryBuffer> allocCompressedBuffers(long numChunks,
                                                                    Cuda.Stream stream) {
    final long chunksPerBuffer = targetIntermediateBufferSize / maxOutputChunkSize;
    final long numBuffers = ceilingDivide(numChunks, chunksPerBuffer);
    if (numBuffers > Integer.MAX_VALUE) {
      throw new IllegalStateException("Too many chunks");
    }

    CloseableArray<DeviceMemoryBuffer> buffers = CloseableArray.wrap(
        new DeviceMemoryBuffer[(int) numBuffers]);
    try {
      // allocate all of the max-chunks intermediate compressed buffers
      for (int i = 0; i < buffers.size() - 1; ++i) {
        buffers.set(i, DeviceMemoryBuffer.allocate(chunksPerBuffer * maxOutputChunkSize, stream));
      }
      // allocate the tail intermediate compressed buffer that may be smaller than the others
      buffers.set(buffers.size() - 1, DeviceMemoryBuffer.allocate(
          (numChunks - chunksPerBuffer * (buffers.size() - 1)) * maxOutputChunkSize, stream));
      return buffers;
    } catch (Exception e) {
      buffers.close(e);
      throw e;
    }
  }

  private void buildAddrsAndSizes(CloseableArray<BaseDeviceMemoryBuffer> inputs,
                                  long[] inputChunkAddrs,
                                  int[] inputChunkSizes,
                                  CloseableArray<DeviceMemoryBuffer> compressedBuffers,
                                  long[] outputChunkAddrs) {
    // setup the input addresses and sizes
    int chunkIdx = 0;
    for (BaseDeviceMemoryBuffer input : inputs.getArray()) {
      final int numChunksInBuffer = getNumChunksInBuffer(input);
      for (int i = 0; i < numChunksInBuffer; i++) {
        inputChunkAddrs[chunkIdx] = input.getAddress() + i * chunkSize;
        inputChunkSizes[chunkIdx] = (i != numChunksInBuffer - 1) ? chunkSize
            : (int) (input.getLength() - (long) i * chunkSize);
        ++chunkIdx;
      }
    }
    assert chunkIdx == inputChunkAddrs.length;
    assert chunkIdx == inputChunkSizes.length;

    // setup output addresses
    chunkIdx = 0;
    for (DeviceMemoryBuffer buffer : compressedBuffers.getArray()) {
      assert buffer.getLength() % maxOutputChunkSize == 0;
      long numChunksInBuffer = buffer.getLength() / maxOutputChunkSize;
      long baseAddr = buffer.getAddress();
      for (int i = 0; i < numChunksInBuffer; i++) {
        outputChunkAddrs[chunkIdx++] = baseAddr + i * maxOutputChunkSize;
      }
    }
    assert chunkIdx == outputChunkAddrs.length;
  }

  private DeviceMemoryBuffer putAddrsAndSizesOnDevice(long[] inputAddrs,
                                                      int[] inputSizes,
                                                      long[] outputAddrs,
                                                      Cuda.Stream stream) {
    final long totalSize = inputAddrs.length * 8L * 3;
    final long outputAddrsOffset = inputAddrs.length * 8L;
    final long sizesOffset = outputAddrsOffset + inputAddrs.length * 8L;
    try (HostMemoryBuffer hostbuf = HostMemoryBuffer.allocate(totalSize);
         DeviceMemoryBuffer result = DeviceMemoryBuffer.allocate(totalSize)) {
      hostbuf.setLongs(0, inputAddrs, 0, inputAddrs.length);
      hostbuf.setLongs(outputAddrsOffset, outputAddrs, 0, outputAddrs.length);
      for (int i = 0; i < inputSizes.length; i++) {
        hostbuf.setLong(sizesOffset + i * 8L, inputSizes[i]);
      }
      result.copyFromHostBuffer(hostbuf, stream);
      result.incRefCount();
      return result;
    }
  }

  private int[] getOutputChunkSizes(BaseDeviceMemoryBuffer devChunkSizes, Cuda.Stream stream) {
    try (HostMemoryBuffer hostbuf = HostMemoryBuffer.allocate(devChunkSizes.getLength())) {
      hostbuf.copyFromDeviceBuffer(devChunkSizes, stream);
      int numChunks = (int) (devChunkSizes.getLength() / 8);
      int[] result = new int[numChunks];
      for (int i = 0; i < numChunks; i++) {
        long size = hostbuf.getLong(i * 8L);
        assert size < Integer.MAX_VALUE : "output size is too big";
        result[i] = (int) size;
      }
      return result;
    }
  }

  private DeviceMemoryBuffer[] stitchOutput(int[] chunksPerInput,
                                            DeviceMemoryBuffer compressedChunkSizes,
                                            long[] outputChunkAddrs,
                                            int[] outputChunkSizes,
                                            Cuda.Stream stream) {
    final int numOutputs = chunksPerInput.length;
    final long chunkSizesAddr = compressedChunkSizes.getAddress();
    long[] outputBufferSizes = calcOutputBufferSizes(chunksPerInput, outputChunkSizes);
    try (CloseableArray<DeviceMemoryBuffer> outputs =
             CloseableArray.wrap(new DeviceMemoryBuffer[numOutputs])) {
      // Each chunk needs to be copied, and each output needs a copy of the
      // compressed chunk size vector representing the metadata.
      final long totalBuffersToCopy = (long) numOutputs + outputChunkAddrs.length;
      final long inputAddrsOffset = 0;
      final long outputAddrsOffset = totalBuffersToCopy * 8;
      final long sizesOffset = outputAddrsOffset + totalBuffersToCopy * 8;
      int copyBufferIdx = 0;
      int chunkIdx = 0;
      try (HostMemoryBuffer hostAddrsSizes = HostMemoryBuffer.allocate(totalBuffersToCopy * 8 * 3)) {
        for (int outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
          DeviceMemoryBuffer outputBuffer = DeviceMemoryBuffer.allocate(outputBufferSizes[outputIdx]);
          final long outputBufferAddr = outputBuffer.getAddress();
          outputs.set(outputIdx, outputBuffer);
          final long numChunks = chunksPerInput[outputIdx];

          // setup a copy of the metadata at the front of the output buffer
          hostAddrsSizes.setLong(inputAddrsOffset + copyBufferIdx * 8,
              chunkSizesAddr + chunkIdx * 8);
          hostAddrsSizes.setLong(outputAddrsOffset + copyBufferIdx * 8, outputBufferAddr);
          hostAddrsSizes.setLong(sizesOffset + copyBufferIdx * 8, numChunks * 8);
          ++copyBufferIdx;

          // setup copies of the compressed chunks for this output buffer
          long nextChunkAddr = outputBufferAddr + numChunks * 8;
          for (int i = 0; i < numChunks; ++i) {
            hostAddrsSizes.setLong(inputAddrsOffset + copyBufferIdx * 8,
                outputChunkAddrs[chunkIdx]);
            hostAddrsSizes.setLong(outputAddrsOffset + copyBufferIdx * 8, nextChunkAddr);
            final long chunkSize = outputChunkSizes[chunkIdx];
            hostAddrsSizes.setLong(sizesOffset + copyBufferIdx * 8, chunkSize);
            copyBufferIdx++;
            chunkIdx++;
            nextChunkAddr += chunkSize;
          }
        }
        assert copyBufferIdx == totalBuffersToCopy;
        assert chunkIdx == outputChunkAddrs.length;
        assert chunkIdx == outputChunkSizes.length;

        try (DeviceMemoryBuffer devAddrsSizes = DeviceMemoryBuffer.allocate(hostAddrsSizes.getLength())) {
          devAddrsSizes.copyFromHostBuffer(hostAddrsSizes, stream);
          Cuda.multiBufferCopyAsync(totalBuffersToCopy,
              devAddrsSizes.getAddress() + inputAddrsOffset,
              devAddrsSizes.getAddress() + outputAddrsOffset,
              devAddrsSizes.getAddress() + sizesOffset,
              stream);
        }
      }

      return outputs.release();
    }
  }

  private long[] calcOutputBufferSizes(int[] chunksPerInput,
                                       int[] outputChunkSizes) {
    long[] sizes = new long[chunksPerInput.length];
    int chunkIdx = 0;
    for (int i = 0; i < sizes.length; i++) {
      final int chunksInBuffer = chunksPerInput[i];
      final int chunkEndIdx = chunkIdx + chunksInBuffer;
      // metadata stored in front of compressed data
      long bufferSize = METADATA_BYTES_PER_CHUNK * chunksInBuffer;
      // add in the compressed chunk sizes to get the total size
      while (chunkIdx < chunkEndIdx) {
        bufferSize += outputChunkSizes[chunkIdx++];
      }
      sizes[i] = bufferSize;
    }
    assert chunkIdx == outputChunkSizes.length;
    return sizes;
  }
}
