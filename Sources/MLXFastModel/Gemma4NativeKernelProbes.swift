#if canImport(Metal)
import Foundation
import Metal
import MLX

enum Gemma4NativeKernelProbes {
    static func rmsNorm(
        input: MLXArray,
        weight: MLXArray,
        eps: Float
    ) -> MLXArray? {
        guard input.dtype == .bfloat16,
              weight.dtype == .bfloat16,
              input.dim(-1) == weight.size,
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = Gemma4NativeKernelLibrary.load(device: device),
              let pipeline = Gemma4NativeKernelLibrary.pipeline(
                  library: library,
                  device: device,
                  name: input.dim(-1) > 4_096
                    ? "rms_loopedbfloat16"
                    : "rmsbfloat16",
                  boolConstants: [20: true]
              ),
              let inputBuffer = input.asMTLBuffer(device: device, noCopy: true),
              let weightBuffer = weight.asMTLBuffer(device: device, noCopy: true),
              let outputBuffer = device.makeBuffer(
                  length: input.nbytes,
                  options: .storageModeShared)
        else {
            return nil
        }
        var eps = eps
        var axisSize = UInt32(input.dim(-1))
        var weightStride = UInt32(1)
        guard let epsBuffer = device.makeBuffer(
                  bytes: &eps,
                  length: MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let axisBuffer = device.makeBuffer(
                  bytes: &axisSize,
                  length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let strideBuffer = device.makeBuffer(
                  bytes: &weightStride,
                  length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(epsBuffer, offset: 0, index: 3)
        encoder.setBuffer(axisBuffer, offset: 0, index: 4)
        encoder.setBuffer(strideBuffer, offset: 0, index: 5)
        let rows = input.size / input.dim(-1)
        let threads = input.dim(-1) > 4_096
            ? pipeline.maxTotalThreadsPerThreadgroup
            : 32 * ((input.dim(-1) + 127) / 128)
        encoder.dispatchThreads(
            MTLSize(width: rows * threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return MLXArray(
            rawPointer: outputBuffer.contents(),
            input.shape,
            dtype: input.dtype
        ) { [outputBuffer] in
            _ = outputBuffer
        }
    }

    static func quantizedMV(
        input: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray
    ) -> MLXArray? {
        guard input.dtype == .bfloat16,
              weight.dtype == .uint32,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              scales.shape == biases.shape,
              weight.ndim == 2,
              input.ndim == 3,
              input.dim(0) == 1,
              input.dim(1) == 1,
              input.dim(2) == weight.dim(1) * 8,
              scales.shape == [weight.dim(0), input.dim(2) / 64],
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = Gemma4NativeKernelLibrary.load(device: device)
        else {
            return nil
        }
        let inputWidth = input.dim(2)
        let outputWidth = weight.dim(0)
        let fast = inputWidth.isMultiple(of: 512) && outputWidth.isMultiple(of: 8)
        let name = fast
            ? "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0"
            : "affine_qmv_bfloat16_t_gs_64_b_4_batch_0"
        guard let pipeline = Gemma4NativeKernelLibrary.pipeline(
                  library: library,
                  device: device,
                  name: name),
              let weightBuffer = weight.asMTLBuffer(device: device, noCopy: true),
              let scalesBuffer = scales.asMTLBuffer(device: device, noCopy: true),
              let biasesBuffer = biases.asMTLBuffer(device: device, noCopy: true),
              let inputBuffer = input.asMTLBuffer(device: device, noCopy: true),
              let outputBuffer = device.makeBuffer(
                  length: outputWidth * DType.bfloat16.size,
                  options: .storageModeShared)
        else {
            return nil
        }
        var inputWidthValue = Int32(inputWidth)
        var outputWidthValue = Int32(outputWidth)
        guard let inputWidthBuffer = device.makeBuffer(
                  bytes: &inputWidthValue,
                  length: MemoryLayout<Int32>.stride,
                  options: .storageModeShared),
              let outputWidthBuffer = device.makeBuffer(
                  bytes: &outputWidthValue,
                  length: MemoryLayout<Int32>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 1)
        encoder.setBuffer(biasesBuffer, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        encoder.setBuffer(inputWidthBuffer, offset: 0, index: 5)
        encoder.setBuffer(outputWidthBuffer, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: (outputWidth + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return MLXArray(
            rawPointer: outputBuffer.contents(),
            [1, 1, outputWidth],
            dtype: .bfloat16
        ) { [outputBuffer] in
            _ = outputBuffer
        }
    }

    static func rope(
        input: MLXArray,
        offset: Int,
        base: Float
    ) -> MLXArray? {
        guard input.dtype == .bfloat16,
              input.ndim == 4,
              input.dim(0) == 1,
              input.dim(2) == 1,
              input.dim(3) == 256,
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = Gemma4NativeKernelLibrary.load(device: device),
              let pipeline = Gemma4NativeKernelLibrary.pipeline(
                  library: library,
                  device: device,
                  name: "rope_single_bfloat16",
                  boolConstants: [1: true, 2: false, 3: false]
              ),
              let inputBuffer = input.asMTLBuffer(device: device, noCopy: true),
              let outputBuffer = device.makeBuffer(
                  length: input.nbytes,
                  options: .storageModeShared)
        else {
            return nil
        }
        var offsetValue = Int32(offset)
        var scale: Float = 1
        var stride = Int64(256)
        var log2Base = log2(base)
        guard let offsetBuffer = device.makeBuffer(
                  bytes: &offsetValue,
                  length: MemoryLayout<Int32>.stride,
                  options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(
                  bytes: &scale,
                  length: MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let strideBuffer = device.makeBuffer(
                  bytes: &stride,
                  length: MemoryLayout<Int64>.stride,
                  options: .storageModeShared),
              let baseBuffer = device.makeBuffer(
                  bytes: &log2Base,
                  length: MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(offsetBuffer, offset: 0, index: 2)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 3)
        encoder.setBuffer(strideBuffer, offset: 0, index: 4)
        encoder.setBuffer(baseBuffer, offset: 0, index: 10)
        encoder.dispatchThreads(
            MTLSize(width: 128, height: input.dim(1), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 32, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return MLXArray(
            rawPointer: outputBuffer.contents(),
            input.shape,
            dtype: input.dtype
        ) { [outputBuffer] in
            _ = outputBuffer
        }
    }

    static func slidingAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float
    ) -> MLXArray? {
        guard queries.dtype == .bfloat16,
              keys.dtype == .bfloat16,
              values.dtype == .bfloat16,
              queries.ndim == 4,
              keys.ndim == 4,
              values.shape == keys.shape,
              queries.shape == [1, 32, 1, 256],
              keys.dim(0) == 1,
              keys.dim(1) == 16,
              keys.dim(3) == 256,
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = Gemma4NativeKernelLibrary.load(device: device),
              let pipeline = Gemma4NativeKernelLibrary.pipeline(
                  library: library,
                  device: device,
                  name: "sdpa_vector_bfloat16_t_256_256",
                  boolConstants: [
                      20: false, 21: false, 22: false,
                      23: false, 24: false, 25: false,
                  ],
                  intConstants: [26: 0]
              ),
              let queryBuffer = queries.asMTLBuffer(device: device, noCopy: true),
              let keyBuffer = keys.asMTLBuffer(device: device, noCopy: true),
              let valueBuffer = values.asMTLBuffer(device: device, noCopy: true),
              let outputBuffer = device.makeBuffer(
                  length: queries.nbytes,
                  options: .storageModeShared)
        else {
            return nil
        }
        var gqaFactor = Int32(2)
        var length = Int32(keys.dim(2))
        var keyHeadStride = UInt(keys.dim(2) * 256)
        var keySequenceStride = UInt(256)
        var valueHeadStride = UInt(keys.dim(2) * 256)
        var valueSequenceStride = UInt(256)
        var scale = scale
        guard let gqaBuffer = device.makeBuffer(
                  bytes: &gqaFactor,
                  length: MemoryLayout<Int32>.stride,
                  options: .storageModeShared),
              let lengthBuffer = device.makeBuffer(
                  bytes: &length,
                  length: MemoryLayout<Int32>.stride,
                  options: .storageModeShared),
              let keyHeadBuffer = device.makeBuffer(
                  bytes: &keyHeadStride,
                  length: MemoryLayout<UInt>.stride,
                  options: .storageModeShared),
              let keySequenceBuffer = device.makeBuffer(
                  bytes: &keySequenceStride,
                  length: MemoryLayout<UInt>.stride,
                  options: .storageModeShared),
              let valueHeadBuffer = device.makeBuffer(
                  bytes: &valueHeadStride,
                  length: MemoryLayout<UInt>.stride,
                  options: .storageModeShared),
              let valueSequenceBuffer = device.makeBuffer(
                  bytes: &valueSequenceStride,
                  length: MemoryLayout<UInt>.stride,
                  options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(
                  bytes: &scale,
                  length: MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(valueBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBuffer(gqaBuffer, offset: 0, index: 4)
        encoder.setBuffer(lengthBuffer, offset: 0, index: 5)
        encoder.setBuffer(keyHeadBuffer, offset: 0, index: 6)
        encoder.setBuffer(keySequenceBuffer, offset: 0, index: 7)
        encoder.setBuffer(valueHeadBuffer, offset: 0, index: 8)
        encoder.setBuffer(valueSequenceBuffer, offset: 0, index: 9)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 10)
        encoder.dispatchThreadgroups(
            MTLSize(width: 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return MLXArray(
            rawPointer: outputBuffer.contents(),
            queries.shape,
            dtype: queries.dtype
        ) { [outputBuffer] in
            _ = outputBuffer
        }
    }
}
#endif
