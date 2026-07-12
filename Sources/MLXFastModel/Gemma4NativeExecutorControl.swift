#if canImport(Metal)
import Foundation
import Metal
import MLX

final class Gemma4NativeExecutorControl {
    static let commandCount = 640

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState
    private let first: any MTLBuffer
    private let second: any MTLBuffer
    private let indirect: any MTLIndirectCommandBuffer

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else {
            return nil
        }
        let source = """
            #include <metal_stdlib>
            using namespace metal;

            kernel void mlxfast_icb_control(
                const device uint* input [[buffer(0)]],
                device uint* output [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
                if (index == 0) {
                    output[0] = input[0] + 1;
                }
            }
            """
        guard let library = try? device.makeLibrary(source: source, options: nil),
              let function = library.makeFunction(name: "mlxfast_icb_control")
        else {
            return nil
        }
        let pipelineDescriptor = MTLComputePipelineDescriptor()
        pipelineDescriptor.computeFunction = function
        pipelineDescriptor.supportIndirectCommandBuffers = true
        var reflection: MTLAutoreleasedComputePipelineReflection?
        guard let pipeline = try? device.makeComputePipelineState(
                  descriptor: pipelineDescriptor,
                  options: [],
                  reflection: &reflection),
              let first = device.makeBuffer(
                  length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let second = device.makeBuffer(
                  length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared)
        else {
            return nil
        }

        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = .concurrentDispatch
        descriptor.inheritBuffers = false
        descriptor.inheritPipelineState = false
        descriptor.maxKernelBufferBindCount = 2
        guard let indirect = device.makeIndirectCommandBuffer(
            descriptor: descriptor,
            maxCommandCount: Self.commandCount,
            options: [])
        else {
            return nil
        }

        for index in 0..<Self.commandCount {
            let command = indirect.indirectComputeCommandAt(index)
            if index > 0 {
                command.setBarrier()
            }
            command.setComputePipelineState(pipeline)
            let input = index.isMultiple(of: 2) ? first : second
            let output = index.isMultiple(of: 2) ? second : first
            command.setKernelBuffer(input, offset: 0, at: 0)
            command.setKernelBuffer(output, offset: 0, at: 1)
            command.concurrentDispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }

        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.first = first
        self.second = second
        self.indirect = indirect
    }

    func runDirect() -> UInt32 {
        reset()
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            preconditionFailure("unable to create direct Metal command buffer")
        }
        for index in 0..<Self.commandCount {
            let input = index.isMultiple(of: 2) ? first : second
            let output = index.isMultiple(of: 2) ? second : first
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            if index + 1 < Self.commandCount {
                encoder.memoryBarrier(resources: [first, second])
            }
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return result()
    }

    func runIndirect() -> UInt32 {
        reset()
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            preconditionFailure("unable to create indirect Metal command buffer")
        }
        encoder.useResource(first, usage: [.read, .write])
        encoder.useResource(second, usage: [.read, .write])
        encoder.executeCommandsInBuffer(
            indirect,
            range: 0..<Self.commandCount)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return result()
    }

    private func reset() {
        first.contents().assumingMemoryBound(to: UInt32.self).pointee = 0
        second.contents().assumingMemoryBound(to: UInt32.self).pointee = 0
    }

    private func result() -> UInt32 {
        let buffer = Self.commandCount.isMultiple(of: 2) ? first : second
        return buffer.contents().assumingMemoryBound(to: UInt32.self).pointee
    }

    static func roundTrip(_ input: MLXArray) -> MLXArray? {
        guard input.dtype == .bfloat16,
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let inputBuffer = input.asMTLBuffer(device: device, noCopy: true),
              let outputBuffer = device.makeBuffer(
                  length: input.nbytes,
                  options: .storageModeShared)
        else {
            return nil
        }
        let source = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void mlxfast_interop_copy(
                const device bfloat* input [[buffer(0)]],
                device bfloat* output [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
                output[index] = input[index];
            }
            """
        guard let library = try? device.makeLibrary(source: source, options: nil),
              let function = library.makeFunction(name: "mlxfast_interop_copy"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return nil
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: input.size, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
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
}
#endif
