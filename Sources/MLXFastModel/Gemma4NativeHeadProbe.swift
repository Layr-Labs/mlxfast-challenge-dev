#if canImport(Metal)
import Foundation
import Metal
import MLX

enum Gemma4NativeHeadProbeError: Error, CustomStringConvertible {
    case invalidInput(String)
    case unavailable(String)
    case commandFailed(String)

    var description: String {
        switch self {
        case .invalidInput(let message):
            return "invalid native token-boundary input: \(message)"
        case .unavailable(let message):
            return "native token-boundary resource unavailable: \(message)"
        case .commandFailed(let message):
            return "native token-boundary command failed: \(message)"
        }
    }
}

/// Direct-Metal parity probe for the quantized tied-token embedding boundary.
final class Gemma4NativeEmbeddingProbe {
    typealias ProbeError = Gemma4NativeHeadProbeError

    private static let hiddenSize = 5_376
    private static let packedWordsPerRow = hiddenSize / 8

    let registryID: UInt64
    let stableOutputBuffer: any MTLBuffer

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState
    private let weight: any MTLBuffer
    private let scales: any MTLBuffer
    private let biases: any MTLBuffer
    private let embeddingScale: any MTLBuffer
    private let retainedArrays: [MLXArray]

    convenience init(
        embedding: Gemma4LinearWeight,
        usePrivateStorage: Bool = false
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
              let queue = device.makeCommandQueue()
        else {
            throw ProbeError.unavailable("unified-memory Metal device or command queue")
        }
        try self.init(
            embedding: embedding,
            device: device,
            queue: queue,
            usePrivateStorage: usePrivateStorage
        )
    }

    init(
        embedding: Gemma4LinearWeight,
        device: any MTLDevice,
        queue: any MTLCommandQueue,
        usePrivateStorage: Bool = false
    ) throws {
        let arrays = try Gemma4NativeHeadProbeUtilities.validateTiedWeight(embedding)
        try Gemma4NativeHeadProbeUtilities.validateContext(device: device, queue: queue)

        let helperLibrary = try Gemma4NativeHeadProbeUtilities.makeHelperLibrary(device: device)
        let pipeline = try Gemma4NativeHeadProbeUtilities.helperPipeline(
            named: "mlxfast_tied_embedding_bfloat16",
            library: helperLibrary,
            device: device
        )

        self.device = device
        self.queue = queue
        self.registryID = device.registryID
        self.pipeline = pipeline
        self.weight = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            arrays.weight, named: "embed_tokens.weight", device: device)
        self.scales = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            arrays.scales, named: "embed_tokens.scales", device: device)
        self.biases = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            arrays.biases, named: "embed_tokens.biases", device: device)
        let scale = Float(Self.hiddenSize).squareRoot()
        self.embeddingScale = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            Gemma4NativeHeadProbeUtilities.bfloat16Bits(scale),
            name: "BF16 embedding scale",
            device: device
        )
        self.stableOutputBuffer = try Gemma4NativeHeadProbeUtilities.storageBuffer(
            length: Self.hiddenSize * DType.bfloat16.size,
            name: "Gemma4NativeEmbeddingProbe.stableOutput",
            options: usePrivateStorage ? .storageModePrivate : .storageModeShared,
            device: device
        )
        self.retainedArrays = [arrays.weight, arrays.scales, arrays.biases]
    }

    /// Allocates an additional private BF16 output for a native output ring.
    func makePrivateOutputBuffer() throws -> any MTLBuffer {
        try Gemma4NativeHeadProbeUtilities.storageBuffer(
            length: Self.hiddenSize * DType.bfloat16.size,
            name: "Gemma4NativeEmbeddingProbe.privateOutput",
            options: .storageModePrivate,
            device: device
        )
    }

    /// Appends the embedding dispatch to an existing encoder without ending it.
    @discardableResult
    func encode(
        into encoder: any Gemma4NativeCommandEncoder,
        tokenBuffer: any MTLBuffer,
        outputBuffer: (any MTLBuffer)? = nil
    ) throws -> any MTLBuffer {
        let output = outputBuffer ?? stableOutputBuffer
        try Gemma4NativeHeadProbeUtilities.validateEncoder(
            encoder, registryID: registryID)
        try Gemma4NativeHeadProbeUtilities.validateBuffer(
            tokenBuffer,
            minimumLength: MemoryLayout<Int32>.stride,
            name: "embedding token",
            registryID: registryID
        )
        try Gemma4NativeHeadProbeUtilities.validateBuffer(
            output,
            minimumLength: Self.hiddenSize * DType.bfloat16.size,
            name: "embedding output",
            registryID: registryID
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        encoder.setBuffer(biases, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBuffer(embeddingScale, offset: 0, index: 5)
        let threadWidth = min(224, pipeline.maxTotalThreadsPerThreadgroup)
        guard Self.packedWordsPerRow.isMultiple(of: threadWidth) else {
            throw ProbeError.unavailable(
                "embedding pipeline cannot express an exact uniform dispatch")
        }
        encoder.dispatchThreads(
            MTLSize(width: Self.packedWordsPerRow, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: 1, depth: 1)
        )
        encoder.memoryBarrier(resources: [output])
        return output
    }

    /// Executes one fused gather/dequantize/BF16-scale dispatch, then waits once.
    func run(token: MLXArray) throws -> MLXArray {
        guard token.dtype == .int32, token.shape == [1, 1] else {
            throw ProbeError.invalidInput("token must be Int32 [1,1]")
        }
        guard let tokenBuffer = token.asMTLBuffer(device: device, noCopy: true)
            ?? token.asMTLBuffer(device: device, noCopy: false)
        else {
            throw ProbeError.unavailable("token Metal buffer")
        }
        let output = try Gemma4NativeHeadProbeUtilities.sharedBuffer(
            length: Self.hiddenSize * DType.bfloat16.size,
            name: "Gemma4NativeEmbeddingProbe.output",
            device: device
        )
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.unavailable("embedding command buffer or compute encoder")
        }

        commandBuffer.label = "Gemma4NativeEmbeddingProbe"
        encoder.label = "Gemma4NativeEmbeddingProbe.direct"
        let encodedOutput = try encode(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            tokenBuffer: tokenBuffer,
            outputBuffer: output
        )
        encoder.endEncoding()

        withExtendedLifetime(token) {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        try Gemma4NativeHeadProbeUtilities.requireCompleted(commandBuffer)
        return try Gemma4NativeHeadProbeUtilities.makeMLXAlias(
            buffer: encodedOutput,
            shape: [1, 1, Self.hiddenSize],
            dtype: .bfloat16
        )
    }
}

/// Direct-Metal parity probe for final RMSNorm, tied QMV, and FP32 softcap.
final class Gemma4NativeHeadProbe {
    typealias ProbeError = Gemma4NativeHeadProbeError

    private static let vocabSize = 262_144
    private static let hiddenSize = 5_376

    let registryID: UInt64
    let stableOutputBuffer: any MTLBuffer

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let rmsPipeline: any MTLComputePipelineState
    private let qmvPipeline: any MTLComputePipelineState
    private let packed13QMVPipeline: (any MTLComputePipelineState)?
    private let softcapPipeline: any MTLComputePipelineState

    private let weight: any MTLBuffer
    private let scales: any MTLBuffer
    private let biases: any MTLBuffer
    private let packed13Indices: (any MTLBuffer)?
    private let packed13LUT: (any MTLBuffer)?
    private let finalNorm: any MTLBuffer
    private let normalizedHidden: any MTLBuffer
    private let bf16Logits: any MTLBuffer

    private let epsilon: any MTLBuffer
    private let normAxis: any MTLBuffer
    private let normWeightStride: any MTLBuffer
    private let inputWidth: any MTLBuffer
    private let outputWidth: any MTLBuffer
    private let softcap: any MTLBuffer
    private let retainedArrays: [MLXArray]

    convenience init(
        embedding: Gemma4LinearWeight,
        finalNorm: MLXArray,
        rmsNormEps: Float = 1e-6,
        logitSoftcap: Float = 30,
        usePacked13: Bool? = nil,
        usePrivateStorage: Bool = false
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
              let queue = device.makeCommandQueue()
        else {
            throw ProbeError.unavailable("unified-memory Metal device or command queue")
        }
        try self.init(
            embedding: embedding,
            finalNorm: finalNorm,
            device: device,
            queue: queue,
            rmsNormEps: rmsNormEps,
            logitSoftcap: logitSoftcap,
            usePacked13: usePacked13,
            usePrivateStorage: usePrivateStorage
        )
    }

    init(
        embedding: Gemma4LinearWeight,
        finalNorm: MLXArray,
        device: any MTLDevice,
        queue: any MTLCommandQueue,
        rmsNormEps: Float = 1e-6,
        logitSoftcap: Float = 30,
        usePacked13: Bool? = nil,
        usePrivateStorage: Bool = false
    ) throws {
        let arrays = try Gemma4NativeHeadProbeUtilities.validateTiedWeight(embedding)
        guard finalNorm.dtype == .bfloat16,
              finalNorm.ndim == 1,
              finalNorm.shape == [Self.hiddenSize]
        else {
            throw ProbeError.invalidInput("final norm must be BF16 [5376]")
        }
        guard rmsNormEps.isFinite, rmsNormEps > 0 else {
            throw ProbeError.invalidInput("RMS epsilon must be finite and positive")
        }
        guard logitSoftcap.isFinite, logitSoftcap == 30 else {
            throw ProbeError.invalidInput("promoted final logit softcap must be exactly 30")
        }
        try Gemma4NativeHeadProbeUtilities.validateContext(device: device, queue: queue)
        guard let trustedLibrary = Gemma4NativeKernelLibrary.load(device: device) else {
            throw ProbeError.unavailable("MLX metallib for injected Metal device")
        }
        guard let rmsPipeline = Gemma4NativeKernelLibrary.pipeline(
                  library: trustedLibrary,
                  device: device,
                  name: "rms_loopedbfloat16",
                  boolConstants: [20: true]
              ),
              let qmvPipeline = Gemma4NativeKernelLibrary.pipeline(
                  library: trustedLibrary,
                  device: device,
                  name: "affine_qmv_bfloat16_t_gs_64_b_4_batch_0"
              )
        else {
            throw ProbeError.unavailable("trusted final RMSNorm or generic affine QMV pipeline")
        }
        let helperLibrary = try Gemma4NativeHeadProbeUtilities.makeHelperLibrary(device: device)
        let packed13Enabled = try usePacked13
            ?? Gemma4NativeHeadProbeUtilities.packed13HeadEnabled()
        let packed13Metadata = try packed13Enabled
            ? Gemma4NativeHeadProbeUtilities.makePacked13Metadata(
                scales: arrays.scales,
                biases: arrays.biases
            )
            : nil
        let packed13QMVPipeline = try packed13Enabled
            ? Gemma4NativeHeadProbeUtilities.helperPipeline(
                named: "mlxfast_tied_head_packed13_bfloat16",
                library: helperLibrary,
                device: device
            )
            : nil
        let softcapPipeline = try Gemma4NativeHeadProbeUtilities.helperPipeline(
            named: "mlxfast_logit_softcap_float32",
            library: helperLibrary,
            device: device
        )

        self.device = device
        self.queue = queue
        self.registryID = device.registryID
        self.rmsPipeline = rmsPipeline
        self.qmvPipeline = qmvPipeline
        self.packed13QMVPipeline = packed13QMVPipeline
        self.softcapPipeline = softcapPipeline
        self.weight = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            arrays.weight, named: "tied_head.weight", device: device)
        self.scales = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            arrays.scales, named: "tied_head.scales", device: device)
        self.biases = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            arrays.biases, named: "tied_head.biases", device: device)
        self.packed13Indices = try packed13Metadata.map {
            try Gemma4NativeHeadProbeUtilities.metalBuffer(
                $0.indices,
                named: "tied_head.packed13_indices",
                device: device
            )
        }
        self.packed13LUT = try packed13Metadata.map {
            try Gemma4NativeHeadProbeUtilities.metalBuffer(
                $0.lut,
                named: "tied_head.packed13_lut",
                device: device
            )
        }
        self.finalNorm = try Gemma4NativeHeadProbeUtilities.metalBuffer(
            finalNorm, named: "final_norm.weight", device: device)
        let scratchOptions: MTLResourceOptions = usePrivateStorage
            ? .storageModePrivate
            : .storageModeShared
        self.normalizedHidden = try Gemma4NativeHeadProbeUtilities.storageBuffer(
            length: Self.hiddenSize * DType.bfloat16.size,
            name: "Gemma4NativeHeadProbe.normalizedHidden",
            options: scratchOptions,
            device: device
        )
        self.bf16Logits = try Gemma4NativeHeadProbeUtilities.storageBuffer(
            length: Self.vocabSize * DType.bfloat16.size,
            name: "Gemma4NativeHeadProbe.bf16Logits",
            options: scratchOptions,
            device: device
        )
        self.epsilon = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            rmsNormEps, name: "final RMS epsilon", device: device)
        self.normAxis = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            UInt32(Self.hiddenSize), name: "final RMS axis", device: device)
        self.normWeightStride = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            UInt32(1), name: "final RMS weight stride", device: device)
        self.inputWidth = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            Int32(Self.hiddenSize), name: "head input width", device: device)
        self.outputWidth = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            Int32(Self.vocabSize), name: "head output width", device: device)
        self.softcap = try Gemma4NativeHeadProbeUtilities.scalarBuffer(
            logitSoftcap, name: "logit softcap", device: device)
        self.stableOutputBuffer = try Gemma4NativeHeadProbeUtilities.sharedBuffer(
            length: Self.vocabSize * DType.float32.size,
            name: "Gemma4NativeHeadProbe.stableOutput",
            device: device
        )
        var retainedArrays = [arrays.weight, arrays.scales, arrays.biases, finalNorm]
        if let packed13Metadata {
            retainedArrays.append(packed13Metadata.indices)
            retainedArrays.append(packed13Metadata.lut)
        }
        self.retainedArrays = retainedArrays
    }

    /// Appends final RMSNorm, tied QMV, and softcap to an existing encoder.
    @discardableResult
    func encode(
        into encoder: any Gemma4NativeCommandEncoder,
        hiddenBuffer: any MTLBuffer,
        logitsBuffer: (any MTLBuffer)? = nil
    ) throws -> any MTLBuffer {
        let logits = logitsBuffer ?? stableOutputBuffer
        try Gemma4NativeHeadProbeUtilities.validateEncoder(
            encoder, registryID: registryID)
        try Gemma4NativeHeadProbeUtilities.validateBuffer(
            hiddenBuffer,
            minimumLength: Self.hiddenSize * DType.bfloat16.size,
            name: "final hidden",
            registryID: registryID
        )
        try Gemma4NativeHeadProbeUtilities.validateBuffer(
            logits,
            minimumLength: Self.vocabSize * DType.float32.size,
            name: "final logits",
            registryID: registryID
        )

        encoder.setComputePipelineState(rmsPipeline)
        encoder.setBuffer(hiddenBuffer, offset: 0, index: 0)
        encoder.setBuffer(finalNorm, offset: 0, index: 1)
        encoder.setBuffer(normalizedHidden, offset: 0, index: 2)
        encoder.setBuffer(epsilon, offset: 0, index: 3)
        encoder.setBuffer(normAxis, offset: 0, index: 4)
        encoder.setBuffer(normWeightStride, offset: 0, index: 5)
        let rmsThreads = rmsPipeline.maxTotalThreadsPerThreadgroup
        encoder.dispatchThreads(
            MTLSize(width: rmsThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rmsThreads, height: 1, depth: 1)
        )
        encoder.memoryBarrier(resources: [normalizedHidden])

        return encodeHead(
            into: encoder,
            normalizedHiddenBuffer: normalizedHidden,
            logitsBuffer: logits
        )
    }

    /// Appends tied QMV and softcap when the trunk already produced final RMSNorm.
    @discardableResult
    func encodeNormalized(
        into encoder: any Gemma4NativeCommandEncoder,
        normalizedHiddenBuffer: any MTLBuffer,
        logitsBuffer: (any MTLBuffer)? = nil
    ) throws -> any MTLBuffer {
        let logits = logitsBuffer ?? stableOutputBuffer
        try Gemma4NativeHeadProbeUtilities.validateEncoder(
            encoder, registryID: registryID)
        try Gemma4NativeHeadProbeUtilities.validateBuffer(
            normalizedHiddenBuffer,
            minimumLength: Self.hiddenSize * DType.bfloat16.size,
            name: "final normalized hidden",
            registryID: registryID
        )
        try Gemma4NativeHeadProbeUtilities.validateBuffer(
            logits,
            minimumLength: Self.vocabSize * DType.float32.size,
            name: "final logits",
            registryID: registryID
        )
        return encodeHead(
            into: encoder,
            normalizedHiddenBuffer: normalizedHiddenBuffer,
            logitsBuffer: logits
        )
    }

    private func encodeHead(
        into encoder: any Gemma4NativeCommandEncoder,
        normalizedHiddenBuffer: any MTLBuffer,
        logitsBuffer: any MTLBuffer
    ) -> any MTLBuffer {

        if let packed13QMVPipeline, let packed13Indices, let packed13LUT {
            encoder.setComputePipelineState(packed13QMVPipeline)
            encoder.setBuffer(weight, offset: 0, index: 0)
            encoder.setBuffer(packed13Indices, offset: 0, index: 1)
            encoder.setBuffer(packed13LUT, offset: 0, index: 2)
            encoder.setBuffer(normalizedHiddenBuffer, offset: 0, index: 3)
            encoder.setBuffer(bf16Logits, offset: 0, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: Self.vocabSize / 16, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )
        } else {
            encoder.setComputePipelineState(qmvPipeline)
            encoder.setBuffer(weight, offset: 0, index: 0)
            encoder.setBuffer(scales, offset: 0, index: 1)
            encoder.setBuffer(biases, offset: 0, index: 2)
            encoder.setBuffer(normalizedHiddenBuffer, offset: 0, index: 3)
            encoder.setBuffer(bf16Logits, offset: 0, index: 4)
            encoder.setBuffer(inputWidth, offset: 0, index: 5)
            encoder.setBuffer(outputWidth, offset: 0, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: Self.vocabSize / 8, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
            )
        }
        encoder.memoryBarrier(resources: [bf16Logits])

        encoder.setComputePipelineState(softcapPipeline)
        encoder.setBuffer(bf16Logits, offset: 0, index: 0)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 1)
        encoder.setBuffer(softcap, offset: 0, index: 2)
        let softcapThreads = min(256, softcapPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: Self.vocabSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: softcapThreads, height: 1, depth: 1)
        )
        return logitsBuffer
    }

    /// Executes final norm, tied generic QMV, and softcap in one command buffer.
    func run(hidden: MLXArray) throws -> MLXArray {
        guard hidden.dtype == .bfloat16, hidden.shape == [1, 1, Self.hiddenSize] else {
            throw ProbeError.invalidInput("hidden must be BF16 [1,1,5376]")
        }
        guard let hiddenBuffer = hidden.asMTLBuffer(device: device, noCopy: true) else {
            throw ProbeError.unavailable("no-copy final hidden alias")
        }
        let logits = try Gemma4NativeHeadProbeUtilities.sharedBuffer(
            length: Self.vocabSize * DType.float32.size,
            name: "Gemma4NativeHeadProbe.logits",
            device: device
        )
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.unavailable("head command buffer or compute encoder")
        }

        commandBuffer.label = "Gemma4NativeHeadProbe"
        encoder.label = "Gemma4NativeHeadProbe.direct"
        let encodedLogits = try encode(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            hiddenBuffer: hiddenBuffer,
            logitsBuffer: logits
        )
        encoder.endEncoding()

        withExtendedLifetime(hidden) {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        try Gemma4NativeHeadProbeUtilities.requireCompleted(commandBuffer)
        return try Gemma4NativeHeadProbeUtilities.makeMLXAlias(
            buffer: encodedLogits,
            shape: [1, 1, Self.vocabSize],
            dtype: .float32
        )
    }

    /// Synchronous parity wrapper for a caller-provided final-normalized row.
    func runNormalized(normalizedHidden: MLXArray) throws -> MLXArray {
        guard normalizedHidden.dtype == .bfloat16,
              normalizedHidden.shape == [1, 1, Self.hiddenSize],
              let normalizedBuffer = normalizedHidden.asMTLBuffer(
                device: device,
                noCopy: true
              )
        else {
            throw ProbeError.invalidInput(
                "normalized hidden must be contiguous BF16 [1,1,5376]")
        }
        let logits = try Gemma4NativeHeadProbeUtilities.sharedBuffer(
            length: Self.vocabSize * DType.float32.size,
            name: "Gemma4NativeHeadProbe.normalizedInputLogits",
            device: device
        )
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.unavailable(
                "normalized head command buffer or compute encoder")
        }

        commandBuffer.label = "Gemma4NativeHeadProbe.normalizedInput"
        encoder.label = "Gemma4NativeHeadProbe.normalizedInput.direct"
        let encodedLogits = try encodeNormalized(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            normalizedHiddenBuffer: normalizedBuffer,
            logitsBuffer: logits
        )
        encoder.endEncoding()

        withExtendedLifetime(normalizedHidden) {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        try Gemma4NativeHeadProbeUtilities.requireCompleted(commandBuffer)
        return try Gemma4NativeHeadProbeUtilities.makeMLXAlias(
            buffer: encodedLogits,
            shape: [1, 1, Self.vocabSize],
            dtype: .float32
        )
    }
}

private struct Gemma4NativeTiedWeightArrays {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

private struct Gemma4NativePacked13Metadata {
    let indices: MLXArray
    let lut: MLXArray
}

private enum Gemma4NativeHeadProbeUtilities {
    typealias ProbeError = Gemma4NativeHeadProbeError

    private static let vocabSize = 262_144
    private static let hiddenSize = 5_376
    private static let groupsPerRow = hiddenSize / 64
    private static let packed13WordsPerRow = 35
    private static let packed13LUTCount = 6_224

    static func packed13HeadEnabled() throws -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_NATIVE_PACKED13_HEAD"
        ] else {
            return true
        }
        switch raw.lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            throw ProbeError.invalidInput(
                "DARKBLOOM_NATIVE_PACKED13_HEAD must be 0 or 1")
        }
    }

    static func makePacked13Metadata(
        scales: MLXArray,
        biases: MLXArray
    ) throws -> Gemma4NativePacked13Metadata {
        let metadata = makeIndexedAffineMetadata(scales: scales, biases: biases)
        eval(metadata.indices, metadata.lut)
        guard metadata.indices.dtype == .uint16,
              metadata.indices.shape == [vocabSize, groupsPerRow],
              metadata.lut.dtype == .uint32,
              metadata.lut.shape == [packed13LUTCount]
        else {
            throw ProbeError.invalidInput(
                "tied-head indexed metadata does not match frozen packed13 geometry")
        }

        let indices = metadata.indices.asArray(UInt16.self)
        let expectedCount = vocabSize * groupsPerRow
        guard indices.count == expectedCount else {
            throw ProbeError.invalidInput("tied-head indexed metadata has invalid size")
        }
        var packed = Array(
            repeating: UInt32(0),
            count: vocabSize * packed13WordsPerRow
        )
        for row in 0..<vocabSize {
            let sourceBase = row * groupsPerRow
            let wordBase = row * packed13WordsPerRow
            for column in 0..<groupsPerRow {
                let index = UInt32(indices[sourceBase + column])
                guard index < packed13LUTCount else {
                    throw ProbeError.invalidInput(
                        "tied-head packed13 index exceeds LUT at row \(row), column \(column)")
                }
                let bitOffset = column * 13
                let word = wordBase + (bitOffset >> 5)
                let shift = bitOffset & 31
                packed[word] |= index << UInt32(shift)
                if shift > 19 {
                    packed[word + 1] |= index >> UInt32(32 - shift)
                }
            }
        }

        for row in 0..<vocabSize {
            let sourceBase = row * groupsPerRow
            let wordBase = row * packed13WordsPerRow
            guard packed[wordBase + packed13WordsPerRow - 1] & 0xfffffff0 == 0 else {
                throw ProbeError.invalidInput(
                    "tied-head packed13 row \(row) has nonzero padding bits")
            }
            for column in 0..<groupsPerRow {
                let bitOffset = column * 13
                let word = wordBase + (bitOffset >> 5)
                let shift = bitOffset & 31
                var index = packed[word] >> UInt32(shift)
                if shift > 19 {
                    index |= packed[word + 1] << UInt32(32 - shift)
                }
                guard UInt16(index & 0x1fff) == indices[sourceBase + column] else {
                    throw ProbeError.invalidInput(
                        "tied-head packed13 reconstruction differs at row \(row), column \(column)")
                }
            }
        }

        let packedArray = MLXArray(
            packed,
            [vocabSize, packed13WordsPerRow]
        )
        eval(packedArray)
        return Gemma4NativePacked13Metadata(
            indices: packedArray,
            lut: metadata.lut
        )
    }

    static func validateContext(
        device: any MTLDevice,
        queue: any MTLCommandQueue
    ) throws {
        guard device.hasUnifiedMemory else {
            throw ProbeError.unavailable("injected Metal device does not use unified memory")
        }
        guard queue.device.registryID == device.registryID else {
            throw ProbeError.invalidInput(
                "injected command queue belongs to a different Metal device")
        }
    }

    static func validateEncoder(
        _ encoder: any Gemma4NativeCommandEncoder,
        registryID: UInt64
    ) throws {
        guard encoder.device.registryID == registryID else {
            throw ProbeError.invalidInput(
                "compute encoder belongs to a different Metal device")
        }
    }

    static func validateBuffer(
        _ buffer: any MTLBuffer,
        minimumLength: Int,
        name: String,
        registryID: UInt64
    ) throws {
        guard buffer.device.registryID == registryID else {
            throw ProbeError.invalidInput("\(name) belongs to a different Metal device")
        }
        guard buffer.length >= minimumLength else {
            throw ProbeError.invalidInput(
                "\(name) has \(buffer.length) bytes; expected at least \(minimumLength)")
        }
    }

    static func validateTiedWeight(
        _ embedding: Gemma4LinearWeight
    ) throws -> Gemma4NativeTiedWeightArrays {
        guard embedding.logicalShape == [vocabSize, hiddenSize],
              embedding.groupSize == 64,
              embedding.bits == 4,
              embedding.weight.dtype == .uint32,
              embedding.weight.shape == [vocabSize, hiddenSize / 8],
              let scales = embedding.scales,
              scales.dtype == .bfloat16,
              scales.shape == [vocabSize, hiddenSize / 64],
              let biases = embedding.biases,
              biases.dtype == .bfloat16,
              biases.shape == scales.shape
        else {
            throw ProbeError.invalidInput(
                "tied embedding must be affine U32/BF16 4-bit group-64 [262144,5376]")
        }
        return Gemma4NativeTiedWeightArrays(
            weight: embedding.weight,
            scales: scales,
            biases: biases
        )
    }

    static func metalBuffer(
        _ array: MLXArray,
        named name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true),
              buffer.length >= array.nbytes
        else {
            throw ProbeError.unavailable("no-copy Metal alias for \(name)")
        }
        return buffer
    }

    static func scalarBuffer<T>(
        _ value: T,
        name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        var value = value
        let buffer: (any MTLBuffer)? = withUnsafeBytes(of: &value) { bytes in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: baseAddress,
                length: bytes.count,
                options: .storageModeShared
            )
        }
        guard let buffer else {
            throw ProbeError.unavailable("constant buffer \(name)")
        }
        return buffer
    }

    static func sharedBuffer(
        length: Int,
        name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        try storageBuffer(
            length: length,
            name: name,
            options: .storageModeShared,
            device: device
        )
    }

    static func storageBuffer(
        length: Int,
        name: String,
        options: MTLResourceOptions,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        guard length > 0,
              let buffer = device.makeBuffer(length: length, options: options)
        else {
            throw ProbeError.unavailable("Metal buffer \(name)")
        }
        buffer.label = name
        return buffer
    }

    static func makeMLXAlias(
        buffer: any MTLBuffer,
        shape: [Int],
        dtype: DType
    ) throws -> MLXArray {
        let elementCount = shape.reduce(1, *)
        guard buffer.storageMode == .shared,
              elementCount > 0,
              elementCount * dtype.size <= buffer.length
        else {
            throw ProbeError.invalidInput("native output buffer cannot back requested MLX shape")
        }

        // The unmanaged retain survives MLX's pinned leaked finalizer state
        // without placing a strong MTLBuffer capture in that state.
        let owner = Unmanaged.passRetained(buffer as AnyObject)
        return MLXArray(
            rawPointer: buffer.contents(),
            shape,
            dtype: dtype
        ) {
            owner.release()
        }
    }

    static func requireCompleted(_ commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.status == .completed else {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "status \(commandBuffer.status.rawValue)")
        }
    }

    static func bfloat16Bits(_ value: Float) -> UInt16 {
        var bits = value.bitPattern
        bits &+= 0x7fff &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    static func helperPipeline(
        named name: String,
        library: any MTLLibrary,
        device: any MTLDevice
    ) throws -> any MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw ProbeError.unavailable("helper function \(name)")
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.label = name
        descriptor.computeFunction = function
        descriptor.supportIndirectCommandBuffers = true
        var reflection: MTLAutoreleasedComputePipelineReflection?
        do {
            let pipeline = try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: &reflection
            )
            guard pipeline.supportIndirectCommandBuffers else {
                throw ProbeError.unavailable("helper pipeline \(name) is not ICB-capable")
            }
            return pipeline
        } catch {
            if let error = error as? ProbeError {
                throw error
            }
            throw ProbeError.unavailable("helper pipeline \(name): \(error)")
        }
    }

    static func makeHelperLibrary(device: any MTLDevice) throws -> any MTLLibrary {
        let source = """
            #include <metal_stdlib>
            #include <metal_math>
            #include <metal_simdgroup>
            using namespace metal;

            kernel void mlxfast_tied_embedding_bfloat16(
                const device int* token [[buffer(0)]],
                const device uint* weights [[buffer(1)]],
                const device bfloat* scales [[buffer(2)]],
                const device bfloat* biases [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                constant bfloat& embedding_scale [[buffer(5)]],
                uint packed_index [[thread_position_in_grid]]) {
                constexpr int vocab_size = 262144;
                constexpr uint packed_words_per_row = 672;
                constexpr uint groups_per_row = 84;
                constexpr uint words_per_group = 8;
                constexpr uint values_per_word = 8;

                const int raw_token = token[0];
                const int row = raw_token < 0 ? raw_token + vocab_size : raw_token;
                const uint output_base = packed_index * values_per_word;
                if (row < 0 || row >= vocab_size) {
                    #pragma clang loop unroll(full)
                    for (uint lane = 0; lane < values_per_word; ++lane) {
                        output[output_base + lane] = static_cast<bfloat>(0.0f);
                    }
                    return;
                }

                const size_t row_word =
                    size_t(row) * packed_words_per_row + packed_index;
                const size_t metadata_index =
                    size_t(row) * groups_per_row + packed_index / words_per_group;
                const uint packed = weights[row_word];
                const bfloat scale = scales[metadata_index];
                const bfloat bias = biases[metadata_index];
                #pragma clang loop unroll(full)
                for (uint lane = 0; lane < values_per_word; ++lane) {
                    const uchar code = static_cast<uchar>(
                        (packed >> (4 * lane)) & 0x0f);
                    const bfloat dequantized = scale * code + bias;
                    output[output_base + lane] = dequantized * embedding_scale;
                }
            }

            inline ushort mlxfast_tied_head_extract_packed13(
                const device uint* words,
                uint column
            ) {
                const uint bit_offset = column * 13;
                const uint word_index = bit_offset >> 5;
                const uint shift = bit_offset & 31;
                uint value = words[word_index] >> shift;
                if (shift > 19) {
                    value |= words[word_index + 1] << (32 - shift);
                }
                return static_cast<ushort>(value & 0x1fff);
            }

            inline float mlxfast_tied_head_pair_scale(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float mlxfast_tied_head_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float mlxfast_tied_head_load_values(
                const device bfloat* input,
                thread float* values
            ) {
                float sum = 0;
                for (int index = 0; index < 8; index += 4) {
                    sum += input[index] + input[index + 1]
                        + input[index + 2] + input[index + 3];
                    values[index] = input[index];
                    values[index + 1] = input[index + 1] / 16.0f;
                    values[index + 2] = input[index + 2] / 256.0f;
                    values[index + 3] = input[index + 3] / 4096.0f;
                }
                return sum;
            }

            inline float mlxfast_tied_head_qdot_4bit(
                const device uchar* weight,
                const thread float* values,
                float scale,
                float bias,
                float input_sum
            ) {
                const device ushort* packed =
                    reinterpret_cast<const device ushort*>(weight);
                float accumulator = 0;
                for (int index = 0; index < 2; ++index) {
                    accumulator +=
                        (values[4 * index] * (packed[index] & 0x000f)
                        + values[4 * index + 1] * (packed[index] & 0x00f0)
                        + values[4 * index + 2] * (packed[index] & 0x0f00)
                        + values[4 * index + 3] * (packed[index] & 0xf000));
                }
                return scale * accumulator + input_sum * bias;
            }

            kernel void mlxfast_tied_head_packed13_bfloat16(
                const device uint* weight [[buffer(0)]],
                const device uint* packed_indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* input [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint lane [[thread_index_in_simdgroup]],
                uint simd_group [[simdgroup_index_in_threadgroup]],
                uint3 threadgroup_position [[threadgroup_position_in_grid]]) {
                constexpr int kPackedWordsPerRow = 35;
                constexpr int kWeightBytesPerRow = 2688;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kSIMDGroupsPerThreadgroup = 4;

                const int output_row =
                    threadgroup_position.y
                        * kRowsPerSIMD * kSIMDGroupsPerThreadgroup
                    + simd_group * kRowsPerSIMD;
                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow + lane * 4;
                const device uint* row_packed_indices =
                    packed_indices + output_row * kPackedWordsPerRow;
                const device bfloat* x = input + lane * 8;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < 21; ++block) {
                    float values[8];
                    const float input_sum =
                        mlxfast_tied_head_load_values(x, values);
                    const uint metadata_column = block * 4 + lane / 8;
                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index =
                            mlxfast_tied_head_extract_packed13(
                                row_packed_indices + row * kPackedWordsPerRow,
                                metadata_column);
                        const uint pair = lut[metadata_index];
                        result[row] += mlxfast_tied_head_qdot_4bit(
                            row_weight,
                            values,
                            mlxfast_tied_head_pair_scale(pair),
                            mlxfast_tied_head_pair_bias(pair),
                            input_sum);
                    }
                    weight_bytes += 128;
                    x += 256;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (lane == 0) {
                        output[output_row + row] =
                            static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_logit_softcap_float32(
                const device bfloat* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant float& cap [[buffer(2)]],
                uint index [[thread_position_in_grid]]) {
                const float promoted = static_cast<float>(input[index]);
                const float divided = promoted / cap;
                const float activated = metal::precise::tanh(divided);
                output[index] = activated * cap;
            }
            """
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.fastMathEnabled = false
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw ProbeError.unavailable("token-boundary helper MSL library: \(error)")
        }
    }
}
#endif
