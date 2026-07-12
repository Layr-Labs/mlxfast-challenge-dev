#if canImport(Metal)
import Foundation
import Metal
import MLX

enum Gemma4NativeWholeTokenProbeError: Error, CustomStringConvertible {
    case invalidInput(String)
    case unavailable(String)
    case outputSlotLeased(Int)
    case commandFailed(String)

    var description: String {
        switch self {
        case .invalidInput(let message):
            return "invalid native whole-token input: \(message)"
        case .unavailable(let message):
            return "native whole-token resource unavailable: \(message)"
        case .outputSlotLeased(let slot):
            return "native whole-token logits slot \(slot) is still leased by an MLXArray"
        case .commandFailed(let message):
            return "native whole-token command failed: \(message)"
        }
    }
}

/// Diagnostic composition of the native embedding, 60-layer trunk, and head.
/// The probe owns its output storage, so it must outlive every returned alias.
final class Gemma4NativeWholeTokenProbe {
    typealias ProbeError = Gemma4NativeWholeTokenProbeError

    private static let vocabSize = 262_144
    private static let hiddenSize = 5_376

    private struct IndirectGeometry: Hashable {
        let fullAttentionBlocks: Int
        let usesSlidingTwoPass: Bool
        let logitsSlot: Int

        init(position: Int, logitsSlot: Int) {
            let activeLength = position + 1
            self.fullAttentionBlocks = (activeLength + 15) / 16
            self.usesSlidingTwoPass = activeLength >= 1_024
            self.logitsSlot = logitsSlot
        }
    }

    private let trunk: Gemma4NativeTrunkProbe
    private let embedding: Gemma4NativeEmbeddingProbe
    private let head: Gemma4NativeHeadProbe
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let tokenBuffer: any MTLBuffer
    private let logitsBuffers: [any MTLBuffer]
    private let residencyCollector: (any Gemma4NativeResidencyCollecting)?
    private let outputLeases = Gemma4NativeWholeTokenOutputLeases()
    private let executionLock = NSLock()
    private var indirectStreams: [IndirectGeometry: [Gemma4NativeIndirectCommandStream]] = [:]

    private var hasCompletedToken = false
    private var nextLogitsSlot = 0
    private var invalidationReason: String?
    private(set) var position: Int

    var cachedIndirectCommandCounts: [Int] {
        executionLock.lock()
        defer { executionLock.unlock() }
        return indirectStreams.values.map { streams in
            streams.reduce(0) { $0 + $1.commandCount }
        }.sorted()
    }

    init(
        residentWeights: Gemma4NativeResidentWeights,
        config: Gemma4Config,
        priorKeys: [MLXArray],
        priorValues: [MLXArray],
        position: Int,
        useSharedCacheStorage: Bool = false
    ) throws {
        guard config.vocabSize == Self.vocabSize,
              config.hiddenSize == Self.hiddenSize,
              config.numHiddenLayers == 60,
              config.tieWordEmbeddings
        else {
            throw ProbeError.invalidInput(
                "expected the frozen 60-layer, tied [262144,5376] Gemma 4 configuration")
        }

        // The trunk establishes the one device/queue pair for every component.
        let trunk = try Gemma4NativeTrunkProbe(
            residentWeights: residentWeights,
            config: config,
            priorKeys: priorKeys,
            priorValues: priorValues,
            positionOffset: position,
            usePrivateStorage: true,
            useSharedCacheStorage: useSharedCacheStorage
        )
        let device = trunk.metalDevice
        let queue = trunk.commandQueue
        let modelWeights = residentWeights.model
        guard modelWeights.lmHead == nil else {
            throw ProbeError.invalidInput("whole-token probe requires tied token weights")
        }
        let embedding = try Gemma4NativeEmbeddingProbe(
            embedding: modelWeights.embedTokens,
            device: device,
            queue: queue,
            usePrivateStorage: true
        )
        let head = try Gemma4NativeHeadProbe(
            embedding: modelWeights.embedTokens,
            finalNorm: modelWeights.finalNorm,
            device: device,
            queue: queue,
            rmsNormEps: Float(config.rmsNormEps),
            logitSoftcap: Float(config.finalLogitSoftcapping),
            usePrivateStorage: true
        )
        guard embedding.registryID == trunk.metalRegistryID,
              head.registryID == trunk.metalRegistryID
        else {
            throw ProbeError.invalidInput("native components do not share one Metal device")
        }

        let tokenBuffer = try Self.sharedBuffer(
            length: MemoryLayout<Int32>.stride,
            name: "Gemma4NativeWholeTokenProbe.token",
            device: device
        )
        let logitsBuffers = try (0..<2).map { slot in
            try Self.sharedBuffer(
                length: Self.vocabSize * DType.float32.size,
                name: "Gemma4NativeWholeTokenProbe.logits.\(slot)",
                device: device
            )
        }
        let residencyCollector: (any Gemma4NativeResidencyCollecting)?
        let residencyEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_NATIVE_RESIDENCY_SET"
        ] {
            residencyEnabled = !["0", "false", "no", "off"]
                .contains(raw.lowercased())
        } else {
            residencyEnabled = true
        }
        if #available(macOS 15.0, *),
           residencyEnabled
        {
            residencyCollector = Gemma4NativeResidencyCollector(device: device)
        } else {
            residencyCollector = nil
        }
        residencyCollector?.collect(tokenBuffer)
        for buffer in logitsBuffers {
            residencyCollector?.collect(buffer)
        }

        self.trunk = trunk
        self.embedding = embedding
        self.head = head
        self.device = device
        self.queue = queue
        self.tokenBuffer = tokenBuffer
        self.logitsBuffers = logitsBuffers
        self.residencyCollector = residencyCollector
        self.position = position
    }

    /// Executes one token entirely in one Metal encoder and synchronizes once.
    /// Calls are serialized; a selected logits slot is never reused while an
    /// MLX alias can still refer to it.
    func run(token: Int32) throws -> MLXArray {
        let useIndirect: Bool
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_ICB"] {
            useIndirect = ["1", "true", "yes", "on"].contains(raw.lowercased())
        } else {
            useIndirect = false
        }
        return try run(token: token, useIndirect: useIndirect)
    }

    func runIndirect(token: Int32) throws -> MLXArray {
        try run(token: token, useIndirect: true)
    }

    func resetCaches(
        priorKeys: [MLXArray],
        priorValues: [MLXArray],
        position: Int
    ) throws {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard !hasCompletedToken,
              invalidationReason == nil,
              indirectStreams.isEmpty
        else {
            throw ProbeError.invalidInput(
                "whole-token cache import is allowed only before first execution")
        }
        try trunk.resetCaches(
            priorKeys: priorKeys,
            priorValues: priorValues,
            position: position
        )
        self.position = position
    }

    func prefillCacheArrays() throws -> (keys: [MLXArray], values: [MLXArray]) {
        try trunk.prefillCacheArrays()
    }

    func adoptCaches(
        keys: [MLXArray],
        values: [MLXArray],
        position: Int
    ) throws {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard !hasCompletedToken,
              invalidationReason == nil,
              indirectStreams.isEmpty
        else {
            throw ProbeError.invalidInput(
                "whole-token cache adoption is allowed only before first execution")
        }
        try trunk.adoptCaches(
            keys: keys,
            values: values,
            position: position,
            residencyCollector: residencyCollector
        )
        residencyCollector?.activate(on: queue)
        self.position = position
    }

    func adoptCombinedCaches(
        storage: [MLXArray],
        position: Int
    ) throws {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard !hasCompletedToken,
              invalidationReason == nil,
              indirectStreams.isEmpty
        else {
            throw ProbeError.invalidInput(
                "whole-token combined cache adoption is allowed only before first execution")
        }
        try trunk.adoptCombinedCaches(
            storage: storage,
            position: position,
            residencyCollector: residencyCollector
        )
        residencyCollector?.activate(on: queue)
        self.position = position
    }

    func activatePrefillCaches(position: Int) throws {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard !hasCompletedToken,
              invalidationReason == nil,
              indirectStreams.isEmpty
        else {
            throw ProbeError.invalidInput(
                "whole-token prefill activation is allowed only before first execution")
        }
        try trunk.activatePrefillCaches(position: position)
        self.position = position
    }

    func warmupAndReset(position: Int) throws {
        executionLock.lock()
        defer { executionLock.unlock() }
        guard !hasCompletedToken,
              invalidationReason == nil,
              indirectStreams.isEmpty,
              self.position == 0,
              position >= 0,
              position < 1_024
        else {
            throw ProbeError.invalidInput(
                "native warmup requires a fresh position-zero executor")
        }
        try trunk.activatePrefillCaches(position: position)
        self.position = position
        tokenBuffer.contents().storeBytes(of: Int32(2), as: Int32.self)
        try executeDirect(slot: 0)
        residencyCollector?.activate(on: queue)
        try trunk.rewindAfterWarmup()
        self.position = 0
        hasCompletedToken = false
        nextLogitsSlot = 0
    }

    private func run(token: Int32, useIndirect: Bool) throws -> MLXArray {
        executionLock.lock()
        defer { executionLock.unlock() }

        if let invalidationReason {
            throw ProbeError.commandFailed(
                "request was invalidated after native cache mutation: \(invalidationReason)")
        }

        if hasCompletedToken {
            try trunk.prepare(position: position)
        }

        let slot = nextLogitsSlot
        guard outputLeases.acquire(slot: slot) else {
            throw ProbeError.outputSlotLeased(slot)
        }
        var leaseTransferred = false
        defer {
            if !leaseTransferred {
                outputLeases.release(slot: slot)
            }
        }

        tokenBuffer.contents().storeBytes(of: token, as: Int32.self)
        if useIndirect {
            do {
                try executeIndirect(slot: slot)
            } catch {
                invalidationReason = String(describing: error)
                throw error
            }
        } else {
            do {
                try executeDirect(slot: slot)
            } catch {
                invalidationReason = String(describing: error)
                throw error
            }
        }

        position += 1
        hasCompletedToken = true
        nextLogitsSlot = 1 - slot
        let output = makeLogitsAlias(slot: slot)
        leaseTransferred = true
        return output
    }

    private func executeDirect(slot: Int) throws {
        try trunk.beginTokenEncoding()
        let layerRanges = Gemma4NativeTrunkProbe.groupedLayerRanges
        var commandBuffers: [any MTLCommandBuffer] = []
        defer {
            for commandBuffer in commandBuffers {
                commandBuffer.waitUntilCompleted()
            }
        }
        commandBuffers.reserveCapacity(layerRanges.count)
        var current: Gemma4NativeEncodedTrunkState?
        var embeddedHidden: (any MTLBuffer)?
        for (groupIndex, range) in layerRanges.enumerated() {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder()
            else {
                throw ProbeError.unavailable(
                    "command buffer or compute encoder for group \(groupIndex)")
            }
            commandBuffer.label =
                "Gemma4NativeWholeTokenProbe.position.\(position).group.\(groupIndex)"
            encoder.label = "Gemma4NativeWholeTokenProbe.layers.\(range)"
            do {
                if groupIndex == 0 {
                    embeddedHidden = try embedding.encode(
                        into: Gemma4NativeDirectCommandEncoder(
                            encoder,
                            residencyCollector: residencyCollector
                        ),
                        tokenBuffer: tokenBuffer,
                        outputBuffer: embedding.stableOutputBuffer
                    )
                }
                guard let input = current?.hidden ?? embeddedHidden else {
                    throw ProbeError.invalidInput("missing grouped hidden input")
                }
                current = try trunk.encodeLayers(
                    range: range,
                    into: Gemma4NativeDirectCommandEncoder(
                        encoder,
                        residencyCollector: residencyCollector
                    ),
                    inputBuffer: input,
                    normalizedInputBuffer: current?.nextInputNormalized
                )
                if groupIndex + 1 == layerRanges.count {
                    guard let finalNormalized = current?.nextInputNormalized else {
                        throw ProbeError.invalidInput(
                            "missing final normalized trunk output")
                    }
                    _ = try head.encodeNormalized(
                        into: Gemma4NativeDirectCommandEncoder(
                            encoder,
                            residencyCollector: residencyCollector
                        ),
                        normalizedHiddenBuffer: finalNormalized,
                        logitsBuffer: logitsBuffers[slot]
                    )
                }
                encoder.endEncoding()
            } catch {
                encoder.endEncoding()
                throw error
            }
            commandBuffer.commit()
            commandBuffers.append(commandBuffer)
        }

        guard let finalCommandBuffer = commandBuffers.last else {
            throw ProbeError.unavailable("no grouped command buffers encoded")
        }
        finalCommandBuffer.waitUntilCompleted()
        for commandBuffer in commandBuffers where commandBuffer.status != .completed {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "status \(commandBuffer.status.rawValue)")
        }
    }

    private func executeIndirect(slot: Int) throws {
        let geometry = IndirectGeometry(position: position, logitsSlot: slot)
        let streams: [Gemma4NativeIndirectCommandStream]
        if let cached = indirectStreams[geometry] {
            streams = cached
        } else {
            let recorded = try recordIndirectStreams(slot: slot)
            indirectStreams[geometry] = recorded
            streams = recorded
        }

        try trunk.beginReplay()
        var commandBuffers: [any MTLCommandBuffer] = []
        defer {
            for commandBuffer in commandBuffers {
                commandBuffer.waitUntilCompleted()
            }
        }
        commandBuffers.reserveCapacity(streams.count)
        for (groupIndex, stream) in streams.enumerated() {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder()
            else {
                throw ProbeError.unavailable(
                    "ICB command buffer or compute encoder for group \(groupIndex)")
            }
            commandBuffer.label =
                "Gemma4NativeWholeTokenProbe.ICB.position.\(position).group.\(groupIndex)"
            encoder.label = "Gemma4NativeWholeTokenProbe.ICB.group.\(groupIndex)"
            stream.execute(into: encoder)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffers.append(commandBuffer)
        }
        guard let finalCommandBuffer = commandBuffers.last else {
            throw ProbeError.unavailable("no grouped ICB command buffers encoded")
        }
        finalCommandBuffer.waitUntilCompleted()
        for (groupIndex, commandBuffer) in commandBuffers.enumerated()
            where commandBuffer.status != .completed
        {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "ICB group \(groupIndex) status \(commandBuffer.status.rawValue)")
        }
        try trunk.completeReplay()
    }

    private func recordIndirectStreams(
        slot: Int
    ) throws -> [Gemma4NativeIndirectCommandStream] {
        var streams: [Gemma4NativeIndirectCommandStream] = []
        streams.reserveCapacity(Gemma4NativeTrunkProbe.groupedLayerRanges.count)
        var current: Gemma4NativeEncodedTrunkState?
        var embeddedHidden: (any MTLBuffer)?
        for (groupIndex, range) in Gemma4NativeTrunkProbe.groupedLayerRanges.enumerated() {
            let recorder = try Gemma4NativeIndirectCommandRecorder(device: device)
            if groupIndex == 0 {
                embeddedHidden = try embedding.encode(
                    into: recorder,
                    tokenBuffer: tokenBuffer,
                    outputBuffer: embedding.stableOutputBuffer
                )
            }
            guard let input = current?.hidden ?? embeddedHidden else {
                throw ProbeError.invalidInput("missing grouped ICB hidden input")
            }
            current = try trunk.recordLayers(
                range: range,
                into: recorder,
                inputBuffer: input,
                normalizedInputBuffer: current?.nextInputNormalized
            )
            if groupIndex + 1 == Gemma4NativeTrunkProbe.groupedLayerRanges.count {
                guard let finalNormalized = current?.nextInputNormalized else {
                    throw ProbeError.invalidInput(
                        "missing grouped ICB final normalized output")
                }
                _ = try head.encodeNormalized(
                    into: recorder,
                    normalizedHiddenBuffer: finalNormalized,
                    logitsBuffer: logitsBuffers[slot]
                )
            }
            streams.append(try recorder.finish())
        }
        return streams
    }

    private func makeLogitsAlias(slot: Int) -> MLXArray {
        let buffer = logitsBuffers[slot]
        let lease = Gemma4NativeWholeTokenOutputLease(
            buffer: buffer,
            leases: outputLeases,
            slot: slot
        )
        let leaseOwner = Unmanaged.passRetained(lease)
        return MLXArray(
            rawPointer: buffer.contents(),
            [1, 1, Self.vocabSize],
            dtype: .float32
        ) {
            leaseOwner.release()
        }
    }

    private static func sharedBuffer(
        length: Int,
        name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        guard length > 0,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared)
        else {
            throw ProbeError.unavailable("shared buffer \(name)")
        }
        buffer.label = name
        return buffer
    }
}

private final class Gemma4NativeWholeTokenOutputLease {
    private let buffer: any MTLBuffer
    private let leases: Gemma4NativeWholeTokenOutputLeases
    private let slot: Int

    init(
        buffer: any MTLBuffer,
        leases: Gemma4NativeWholeTokenOutputLeases,
        slot: Int
    ) {
        self.buffer = buffer
        self.leases = leases
        self.slot = slot
    }

    deinit {
        _ = buffer
        leases.release(slot: slot)
    }
}

private final class Gemma4NativeWholeTokenOutputLeases {
    private let lock = NSLock()
    private var leased = [false, false]

    func acquire(slot: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard leased.indices.contains(slot), !leased[slot] else {
            return false
        }
        leased[slot] = true
        return true
    }

    func release(slot: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard leased.indices.contains(slot) else { return }
        leased[slot] = false
    }
}
#endif
