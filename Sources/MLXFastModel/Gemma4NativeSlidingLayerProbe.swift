#if canImport(Metal)
import Foundation
import Metal
import MLX
import MLXLMCommon

struct Gemma4NativeEncodedLayerState {
    let output: any MTLBuffer
    let nextInputNormalized: any MTLBuffer
    let keyCache: any MTLBuffer
    let valueCache: any MTLBuffer
    let activeCacheLength: Int
    let cacheCapacity: Int
}

struct Gemma4NativeEncodedTrunkState {
    let hidden: any MTLBuffer
    let nextInputNormalized: any MTLBuffer
}

struct Gemma4NativeAdoptedCacheBuffers {
    let keys: any MTLBuffer
    let values: any MTLBuffer
}

struct Gemma4NativeSlidingCacheState {
    let keys: MLXArray
    let values: MLXArray
    let position: Int
    let writeIndex: Int
    let activeLength: Int
    let capacity: Int
    let sourceStart: Int

    init(
        cache: any KVCache,
        expectedCapacity: Int
    ) throws {
        if let compilable = cache as? CompilableRotatingKVCache {
            let innerState = compilable.innerState()
            guard innerState.count == 4,
                  innerState[2].dtype == .int32,
                  innerState[2].shape == [1],
                  innerState[3].dtype == .int32,
                  innerState[3].shape == [1]
            else {
                throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                    "compilable sliding cache requires [keys,values,idx,offset] inner state")
            }
            eval(innerState)
            let parentMeta = compilable.metaState
            guard parentMeta.count == 5 else {
                throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                    "compilable sliding cache parent metadata is malformed")
            }
            let liveWriteIndex = innerState[2].item(Int32.self)
            let livePosition = innerState[3].item(Int32.self)
            try self.init(
                state: Array(innerState.prefix(2)),
                metaState: [
                    parentMeta[0],
                    parentMeta[1],
                    parentMeta[2],
                    String(livePosition),
                    String(liveWriteIndex),
                ],
                expectedCapacity: expectedCapacity
            )
            return
        }
        if cache is Gemma4CombinedKVCache {
            try self.init(
                state: cache.state,
                metaState: cache.metaState,
                expectedCapacity: expectedCapacity
            )
            return
        }
        guard cache is RotatingKVCache else {
            throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                "native sliding restore requires RotatingKVCache state")
        }
        try self.init(
            state: cache.state,
            metaState: cache.metaState,
            expectedCapacity: expectedCapacity
        )
    }

    init(
        state: [MLXArray],
        metaState: [String],
        expectedCapacity: Int
    ) throws {
        guard state.count == 2,
              metaState.count == 5,
              let keep = Int(metaState[0]),
              let capacity = Int(metaState[1]),
              let step = Int(metaState[2]),
              let position = Int(metaState[3]),
              let rawWriteIndex = Int(metaState[4])
        else {
            throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                "sliding cache restore requires two arrays and integer [keep,maxSize,step,offset,idx] metadata")
        }
        let keys = state[0]
        let values = state[1]
        guard keep == 0,
              capacity == expectedCapacity,
              capacity == 1_024,
              step > 0,
              position >= 0,
              position < 4_096,
              rawWriteIndex >= 0,
              keys.dtype == .bfloat16,
              keys.ndim == 4,
              keys.dim(0) == 1,
              keys.dim(1) == 16,
              keys.dim(3) == 256,
              values.dtype == .bfloat16,
              values.shape == keys.shape
        else {
            throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                "restored sliding cache does not match the fixed BF16 keep-0 ring contract")
        }

        let storageLength = keys.dim(2)
        let activeLength: Int
        let writeIndex: Int
        let sourceStart: Int
        if storageLength > capacity {
            guard position >= capacity,
                  rawWriteIndex == storageLength
            else {
                throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                    "overlong sliding cache does not match pinned chunk-prefill state")
            }
            activeLength = capacity
            writeIndex = 0
            sourceStart = storageLength - capacity
        } else {
            activeLength = min(position, capacity)
            writeIndex = rawWriteIndex == capacity ? 0 : rawWriteIndex
            sourceStart = 0
            guard writeIndex < capacity,
                  storageLength == capacity
                    || (position < capacity && storageLength == position),
                  position < capacity ? writeIndex == position : true
            else {
                throw Gemma4NativeSlidingLayerProbe.ProbeError.invalidInput(
                    "restored sliding cache storage, offset, and write index are inconsistent")
            }
        }

        self.keys = keys
        self.values = values
        self.position = position
        self.writeIndex = writeIndex
        self.activeLength = activeLength
        self.capacity = capacity
        self.sourceStart = sourceStart
    }
}

/// Diagnostic direct-Metal implementation of one Gemma 4 sliding decode layer.
///
/// This keeps explicit scratch except for the two promoted layer-boundary
/// fusions and their normalized-sibling lifetime reuse. It remains a parity
/// probe for the command sequence that a later ICB executor will replay.
final class Gemma4NativeSlidingLayerProbe {
    struct Result {
        let hidden: MLXArray
        let nextInputNormalized: MLXArray
        /// Capacity-shaped caches. Only `0..<activeCacheLength` is active.
        let keyCache: MLXArray
        let valueCache: MLXArray
        let activeCacheLength: Int
        let cacheCapacity: Int
    }

    enum ProbeError: Error, CustomStringConvertible {
        case invalidInput(String)
        case unavailable(String)
        case commandFailed(String)

        var description: String {
            switch self {
            case .invalidInput(let message):
                return "invalid native layer input: \(message)"
            case .unavailable(let message):
                return "native layer resource unavailable: \(message)"
            case .commandFailed(let message):
                return "native layer command failed: \(message)"
            }
        }
    }

    private static let hiddenSize = 5_376
    private static let intermediateSize = 21_504
    private static let queryHeads = 32
    private static let keyValueHeads = 16
    private static let headDimension = 256
    private static let queryWidth = queryHeads * headDimension
    private static let keyValueWidth = keyValueHeads * headDimension
    private static let bytesPerBF16 = 2

    private struct ProjectionBuffers {
        let weight: any MTLBuffer
        let scales: any MTLBuffer
        let biases: any MTLBuffer
        let inputWidth: Int
        let outputWidth: Int
    }

    private struct NormBuffers {
        let input: any MTLBuffer
        let query: any MTLBuffer
        let key: any MTLBuffer
        let postAttention: any MTLBuffer
        let preFeedForward: any MTLBuffer
        let postFeedForward: any MTLBuffer
        let layerScalar: any MTLBuffer
        let nextInput: any MTLBuffer
    }

    private struct Pipelines {
        let rmsLooped: any MTLComputePipelineState
        let qmv: any MTLComputePipelineState
        let qmvFast: any MTLComputePipelineState
        let indexedSlidingQKV: any MTLComputePipelineState
        let indexedOutput: any MTLComputePipelineState
        let packedIndexedOutput: any MTLComputePipelineState
        let indexedGateUpActivation: any MTLComputePipelineState
        let indexedDown: any MTLComputePipelineState
        let packedIndexedDown: any MTLComputePipelineState
        let fusedAttentionPreparation: any MTLComputePipelineState
        let fusedAttentionPreparationAndAppend: any MTLComputePipelineState
        let sdpa: any MTLComputePipelineState
        let sdpaFirstPass: any MTLComputePipelineState
        let sdpaSecondPass: any MTLComputePipelineState
        let cacheAppend: any MTLComputePipelineState
        let activatedProduct: any MTLComputePipelineState
        let attentionToMLPBoundary: any MTLComputePipelineState
        let mlpToNextBoundary: any MTLComputePipelineState
    }

    private struct ScratchBuffers {
        let hiddenNorm: any MTLBuffer
        let rawQuery: any MTLBuffer
        let rawKey: any MTLBuffer
        let rawValue: any MTLBuffer
        let normalizedValue: any MTLBuffer
        let ropedQuery: any MTLBuffer
        let ropedKey: any MTLBuffer
        let attention: any MTLBuffer
        let sdpaPartials: any MTLBuffer
        let sdpaSums: any MTLBuffer
        let sdpaMaxs: any MTLBuffer
        let attentionProjection: any MTLBuffer
        let attentionResidual: any MTLBuffer
        let preFeedForward: any MTLBuffer
        let gate: any MTLBuffer
        let up: any MTLBuffer
        let activated: any MTLBuffer
        let down: any MTLBuffer
        let output: any MTLBuffer
        var keyCache: any MTLBuffer
        var valueCache: any MTLBuffer
    }

    private struct ConstantBuffers {
        let epsilon: any MTLBuffer
        let hiddenAxis: any MTLBuffer
        let weightStride: any MTLBuffer

        let width4K: any MTLBuffer
        let width5376: any MTLBuffer
        let width8K: any MTLBuffer
        let width21504: any MTLBuffer

        let ropeOffset: any MTLBuffer

        let gqaFactor: any MTLBuffer
        let activeCacheLength: any MTLBuffer
        let keyHeadStride: any MTLBuffer
        let keySequenceStride: any MTLBuffer
        let valueHeadStride: any MTLBuffer
        let valueSequenceStride: any MTLBuffer
        let attentionScale: any MTLBuffer
        let sdpaBlocks: any MTLBuffer

        let appendIndex: any MTLBuffer
        let cacheCapacity: any MTLBuffer
    }

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let trustedLibrary: any MTLLibrary
    private let helperLibrary: any MTLLibrary
    private let pipelines: Pipelines
    private var scratch: ScratchBuffers
    private let constants: ConstantBuffers
    private let norms: NormBuffers
    private let queryProjection: ProjectionBuffers
    private let keyProjection: ProjectionBuffers
    private let valueProjection: ProjectionBuffers
    private let outputProjection: ProjectionBuffers
    private let gateProjection: ProjectionBuffers
    private let upProjection: ProjectionBuffers
    private let downProjection: ProjectionBuffers
    private let indexedQueryProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedKeyProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedValueProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedOutputProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedGateProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedUpProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedDownProjection: Gemma4NativeIndexedProjectionBuffers
    private let ropeCosines: any MTLBuffer
    private let ropeSines: any MTLBuffer
    private let usePromotedProjectionKernels: Bool
    private let fuseAttentionCacheAppend: Bool
    private let usePrivateStorage: Bool
    private let retainedArrays: [MLXArray]
    private var activeCacheLength: Int
    private var cacheCapacity: Int
    private let maximumCacheCapacity: Int
    private var pendingCacheGrowth: (
        key: any MTLBuffer,
        value: any MTLBuffer
    )?
    private var currentPosition: Int
    private var currentWriteIndex: Int
    private var encodedCurrentPosition = false

    convenience init(
        blockWeights: Gemma4BlockWeights,
        attentionWeights: Gemma4AttentionWeights,
        mlpWeights: Gemma4MLPWeights,
        cacheState: Gemma4NativeSlidingCacheState,
        rmsNormEps: Float = 1e-6,
        ropeBase: Float = 10_000,
        nextInputNormWeight: MLXArray? = nil,
        metalDevice: (any MTLDevice)? = nil,
        commandQueue: (any MTLCommandQueue)? = nil,
        usePromotedProjectionKernels: Bool = true,
        usePrivateStorage: Bool = false,
        useSharedCacheStorage: Bool = false
    ) throws {
        try self.init(
            blockWeights: blockWeights,
            attentionWeights: attentionWeights,
            mlpWeights: mlpWeights,
            priorKeys: cacheState.keys,
            priorValues: cacheState.values,
            positionOffset: cacheState.position,
            cacheCapacity: cacheState.capacity,
            rmsNormEps: rmsNormEps,
            ropeBase: ropeBase,
            nextInputNormWeight: nextInputNormWeight,
            metalDevice: metalDevice,
            commandQueue: commandQueue,
            usePromotedProjectionKernels: usePromotedProjectionKernels,
            usePrivateStorage: usePrivateStorage,
            useSharedCacheStorage: useSharedCacheStorage,
            restoredActiveLength: cacheState.activeLength,
            restoredWriteIndex: cacheState.writeIndex,
            restoredSourceStart: cacheState.sourceStart
        )
    }

    init(
        blockWeights: Gemma4BlockWeights,
        attentionWeights: Gemma4AttentionWeights,
        mlpWeights: Gemma4MLPWeights,
        priorKeys: MLXArray,
        priorValues: MLXArray,
        positionOffset: Int,
        cacheCapacity requestedCapacity: Int = 1_024,
        rmsNormEps: Float = 1e-6,
        ropeBase: Float = 10_000,
        nextInputNormWeight: MLXArray? = nil,
        metalDevice: (any MTLDevice)? = nil,
        commandQueue: (any MTLCommandQueue)? = nil,
        usePromotedProjectionKernels: Bool = true,
        usePrivateStorage: Bool = false,
        useSharedCacheStorage: Bool = false,
        qMetadata: IndexedAffineMetadata? = nil,
        kMetadata: IndexedAffineMetadata? = nil,
        vMetadata: IndexedAffineMetadata? = nil,
        outputMetadata: IndexedAffineMetadata? = nil,
        gateMetadata: IndexedAffineMetadata? = nil,
        upMetadata: IndexedAffineMetadata? = nil,
        downMetadata: IndexedAffineMetadata? = nil,
        restoredActiveLength: Int? = nil,
        restoredWriteIndex: Int? = nil,
        restoredSourceStart: Int? = nil
    ) throws {
        guard priorKeys.dtype == .bfloat16,
              priorKeys.ndim == 4,
              priorKeys.dim(0) == 1,
              priorKeys.dim(1) == Self.keyValueHeads,
              priorKeys.dim(3) == Self.headDimension,
              priorValues.dtype == .bfloat16,
              priorValues.shape == priorKeys.shape
        else {
            throw ProbeError.invalidInput(
                "prior K/V must be BF16 [1,16,length,256] with equal shapes")
        }
        let sourceLength = priorKeys.dim(2)
        let sourceStart = restoredSourceStart ?? 0
        let priorActiveLength = restoredActiveLength ?? sourceLength
        let activeCacheLength = min(priorActiveLength + 1, requestedCapacity)
        let cacheCapacity = max(requestedCapacity, activeCacheLength)
        let appendIndex = restoredWriteIndex ?? (sourceLength % cacheCapacity)
        guard priorActiveLength >= 0,
              priorActiveLength <= sourceLength,
              sourceStart >= 0,
              sourceStart + priorActiveLength <= sourceLength,
              priorActiveLength <= cacheCapacity,
              positionOffset >= priorActiveLength,
              positionOffset < 4_096,
              appendIndex >= 0,
              appendIndex < cacheCapacity,
              activeCacheLength <= 1_024,
              cacheCapacity <= 1_024
        else {
            throw ProbeError.invalidInput(
                "probe requires a valid cache state within the 4096-position RoPE table")
        }
        guard restoredActiveLength != nil
                || (positionOffset == sourceLength && sourceLength < cacheCapacity),
              (restoredActiveLength == nil) == (restoredWriteIndex == nil),
              (restoredActiveLength == nil) == (restoredSourceStart == nil)
        else {
            throw ProbeError.invalidInput(
                "canonical cache requires position==length; restored cache requires active length and write index")
        }
        guard positionOffset >= 0, positionOffset <= Int(Int32.max) else {
            throw ProbeError.invalidInput("position offset is outside Int32 range")
        }
        guard activeCacheLength <= Int(Int32.max), cacheCapacity <= Int(UInt32.max) else {
            throw ProbeError.invalidInput("cache length is outside the native ABI range")
        }
        guard rmsNormEps == 1e-6, ropeBase == 10_000 else {
            throw ProbeError.invalidInput(
                "RMS epsilon must be 1e-6 and sliding RoPE base must be 10000")
        }
        guard let valueWeight = attentionWeights.vProj else {
            throw ProbeError.invalidInput("a sliding layer requires v_proj weights")
        }

        try Self.validateNorm(blockWeights.inputLayerNorm, size: Self.hiddenSize, name: "input norm")
        try Self.validateNorm(blockWeights.postAttentionLayerNorm, size: Self.hiddenSize, name: "post-attention norm")
        try Self.validateNorm(blockWeights.preFeedForwardLayerNorm, size: Self.hiddenSize, name: "pre-FFN norm")
        try Self.validateNorm(blockWeights.postFeedForwardLayerNorm, size: Self.hiddenSize, name: "post-FFN norm")
        try Self.validateNorm(blockWeights.layerScalar, size: 1, name: "layer scalar")
        let ownedNextInputNorm = nextInputNormWeight ?? blockWeights.inputLayerNorm
        try Self.validateNorm(ownedNextInputNorm, size: Self.hiddenSize, name: "next input norm")
        try Self.validateNorm(attentionWeights.qNorm, size: Self.headDimension, name: "query norm")
        try Self.validateNorm(attentionWeights.kNorm, size: Self.headDimension, name: "key norm")

        guard let device = metalDevice ?? MTLCreateSystemDefaultDevice(),
              device.hasUnifiedMemory,
              let queue = commandQueue ?? device.makeCommandQueue(),
              let trustedLibrary = Gemma4NativeKernelLibrary.load(device: device)
        else {
            throw ProbeError.unavailable("Metal device, command queue, or MLX metallib")
        }
        let helperLibrary = try Self.makeHelperLibrary(device: device)
        let pipelines = try Self.makePipelines(
            device: device,
            trustedLibrary: trustedLibrary,
            helperLibrary: helperLibrary
        )

        let queryProjection = try Self.makeProjection(
            attentionWeights.qProj,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.queryWidth,
            name: "q_proj",
            device: device
        )
        let keyProjection = try Self.makeProjection(
            attentionWeights.kProj,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.keyValueWidth,
            name: "k_proj",
            device: device
        )
        let valueProjection = try Self.makeProjection(
            valueWeight,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.keyValueWidth,
            name: "v_proj",
            device: device
        )
        let outputProjection = try Self.makeProjection(
            attentionWeights.oProj,
            inputWidth: Self.queryWidth,
            outputWidth: Self.hiddenSize,
            name: "o_proj",
            device: device
        )
        let gateProjection = try Self.makeProjection(
            mlpWeights.gate,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.intermediateSize,
            name: "gate_proj",
            device: device
        )
        let upProjection = try Self.makeProjection(
            mlpWeights.up,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.intermediateSize,
            name: "up_proj",
            device: device
        )
        let downProjection = try Self.makeProjection(
            mlpWeights.down,
            inputWidth: Self.intermediateSize,
            outputWidth: Self.hiddenSize,
            name: "down_proj",
            device: device
        )
        func commonProjection(
            _ projection: ProjectionBuffers
        ) -> Gemma4NativeProjectionBuffers {
            Gemma4NativeProjectionBuffers(
                weight: projection.weight,
                scales: projection.scales,
                biases: projection.biases,
                inputWidth: projection.inputWidth,
                outputWidth: projection.outputWidth
            )
        }
        let indexedQueryProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: attentionWeights.qProj,
            raw: commonProjection(queryProjection),
            name: "q_proj",
            device: device,
            metadata: qMetadata
        )
        let indexedKeyProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: attentionWeights.kProj,
            raw: commonProjection(keyProjection),
            name: "k_proj",
            device: device,
            metadata: kMetadata
        )
        let indexedValueProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: valueWeight,
            raw: commonProjection(valueProjection),
            name: "v_proj",
            device: device,
            metadata: vMetadata
        )
        let indexedOutputProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: attentionWeights.oProj,
            raw: commonProjection(outputProjection),
            name: "o_proj",
            device: device,
            allowPacked12: true,
            metadata: outputMetadata
        )
        let indexedGateProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: mlpWeights.gate,
            raw: commonProjection(gateProjection),
            name: "gate_proj",
            device: device,
            metadata: gateMetadata
        )
        let indexedUpProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: mlpWeights.up,
            raw: commonProjection(upProjection),
            name: "up_proj",
            device: device,
            metadata: upMetadata
        )
        let indexedDownProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: mlpWeights.down,
            raw: commonProjection(downProjection),
            name: "down_proj",
            device: device,
            allowPacked12: true,
            metadata: downMetadata
        )

        let ropeTables = gemma4MaterializedAttentionRopeTables(isSliding: true)
        let ropeCosines = try Self.metalBuffer(
            ropeTables.cosines,
            named: "sliding RoPE cosine table",
            device: device
        )
        let ropeSines = try Self.metalBuffer(
            ropeTables.sines,
            named: "sliding RoPE sine table",
            device: device
        )
        let norms = NormBuffers(
            input: try Self.metalBuffer(blockWeights.inputLayerNorm, named: "input norm", device: device),
            query: try Self.metalBuffer(attentionWeights.qNorm, named: "query norm", device: device),
            key: try Self.metalBuffer(attentionWeights.kNorm, named: "key norm", device: device),
            postAttention: try Self.metalBuffer(
                blockWeights.postAttentionLayerNorm,
                named: "post-attention norm",
                device: device
            ),
            preFeedForward: try Self.metalBuffer(
                blockWeights.preFeedForwardLayerNorm,
                named: "pre-FFN norm",
                device: device
            ),
            postFeedForward: try Self.metalBuffer(
                blockWeights.postFeedForwardLayerNorm,
                named: "post-FFN norm",
                device: device
            ),
            layerScalar: try Self.metalBuffer(
                blockWeights.layerScalar,
                named: "layer scalar",
                device: device
            ),
            nextInput: try Self.metalBuffer(
                ownedNextInputNorm,
                named: "next input norm",
                device: device
            )
        )

        let scratch = try Self.makeScratch(
            device: device,
            cacheCapacity: cacheCapacity,
            usePrivateStorage: usePrivateStorage,
            useSharedCacheStorage: useSharedCacheStorage
        )
        let constants = try Self.makeConstants(
            device: device,
            epsilon: rmsNormEps,
            positionOffset: positionOffset,
            appendIndex: appendIndex,
            activeLength: activeCacheLength,
            cacheCapacity: cacheCapacity
        )
        try Gemma4NativeLayerUtilities.importCache(
            keys: priorKeys,
            values: priorValues,
            heads: Self.keyValueHeads,
            headDimension: Self.headDimension,
            sourceStart: sourceStart,
            copyLength: priorActiveLength,
            capacity: cacheCapacity,
            keyDestination: scratch.keyCache,
            valueDestination: scratch.valueCache,
            device: device,
            queue: queue
        )

        self.device = device
        self.queue = queue
        self.trustedLibrary = trustedLibrary
        self.helperLibrary = helperLibrary
        self.pipelines = pipelines
        self.scratch = scratch
        self.constants = constants
        self.norms = norms
        self.queryProjection = queryProjection
        self.keyProjection = keyProjection
        self.valueProjection = valueProjection
        self.outputProjection = outputProjection
        self.gateProjection = gateProjection
        self.upProjection = upProjection
        self.downProjection = downProjection
        self.indexedQueryProjection = indexedQueryProjection
        self.indexedKeyProjection = indexedKeyProjection
        self.indexedValueProjection = indexedValueProjection
        self.indexedOutputProjection = indexedOutputProjection
        self.indexedGateProjection = indexedGateProjection
        self.indexedUpProjection = indexedUpProjection
        self.indexedDownProjection = indexedDownProjection
        self.ropeCosines = ropeCosines
        self.ropeSines = ropeSines
        self.usePromotedProjectionKernels = usePromotedProjectionKernels
        self.fuseAttentionCacheAppend =
            Gemma4NativeLayerUtilities.nativeFusedAttentionCacheAppendEnabled()
        self.usePrivateStorage = usePrivateStorage
        self.retainedArrays = [
            blockWeights.inputLayerNorm,
            blockWeights.postAttentionLayerNorm,
            blockWeights.preFeedForwardLayerNorm,
            blockWeights.postFeedForwardLayerNorm,
            blockWeights.layerScalar,
            ownedNextInputNorm,
            attentionWeights.qNorm,
            attentionWeights.kNorm,
            ropeTables.cosines,
            ropeTables.sines,
            attentionWeights.qProj.weight,
            attentionWeights.qProj.scales!,
            attentionWeights.qProj.biases!,
            attentionWeights.kProj.weight,
            attentionWeights.kProj.scales!,
            attentionWeights.kProj.biases!,
            valueWeight.weight,
            valueWeight.scales!,
            valueWeight.biases!,
            attentionWeights.oProj.weight,
            attentionWeights.oProj.scales!,
            attentionWeights.oProj.biases!,
            mlpWeights.gate.weight,
            mlpWeights.gate.scales!,
            mlpWeights.gate.biases!,
            mlpWeights.up.weight,
            mlpWeights.up.scales!,
            mlpWeights.up.biases!,
            mlpWeights.down.weight,
            mlpWeights.down.scales!,
            mlpWeights.down.biases!,
            priorKeys,
            priorValues,
        ]
        self.activeCacheLength = activeCacheLength
        self.cacheCapacity = cacheCapacity
        self.maximumCacheCapacity = cacheCapacity
        self.currentPosition = positionOffset
        self.currentWriteIndex = appendIndex
    }

    func run(hidden: MLXArray) throws -> Result {
        guard !usePrivateStorage else {
            throw ProbeError.invalidInput(
                "sliding run() cannot create MLX aliases for private storage")
        }
        guard hidden.dtype == .bfloat16, hidden.shape == [1, 1, Self.hiddenSize] else {
            throw ProbeError.invalidInput("hidden must be BF16 [1,1,5376]")
        }
        guard let hiddenBuffer = hidden.asMTLBuffer(device: device, noCopy: true),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.unavailable("hidden alias, command buffer, or compute encoder")
        }

        commandBuffer.label = "Gemma4NativeSlidingLayerProbe"
        encoder.label = "Gemma4NativeSlidingLayerProbe.direct"

        let encoded = try encode(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            inputBuffer: hiddenBuffer
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "status \(commandBuffer.status.rawValue)")
        }

        return Result(
            hidden: makeMLXAlias(buffer: encoded.output, shape: [1, 1, Self.hiddenSize]),
            nextInputNormalized: makeMLXAlias(
                buffer: encoded.nextInputNormalized,
                shape: [1, 1, Self.hiddenSize]
            ),
            keyCache: makeMLXAlias(
                buffer: encoded.keyCache,
                shape: [1, Self.keyValueHeads, encoded.cacheCapacity, Self.headDimension]
            ),
            valueCache: makeMLXAlias(
                buffer: encoded.valueCache,
                shape: [1, Self.keyValueHeads, encoded.cacheCapacity, Self.headDimension]
            ),
            activeCacheLength: encoded.activeCacheLength,
            cacheCapacity: encoded.cacheCapacity
        )
    }

    var metalRegistryID: UInt64 {
        device.registryID
    }

    var fixedCacheCapacity: Int {
        maximumCacheCapacity
    }

    func prefillCacheArrays() throws -> (MLXArray, MLXArray) {
        guard scratch.keyCache.storageMode == .shared,
              scratch.valueCache.storageMode == .shared
        else {
            throw ProbeError.invalidInput(
                "sliding prefill aliases require shared cache storage")
        }
        let shape = [1, Self.keyValueHeads, cacheCapacity, Self.headDimension]
        return (
            makeMLXAlias(buffer: scratch.keyCache, shape: shape),
            makeMLXAlias(buffer: scratch.valueCache, shape: shape)
        )
    }

    /// Caller must ensure no command buffer referencing this plan is in flight.
    func resetCache(
        keys: MLXArray,
        values: MLXArray,
        position: Int
    ) throws {
        guard position >= 0,
              position < cacheCapacity,
              position + 1 <= 1_024,
              keys.dtype == .bfloat16,
              keys.shape == [1, Self.keyValueHeads, position, Self.headDimension],
              values.dtype == .bfloat16,
              values.shape == keys.shape
        else {
            throw ProbeError.invalidInput(
                "sliding reset cache must be BF16 [1,16,position,256] below capacity")
        }
        try Gemma4NativeLayerUtilities.importCache(
            keys: keys,
            values: values,
            heads: Self.keyValueHeads,
            headDimension: Self.headDimension,
            copyLength: position,
            capacity: cacheCapacity,
            keyDestination: scratch.keyCache,
            valueDestination: scratch.valueCache,
            device: device,
            queue: queue
        )
        try applyResetState(position: position)
    }

    func encodePrivateCacheReset(
        keys: MLXArray,
        values: MLXArray,
        position: Int,
        blit: any MTLBlitCommandEncoder
    ) throws {
        guard usePrivateStorage,
              scratch.keyCache.storageMode == .private,
              scratch.valueCache.storageMode == .private,
              position >= 0,
              position < cacheCapacity,
              position + 1 <= 1_024,
              keys.dtype == .bfloat16,
              keys.shape == [1, Self.keyValueHeads, position, Self.headDimension],
              values.dtype == .bfloat16,
              values.shape == keys.shape,
              let keySource = keys.asMTLBuffer(device: device, noCopy: true),
              let valueSource = values.asMTLBuffer(device: device, noCopy: true)
        else {
            throw ProbeError.invalidInput(
                "private sliding reset cache is not a contiguous BF16 canonical prefix")
        }
        let sourceHeadBytes = position * Self.headDimension * Self.bytesPerBF16
        let destinationHeadBytes = cacheCapacity * Self.headDimension * Self.bytesPerBF16
        for head in 0..<Self.keyValueHeads where sourceHeadBytes > 0 {
            blit.copy(
                from: keySource,
                sourceOffset: head * sourceHeadBytes,
                to: scratch.keyCache,
                destinationOffset: head * destinationHeadBytes,
                size: sourceHeadBytes
            )
            blit.copy(
                from: valueSource,
                sourceOffset: head * sourceHeadBytes,
                to: scratch.valueCache,
                destinationOffset: head * destinationHeadBytes,
                size: sourceHeadBytes
            )
        }
    }

    func commitPrivateCacheReset(position: Int) throws {
        try applyResetState(position: position)
    }

    func adoptCacheBuffers(
        _ buffers: Gemma4NativeAdoptedCacheBuffers,
        position: Int,
        residencyCollector: (any Gemma4NativeResidencyCollecting)? = nil
    ) throws {
        let bytesPerHeadPosition = Self.headDimension * Self.bytesPerBF16
        let cacheHeadBytes = Self.keyValueHeads * bytesPerHeadPosition
        let adoptedCapacity = buffers.keys.length / cacheHeadBytes
        guard buffers.keys.device.registryID == device.registryID,
              buffers.values.device.registryID == device.registryID,
              buffers.keys.length == adoptedCapacity * cacheHeadBytes,
              buffers.values.length == buffers.keys.length,
              adoptedCapacity > position,
              adoptedCapacity <= maximumCacheCapacity,
              position >= 0,
              pendingCacheGrowth == nil
        else {
            throw ProbeError.invalidInput("adopted sliding cache buffers are invalid")
        }
        residencyCollector?.remove(scratch.keyCache)
        residencyCollector?.remove(scratch.valueCache)
        scratch.keyCache = buffers.keys
        scratch.valueCache = buffers.values
        residencyCollector?.collect(buffers.keys)
        residencyCollector?.collect(buffers.values)
        cacheCapacity = adoptedCapacity
        try updateCacheCapacityConstants(adoptedCapacity)
        try applyResetState(position: position)
    }

    func needsCacheGrowth(position: Int) -> Bool {
        position >= cacheCapacity && cacheCapacity < maximumCacheCapacity
    }

    func encodeCacheGrowth(
        position: Int,
        blit: any MTLBlitCommandEncoder
    ) throws {
        guard needsCacheGrowth(position: position),
              position == cacheCapacity,
              usePrivateStorage,
              pendingCacheGrowth == nil
        else {
            throw ProbeError.invalidInput("invalid sliding cache growth request")
        }
        let bytesPerHead = cacheCapacity * Self.headDimension * Self.bytesPerBF16
        let grownHeadBytes = maximumCacheCapacity * Self.headDimension * Self.bytesPerBF16
        let grownLength = Self.keyValueHeads * grownHeadBytes
        guard let key = device.makeBuffer(length: grownLength, options: .storageModePrivate),
              let value = device.makeBuffer(length: grownLength, options: .storageModePrivate)
        else {
            throw ProbeError.unavailable("grown sliding cache buffers")
        }
        for head in 0..<Self.keyValueHeads {
            blit.copy(
                from: scratch.keyCache,
                sourceOffset: head * bytesPerHead,
                to: key,
                destinationOffset: head * grownHeadBytes,
                size: bytesPerHead
            )
            blit.copy(
                from: scratch.valueCache,
                sourceOffset: head * bytesPerHead,
                to: value,
                destinationOffset: head * grownHeadBytes,
                size: bytesPerHead
            )
        }
        pendingCacheGrowth = (key, value)
    }

    func commitCacheGrowth() throws {
        guard let pendingCacheGrowth else {
            throw ProbeError.invalidInput("sliding cache growth was not encoded")
        }
        scratch.keyCache = pendingCacheGrowth.key
        scratch.valueCache = pendingCacheGrowth.value
        cacheCapacity = maximumCacheCapacity
        self.pendingCacheGrowth = nil
        try updateCacheCapacityConstants(cacheCapacity)
    }

    private func updateCacheCapacityConstants(_ capacity: Int) throws {
        try Gemma4NativeLayerUtilities.writeScalar(
            UInt32(capacity),
            to: constants.cacheCapacity,
            name: "sliding cache capacity"
        )
        let headStride = UInt(capacity * Self.headDimension)
        try Gemma4NativeLayerUtilities.writeScalar(
            headStride,
            to: constants.keyHeadStride,
            name: "sliding key head stride"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            headStride,
            to: constants.valueHeadStride,
            name: "sliding value head stride"
        )
    }

    private func applyResetState(position: Int) throws {
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(position),
            to: constants.ropeOffset,
            name: "sliding reset RoPE offset"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(position + 1),
            to: constants.activeCacheLength,
            name: "sliding reset active cache length"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            UInt32(position),
            to: constants.appendIndex,
            name: "sliding reset append index"
        )
        currentPosition = position
        currentWriteIndex = position
        activeCacheLength = position + 1
        encodedCurrentPosition = false
    }

    /// Must be called only while no command buffer referencing this plan's
    /// shared dynamic constants is in flight.
    func prepare(position: Int) throws {
        if position == currentPosition, !encodedCurrentPosition {
            return
        }
        guard encodedCurrentPosition,
              position == currentPosition + 1,
              position >= 0,
              position < 4_096
        else {
            throw ProbeError.invalidInput(
                "sliding position must advance by one")
        }
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(position),
            to: constants.ropeOffset,
            name: "sliding RoPE offset"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(min(position + 1, cacheCapacity)),
            to: constants.activeCacheLength,
            name: "sliding active cache length"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            UInt32((currentWriteIndex + 1) % cacheCapacity),
            to: constants.appendIndex,
            name: "sliding append index"
        )
        currentPosition = position
        currentWriteIndex = (currentWriteIndex + 1) % cacheCapacity
        activeCacheLength = min(position + 1, cacheCapacity)
        encodedCurrentPosition = false
    }

    func encode(
        into encoder: any Gemma4NativeCommandEncoder,
        inputBuffer: any MTLBuffer,
        normalizedInputBuffer: (any MTLBuffer)? = nil,
        consumePosition: Bool = true
    ) throws -> Gemma4NativeEncodedLayerState {
        guard !encodedCurrentPosition,
              inputBuffer.device.registryID == device.registryID,
              inputBuffer.length >= Self.hiddenSize * Self.bytesPerBF16,
              normalizedInputBuffer?.device.registryID == nil
                || normalizedInputBuffer?.device.registryID == device.registryID,
              normalizedInputBuffer?.length == nil
                || normalizedInputBuffer!.length >= Self.hiddenSize * Self.bytesPerBF16
        else {
            throw ProbeError.invalidInput(
                "sliding encoder input must be a same-device BF16 hidden buffer")
        }

        let projectionInput: any MTLBuffer
        if let normalizedInputBuffer {
            projectionInput = normalizedInputBuffer
        } else {
            // The trunk supplies this normalization from the preceding layer.
            encodeRMS(
                encoder: encoder,
                pipeline: pipelines.rmsLooped,
                input: inputBuffer,
                weight: norms.input,
                output: scratch.hiddenNorm,
                rows: 1,
                axis: constants.hiddenAxis,
                weightStride: constants.weightStride,
                looped: true
            )
            encoder.memoryBarrier(resources: [scratch.hiddenNorm])
            projectionInput = scratch.hiddenNorm
        }

        if usePromotedProjectionKernels {
            encodeIndexedSlidingQKV(encoder: encoder, input: projectionInput)
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: queryProjection,
                input: projectionInput,
                output: scratch.rawQuery,
                inputWidth: constants.width5376,
                outputWidth: constants.width8K
            )
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: keyProjection,
                input: projectionInput,
                output: scratch.rawKey,
                inputWidth: constants.width5376,
                outputWidth: constants.width4K
            )
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: valueProjection,
                input: projectionInput,
                output: scratch.rawValue,
                inputWidth: constants.width5376,
                outputWidth: constants.width4K
            )
        }
        encoder.memoryBarrier(resources: [
            projectionInput, scratch.rawQuery, scratch.rawKey, scratch.rawValue,
        ])

        if fuseAttentionCacheAppend {
            // S4-S9: promoted Q/K/V RMSNorm, Q/K RoPE, and direct cache append.
            Gemma4NativeLayerUtilities.encodeFusedAttentionPreparation(
                encoder: encoder,
                pipeline: pipelines.fusedAttentionPreparationAndAppend,
                rawQuery: scratch.rawQuery,
                rawKey: scratch.rawKey,
                rawValue: scratch.rawValue,
                queryWeight: norms.query,
                keyWeight: norms.key,
                position: constants.ropeOffset,
                ropeCosines: ropeCosines,
                ropeSines: ropeSines,
                queries: scratch.ropedQuery,
                keys: scratch.keyCache,
                values: scratch.valueCache,
                appendIndex: constants.appendIndex,
                cacheCapacity: constants.cacheCapacity,
                threads: Self.headDimension / 4,
                rows: Self.queryHeads + 2 * Self.keyValueHeads
            )
            encoder.memoryBarrier(resources: [
                scratch.ropedQuery, scratch.keyCache, scratch.valueCache,
            ])
        } else {
            Gemma4NativeLayerUtilities.encodeFusedAttentionPreparation(
                encoder: encoder,
                pipeline: pipelines.fusedAttentionPreparation,
                rawQuery: scratch.rawQuery,
                rawKey: scratch.rawKey,
                rawValue: scratch.rawValue,
                queryWeight: norms.query,
                keyWeight: norms.key,
                position: constants.ropeOffset,
                ropeCosines: ropeCosines,
                ropeSines: ropeSines,
                queries: scratch.ropedQuery,
                keys: scratch.ropedKey,
                values: scratch.normalizedValue,
                threads: Self.headDimension / 4,
                rows: Self.queryHeads + 2 * Self.keyValueHeads
            )
            encoder.memoryBarrier(resources: [
                scratch.ropedQuery, scratch.ropedKey, scratch.normalizedValue,
            ])
            encoder.setComputePipelineState(pipelines.cacheAppend)
            encoder.setBuffer(scratch.ropedKey, offset: 0, index: 0)
            encoder.setBuffer(scratch.normalizedValue, offset: 0, index: 1)
            encoder.setBuffer(scratch.keyCache, offset: 0, index: 2)
            encoder.setBuffer(scratch.valueCache, offset: 0, index: 3)
            encoder.setBuffer(constants.appendIndex, offset: 0, index: 4)
            encoder.setBuffer(constants.cacheCapacity, offset: 0, index: 5)
            dispatchLinear(
                encoder: encoder,
                pipeline: pipelines.cacheAppend,
                count: Self.keyValueWidth
            )
            encoder.memoryBarrier(resources: [scratch.keyCache, scratch.valueCache])
        }

        if activeCacheLength == cacheCapacity, cacheCapacity == 1_024 {
            // Pinned MLX uses 64-block two-pass SDPA on M5 `s` at N == 1024.
            encoder.setComputePipelineState(pipelines.sdpaFirstPass)
            encoder.setBuffer(scratch.ropedQuery, offset: 0, index: 0)
            encoder.setBuffer(scratch.keyCache, offset: 0, index: 1)
            encoder.setBuffer(scratch.valueCache, offset: 0, index: 2)
            encoder.setBuffer(scratch.sdpaPartials, offset: 0, index: 3)
            encoder.setBuffer(scratch.sdpaSums, offset: 0, index: 4)
            encoder.setBuffer(scratch.sdpaMaxs, offset: 0, index: 5)
            encoder.setBuffer(constants.activeCacheLength, offset: 0, index: 7)
            encoder.setBuffer(constants.keyHeadStride, offset: 0, index: 8)
            encoder.setBuffer(constants.keySequenceStride, offset: 0, index: 9)
            encoder.setBuffer(constants.valueHeadStride, offset: 0, index: 10)
            encoder.setBuffer(constants.valueSequenceStride, offset: 0, index: 11)
            encoder.setBuffer(constants.attentionScale, offset: 0, index: 12)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.keyValueHeads, height: 1, depth: 64),
                threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
            )
            encoder.memoryBarrier(resources: [
                scratch.sdpaPartials, scratch.sdpaSums, scratch.sdpaMaxs,
            ])

            encoder.setComputePipelineState(pipelines.sdpaSecondPass)
            encoder.setBuffer(scratch.sdpaPartials, offset: 0, index: 0)
            encoder.setBuffer(scratch.sdpaSums, offset: 0, index: 1)
            encoder.setBuffer(scratch.sdpaMaxs, offset: 0, index: 2)
            encoder.setBuffer(scratch.attention, offset: 0, index: 3)
            encoder.setBuffer(constants.sdpaBlocks, offset: 0, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.queryHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
            )
        } else {
            encoder.setComputePipelineState(pipelines.sdpa)
            encoder.setBuffer(scratch.ropedQuery, offset: 0, index: 0)
            encoder.setBuffer(scratch.keyCache, offset: 0, index: 1)
            encoder.setBuffer(scratch.valueCache, offset: 0, index: 2)
            encoder.setBuffer(scratch.attention, offset: 0, index: 3)
            encoder.setBuffer(constants.gqaFactor, offset: 0, index: 4)
            encoder.setBuffer(constants.activeCacheLength, offset: 0, index: 5)
            encoder.setBuffer(constants.keyHeadStride, offset: 0, index: 6)
            encoder.setBuffer(constants.keySequenceStride, offset: 0, index: 7)
            encoder.setBuffer(constants.valueHeadStride, offset: 0, index: 8)
            encoder.setBuffer(constants.valueSequenceStride, offset: 0, index: 9)
            encoder.setBuffer(constants.attentionScale, offset: 0, index: 10)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.queryHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
            )
        }
        encoder.memoryBarrier(resources: [scratch.attention])

        if usePromotedProjectionKernels {
            encodeIndexedProjection(
                encoder: encoder,
                pipeline: pipelines.indexedOutput,
                packedPipeline: pipelines.packedIndexedOutput,
                projection: indexedOutputProjection,
                input: scratch.attention,
                output: scratch.attentionProjection
            )
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmvFast,
                projection: outputProjection,
                input: scratch.attention,
                output: scratch.attentionProjection,
                inputWidth: constants.width8K,
                outputWidth: constants.width5376
            )
        }
        encoder.memoryBarrier(resources: [scratch.attentionProjection])

        // S12-S14: promoted post-attention norm, residual, and pre-FFN norm.
        encoder.setComputePipelineState(pipelines.attentionToMLPBoundary)
        encoder.setBuffer(scratch.attentionProjection, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(norms.postAttention, offset: 0, index: 2)
        encoder.setBuffer(norms.preFeedForward, offset: 0, index: 3)
        encoder.setBuffer(scratch.attentionResidual, offset: 0, index: 4)
        encoder.setBuffer(scratch.preFeedForward, offset: 0, index: 5)
        dispatchBoundary(encoder: encoder, pipeline: pipelines.attentionToMLPBoundary)
        encoder.memoryBarrier(resources: [
            scratch.attentionResidual, scratch.preFeedForward,
        ])

        if usePromotedProjectionKernels {
            encodeIndexedGateUpActivation(
                encoder: encoder,
                input: scratch.preFeedForward
            )
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: gateProjection,
                input: scratch.preFeedForward,
                output: scratch.gate,
                inputWidth: constants.width5376,
                outputWidth: constants.width21504
            )
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: upProjection,
                input: scratch.preFeedForward,
                output: scratch.up,
                inputWidth: constants.width5376,
                outputWidth: constants.width21504
            )
            encoder.memoryBarrier(resources: [scratch.gate, scratch.up])

            encoder.setComputePipelineState(pipelines.activatedProduct)
            encoder.setBuffer(scratch.gate, offset: 0, index: 0)
            encoder.setBuffer(scratch.up, offset: 0, index: 1)
            encoder.setBuffer(scratch.activated, offset: 0, index: 2)
            dispatchLinear(
                encoder: encoder,
                pipeline: pipelines.activatedProduct,
                count: Self.intermediateSize
            )
        }
        encoder.memoryBarrier(resources: [scratch.activated])

        if usePromotedProjectionKernels {
            encodeIndexedProjection(
                encoder: encoder,
                pipeline: pipelines.indexedDown,
                packedPipeline: pipelines.packedIndexedDown,
                projection: indexedDownProjection,
                input: scratch.activated,
                output: scratch.down
            )
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmvFast,
                projection: downProjection,
                input: scratch.activated,
                output: scratch.down,
                inputWidth: constants.width21504,
                outputWidth: constants.width5376
            )
        }
        encoder.memoryBarrier(resources: [scratch.down])

        // S16-S17 and the next layer's S0 share the promoted boundary.
        encoder.setComputePipelineState(pipelines.mlpToNextBoundary)
        encoder.setBuffer(scratch.down, offset: 0, index: 0)
        encoder.setBuffer(scratch.attentionResidual, offset: 0, index: 1)
        encoder.setBuffer(norms.postFeedForward, offset: 0, index: 2)
        encoder.setBuffer(norms.layerScalar, offset: 0, index: 3)
        encoder.setBuffer(norms.nextInput, offset: 0, index: 4)
        encoder.setBuffer(scratch.output, offset: 0, index: 5)
        // hiddenNorm is dead after QKV and becomes the normalized sibling.
        encoder.setBuffer(scratch.hiddenNorm, offset: 0, index: 6)
        dispatchBoundary(encoder: encoder, pipeline: pipelines.mlpToNextBoundary)
        encoder.memoryBarrier(resources: [scratch.output, scratch.hiddenNorm])
        if consumePosition {
            encodedCurrentPosition = true
        }

        return Gemma4NativeEncodedLayerState(
            output: scratch.output,
            nextInputNormalized: scratch.hiddenNorm,
            keyCache: scratch.keyCache,
            valueCache: scratch.valueCache,
            activeCacheLength: activeCacheLength,
            cacheCapacity: cacheCapacity
        )
    }

    func markReplayCompleted() throws {
        guard !encodedCurrentPosition else {
            throw ProbeError.invalidInput(
                "sliding replay cannot consume an already encoded position")
        }
        encodedCurrentPosition = true
    }

    private func encodeIndexedSlidingQKV(
        encoder: any Gemma4NativeCommandEncoder,
        input: any MTLBuffer
    ) {
        encoder.setComputePipelineState(pipelines.indexedSlidingQKV)
        encoder.setBuffer(indexedQueryProjection.weight, offset: 0, index: 0)
        encoder.setBuffer(indexedQueryProjection.indices, offset: 0, index: 1)
        encoder.setBuffer(indexedQueryProjection.lut, offset: 0, index: 2)
        encoder.setBuffer(indexedKeyProjection.weight, offset: 0, index: 3)
        encoder.setBuffer(indexedKeyProjection.indices, offset: 0, index: 4)
        encoder.setBuffer(indexedKeyProjection.lut, offset: 0, index: 5)
        encoder.setBuffer(indexedValueProjection.weight, offset: 0, index: 6)
        encoder.setBuffer(indexedValueProjection.indices, offset: 0, index: 7)
        encoder.setBuffer(indexedValueProjection.lut, offset: 0, index: 8)
        encoder.setBuffer(input, offset: 0, index: 9)
        encoder.setBuffer(scratch.rawQuery, offset: 0, index: 10)
        encoder.setBuffer(scratch.rawKey, offset: 0, index: 11)
        encoder.setBuffer(scratch.rawValue, offset: 0, index: 12)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 4_096, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
    }

    private func encodeIndexedGateUpActivation(
        encoder: any Gemma4NativeCommandEncoder,
        input: any MTLBuffer
    ) {
        encoder.setComputePipelineState(pipelines.indexedGateUpActivation)
        encoder.setBuffer(indexedGateProjection.weight, offset: 0, index: 0)
        encoder.setBuffer(indexedGateProjection.indices, offset: 0, index: 1)
        encoder.setBuffer(indexedGateProjection.lut, offset: 0, index: 2)
        encoder.setBuffer(indexedUpProjection.weight, offset: 0, index: 3)
        encoder.setBuffer(indexedUpProjection.indices, offset: 0, index: 4)
        encoder.setBuffer(indexedUpProjection.lut, offset: 0, index: 5)
        encoder.setBuffer(input, offset: 0, index: 6)
        encoder.setBuffer(scratch.activated, offset: 0, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 10_752, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
    }

    private func encodeIndexedProjection(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        packedPipeline: any MTLComputePipelineState,
        projection: Gemma4NativeIndexedProjectionBuffers,
        input: any MTLBuffer,
        output: any MTLBuffer
    ) {
        encoder.setComputePipelineState(
            projection.packedIndices == nil ? pipeline : packedPipeline)
        encoder.setBuffer(projection.weight, offset: 0, index: 0)
        encoder.setBuffer(
            projection.packedIndices ?? projection.indices,
            offset: 0,
            index: 1
        )
        encoder.setBuffer(projection.lut, offset: 0, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 1_344, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
    }

    private func encodeRMS(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        input: any MTLBuffer,
        weight: any MTLBuffer,
        output: any MTLBuffer,
        rows: Int,
        axis: any MTLBuffer,
        weightStride: any MTLBuffer,
        looped: Bool
    ) {
        Gemma4NativeLayerUtilities.encodeRMS(
            encoder: encoder,
            pipeline: pipeline,
            input: input,
            weight: weight,
            output: output,
            epsilon: constants.epsilon,
            axis: axis,
            weightStride: weightStride,
            rows: rows,
            threads: looped ? pipeline.maxTotalThreadsPerThreadgroup : 64
        )
    }

    private func encodeQMV(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        projection: ProjectionBuffers,
        input: any MTLBuffer,
        output: any MTLBuffer,
        inputWidth: any MTLBuffer,
        outputWidth: any MTLBuffer
    ) {
        Gemma4NativeLayerUtilities.encodeQMV(
            encoder: encoder,
            pipeline: pipeline,
            weight: projection.weight,
            scales: projection.scales,
            biases: projection.biases,
            input: input,
            output: output,
            inputWidth: inputWidth,
            outputWidth: outputWidth,
            outputElements: projection.outputWidth
        )
    }

    private func dispatchLinear(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        count: Int
    ) {
        Gemma4NativeLayerUtilities.dispatchLinear(
            encoder: encoder,
            pipeline: pipeline,
            count: count
        )
    }

    private func dispatchBoundary(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState
    ) {
        encoder.dispatchThreads(
            MTLSize(width: 1_024, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
        )
    }

    private func makeMLXAlias(buffer: any MTLBuffer, shape: [Int]) -> MLXArray {
        Gemma4NativeLayerUtilities.makeMLXAlias(buffer: buffer, shape: shape)
    }

    private static func validateNorm(
        _ array: MLXArray,
        size: Int,
        name: String
    ) throws {
        guard array.dtype == .bfloat16, array.ndim == 1, array.size == size else {
            throw ProbeError.invalidInput("\(name) must be BF16 [\(size)]")
        }
    }

    private static func makeProjection(
        _ projection: Gemma4LinearWeight,
        inputWidth: Int,
        outputWidth: Int,
        name: String,
        device: any MTLDevice
    ) throws -> ProjectionBuffers {
        guard projection.logicalShape == [outputWidth, inputWidth],
              projection.groupSize == 64,
              projection.bits == 4,
              projection.weight.dtype == .uint32,
              projection.weight.shape == [outputWidth, inputWidth / 8],
              let scales = projection.scales,
              scales.dtype == .bfloat16,
              scales.shape == [outputWidth, inputWidth / 64],
              let biases = projection.biases,
              biases.dtype == .bfloat16,
              biases.shape == scales.shape
        else {
            throw ProbeError.invalidInput(
                "\(name) must be affine U32/BF16 4-bit group-64 [\(outputWidth),\(inputWidth)]")
        }
        return ProjectionBuffers(
            weight: try metalBuffer(projection.weight, named: "\(name).weight", device: device),
            scales: try metalBuffer(scales, named: "\(name).scales", device: device),
            biases: try metalBuffer(biases, named: "\(name).biases", device: device),
            inputWidth: inputWidth,
            outputWidth: outputWidth
        )
    }

    private static func metalBuffer(
        _ array: MLXArray,
        named name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        try Gemma4NativeLayerUtilities.metalBuffer(
            array,
            named: name,
            device: device
        )
    }

    private static func makeScratch(
        device: any MTLDevice,
        cacheCapacity: Int,
        usePrivateStorage: Bool,
        useSharedCacheStorage: Bool
    ) throws -> ScratchBuffers {
        let options: MTLResourceOptions = usePrivateStorage
            ? .storageModePrivate
            : .storageModeShared
        func buffer(elements: Int, name: String) throws -> any MTLBuffer {
            guard let value = device.makeBuffer(
                length: elements * bytesPerBF16,
                options: options
            ) else {
                throw ProbeError.unavailable("scratch buffer \(name)")
            }
            value.label = "Gemma4NativeSlidingLayerProbe.\(name)"
            return value
        }
        func floatBuffer(elements: Int, name: String) throws -> any MTLBuffer {
            guard let value = device.makeBuffer(
                length: elements * MemoryLayout<Float>.stride,
                options: options
            ) else {
                throw ProbeError.unavailable("scratch buffer \(name)")
            }
            value.label = "Gemma4NativeSlidingLayerProbe.\(name)"
            return value
        }
        func cacheBuffer(elements: Int, name: String) throws -> any MTLBuffer {
            guard let value = device.makeBuffer(
                length: elements * bytesPerBF16,
                options: useSharedCacheStorage ? .storageModeShared : options
            ) else {
                throw ProbeError.unavailable("scratch buffer \(name)")
            }
            value.label = "Gemma4NativeSlidingLayerProbe.\(name)"
            return value
        }

        let cacheElements = keyValueHeads * cacheCapacity * headDimension
        let twoPassBlocks = 64
        return ScratchBuffers(
            hiddenNorm: try buffer(elements: hiddenSize, name: "hiddenNorm"),
            rawQuery: try buffer(elements: queryWidth, name: "rawQuery"),
            rawKey: try buffer(elements: keyValueWidth, name: "rawKey"),
            rawValue: try buffer(elements: keyValueWidth, name: "rawValue"),
            normalizedValue: try buffer(elements: keyValueWidth, name: "normalizedValue"),
            ropedQuery: try buffer(elements: queryWidth, name: "ropedQuery"),
            ropedKey: try buffer(elements: keyValueWidth, name: "ropedKey"),
            attention: try buffer(elements: queryWidth, name: "attention"),
            sdpaPartials: try buffer(
                elements: queryHeads * twoPassBlocks * headDimension,
                name: "sdpaPartials"
            ),
            sdpaSums: try floatBuffer(
                elements: queryHeads * twoPassBlocks,
                name: "sdpaSums"
            ),
            sdpaMaxs: try floatBuffer(
                elements: queryHeads * twoPassBlocks,
                name: "sdpaMaxs"
            ),
            attentionProjection: try buffer(elements: hiddenSize, name: "attentionProjection"),
            attentionResidual: try buffer(elements: hiddenSize, name: "attentionResidual"),
            preFeedForward: try buffer(elements: hiddenSize, name: "preFeedForward"),
            gate: try buffer(elements: intermediateSize, name: "gate"),
            up: try buffer(elements: intermediateSize, name: "up"),
            activated: try buffer(elements: intermediateSize, name: "activated"),
            down: try buffer(elements: hiddenSize, name: "down"),
            output: try buffer(elements: hiddenSize, name: "output"),
            keyCache: try cacheBuffer(elements: cacheElements, name: "keyCache"),
            valueCache: try cacheBuffer(elements: cacheElements, name: "valueCache")
        )
    }

    private static func makeConstants(
        device: any MTLDevice,
        epsilon: Float,
        positionOffset: Int,
        appendIndex: Int,
        activeLength: Int,
        cacheCapacity: Int
    ) throws -> ConstantBuffers {
        let headStride = UInt(cacheCapacity * headDimension)
        return ConstantBuffers(
            epsilon: try scalarBuffer(epsilon, name: "epsilon", device: device),
            hiddenAxis: try scalarBuffer(UInt32(hiddenSize), name: "hidden axis", device: device),
            weightStride: try scalarBuffer(UInt32(1), name: "weight stride", device: device),
            width4K: try scalarBuffer(Int32(keyValueWidth), name: "width 4096", device: device),
            width5376: try scalarBuffer(Int32(hiddenSize), name: "width 5376", device: device),
            width8K: try scalarBuffer(Int32(queryWidth), name: "width 8192", device: device),
            width21504: try scalarBuffer(Int32(intermediateSize), name: "width 21504", device: device),
            ropeOffset: try scalarBuffer(Int32(positionOffset), name: "RoPE offset", device: device),
            gqaFactor: try scalarBuffer(Int32(2), name: "GQA factor", device: device),
            activeCacheLength: try scalarBuffer(
                Int32(activeLength), name: "active cache length", device: device),
            keyHeadStride: try scalarBuffer(headStride, name: "key head stride", device: device),
            keySequenceStride: try scalarBuffer(
                UInt(headDimension), name: "key sequence stride", device: device),
            valueHeadStride: try scalarBuffer(headStride, name: "value head stride", device: device),
            valueSequenceStride: try scalarBuffer(
                UInt(headDimension), name: "value sequence stride", device: device),
            attentionScale: try scalarBuffer(Float(1), name: "attention scale", device: device),
            sdpaBlocks: try scalarBuffer(Int32(64), name: "SDPA blocks", device: device),
            appendIndex: try scalarBuffer(
                UInt32(appendIndex), name: "append index", device: device),
            cacheCapacity: try scalarBuffer(
                UInt32(cacheCapacity), name: "cache capacity", device: device)
        )
    }

    private static func scalarBuffer<T>(
        _ value: T,
        name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        try Gemma4NativeLayerUtilities.scalarBuffer(
            value,
            name: name,
            device: device
        )
    }

    private static func makePipelines(
        device: any MTLDevice,
        trustedLibrary: any MTLLibrary,
        helperLibrary: any MTLLibrary
    ) throws -> Pipelines {
        func trusted(
            _ name: String,
            boolConstants: [Int: Bool] = [:],
            intConstants: [Int: Int32] = [:]
        ) throws -> any MTLComputePipelineState {
            guard let pipeline = Gemma4NativeKernelLibrary.pipeline(
                library: trustedLibrary,
                device: device,
                name: name,
                boolConstants: boolConstants,
                intConstants: intConstants
            ) else {
                throw ProbeError.unavailable("trusted pipeline \(name)")
            }
            return pipeline
        }

        return Pipelines(
            rmsLooped: try trusted("rms_loopedbfloat16", boolConstants: [20: true]),
            qmv: try trusted("affine_qmv_bfloat16_t_gs_64_b_4_batch_0"),
            qmvFast: try trusted("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0"),
            indexedSlidingQKV: try helperPipeline(
                named: "mlxfast_indexed_sliding_qkv_qmv_5376_v1",
                library: helperLibrary,
                device: device
            ),
            indexedOutput: try helperPipeline(
                named: "mlxfast_indexed_output_qmv_fast_8192_v1",
                library: helperLibrary,
                device: device
            ),
            packedIndexedOutput: try helperPipeline(
                named: "mlxfast_packed12_indexed_output_qmv_fast_8192_v1",
                library: helperLibrary,
                device: device
            ),
            indexedGateUpActivation: try helperPipeline(
                named: "mlxfast_indexed_fused_gate_up_activation_qmv_5376_v1",
                library: helperLibrary,
                device: device
            ),
            indexedDown: try helperPipeline(
                named: "mlxfast_indexed_down_qmv_21504_v1",
                library: helperLibrary,
                device: device
            ),
            packedIndexedDown: try helperPipeline(
                named: "mlxfast_packed12_indexed_down_qmv_21504_v1",
                library: helperLibrary,
                device: device
            ),
            fusedAttentionPreparation: try helperPipeline(
                named: "mlxfast_fused_sliding_attention_rms_rope_table_256_v4",
                library: helperLibrary,
                device: device
            ),
            fusedAttentionPreparationAndAppend: try helperPipeline(
                named: "mlxfast_native_fused_sliding_attention_prep_append_256_v1",
                library: helperLibrary,
                device: device
            ),
            sdpa: try trusted(
                "sdpa_vector_bfloat16_t_256_256",
                boolConstants: [
                    20: false, 21: false, 22: false,
                    23: false, 24: false, 25: false,
                ],
                intConstants: [26: 0]
            ),
            sdpaFirstPass: try trusted(
                "sdpa_vector_2pass_1_bfloat16_t_256_256",
                boolConstants: [
                    20: false, 21: false, 22: false,
                    23: false, 24: false, 25: false,
                ],
                intConstants: [26: 64]
            ),
            sdpaSecondPass: try trusted(
                "sdpa_vector_2pass_2_bfloat16_t_256"
            ),
            cacheAppend: try helperPipeline(
                named: "mlxfast_sliding_cache_append",
                library: helperLibrary,
                device: device
            ),
            activatedProduct: try helperPipeline(
                named: "mlxfast_bf16_activated_product",
                library: helperLibrary,
                device: device
            ),
            attentionToMLPBoundary: try helperPipeline(
                named: "mlxfast_fused_attention_to_mlp_boundary_5376_v1",
                library: helperLibrary,
                device: device
            ),
            mlpToNextBoundary: try helperPipeline(
                named: "mlxfast_fused_mlp_to_next_boundary_5376_v1",
                library: helperLibrary,
                device: device
            )
        )
    }

    private static func makeHelperLibrary(device: any MTLDevice) throws -> any MTLLibrary {
        try Gemma4NativeLayerUtilities.makeHelperLibrary(device: device)
    }

    private static func helperPipeline(
        named name: String,
        library: any MTLLibrary,
        device: any MTLDevice
    ) throws -> any MTLComputePipelineState {
        try Gemma4NativeLayerUtilities.helperPipeline(
            named: name,
            library: library,
            device: device
        )
    }

}

private struct Gemma4NativeProjectionBuffers {
    let weight: any MTLBuffer
    let scales: any MTLBuffer
    let biases: any MTLBuffer
    let inputWidth: Int
    let outputWidth: Int
}

// Native fixed-12 port authored by GPT 5.6 Sol through Gaj's OpenCode Harness.
private struct Gemma4NativeIndexedProjectionBuffers {
    let weight: any MTLBuffer
    let indices: any MTLBuffer
    let packedIndices: (any MTLBuffer)?
    let lut: any MTLBuffer
    // Strong owners for the no-copy metadata wrappers.
    let indicesArray: MLXArray
    let packedIndicesArray: MLXArray?
    let lutArray: MLXArray
}

private enum Gemma4NativeLayerUtilities {
    typealias ProbeError = Gemma4NativeSlidingLayerProbe.ProbeError

    static func encodeFusedAttentionPreparation(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        rawQuery: any MTLBuffer,
        rawKey: any MTLBuffer,
        rawValue: any MTLBuffer,
        queryWeight: any MTLBuffer,
        keyWeight: any MTLBuffer,
        position: any MTLBuffer,
        ropeCosines: any MTLBuffer,
        ropeSines: any MTLBuffer,
        queries: any MTLBuffer,
        keys: any MTLBuffer,
        values: any MTLBuffer,
        appendIndex: (any MTLBuffer)? = nil,
        cacheCapacity: (any MTLBuffer)? = nil,
        threads: Int,
        rows: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(rawQuery, offset: 0, index: 0)
        encoder.setBuffer(rawKey, offset: 0, index: 1)
        encoder.setBuffer(rawValue, offset: 0, index: 2)
        encoder.setBuffer(queryWeight, offset: 0, index: 3)
        encoder.setBuffer(keyWeight, offset: 0, index: 4)
        encoder.setBuffer(position, offset: 0, index: 5)
        encoder.setBuffer(ropeCosines, offset: 0, index: 6)
        encoder.setBuffer(ropeSines, offset: 0, index: 7)
        encoder.setBuffer(queries, offset: 0, index: 8)
        encoder.setBuffer(keys, offset: 0, index: 9)
        encoder.setBuffer(values, offset: 0, index: 10)
        if let appendIndex, let cacheCapacity {
            encoder.setBuffer(appendIndex, offset: 0, index: 11)
            encoder.setBuffer(cacheCapacity, offset: 0, index: 12)
        }
        encoder.dispatchThreads(
            MTLSize(width: threads, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    static func encodeRMS(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        input: any MTLBuffer,
        weight: any MTLBuffer,
        output: any MTLBuffer,
        epsilon: any MTLBuffer,
        axis: any MTLBuffer,
        weightStride: any MTLBuffer,
        rows: Int,
        threads: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(epsilon, offset: 0, index: 3)
        encoder.setBuffer(axis, offset: 0, index: 4)
        encoder.setBuffer(weightStride, offset: 0, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: rows * threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    static func encodeQMV(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        weight: any MTLBuffer,
        scales: any MTLBuffer,
        biases: any MTLBuffer,
        input: any MTLBuffer,
        output: any MTLBuffer,
        inputWidth: any MTLBuffer,
        outputWidth: any MTLBuffer,
        outputElements: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: 0, index: 1)
        encoder.setBuffer(biases, offset: 0, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBuffer(inputWidth, offset: 0, index: 5)
        encoder.setBuffer(outputWidth, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: (outputElements + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
    }

    static func dispatchLinear(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        count: Int
    ) {
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    static func makeMLXAlias(buffer: any MTLBuffer, shape: [Int]) -> MLXArray {
        // Avoid a strong buffer capture in the pinned MLX finalizer-state leak.
        let owner = Unmanaged.passRetained(buffer as AnyObject)
        return MLXArray(
            rawPointer: buffer.contents(),
            shape,
            dtype: .bfloat16
        ) {
            owner.release()
        }
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

    static func arrayBuffer<T>(
        _ values: [T],
        name: String,
        device: any MTLDevice
    ) throws -> any MTLBuffer {
        guard !values.isEmpty else {
            throw ProbeError.invalidInput("constant array \(name) must not be empty")
        }
        let buffer: (any MTLBuffer)? = values.withUnsafeBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: baseAddress,
                length: pointer.count * MemoryLayout<T>.stride,
                options: .storageModeShared
            )
        }
        guard let buffer else {
            throw ProbeError.unavailable("constant array buffer \(name)")
        }
        return buffer
    }

    static func writeScalar<T>(
        _ value: T,
        to buffer: any MTLBuffer,
        name: String
    ) throws {
        guard buffer.storageMode == .shared,
              buffer.length >= MemoryLayout<T>.stride
        else {
            throw ProbeError.invalidInput(
                "dynamic scalar buffer \(name) is not writable shared storage")
        }
        var value = value
        withUnsafeBytes(of: &value) { bytes in
            buffer.contents().copyMemory(
                from: bytes.baseAddress!,
                byteCount: bytes.count
            )
        }
    }

    static func writeArray<T>(
        _ values: [T],
        to buffer: any MTLBuffer,
        name: String
    ) throws {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard !values.isEmpty,
              buffer.storageMode == .shared,
              buffer.length >= byteCount
        else {
            throw ProbeError.invalidInput(
                "dynamic array buffer \(name) is not writable shared storage")
        }
        values.withUnsafeBufferPointer { pointer in
            buffer.contents().copyMemory(
                from: pointer.baseAddress!,
                byteCount: byteCount
            )
        }
    }

    static func scratchBuffer(
        elements: Int,
        name: String,
        device: any MTLDevice,
        storageMode: MTLStorageMode = .shared
    ) throws -> any MTLBuffer {
        let options: MTLResourceOptions = storageMode == .private
            ? .storageModePrivate
            : .storageModeShared
        guard elements > 0,
              let buffer = device.makeBuffer(
                  length: elements * 2,
                  options: options
              )
        else {
            throw ProbeError.unavailable("scratch buffer \(name)")
        }
        buffer.label = "Gemma4NativeLayerProbe.\(name)"
        return buffer
    }

    static func makeProjection(
        _ projection: Gemma4LinearWeight,
        inputWidth: Int,
        outputWidth: Int,
        name: String,
        device: any MTLDevice
    ) throws -> Gemma4NativeProjectionBuffers {
        guard projection.logicalShape == [outputWidth, inputWidth],
              projection.groupSize == 64,
              projection.bits == 4,
              projection.weight.dtype == .uint32,
              projection.weight.shape == [outputWidth, inputWidth / 8],
              let scales = projection.scales,
              scales.dtype == .bfloat16,
              scales.shape == [outputWidth, inputWidth / 64],
              let biases = projection.biases,
              biases.dtype == .bfloat16,
              biases.shape == scales.shape
        else {
            throw ProbeError.invalidInput(
                "\(name) must be affine U32/BF16 4-bit group-64 [\(outputWidth),\(inputWidth)]")
        }
        return Gemma4NativeProjectionBuffers(
            weight: try metalBuffer(projection.weight, named: "\(name).weight", device: device),
            scales: try metalBuffer(scales, named: "\(name).scales", device: device),
            biases: try metalBuffer(biases, named: "\(name).biases", device: device),
            inputWidth: inputWidth,
            outputWidth: outputWidth
        )
    }

    static func makeIndexedProjection(
        projection: Gemma4LinearWeight,
        raw: Gemma4NativeProjectionBuffers,
        name: String,
        device: any MTLDevice,
        allowPacked12: Bool = false,
        metadata suppliedMetadata: IndexedAffineMetadata? = nil
    ) throws -> Gemma4NativeIndexedProjectionBuffers {
        guard let scales = projection.scales,
              let biases = projection.biases,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              scales.shape == biases.shape
        else {
            throw ProbeError.invalidInput(
                "indexed projection \(name) requires equal BF16 scales and biases")
        }
        let metadata = suppliedMetadata ?? makeIndexedAffineMetadata(
            scales: scales,
            biases: biases
        )
        eval(metadata.indices, metadata.lut)
        guard metadata.indices.dtype == .uint16,
              metadata.indices.shape == scales.shape,
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...65_536).contains(metadata.lut.size)
        else {
            throw ProbeError.invalidInput(
                "derived indexed metadata for \(name) has invalid shape or dtype")
        }
        let packedIndicesArray = allowPacked12 && nativePackedIndicesEnabled()
            ? makePacked12Indices(metadata: metadata, raw: raw)
            : nil
        if let packedIndicesArray {
            eval(packedIndicesArray)
        }
        return Gemma4NativeIndexedProjectionBuffers(
            weight: raw.weight,
            indices: try metalBuffer(
                metadata.indices,
                named: "\(name).metadata_indices",
                device: device
            ),
            packedIndices: try packedIndicesArray.map {
                try metalBuffer(
                    $0,
                    named: "\(name).packed12_metadata_indices",
                    device: device
                )
            },
            lut: try metalBuffer(
                metadata.lut,
                named: "\(name).metadata_lut",
                device: device
            ),
            indicesArray: metadata.indices,
            packedIndicesArray: packedIndicesArray,
            lutArray: metadata.lut
        )
    }

    private static func nativePackedIndicesEnabled() -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_NATIVE_PACKED_INDICES"
        ] else {
            return true
        }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }

    static func nativeFusedAttentionCacheAppendEnabled() -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_NATIVE_FUSED_ATTENTION_CACHE_APPEND"
        ] else {
            return false
        }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }

    private static func makePacked12Indices(
        metadata: IndexedAffineMetadata,
        raw: Gemma4NativeProjectionBuffers
    ) -> MLXArray? {
        guard metadata.indices.dtype == .uint16,
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size),
              raw.outputWidth == 5_376
        else {
            return nil
        }

        if raw.inputWidth == 21_504 {
            guard metadata.indices.shape == [5_376, 336],
                  let packed = gemma4Pack12BitIndices(
                      metadata.indices.asArray(UInt16.self),
                      rows: 5_376,
                      groupsPerRow: 336
                  )
            else {
                return nil
            }
            return MLXArray(packed, [5_376, 126])
        }

        let groupsPerRow: Int
        switch raw.inputWidth {
        case 8_192:
            groupsPerRow = 128
        case 16_384:
            groupsPerRow = 256
        default:
            return nil
        }
        guard metadata.indices.shape == [5_376, groupsPerRow] else {
            return nil
        }
        let (rowBits, rowBitsOverflow) = groupsPerRow.multipliedReportingOverflow(
            by: 12)
        guard !rowBitsOverflow, rowBits.isMultiple(of: 32) else { return nil }
        let wordsPerRow = rowBits / 32
        let (wordCount, wordCountOverflow) = 5_376.multipliedReportingOverflow(
            by: wordsPerRow)
        guard !wordCountOverflow else { return nil }
        let indices = metadata.indices.asArray(UInt16.self)
        guard indices.allSatisfy({ Int($0) < metadata.lut.size }),
              let packed = gemma4Pack12BitIndices(
                  indices,
                  rows: 5_376,
                  groupsPerRow: groupsPerRow
              ),
              packed.count == wordCount
        else {
            return nil
        }
        return MLXArray(packed, [5_376, wordsPerRow])
    }

    static func importCache(
        keys: MLXArray,
        values: MLXArray,
        heads: Int,
        headDimension: Int,
        sourceStart: Int = 0,
        copyLength: Int,
        capacity: Int,
        keyDestination: any MTLBuffer,
        valueDestination: any MTLBuffer,
        device: any MTLDevice,
        queue: any MTLCommandQueue
    ) throws {
        guard keyDestination.storageMode == valueDestination.storageMode else {
            throw ProbeError.invalidInput("K/V cache storage modes differ")
        }
        let sourceLength = keys.dim(2)
        guard sourceStart >= 0,
              copyLength >= 0,
              sourceStart + copyLength <= sourceLength,
              copyLength <= capacity
        else {
            throw ProbeError.invalidInput("K/V cache import range is invalid")
        }
        if copyLength == 0 {
            if keyDestination.storageMode != .private {
                keyDestination.contents().initializeMemory(
                    as: UInt8.self,
                    repeating: 0,
                    count: keyDestination.length
                )
                valueDestination.contents().initializeMemory(
                    as: UInt8.self,
                    repeating: 0,
                    count: valueDestination.length
                )
            }
            return
        }
        guard let keySource = keys.asMTLBuffer(device: device, noCopy: true),
              let valueSource = values.asMTLBuffer(device: device, noCopy: true)
        else {
            throw ProbeError.unavailable("no-copy prior cache aliases")
        }
        let sourceHeadBytes = sourceLength * headDimension * 2
        let sourceStartBytes = sourceStart * headDimension * 2
        let copyBytes = copyLength * headDimension * 2
        let destinationHeadBytes = capacity * headDimension * 2
        if keyDestination.storageMode == .private {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder()
            else {
                throw ProbeError.unavailable("private cache import command buffer")
            }
            commandBuffer.label = "Gemma4NativeLayerProbe.privateCacheImport"
            blit.label = "Gemma4NativeLayerProbe.privateCacheImport.blit"
            for head in 0..<heads {
                blit.copy(
                    from: keySource,
                    sourceOffset: head * sourceHeadBytes + sourceStartBytes,
                    to: keyDestination,
                    destinationOffset: head * destinationHeadBytes,
                    size: copyBytes
                )
                blit.copy(
                    from: valueSource,
                    sourceOffset: head * sourceHeadBytes + sourceStartBytes,
                    to: valueDestination,
                    destinationOffset: head * destinationHeadBytes,
                    size: copyBytes
                )
            }
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            guard commandBuffer.status == .completed else {
                throw ProbeError.commandFailed(
                    commandBuffer.error?.localizedDescription
                        ?? "private cache import status \(commandBuffer.status.rawValue)")
            }
        } else {
            keyDestination.contents().initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: keyDestination.length
            )
            valueDestination.contents().initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: valueDestination.length
            )
            for head in 0..<heads {
                keyDestination.contents().advanced(by: head * destinationHeadBytes).copyMemory(
                    from: keySource.contents().advanced(
                        by: head * sourceHeadBytes + sourceStartBytes),
                    byteCount: copyBytes
                )
                valueDestination.contents().advanced(by: head * destinationHeadBytes).copyMemory(
                    from: valueSource.contents().advanced(
                        by: head * sourceHeadBytes + sourceStartBytes),
                    byteCount: copyBytes
                )
            }
        }
    }

    static func trustedPipeline(
        named name: String,
        library: any MTLLibrary,
        device: any MTLDevice,
        boolConstants: [Int: Bool] = [:],
        intConstants: [Int: Int32] = [:]
    ) throws -> any MTLComputePipelineState {
        guard let pipeline = Gemma4NativeKernelLibrary.pipeline(
            library: library,
            device: device,
            name: name,
            boolConstants: boolConstants,
            intConstants: intConstants
        ) else {
            throw ProbeError.unavailable("trusted pipeline \(name)")
        }
        return pipeline
    }

    static func makeHelperLibrary(device: any MTLDevice) throws -> any MTLLibrary {
        let source = """
            #include <metal_stdlib>
            #include <metal_math>
            #include <metal_simdgroup>
            using namespace metal;

            kernel void mlxfast_fused_sliding_attention_rms_rope_table_256_v4(
                const device bfloat* raw_q [[buffer(0)]],
                const device bfloat* raw_k [[buffer(1)]],
                const device bfloat* raw_v [[buffer(2)]],
                const device bfloat* q_weight [[buffer(3)]],
                const device bfloat* k_weight [[buffer(4)]],
                constant int& position [[buffer(5)]],
                const device float* rope_cosines [[buffer(6)]],
                const device float* rope_sines [[buffer(7)]],
                device bfloat* queries [[buffer(8)]],
                device bfloat* keys [[buffer(9)]],
                device bfloat* values [[buffer(10)]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
            \(gemma4FusedAttentionRMSKernelBody(
                headDim: 256,
                kvHeads: 16,
                sharesFullKVReduction: false
            ))
            }

            kernel void mlxfast_fused_full_attention_rms_rope_table_shared_kv_512_v5(
                const device bfloat* raw_q [[buffer(0)]],
                const device bfloat* raw_k [[buffer(1)]],
                const device bfloat* raw_v [[buffer(2)]],
                const device bfloat* q_weight [[buffer(3)]],
                const device bfloat* k_weight [[buffer(4)]],
                constant int& position [[buffer(5)]],
                const device float* rope_cosines [[buffer(6)]],
                const device float* rope_sines [[buffer(7)]],
                device bfloat* queries [[buffer(8)]],
                device bfloat* keys [[buffer(9)]],
                device bfloat* values [[buffer(10)]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
            \(gemma4FusedAttentionRMSKernelBody(
                headDim: 512,
                kvHeads: 4,
                sharesFullKVReduction: true
            ))
            }

            kernel void mlxfast_native_fused_sliding_attention_prep_append_256_v1(
                const device bfloat* raw_q [[buffer(0)]],
                const device bfloat* raw_k [[buffer(1)]],
                const device bfloat* raw_v [[buffer(2)]],
                const device bfloat* q_weight [[buffer(3)]],
                const device bfloat* k_weight [[buffer(4)]],
                constant int& position [[buffer(5)]],
                const device float* rope_cosines [[buffer(6)]],
                const device float* rope_sines [[buffer(7)]],
                device bfloat* queries [[buffer(8)]],
                device bfloat* keys [[buffer(9)]],
                device bfloat* values [[buffer(10)]],
                constant uint& append_index [[buffer(11)]],
                constant uint& capacity [[buffer(12)]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
            \(gemma4FusedAttentionRMSKernelBody(
                headDim: 256,
                kvHeads: 16,
                sharesFullKVReduction: false,
                writesDirectlyToCache: true
            ))
            }

            kernel void mlxfast_native_fused_full_attention_prep_append_512_v1(
                const device bfloat* raw_q [[buffer(0)]],
                const device bfloat* raw_k [[buffer(1)]],
                const device bfloat* raw_v [[buffer(2)]],
                const device bfloat* q_weight [[buffer(3)]],
                const device bfloat* k_weight [[buffer(4)]],
                constant int& position [[buffer(5)]],
                const device float* rope_cosines [[buffer(6)]],
                const device float* rope_sines [[buffer(7)]],
                device bfloat* queries [[buffer(8)]],
                device bfloat* keys [[buffer(9)]],
                device bfloat* values [[buffer(10)]],
                constant uint& append_index [[buffer(11)]],
                constant uint& capacity [[buffer(12)]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
            \(gemma4FusedAttentionRMSKernelBody(
                headDim: 512,
                kvHeads: 4,
                sharesFullKVReduction: true,
                writesDirectlyToCache: true,
                scalesFullQueries: true
            ))
            }

            inline float gemma4_qkv_pair_scale(uint pair) {
                return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_qkv_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_qkv_load_values(
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

            inline float gemma4_qkv_qdot_4bit(
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

            inline float gemma4_full_qk_pair_scale(uint pair) {
                return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_full_qk_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_full_qk_load_values(
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

            inline float gemma4_full_qk_qdot_4bit(
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

            inline float gemma4_pair_scale(uint pair) {
                return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_load_qmv_values(
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

            inline float gemma4_qdot_4bit(
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

            inline float gemma4_down_pair_scale(uint pair) {
                return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_down_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_down_load_values(
                const device bfloat* input,
                thread float* values
            ) {
                float sum = 0;
                for (int index = 0; index < 16; index += 4) {
                    sum += input[index] + input[index + 1]
                        + input[index + 2] + input[index + 3];
                    values[index] = input[index];
                    values[index + 1] = input[index + 1] / 16.0f;
                    values[index + 2] = input[index + 2] / 256.0f;
                    values[index + 3] = input[index + 3] / 4096.0f;
                }
                return sum;
            }

            inline float gemma4_down_qdot_4bit(
                const device uchar* weight,
                const thread float* values,
                float scale,
                float bias,
                float input_sum
            ) {
                const device ushort* packed =
                    reinterpret_cast<const device ushort*>(weight);
                float accumulator = 0;
                for (int index = 0; index < 4; ++index) {
                    accumulator +=
                        (values[4 * index] * (packed[index] & 0x000f)
                        + values[4 * index + 1] * (packed[index] & 0x00f0)
                        + values[4 * index + 2] * (packed[index] & 0x0f00)
                        + values[4 * index + 3] * (packed[index] & 0xf000));
                }
                return scale * accumulator + input_sum * bias;
            }

            inline float gemma4_output_pair_scale(uint pair) {
                return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_output_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_output_load_values(
                const device bfloat* input,
                thread float* values
            ) {
                float sum = 0;
                for (int index = 0; index < 16; index += 4) {
                    sum += input[index] + input[index + 1]
                        + input[index + 2] + input[index + 3];
                    values[index] = input[index];
                    values[index + 1] = input[index + 1] / 16.0f;
                    values[index + 2] = input[index + 2] / 256.0f;
                    values[index + 3] = input[index + 3] / 4096.0f;
                }
                return sum;
            }

            inline float gemma4_output_qdot_4bit(
                const device uchar* weight,
                const thread float* values,
                float scale,
                float bias,
                float input_sum
            ) {
                const device ushort* packed =
                    reinterpret_cast<const device ushort*>(weight);
                float accumulator = 0;
                for (int index = 0; index < 4; ++index) {
                    accumulator +=
                        (values[4 * index] * (packed[index] & 0x000f)
                        + values[4 * index + 1] * (packed[index] & 0x00f0)
                        + values[4 * index + 2] * (packed[index] & 0x0f00)
                        + values[4 * index + 3] * (packed[index] & 0xf000));
                }
                return scale * accumulator + input_sum * bias;
            }

            kernel void mlxfast_indexed_sliding_qkv_qmv_5376_v1(
                const device uint* q_weight [[buffer(0)]],
                const device ushort* q_indices [[buffer(1)]],
                const device uint* q_lut [[buffer(2)]],
                const device uint* k_weight [[buffer(3)]],
                const device ushort* k_indices [[buffer(4)]],
                const device uint* k_lut [[buffer(5)]],
                const device uint* v_weight [[buffer(6)]],
                const device ushort* v_indices [[buffer(7)]],
                const device uint* v_lut [[buffer(8)]],
                const device bfloat* x [[buffer(9)]],
                device bfloat* q_output [[buffer(10)]],
                device bfloat* k_output [[buffer(11)]],
                device bfloat* v_output [[buffer(12)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 5376;
                constexpr int kGroupsPerRow = 84;
                constexpr int kWeightBytesPerRow = 2688;
                constexpr int kRowsPerSIMD = 4;

                const int projection = simdgroup_index_in_threadgroup;
                const bool is_q = projection < 2;
                const bool is_k = projection == 2;
                const int output_row = is_q
                    ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
                    : threadgroup_position_in_grid.y * kRowsPerSIMD;

                const device uint* weight = is_q
                    ? q_weight
                    : (is_k ? k_weight : v_weight);
                const device ushort* indices = is_q
                    ? q_indices
                    : (is_k ? k_indices : v_indices);
                const device uint* lut = is_q
                    ? q_lut
                    : (is_k ? k_lut : v_lut);
                device bfloat* output = is_q
                    ? q_output
                    : (is_k ? k_output : v_output);

                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 4;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow
                    + thread_index_in_simdgroup / 8;
                const device bfloat* input = x + thread_index_in_simdgroup * 8;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < 21; ++block) {
                    float values[8];
                    const float input_sum = gemma4_qkv_load_values(input, values);

                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index = row_indices[row * kGroupsPerRow];
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_qkv_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_qkv_pair_scale(pair),
                            gemma4_qkv_pair_bias(pair),
                            input_sum);
                    }

                    weight_bytes += 128;
                    row_indices += 4;
                    input += 256;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_indexed_full_qk_qmv_5376_v1(
                const device uint* q_weight [[buffer(0)]],
                const device ushort* q_indices [[buffer(1)]],
                const device uint* q_lut [[buffer(2)]],
                const device uint* k_weight [[buffer(3)]],
                const device ushort* k_indices [[buffer(4)]],
                const device uint* k_lut [[buffer(5)]],
                const device bfloat* x [[buffer(6)]],
                device bfloat* q_output [[buffer(7)]],
                device bfloat* k_output [[buffer(8)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kGroupsPerRow = 84;
                constexpr int kWeightBytesPerRow = 2688;
                constexpr int kRowsPerSIMD = 4;

                const int projection = simdgroup_index_in_threadgroup;
                const bool is_q = projection < 8;
                const int output_row = is_q
                    ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
                    : threadgroup_position_in_grid.y * kRowsPerSIMD;

                const device uint* weight = is_q ? q_weight : k_weight;
                const device ushort* indices = is_q ? q_indices : k_indices;
                const device uint* lut = is_q ? q_lut : k_lut;
                device bfloat* output = is_q ? q_output : k_output;

                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 4;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow
                    + thread_index_in_simdgroup / 8;
                const device bfloat* input = x + thread_index_in_simdgroup * 8;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < 21; ++block) {
                    float values[8];
                    const float input_sum = gemma4_full_qk_load_values(input, values);

                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index = row_indices[row * kGroupsPerRow];
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_full_qk_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_full_qk_pair_scale(pair),
                            gemma4_full_qk_pair_bias(pair),
                            input_sum);
                    }

                    weight_bytes += 128;
                    row_indices += 4;
                    input += 256;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_indexed_fused_gate_up_activation_qmv_5376_v1(
                const device uint* gate_weight [[buffer(0)]],
                const device ushort* gate_indices [[buffer(1)]],
                const device uint* gate_lut [[buffer(2)]],
                const device uint* up_weight [[buffer(3)]],
                const device ushort* up_indices [[buffer(4)]],
                const device uint* up_lut [[buffer(5)]],
                const device bfloat* x [[buffer(6)]],
                device bfloat* activated [[buffer(7)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 5376;
                constexpr int kGroupsPerRow = 84;
                constexpr int kWeightBytesPerRow = 2688;
                constexpr int kRowsPerSIMD = 4;

                const bool is_up = simdgroup_index_in_threadgroup == 1;
                const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
                const device uint* weight = is_up ? up_weight : gate_weight;
                const device ushort* indices = is_up ? up_indices : gate_indices;

                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 4;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow
                    + thread_index_in_simdgroup / 8;
                const device bfloat* input = x + thread_index_in_simdgroup * 8;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < 21; ++block) {
                    float values[8];
                    const float input_sum = gemma4_load_qmv_values(input, values);

                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index = row_indices[row * kGroupsPerRow];
                        const uint pair = is_up
                            ? up_lut[metadata_index]
                            : gate_lut[metadata_index];
                        result[row] += gemma4_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_pair_scale(pair),
                            gemma4_pair_bias(pair),
                            input_sum);
                    }

                    weight_bytes += 128;
                    row_indices += 4;
                    input += 256;
                }

                threadgroup bfloat projections[2][kRowsPerSIMD];
                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        projections[is_up ? 1 : 0][row] =
                            static_cast<bfloat>(result[row]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simdgroup_index_in_threadgroup == 0
                    && thread_index_in_simdgroup < kRowsPerSIMD
                ) {
                    const int row = thread_index_in_simdgroup;
                    const bfloat gate = projections[0][row];
                    const bfloat up = projections[1][row];

                    const bfloat cubic0 = static_cast<bfloat>(0.044715f) * gate;
                    const bfloat cubic1 = cubic0 * gate;
                    const bfloat cubic2 = cubic1 * gate;
                    const bfloat inner0 = gate + cubic2;
                    const bfloat inner1 =
                        static_cast<bfloat>(0.7978845834732056f) * inner0;
                    const bfloat tanh_value =
                        static_cast<bfloat>(metal::precise::tanh(inner1));
                    const bfloat shifted = static_cast<bfloat>(1.0f) + tanh_value;
                    const bfloat scaled = static_cast<bfloat>(0.5f) * gate;
                    const bfloat gelu = scaled * shifted;
                    activated[output_row + row] = gelu * up;
                }
            }

            kernel void mlxfast_packed12_indexed_down_qmv_21504_v1(
                const device uint* weight [[buffer(0)]],
                const device uint* packed_indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* x [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 21504;
                constexpr int kOutputWidth = 5376;
                constexpr int kGroupsPerRow = 336;
                constexpr int kPackedWordsPerRow = 126;
                constexpr int kWeightBytesPerRow = 10752;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kBlockSize = 512;

                const int output_row =
                    threadgroup_position_in_grid.y * 8
                    + simdgroup_index_in_threadgroup * kRowsPerSIMD;

                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 8;

                // One qmv_fast block consumes eight group-64 metadata entries. Eight
                // 12-bit indexes occupy exactly three U32 words, so the per-lane bit
                // location is invariant across all 42 K blocks.
                const uint lane_group = thread_index_in_simdgroup / 4;
                const uint lane_bit = lane_group * 12;
                const uint lane_word = lane_bit / 32;
                const uint lane_shift = lane_bit % 32;
                const device uint* row_packed_indices =
                    packed_indices + output_row * kPackedWordsPerRow + lane_word;
                const device bfloat* input = x + thread_index_in_simdgroup * 16;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < kInputWidth; block += kBlockSize) {
                    float values[16];
                    const float input_sum = gemma4_down_load_values(input, values);

                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const device uint* row_words =
                            row_packed_indices + row * kPackedWordsPerRow;
                        uint metadata_index = row_words[0] >> lane_shift;
                        if (lane_shift > 20) {
                            metadata_index |= row_words[1] << (32 - lane_shift);
                        }
                        metadata_index &= 0x0fff;
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_down_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_down_pair_scale(pair),
                            gemma4_down_pair_bias(pair),
                            input_sum);
                    }

                    weight_bytes += 256;
                    row_packed_indices += 3;
                    input += kBlockSize;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_indexed_down_qmv_21504_v1(
                const device uint* weight [[buffer(0)]],
                const device ushort* indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* x [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 21504;
                constexpr int kOutputWidth = 5376;
                constexpr int kGroupsPerRow = 336;
                constexpr int kWeightBytesPerRow = 10752;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kBlockSize = 512;

                const int output_row =
                    threadgroup_position_in_grid.y * 8
                    + simdgroup_index_in_threadgroup * kRowsPerSIMD;

                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 8;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow
                    + thread_index_in_simdgroup / 4;
                const device bfloat* input = x + thread_index_in_simdgroup * 16;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < kInputWidth; block += kBlockSize) {
                    float values[16];
                    const float input_sum = gemma4_down_load_values(input, values);

                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index =
                            row_indices[row * kGroupsPerRow];
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_down_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_down_pair_scale(pair),
                            gemma4_down_pair_bias(pair),
                            input_sum);
                    }

                    weight_bytes += 256;
                    row_indices += 8;
                    input += kBlockSize;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_packed12_indexed_output_qmv_fast_8192_v1(
                const device uint* weight [[buffer(0)]],
                const device uint* packed_indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* x [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 8192;
                constexpr int kGroupsPerRow = 128;
                constexpr int kPackedWordsPerRow = 48;
                constexpr int kWeightBytesPerRow = 4096;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kBlockSize = 512;

                const int output_row =
                    threadgroup_position_in_grid.y * 8
                    + simdgroup_index_in_threadgroup * kRowsPerSIMD;
                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 8;

                // Each 512-element block consumes eight group-64 indexes. The block's
                // 96 metadata bits are exactly three U32 words, so no block crosses a
                // row boundary for either supported output geometry.
                const uint lane_group = thread_index_in_simdgroup / 4;
                const uint lane_bit = lane_group * 12;
                const uint lane_word = lane_bit / 32;
                const uint lane_shift = lane_bit % 32;
                const device uint* row_packed_indices =
                    packed_indices + output_row * kPackedWordsPerRow + lane_word;
                const device bfloat* input = x + thread_index_in_simdgroup * 16;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < kInputWidth; block += kBlockSize) {
                    float values[16];
                    const float input_sum = gemma4_output_load_values(input, values);
                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const device uint* row_words =
                            row_packed_indices + row * kPackedWordsPerRow;
                        uint metadata_index = row_words[0] >> lane_shift;
                        if (lane_shift > 20) {
                            metadata_index |= row_words[1] << (32 - lane_shift);
                        }
                        metadata_index &= 0x0fff;
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_output_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_output_pair_scale(pair),
                            gemma4_output_pair_bias(pair),
                            input_sum);
                    }
                    weight_bytes += 256;
                    row_packed_indices += 3;
                    input += kBlockSize;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_packed12_indexed_output_qmv_fast_16384_v1(
                const device uint* weight [[buffer(0)]],
                const device uint* packed_indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* x [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 16384;
                constexpr int kGroupsPerRow = 256;
                constexpr int kPackedWordsPerRow = 96;
                constexpr int kWeightBytesPerRow = 8192;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kBlockSize = 512;

                const int output_row =
                    threadgroup_position_in_grid.y * 8
                    + simdgroup_index_in_threadgroup * kRowsPerSIMD;
                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 8;

                // Each 512-element block consumes eight group-64 indexes. The block's
                // 96 metadata bits are exactly three U32 words, so no block crosses a
                // row boundary for either supported output geometry.
                const uint lane_group = thread_index_in_simdgroup / 4;
                const uint lane_bit = lane_group * 12;
                const uint lane_word = lane_bit / 32;
                const uint lane_shift = lane_bit % 32;
                const device uint* row_packed_indices =
                    packed_indices + output_row * kPackedWordsPerRow + lane_word;
                const device bfloat* input = x + thread_index_in_simdgroup * 16;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < kInputWidth; block += kBlockSize) {
                    float values[16];
                    const float input_sum = gemma4_output_load_values(input, values);
                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const device uint* row_words =
                            row_packed_indices + row * kPackedWordsPerRow;
                        uint metadata_index = row_words[0] >> lane_shift;
                        if (lane_shift > 20) {
                            metadata_index |= row_words[1] << (32 - lane_shift);
                        }
                        metadata_index &= 0x0fff;
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_output_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_output_pair_scale(pair),
                            gemma4_output_pair_bias(pair),
                            input_sum);
                    }
                    weight_bytes += 256;
                    row_packed_indices += 3;
                    input += kBlockSize;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_indexed_output_qmv_fast_8192_v1(
                const device uint* weight [[buffer(0)]],
                const device ushort* indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* x [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 8192;
                constexpr int kGroupsPerRow = 128;
                constexpr int kWeightBytesPerRow = 4096;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kBlockSize = 512;

                const int output_row =
                    threadgroup_position_in_grid.y * 8
                    + simdgroup_index_in_threadgroup * kRowsPerSIMD;
                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 8;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow
                    + thread_index_in_simdgroup / 4;
                const device bfloat* input = x + thread_index_in_simdgroup * 16;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < kInputWidth; block += kBlockSize) {
                    float values[16];
                    const float input_sum = gemma4_output_load_values(input, values);
                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index = row_indices[row * kGroupsPerRow];
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_output_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_output_pair_scale(pair),
                            gemma4_output_pair_bias(pair),
                            input_sum);
                    }
                    weight_bytes += 256;
                    row_indices += 8;
                    input += kBlockSize;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_indexed_output_qmv_fast_16384_v1(
                const device uint* weight [[buffer(0)]],
                const device ushort* indices [[buffer(1)]],
                const device uint* lut [[buffer(2)]],
                const device bfloat* x [[buffer(3)]],
                device bfloat* output [[buffer(4)]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                constexpr int kInputWidth = 16384;
                constexpr int kGroupsPerRow = 256;
                constexpr int kWeightBytesPerRow = 8192;
                constexpr int kRowsPerSIMD = 4;
                constexpr int kBlockSize = 512;

                const int output_row =
                    threadgroup_position_in_grid.y * 8
                    + simdgroup_index_in_threadgroup * kRowsPerSIMD;
                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + thread_index_in_simdgroup * 8;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow
                    + thread_index_in_simdgroup / 4;
                const device bfloat* input = x + thread_index_in_simdgroup * 16;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < kInputWidth; block += kBlockSize) {
                    float values[16];
                    const float input_sum = gemma4_output_load_values(input, values);
                    for (int row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index = row_indices[row * kGroupsPerRow];
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_output_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_output_pair_scale(pair),
                            gemma4_output_pair_bias(pair),
                            input_sum);
                    }
                    weight_bytes += 256;
                    row_indices += 8;
                    input += kBlockSize;
                }

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (thread_index_in_simdgroup == 0) {
                        output[output_row + row] = static_cast<bfloat>(result[row]);
                    }
                }
            }

            kernel void mlxfast_sliding_cache_append(
                const device bfloat* key [[buffer(0)]],
                const device bfloat* value [[buffer(1)]],
                device bfloat* key_cache [[buffer(2)]],
                device bfloat* value_cache [[buffer(3)]],
                constant uint& append_index [[buffer(4)]],
                constant uint& capacity [[buffer(5)]],
                uint index [[thread_position_in_grid]]) {
                constexpr uint head_dimension = 256;
                const uint head = index / head_dimension;
                const uint element = index % head_dimension;
                const uint destination =
                    (head * capacity + append_index) * head_dimension + element;
                key_cache[destination] = key[index];
                value_cache[destination] = value[index];
            }

            kernel void mlxfast_full_cache_append(
                const device bfloat* key [[buffer(0)]],
                const device bfloat* value [[buffer(1)]],
                device bfloat* key_cache [[buffer(2)]],
                device bfloat* value_cache [[buffer(3)]],
                constant uint& append_index [[buffer(4)]],
                constant uint& capacity [[buffer(5)]],
                uint index [[thread_position_in_grid]]) {
                constexpr uint head_dimension = 512;
                const uint head = index / head_dimension;
                const uint element = index % head_dimension;
                const uint destination =
                    (head * capacity + append_index) * head_dimension + element;
                key_cache[destination] = key[index];
                value_cache[destination] = value[index];
            }

            kernel void mlxfast_bf16_multiply_one(
                const device bfloat* input [[buffer(0)]],
                device bfloat* output [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
                const bfloat one = static_cast<bfloat>(1.0f);
                output[index] = one * input[index];
            }

            kernel void mlxfast_fused_attention_to_mlp_boundary_5376_v1(
                const device bfloat* attention_output [[buffer(0)]],
                const device bfloat* residual [[buffer(1)]],
                const device bfloat* post_attention_weight [[buffer(2)]],
                const device bfloat* pre_ffn_weight [[buffer(3)]],
                device bfloat* residual_output [[buffer(4)]],
                device bfloat* normalized_output [[buffer(5)]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
                constexpr uint kWidth = 5376;
                constexpr uint kReads = 4;
                constexpr uint kThreads = 1024;
                constexpr uint kSIMDSize = 32;

                const uint thread_index = thread_position_in_threadgroup.x;
                const uint simd_lane = thread_index_in_simdgroup;
                const uint simd_group = simdgroup_index_in_threadgroup;

                threadgroup float inverse_mean[1];
                threadgroup float local_sums[kSIMDSize];
                threadgroup bfloat residual_row[kWidth];

                float accumulator = 0;
                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const float value = attention_output[base + index];
                            accumulator += value * value;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            if (base + index < kWidth) {
                                const float value = attention_output[base + index];
                                accumulator += value * value;
                            }
                        }
                    }
                }
                accumulator = simd_sum(accumulator);

                if (simd_group == 0) {
                    local_sums[simd_lane] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    local_sums[simd_group] = accumulator;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    accumulator = simd_sum(local_sums[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_mean[0] = metal::precise::rsqrt(
                            accumulator / kWidth + 1.0e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            const bfloat unit_normalized = static_cast<bfloat>(
                                attention_output[dimension] * inverse_mean[0]);
                            const bfloat post_normalized =
                                post_attention_weight[dimension] * unit_normalized;
                            const bfloat combined =
                                residual[dimension] + post_normalized;
                            residual_row[dimension] = combined;
                            residual_output[dimension] = combined;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            if (dimension < kWidth) {
                                const bfloat unit_normalized = static_cast<bfloat>(
                                    attention_output[dimension] * inverse_mean[0]);
                                const bfloat post_normalized =
                                    post_attention_weight[dimension] * unit_normalized;
                                const bfloat combined =
                                    residual[dimension] + post_normalized;
                                residual_row[dimension] = combined;
                                residual_output[dimension] = combined;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                accumulator = 0;
                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const float value = residual_row[base + index];
                            accumulator += value * value;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            if (base + index < kWidth) {
                                const float value = residual_row[base + index];
                                accumulator += value * value;
                            }
                        }
                    }
                }
                accumulator = simd_sum(accumulator);

                if (simd_group == 0) {
                    local_sums[simd_lane] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    local_sums[simd_group] = accumulator;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    accumulator = simd_sum(local_sums[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_mean[0] = metal::precise::rsqrt(
                            accumulator / kWidth + 1.0e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            const bfloat unit_normalized = static_cast<bfloat>(
                                residual_row[dimension] * inverse_mean[0]);
                            normalized_output[dimension] =
                                pre_ffn_weight[dimension] * unit_normalized;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            if (dimension < kWidth) {
                                const bfloat unit_normalized = static_cast<bfloat>(
                                    residual_row[dimension] * inverse_mean[0]);
                                normalized_output[dimension] =
                                    pre_ffn_weight[dimension] * unit_normalized;
                            }
                        }
                    }
                }
            }

            kernel void mlxfast_bf16_activated_product(
                const device bfloat* gate_values [[buffer(0)]],
                const device bfloat* up_values [[buffer(1)]],
                device bfloat* output [[buffer(2)]],
                uint index [[thread_position_in_grid]]) {
                const bfloat gate = gate_values[index];
                const bfloat up = up_values[index];
                const bfloat cubic0 = static_cast<bfloat>(0.044715f) * gate;
                const bfloat cubic1 = cubic0 * gate;
                const bfloat cubic2 = cubic1 * gate;
                const bfloat inner0 = gate + cubic2;
                const bfloat inner1 =
                    static_cast<bfloat>(0.7978845834732056f) * inner0;
                const bfloat tanh_value =
                    static_cast<bfloat>(metal::precise::tanh(inner1));
                const bfloat shifted = static_cast<bfloat>(1.0f) + tanh_value;
                const bfloat scaled = static_cast<bfloat>(0.5f) * gate;
                const bfloat gelu = scaled * shifted;
                output[index] = gelu * up;
            }

            kernel void mlxfast_fused_mlp_to_next_boundary_5376_v1(
                const device bfloat* mlp_output [[buffer(0)]],
                const device bfloat* residual [[buffer(1)]],
                const device bfloat* post_ffn_weight [[buffer(2)]],
                const device bfloat* layer_scalar [[buffer(3)]],
                const device bfloat* next_norm_weight [[buffer(4)]],
                device bfloat* hidden_output [[buffer(5)]],
                device bfloat* next_normalized_output [[buffer(6)]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
                constexpr uint kWidth = 5376;
                constexpr uint kReads = 4;
                constexpr uint kThreads = 1024;
                constexpr uint kSIMDSize = 32;

                const uint thread_index = thread_position_in_threadgroup.x;
                const uint simd_lane = thread_index_in_simdgroup;
                const uint simd_group = simdgroup_index_in_threadgroup;

                threadgroup float inverse_mean[1];
                threadgroup float local_sums[kSIMDSize];
                threadgroup bfloat hidden_row[kWidth];

                float accumulator = 0;
                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const float value = mlp_output[base + index];
                            accumulator += value * value;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            if (base + index < kWidth) {
                                const float value = mlp_output[base + index];
                                accumulator += value * value;
                            }
                        }
                    }
                }
                accumulator = simd_sum(accumulator);

                if (simd_group == 0) {
                    local_sums[simd_lane] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    local_sums[simd_group] = accumulator;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    accumulator = simd_sum(local_sums[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_mean[0] = metal::precise::rsqrt(
                            accumulator / kWidth + 1.0e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            const bfloat unit_normalized = static_cast<bfloat>(
                                mlp_output[dimension] * inverse_mean[0]);
                            const bfloat post_normalized =
                                post_ffn_weight[dimension] * unit_normalized;
                            const bfloat combined =
                                residual[dimension] + post_normalized;
                            const bfloat scaled = combined * layer_scalar[0];
                            hidden_row[dimension] = scaled;
                            hidden_output[dimension] = scaled;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            if (dimension < kWidth) {
                                const bfloat unit_normalized = static_cast<bfloat>(
                                    mlp_output[dimension] * inverse_mean[0]);
                                const bfloat post_normalized =
                                    post_ffn_weight[dimension] * unit_normalized;
                                const bfloat combined =
                                    residual[dimension] + post_normalized;
                                const bfloat scaled = combined * layer_scalar[0];
                                hidden_row[dimension] = scaled;
                                hidden_output[dimension] = scaled;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                accumulator = 0;
                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const float value = hidden_row[base + index];
                            accumulator += value * value;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            if (base + index < kWidth) {
                                const float value = hidden_row[base + index];
                                accumulator += value * value;
                            }
                        }
                    }
                }
                accumulator = simd_sum(accumulator);

                if (simd_group == 0) {
                    local_sums[simd_lane] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane == 0) {
                    local_sums[simd_group] = accumulator;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    accumulator = simd_sum(local_sums[simd_lane]);
                    if (simd_lane == 0) {
                        inverse_mean[0] = metal::precise::rsqrt(
                            accumulator / kWidth + 1.0e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint row_offset = 0;
                     row_offset < kWidth;
                     row_offset += kThreads * kReads) {
                    const uint base = row_offset + thread_index * kReads;
                    if (base + kReads <= kWidth) {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            const bfloat unit_normalized = static_cast<bfloat>(
                                hidden_row[dimension] * inverse_mean[0]);
                            next_normalized_output[dimension] =
                                next_norm_weight[dimension] * unit_normalized;
                        }
                    } else {
                        for (uint index = 0; index < kReads; ++index) {
                            const uint dimension = base + index;
                            if (dimension < kWidth) {
                                const bfloat unit_normalized = static_cast<bfloat>(
                                    hidden_row[dimension] * inverse_mean[0]);
                                next_normalized_output[dimension] =
                                    next_norm_weight[dimension] * unit_normalized;
                            }
                        }
                    }
                }
            }
            """
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.fastMathEnabled = false
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw ProbeError.unavailable("helper MSL library: \(error)")
        }
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
            return try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: &reflection
            )
        } catch {
            throw ProbeError.unavailable("helper pipeline \(name): \(error)")
        }
    }

}

/// Diagnostic direct-Metal implementation of one Gemma 4 full-attention
/// decode layer. Like the sliding probe, it keeps explicit intermediates except
/// where the promoted boundary kernels themselves own the BF16 boundaries.
final class Gemma4NativeFullLayerProbe {
    typealias ProbeError = Gemma4NativeSlidingLayerProbe.ProbeError

    struct Result {
        let hidden: MLXArray
        let nextInputNormalized: MLXArray
        /// Capacity-shaped caches. Only `0..<activeCacheLength` is active.
        let keyCache: MLXArray
        let valueCache: MLXArray
        let activeCacheLength: Int
        let cacheCapacity: Int
    }

    private static let hiddenSize = 5_376
    private static let intermediateSize = 21_504
    private static let queryHeads = 32
    private static let keyValueHeads = 4
    private static let headDimension = 512
    private static let queryWidth = queryHeads * headDimension
    private static let keyValueWidth = keyValueHeads * headDimension
    private static let repeatCount = queryHeads / keyValueHeads

    private struct NormBuffers {
        let input: any MTLBuffer
        let query: any MTLBuffer
        let key: any MTLBuffer
        let postAttention: any MTLBuffer
        let preFeedForward: any MTLBuffer
        let postFeedForward: any MTLBuffer
        let layerScalar: any MTLBuffer
        let nextInput: any MTLBuffer
    }

    private struct Pipelines {
        let rmsLooped: any MTLComputePipelineState
        let qmv: any MTLComputePipelineState
        let qmvFast: any MTLComputePipelineState
        let indexedFullQK: any MTLComputePipelineState
        let indexedOutput: any MTLComputePipelineState
        let packedIndexedOutput: any MTLComputePipelineState
        let indexedGateUpActivation: any MTLComputePipelineState
        let indexedDown: any MTLComputePipelineState
        let packedIndexedDown: any MTLComputePipelineState
        let fusedAttentionPreparation: any MTLComputePipelineState
        let fusedAttentionPreparationAndAppend: any MTLComputePipelineState
        let queryKeyGEMV: any MTLComputePipelineState
        let preciseSoftmax: any MTLComputePipelineState
        let probabilityValueGEMV: any MTLComputePipelineState
        let cacheAppend: any MTLComputePipelineState
        let multiplyOne: any MTLComputePipelineState
        let activatedProduct: any MTLComputePipelineState
        let attentionToMLPBoundary: any MTLComputePipelineState
        let mlpToNextBoundary: any MTLComputePipelineState
    }

    private struct ScratchBuffers {
        let hiddenNorm: any MTLBuffer
        let rawQuery: any MTLBuffer
        let rawKey: any MTLBuffer
        let normalizedValue: any MTLBuffer
        let ropedQuery: any MTLBuffer
        let ropedKey: any MTLBuffer
        let scaledQuery: any MTLBuffer
        let scores: any MTLBuffer
        let probabilities: any MTLBuffer
        let attention: any MTLBuffer
        let attentionProjection: any MTLBuffer
        let attentionResidual: any MTLBuffer
        let preFeedForward: any MTLBuffer
        let gate: any MTLBuffer
        let up: any MTLBuffer
        let activated: any MTLBuffer
        let down: any MTLBuffer
        let output: any MTLBuffer
        var keyCache: any MTLBuffer
        var valueCache: any MTLBuffer
    }

    private struct ConstantBuffers {
        let epsilon: any MTLBuffer
        let hiddenAxis: any MTLBuffer
        let weightStride: any MTLBuffer

        let width2048: any MTLBuffer
        let width5376: any MTLBuffer
        let width16384: any MTLBuffer
        let width21504: any MTLBuffer

        let ropeOffset: any MTLBuffer

        let appendIndex: any MTLBuffer
        let cacheCapacity: any MTLBuffer

        let gemvBatchDimensions: any MTLBuffer
        let gemvBatchShape: any MTLBuffer
        let queryKeyInputSize: any MTLBuffer
        let queryKeyOutputSize: any MTLBuffer
        let queryKeyMatrixLeadingDimension: any MTLBuffer
        let queryKeyVectorBatchStrides: any MTLBuffer
        let queryKeyMatrixBatchStrides: any MTLBuffer
        let probabilityValueInputSize: any MTLBuffer
        let probabilityValueOutputSize: any MTLBuffer
        let probabilityValueMatrixLeadingDimension: any MTLBuffer
        let probabilityValueVectorBatchStrides: any MTLBuffer
        let probabilityValueMatrixBatchStrides: any MTLBuffer
        let softmaxAxisSize: any MTLBuffer
    }

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let trustedLibrary: any MTLLibrary
    private let helperLibrary: any MTLLibrary
    private let pipelines: Pipelines
    private var scratch: ScratchBuffers
    private let constants: ConstantBuffers
    private let norms: NormBuffers
    private let queryProjection: Gemma4NativeProjectionBuffers
    private let keyProjection: Gemma4NativeProjectionBuffers
    private let outputProjection: Gemma4NativeProjectionBuffers
    private let gateProjection: Gemma4NativeProjectionBuffers
    private let upProjection: Gemma4NativeProjectionBuffers
    private let downProjection: Gemma4NativeProjectionBuffers
    private let indexedQueryProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedKeyProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedOutputProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedGateProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedUpProjection: Gemma4NativeIndexedProjectionBuffers
    private let indexedDownProjection: Gemma4NativeIndexedProjectionBuffers
    private let ropeCosines: any MTLBuffer
    private let ropeSines: any MTLBuffer
    private let usePromotedProjectionKernels: Bool
    private let fuseAttentionCacheAppend: Bool
    private let usePrivateStorage: Bool
    private let retainedArrays: [MLXArray]
    private var activeCacheLength: Int
    private var cacheCapacity: Int
    private let maximumCacheCapacity: Int
    private var pendingCacheGrowth: (
        key: any MTLBuffer,
        value: any MTLBuffer
    )?
    private var currentPosition: Int
    private var encodedCurrentPosition = false

    init(
        blockWeights: Gemma4BlockWeights,
        attentionWeights: Gemma4AttentionWeights,
        mlpWeights: Gemma4MLPWeights,
        priorKeys: MLXArray,
        priorValues: MLXArray,
        positionOffset: Int,
        cacheCapacity requestedCapacity: Int = 768,
        rmsNormEps: Float = 1e-6,
        ropeBase: Float = 1_000_000,
        partialRotaryFactor: Float = 0.25,
        nextInputNormWeight: MLXArray? = nil,
        metalDevice: (any MTLDevice)? = nil,
        commandQueue: (any MTLCommandQueue)? = nil,
        usePromotedProjectionKernels: Bool = true,
        usePrivateStorage: Bool = false,
        useSharedCacheStorage: Bool = false,
        qMetadata: IndexedAffineMetadata? = nil,
        kMetadata: IndexedAffineMetadata? = nil,
        outputMetadata: IndexedAffineMetadata? = nil,
        gateMetadata: IndexedAffineMetadata? = nil,
        upMetadata: IndexedAffineMetadata? = nil,
        downMetadata: IndexedAffineMetadata? = nil
    ) throws {
        guard priorKeys.dtype == .bfloat16,
              priorKeys.ndim == 4,
              priorKeys.dim(0) == 1,
              priorKeys.dim(1) == Self.keyValueHeads,
              priorKeys.dim(3) == Self.headDimension,
              priorValues.dtype == .bfloat16,
              priorValues.shape == priorKeys.shape
        else {
            throw ProbeError.invalidInput(
                "full prior K/V must be BF16 [1,4,length,512] with equal shapes")
        }
        let priorLength = priorKeys.dim(2)
        let activeCacheLength = priorLength + 1
        let cacheCapacity = max(requestedCapacity, activeCacheLength)
        guard priorLength >= 0,
              positionOffset == priorLength,
              priorLength < cacheCapacity,
              activeCacheLength <= 4_096,
              cacheCapacity <= 4_096
        else {
            throw ProbeError.invalidInput(
                "full probe requires append-only position==priorLength and block-softmax length <=4096")
        }
        guard positionOffset <= Int(Int32.max),
              rmsNormEps == 1e-6,
              ropeBase == 1_000_000,
              partialRotaryFactor == 0.25
        else {
            throw ProbeError.invalidInput(
                "full position/RMS/RoPE parameters do not match the fixed layer contract")
        }
        guard case .none = attentionWeights.vProj else {
            throw ProbeError.invalidInput("full attention must reuse raw K as raw V")
        }

        try Self.validateNorm(blockWeights.inputLayerNorm, size: Self.hiddenSize, name: "input norm")
        try Self.validateNorm(blockWeights.postAttentionLayerNorm, size: Self.hiddenSize, name: "post-attention norm")
        try Self.validateNorm(blockWeights.preFeedForwardLayerNorm, size: Self.hiddenSize, name: "pre-FFN norm")
        try Self.validateNorm(blockWeights.postFeedForwardLayerNorm, size: Self.hiddenSize, name: "post-FFN norm")
        try Self.validateNorm(blockWeights.layerScalar, size: 1, name: "layer scalar")
        let ownedNextInputNorm = nextInputNormWeight ?? blockWeights.inputLayerNorm
        try Self.validateNorm(ownedNextInputNorm, size: Self.hiddenSize, name: "next input norm")
        try Self.validateNorm(attentionWeights.qNorm, size: Self.headDimension, name: "query norm")
        try Self.validateNorm(attentionWeights.kNorm, size: Self.headDimension, name: "key norm")

        guard let device = metalDevice ?? MTLCreateSystemDefaultDevice(),
              device.hasUnifiedMemory,
              let queue = commandQueue ?? device.makeCommandQueue(),
              let trustedLibrary = Gemma4NativeKernelLibrary.load(device: device)
        else {
            throw ProbeError.unavailable("Metal device, command queue, or MLX metallib")
        }
        let helperLibrary = try Gemma4NativeLayerUtilities.makeHelperLibrary(device: device)
        let pipelines = try Self.makePipelines(
            device: device,
            trustedLibrary: trustedLibrary,
            helperLibrary: helperLibrary
        )

        let queryProjection = try Gemma4NativeLayerUtilities.makeProjection(
            attentionWeights.qProj,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.queryWidth,
            name: "full.q_proj",
            device: device
        )
        let keyProjection = try Gemma4NativeLayerUtilities.makeProjection(
            attentionWeights.kProj,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.keyValueWidth,
            name: "full.k_proj",
            device: device
        )
        let outputProjection = try Gemma4NativeLayerUtilities.makeProjection(
            attentionWeights.oProj,
            inputWidth: Self.queryWidth,
            outputWidth: Self.hiddenSize,
            name: "full.o_proj",
            device: device
        )
        let gateProjection = try Gemma4NativeLayerUtilities.makeProjection(
            mlpWeights.gate,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.intermediateSize,
            name: "full.gate_proj",
            device: device
        )
        let upProjection = try Gemma4NativeLayerUtilities.makeProjection(
            mlpWeights.up,
            inputWidth: Self.hiddenSize,
            outputWidth: Self.intermediateSize,
            name: "full.up_proj",
            device: device
        )
        let downProjection = try Gemma4NativeLayerUtilities.makeProjection(
            mlpWeights.down,
            inputWidth: Self.intermediateSize,
            outputWidth: Self.hiddenSize,
            name: "full.down_proj",
            device: device
        )
        let indexedQueryProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: attentionWeights.qProj,
            raw: queryProjection,
            name: "full.q_proj",
            device: device,
            metadata: qMetadata
        )
        let indexedKeyProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: attentionWeights.kProj,
            raw: keyProjection,
            name: "full.k_proj",
            device: device,
            metadata: kMetadata
        )
        let indexedOutputProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: attentionWeights.oProj,
            raw: outputProjection,
            name: "full.o_proj",
            device: device,
            allowPacked12: true,
            metadata: outputMetadata
        )
        let indexedGateProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: mlpWeights.gate,
            raw: gateProjection,
            name: "full.gate_proj",
            device: device,
            metadata: gateMetadata
        )
        let indexedUpProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: mlpWeights.up,
            raw: upProjection,
            name: "full.up_proj",
            device: device,
            metadata: upMetadata
        )
        let indexedDownProjection = try Gemma4NativeLayerUtilities.makeIndexedProjection(
            projection: mlpWeights.down,
            raw: downProjection,
            name: "full.down_proj",
            device: device,
            allowPacked12: true,
            metadata: downMetadata
        )

        let ropeTables = gemma4MaterializedAttentionRopeTables(isSliding: false)
        let ropeCosines = try Gemma4NativeLayerUtilities.metalBuffer(
            ropeTables.cosines,
            named: "full RoPE cosine table",
            device: device
        )
        let ropeSines = try Gemma4NativeLayerUtilities.metalBuffer(
            ropeTables.sines,
            named: "full RoPE sine table",
            device: device
        )
        let norms = NormBuffers(
            input: try Gemma4NativeLayerUtilities.metalBuffer(
                blockWeights.inputLayerNorm, named: "full input norm", device: device),
            query: try Gemma4NativeLayerUtilities.metalBuffer(
                attentionWeights.qNorm, named: "full query norm", device: device),
            key: try Gemma4NativeLayerUtilities.metalBuffer(
                attentionWeights.kNorm, named: "full key norm", device: device),
            postAttention: try Gemma4NativeLayerUtilities.metalBuffer(
                blockWeights.postAttentionLayerNorm,
                named: "full post-attention norm",
                device: device
            ),
            preFeedForward: try Gemma4NativeLayerUtilities.metalBuffer(
                blockWeights.preFeedForwardLayerNorm,
                named: "full pre-FFN norm",
                device: device
            ),
            postFeedForward: try Gemma4NativeLayerUtilities.metalBuffer(
                blockWeights.postFeedForwardLayerNorm,
                named: "full post-FFN norm",
                device: device
            ),
            layerScalar: try Gemma4NativeLayerUtilities.metalBuffer(
                blockWeights.layerScalar,
                named: "full layer scalar",
                device: device
            ),
            nextInput: try Gemma4NativeLayerUtilities.metalBuffer(
                ownedNextInputNorm,
                named: "full next input norm",
                device: device
            )
        )

        let scratch = try Self.makeScratch(
            device: device,
            cacheCapacity: cacheCapacity,
            usePrivateStorage: usePrivateStorage,
            useSharedCacheStorage: useSharedCacheStorage
        )
        let constants = try Self.makeConstants(
            device: device,
            epsilon: rmsNormEps,
            positionOffset: positionOffset,
            priorLength: priorLength,
            activeLength: activeCacheLength,
            cacheCapacity: cacheCapacity
        )
        try Gemma4NativeLayerUtilities.importCache(
            keys: priorKeys,
            values: priorValues,
            heads: Self.keyValueHeads,
            headDimension: Self.headDimension,
            copyLength: priorLength,
            capacity: cacheCapacity,
            keyDestination: scratch.keyCache,
            valueDestination: scratch.valueCache,
            device: device,
            queue: queue
        )

        self.device = device
        self.queue = queue
        self.trustedLibrary = trustedLibrary
        self.helperLibrary = helperLibrary
        self.pipelines = pipelines
        self.scratch = scratch
        self.constants = constants
        self.norms = norms
        self.queryProjection = queryProjection
        self.keyProjection = keyProjection
        self.outputProjection = outputProjection
        self.gateProjection = gateProjection
        self.upProjection = upProjection
        self.downProjection = downProjection
        self.indexedQueryProjection = indexedQueryProjection
        self.indexedKeyProjection = indexedKeyProjection
        self.indexedOutputProjection = indexedOutputProjection
        self.indexedGateProjection = indexedGateProjection
        self.indexedUpProjection = indexedUpProjection
        self.indexedDownProjection = indexedDownProjection
        self.ropeCosines = ropeCosines
        self.ropeSines = ropeSines
        self.usePromotedProjectionKernels = usePromotedProjectionKernels
        self.fuseAttentionCacheAppend =
            Gemma4NativeLayerUtilities.nativeFusedAttentionCacheAppendEnabled()
        self.usePrivateStorage = usePrivateStorage
        self.retainedArrays = [
            blockWeights.inputLayerNorm,
            blockWeights.postAttentionLayerNorm,
            blockWeights.preFeedForwardLayerNorm,
            blockWeights.postFeedForwardLayerNorm,
            blockWeights.layerScalar,
            ownedNextInputNorm,
            attentionWeights.qNorm,
            attentionWeights.kNorm,
            ropeTables.cosines,
            ropeTables.sines,
            attentionWeights.qProj.weight,
            attentionWeights.qProj.scales!,
            attentionWeights.qProj.biases!,
            attentionWeights.kProj.weight,
            attentionWeights.kProj.scales!,
            attentionWeights.kProj.biases!,
            attentionWeights.oProj.weight,
            attentionWeights.oProj.scales!,
            attentionWeights.oProj.biases!,
            mlpWeights.gate.weight,
            mlpWeights.gate.scales!,
            mlpWeights.gate.biases!,
            mlpWeights.up.weight,
            mlpWeights.up.scales!,
            mlpWeights.up.biases!,
            mlpWeights.down.weight,
            mlpWeights.down.scales!,
            mlpWeights.down.biases!,
            priorKeys,
            priorValues,
        ]
        self.activeCacheLength = activeCacheLength
        self.cacheCapacity = cacheCapacity
        self.maximumCacheCapacity = cacheCapacity
        self.currentPosition = positionOffset
    }

    func run(hidden: MLXArray) throws -> Result {
        guard !usePrivateStorage else {
            throw ProbeError.invalidInput(
                "full run() cannot create MLX aliases for private storage")
        }
        guard hidden.dtype == .bfloat16,
              hidden.shape == [1, 1, Self.hiddenSize],
              let hiddenBuffer = hidden.asMTLBuffer(device: device, noCopy: true),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.invalidInput(
                "full hidden must be contiguous BF16 [1,1,5376] and Metal-encodable")
        }

        commandBuffer.label = "Gemma4NativeFullLayerProbe"
        encoder.label = "Gemma4NativeFullLayerProbe.direct"

        let encoded = try encode(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            inputBuffer: hiddenBuffer
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "status \(commandBuffer.status.rawValue)")
        }

        return Result(
            hidden: Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: encoded.output,
                shape: [1, 1, Self.hiddenSize]
            ),
            nextInputNormalized: Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: encoded.nextInputNormalized,
                shape: [1, 1, Self.hiddenSize]
            ),
            keyCache: Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: encoded.keyCache,
                shape: [1, Self.keyValueHeads, encoded.cacheCapacity, Self.headDimension]
            ),
            valueCache: Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: encoded.valueCache,
                shape: [1, Self.keyValueHeads, encoded.cacheCapacity, Self.headDimension]
            ),
            activeCacheLength: encoded.activeCacheLength,
            cacheCapacity: encoded.cacheCapacity
        )
    }

    var metalRegistryID: UInt64 {
        device.registryID
    }

    var fixedCacheCapacity: Int {
        maximumCacheCapacity
    }

    func prefillCacheArrays() throws -> (MLXArray, MLXArray) {
        guard scratch.keyCache.storageMode == .shared,
              scratch.valueCache.storageMode == .shared
        else {
            throw ProbeError.invalidInput(
                "full prefill aliases require shared cache storage")
        }
        let shape = [1, Self.keyValueHeads, cacheCapacity, Self.headDimension]
        return (
            Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: scratch.keyCache,
                shape: shape
            ),
            Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: scratch.valueCache,
                shape: shape
            )
        )
    }

    /// Caller must ensure no command buffer referencing this plan is in flight.
    func resetCache(
        keys: MLXArray,
        values: MLXArray,
        position: Int
    ) throws {
        let activeLength = position + 1
        guard position >= 0,
              position < cacheCapacity,
              activeLength <= 4_096,
              keys.dtype == .bfloat16,
              keys.shape == [1, Self.keyValueHeads, position, Self.headDimension],
              values.dtype == .bfloat16,
              values.shape == keys.shape
        else {
            throw ProbeError.invalidInput(
                "full reset cache must be BF16 [1,4,position,512] below capacity")
        }
        try Gemma4NativeLayerUtilities.importCache(
            keys: keys,
            values: values,
            heads: Self.keyValueHeads,
            headDimension: Self.headDimension,
            copyLength: position,
            capacity: cacheCapacity,
            keyDestination: scratch.keyCache,
            valueDestination: scratch.valueCache,
            device: device,
            queue: queue
        )
        try applyResetState(position: position)
    }

    func encodePrivateCacheReset(
        keys: MLXArray,
        values: MLXArray,
        position: Int,
        blit: any MTLBlitCommandEncoder
    ) throws {
        let activeLength = position + 1
        guard usePrivateStorage,
              scratch.keyCache.storageMode == .private,
              scratch.valueCache.storageMode == .private,
              position >= 0,
              position < cacheCapacity,
              activeLength <= 4_096,
              keys.dtype == .bfloat16,
              keys.shape == [1, Self.keyValueHeads, position, Self.headDimension],
              values.dtype == .bfloat16,
              values.shape == keys.shape,
              let keySource = keys.asMTLBuffer(device: device, noCopy: true),
              let valueSource = values.asMTLBuffer(device: device, noCopy: true)
        else {
            throw ProbeError.invalidInput(
                "private full reset cache is not a contiguous BF16 canonical prefix")
        }
        let sourceHeadBytes = position * Self.headDimension * 2
        let destinationHeadBytes = cacheCapacity * Self.headDimension * 2
        for head in 0..<Self.keyValueHeads where sourceHeadBytes > 0 {
            blit.copy(
                from: keySource,
                sourceOffset: head * sourceHeadBytes,
                to: scratch.keyCache,
                destinationOffset: head * destinationHeadBytes,
                size: sourceHeadBytes
            )
            blit.copy(
                from: valueSource,
                sourceOffset: head * sourceHeadBytes,
                to: scratch.valueCache,
                destinationOffset: head * destinationHeadBytes,
                size: sourceHeadBytes
            )
        }
    }

    func commitPrivateCacheReset(position: Int) throws {
        try applyResetState(position: position)
    }

    func adoptCacheBuffers(
        _ buffers: Gemma4NativeAdoptedCacheBuffers,
        position: Int,
        residencyCollector: (any Gemma4NativeResidencyCollecting)? = nil
    ) throws {
        let bytesPerHeadPosition = Self.headDimension * 2
        let cacheHeadBytes = Self.keyValueHeads * bytesPerHeadPosition
        let adoptedCapacity = buffers.keys.length / cacheHeadBytes
        guard buffers.keys.device.registryID == device.registryID,
              buffers.values.device.registryID == device.registryID,
              buffers.keys.length == adoptedCapacity * cacheHeadBytes,
              buffers.values.length == buffers.keys.length,
              adoptedCapacity > position,
              adoptedCapacity <= maximumCacheCapacity,
              position >= 0,
              pendingCacheGrowth == nil
        else {
            throw ProbeError.invalidInput("adopted full cache buffers are invalid")
        }
        residencyCollector?.remove(scratch.keyCache)
        residencyCollector?.remove(scratch.valueCache)
        scratch.keyCache = buffers.keys
        scratch.valueCache = buffers.values
        residencyCollector?.collect(buffers.keys)
        residencyCollector?.collect(buffers.values)
        cacheCapacity = adoptedCapacity
        try updateCacheCapacityConstants(adoptedCapacity)
        try applyResetState(position: position)
    }

    func needsCacheGrowth(position: Int) -> Bool {
        position >= cacheCapacity && cacheCapacity < maximumCacheCapacity
    }

    func encodeCacheGrowth(
        position: Int,
        blit: any MTLBlitCommandEncoder
    ) throws {
        guard needsCacheGrowth(position: position),
              position == cacheCapacity,
              usePrivateStorage,
              pendingCacheGrowth == nil
        else {
            throw ProbeError.invalidInput("invalid full cache growth request")
        }
        let bytesPerHead = cacheCapacity * Self.headDimension * 2
        let grownHeadBytes = maximumCacheCapacity * Self.headDimension * 2
        let grownLength = Self.keyValueHeads * grownHeadBytes
        guard let key = device.makeBuffer(length: grownLength, options: .storageModePrivate),
              let value = device.makeBuffer(length: grownLength, options: .storageModePrivate)
        else {
            throw ProbeError.unavailable("grown full cache buffers")
        }
        for head in 0..<Self.keyValueHeads {
            blit.copy(
                from: scratch.keyCache,
                sourceOffset: head * bytesPerHead,
                to: key,
                destinationOffset: head * grownHeadBytes,
                size: bytesPerHead
            )
            blit.copy(
                from: scratch.valueCache,
                sourceOffset: head * bytesPerHead,
                to: value,
                destinationOffset: head * grownHeadBytes,
                size: bytesPerHead
            )
        }
        pendingCacheGrowth = (key, value)
    }

    func commitCacheGrowth() throws {
        guard let pendingCacheGrowth else {
            throw ProbeError.invalidInput("full cache growth was not encoded")
        }
        scratch.keyCache = pendingCacheGrowth.key
        scratch.valueCache = pendingCacheGrowth.value
        cacheCapacity = maximumCacheCapacity
        self.pendingCacheGrowth = nil
        try updateCacheCapacityConstants(cacheCapacity)
    }

    private func updateCacheCapacityConstants(_ capacity: Int) throws {
        try Gemma4NativeLayerUtilities.writeScalar(
            UInt32(capacity),
            to: constants.cacheCapacity,
            name: "full cache capacity"
        )
        let matrixHeadStride = Int64(capacity * Self.headDimension)
        for (buffer, name) in [
            (constants.queryKeyMatrixBatchStrides, "full QK matrix batch strides"),
            (constants.probabilityValueMatrixBatchStrides, "full PV matrix batch strides"),
        ] {
            guard buffer.storageMode == .shared,
                  buffer.length >= 2 * MemoryLayout<Int64>.stride
            else {
                throw ProbeError.invalidInput("\(name) are not writable shared storage")
            }
            let values = buffer.contents().assumingMemoryBound(to: Int64.self)
            values[0] = matrixHeadStride
            values[1] = 0
        }
    }

    private func applyResetState(position: Int) throws {
        let activeLength = position + 1
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(position),
            to: constants.ropeOffset,
            name: "full reset RoPE offset"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            UInt32(position),
            to: constants.appendIndex,
            name: "full reset append index"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(activeLength),
            to: constants.queryKeyOutputSize,
            name: "full reset QK output size"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(activeLength),
            to: constants.probabilityValueInputSize,
            name: "full reset PV input size"
        )
        guard constants.probabilityValueVectorBatchStrides.storageMode == .shared,
              constants.probabilityValueVectorBatchStrides.length
                >= 2 * MemoryLayout<Int64>.stride
        else {
            throw ProbeError.invalidInput(
                "full reset PV vector batch strides are not writable shared storage")
        }
        let batchStrides = constants.probabilityValueVectorBatchStrides.contents()
            .assumingMemoryBound(to: Int64.self)
        batchStrides[0] = Int64(Self.repeatCount * activeLength)
        batchStrides[1] = Int64(activeLength)
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(activeLength),
            to: constants.softmaxAxisSize,
            name: "full reset softmax axis size"
        )
        currentPosition = position
        activeCacheLength = activeLength
        encodedCurrentPosition = false
    }

    /// Must be called only while no command buffer referencing this plan's
    /// shared dynamic constants is in flight.
    func prepare(position: Int) throws {
        if position == currentPosition, !encodedCurrentPosition {
            return
        }
        let activeLength = position + 1
        guard encodedCurrentPosition,
              position == currentPosition + 1,
              position < cacheCapacity,
              activeLength <= 4_096
        else {
            throw ProbeError.invalidInput(
                "full position must advance by one within append/softmax capacity")
        }
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(position),
            to: constants.ropeOffset,
            name: "full RoPE offset"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            UInt32(position),
            to: constants.appendIndex,
            name: "full append index"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(activeLength),
            to: constants.queryKeyOutputSize,
            name: "full QK output size"
        )
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(activeLength),
            to: constants.probabilityValueInputSize,
            name: "full PV input size"
        )
        guard constants.probabilityValueVectorBatchStrides.storageMode == .shared,
              constants.probabilityValueVectorBatchStrides.length
                >= 2 * MemoryLayout<Int64>.stride
        else {
            throw ProbeError.invalidInput(
                "full PV vector batch strides are not writable shared storage")
        }
        let batchStrides = constants.probabilityValueVectorBatchStrides.contents()
            .assumingMemoryBound(to: Int64.self)
        batchStrides[0] = Int64(Self.repeatCount * activeLength)
        batchStrides[1] = Int64(activeLength)
        try Gemma4NativeLayerUtilities.writeScalar(
            Int32(activeLength),
            to: constants.softmaxAxisSize,
            name: "full softmax axis size"
        )
        currentPosition = position
        activeCacheLength = activeLength
        encodedCurrentPosition = false
    }

    func encode(
        into encoder: any Gemma4NativeCommandEncoder,
        inputBuffer: any MTLBuffer,
        normalizedInputBuffer: (any MTLBuffer)? = nil,
        consumePosition: Bool = true
    ) throws -> Gemma4NativeEncodedLayerState {
        guard !encodedCurrentPosition,
              inputBuffer.device.registryID == device.registryID,
              inputBuffer.length >= Self.hiddenSize * 2,
              normalizedInputBuffer?.device.registryID == nil
                || normalizedInputBuffer?.device.registryID == device.registryID,
              normalizedInputBuffer?.length == nil
                || normalizedInputBuffer!.length >= Self.hiddenSize * 2
        else {
            throw ProbeError.invalidInput(
                "full encoder input must be a same-device BF16 hidden buffer")
        }

        let projectionInput: any MTLBuffer
        if let normalizedInputBuffer {
            projectionInput = normalizedInputBuffer
        } else {
            encodeRMS(
                encoder: encoder,
                pipeline: pipelines.rmsLooped,
                input: inputBuffer,
                weight: norms.input,
                output: scratch.hiddenNorm,
                rows: 1,
                axis: constants.hiddenAxis,
                weightStride: constants.weightStride,
                threads: pipelines.rmsLooped.maxTotalThreadsPerThreadgroup
            )
            encoder.memoryBarrier(resources: [scratch.hiddenNorm])
            projectionInput = scratch.hiddenNorm
        }

        if usePromotedProjectionKernels {
            encodeIndexedFullQK(encoder: encoder, input: projectionInput)
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: queryProjection,
                input: projectionInput,
                output: scratch.rawQuery,
                inputWidth: constants.width5376,
                outputWidth: constants.width16384
            )
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: keyProjection,
                input: projectionInput,
                output: scratch.rawKey,
                inputWidth: constants.width5376,
                outputWidth: constants.width2048
            )
        }
        encoder.memoryBarrier(resources: [projectionInput, scratch.rawQuery, scratch.rawKey])

        if fuseAttentionCacheAppend {
            // F4-F9: shared-K/V RMS, Q/K RoPE, direct append, and query scale.
            Gemma4NativeLayerUtilities.encodeFusedAttentionPreparation(
                encoder: encoder,
                pipeline: pipelines.fusedAttentionPreparationAndAppend,
                rawQuery: scratch.rawQuery,
                rawKey: scratch.rawKey,
                rawValue: scratch.rawKey,
                queryWeight: norms.query,
                keyWeight: norms.key,
                position: constants.ropeOffset,
                ropeCosines: ropeCosines,
                ropeSines: ropeSines,
                queries: scratch.scaledQuery,
                keys: scratch.keyCache,
                values: scratch.valueCache,
                appendIndex: constants.appendIndex,
                cacheCapacity: constants.cacheCapacity,
                threads: Self.headDimension / 4,
                rows: Self.queryHeads + Self.keyValueHeads
            )
            encoder.memoryBarrier(resources: [
                scratch.scaledQuery, scratch.keyCache, scratch.valueCache,
            ])
        } else {
            Gemma4NativeLayerUtilities.encodeFusedAttentionPreparation(
                encoder: encoder,
                pipeline: pipelines.fusedAttentionPreparation,
                rawQuery: scratch.rawQuery,
                rawKey: scratch.rawKey,
                rawValue: scratch.rawKey,
                queryWeight: norms.query,
                keyWeight: norms.key,
                position: constants.ropeOffset,
                ropeCosines: ropeCosines,
                ropeSines: ropeSines,
                queries: scratch.ropedQuery,
                keys: scratch.ropedKey,
                values: scratch.normalizedValue,
                threads: Self.headDimension / 4,
                rows: Self.queryHeads + Self.keyValueHeads
            )
            encoder.memoryBarrier(resources: [
                scratch.ropedQuery, scratch.ropedKey, scratch.normalizedValue,
            ])
            encoder.setComputePipelineState(pipelines.cacheAppend)
            encoder.setBuffer(scratch.ropedKey, offset: 0, index: 0)
            encoder.setBuffer(scratch.normalizedValue, offset: 0, index: 1)
            encoder.setBuffer(scratch.keyCache, offset: 0, index: 2)
            encoder.setBuffer(scratch.valueCache, offset: 0, index: 3)
            encoder.setBuffer(constants.appendIndex, offset: 0, index: 4)
            encoder.setBuffer(constants.cacheCapacity, offset: 0, index: 5)
            Gemma4NativeLayerUtilities.dispatchLinear(
                encoder: encoder,
                pipeline: pipelines.cacheAppend,
                count: Self.keyValueWidth
            )
            encoder.memoryBarrier(resources: [scratch.keyCache, scratch.valueCache])

            encoder.setComputePipelineState(pipelines.multiplyOne)
            encoder.setBuffer(scratch.ropedQuery, offset: 0, index: 0)
            encoder.setBuffer(scratch.scaledQuery, offset: 0, index: 1)
            Gemma4NativeLayerUtilities.dispatchLinear(
                encoder: encoder,
                pipeline: pipelines.multiplyOne,
                count: Self.queryWidth
            )
            encoder.memoryBarrier(resources: [scratch.scaledQuery])
        }

        // F10: [4,8] batches of a 512-vector against N cache rows.
        encodeGEMV(
            encoder: encoder,
            pipeline: pipelines.queryKeyGEMV,
            matrix: scratch.keyCache,
            vector: scratch.scaledQuery,
            output: scratch.scores,
            inputSize: constants.queryKeyInputSize,
            outputSize: constants.queryKeyOutputSize,
            matrixLeadingDimension: constants.queryKeyMatrixLeadingDimension,
            vectorBatchStrides: constants.queryKeyVectorBatchStrides,
            matrixBatchStrides: constants.queryKeyMatrixBatchStrides,
            threadgroups: MTLSize(
                width: (activeCacheLength + 15) / 16,
                height: 1,
                depth: Self.queryHeads
            ),
            threads: MTLSize(width: 32, height: 1, depth: 4)
        )
        encoder.memoryBarrier(resources: [scratch.scores])

        // F11: precise FP32 accumulation with a BF16 probability store.
        encoder.setComputePipelineState(pipelines.preciseSoftmax)
        encoder.setBuffer(scratch.scores, offset: 0, index: 0)
        encoder.setBuffer(scratch.probabilities, offset: 0, index: 1)
        encoder.setBuffer(constants.softmaxAxisSize, offset: 0, index: 2)
        let softmaxThreads = 32 * ((activeCacheLength + 127) / 128)
        encoder.dispatchThreads(
            MTLSize(width: Self.queryHeads * softmaxThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: softmaxThreads,
                height: 1,
                depth: 1
            )
        )
        encoder.memoryBarrier(resources: [scratch.probabilities])

        // F12: [4,8] batches of BF16 probabilities against cached V.
        encodeGEMV(
            encoder: encoder,
            pipeline: pipelines.probabilityValueGEMV,
            matrix: scratch.valueCache,
            vector: scratch.probabilities,
            output: scratch.attention,
            inputSize: constants.probabilityValueInputSize,
            outputSize: constants.probabilityValueOutputSize,
            matrixLeadingDimension: constants.probabilityValueMatrixLeadingDimension,
            vectorBatchStrides: constants.probabilityValueVectorBatchStrides,
            matrixBatchStrides: constants.probabilityValueMatrixBatchStrides,
            threadgroups: MTLSize(width: 8, height: 1, depth: Self.queryHeads),
            threads: MTLSize(width: 32, height: 4, depth: 1)
        )
        encoder.memoryBarrier(resources: [scratch.attention])

        if usePromotedProjectionKernels {
            encodeIndexedProjection(
                encoder: encoder,
                pipeline: pipelines.indexedOutput,
                packedPipeline: pipelines.packedIndexedOutput,
                projection: indexedOutputProjection,
                input: scratch.attention,
                output: scratch.attentionProjection
            )
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmvFast,
                projection: outputProjection,
                input: scratch.attention,
                output: scratch.attentionProjection,
                inputWidth: constants.width16384,
                outputWidth: constants.width5376
            )
        }
        encoder.memoryBarrier(resources: [scratch.attentionProjection])

        encoder.setComputePipelineState(pipelines.attentionToMLPBoundary)
        encoder.setBuffer(scratch.attentionProjection, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(norms.postAttention, offset: 0, index: 2)
        encoder.setBuffer(norms.preFeedForward, offset: 0, index: 3)
        encoder.setBuffer(scratch.attentionResidual, offset: 0, index: 4)
        encoder.setBuffer(scratch.preFeedForward, offset: 0, index: 5)
        dispatchBoundary(encoder: encoder, pipeline: pipelines.attentionToMLPBoundary)
        encoder.memoryBarrier(resources: [
            scratch.attentionResidual, scratch.preFeedForward,
        ])

        if usePromotedProjectionKernels {
            encodeIndexedGateUpActivation(
                encoder: encoder,
                input: scratch.preFeedForward
            )
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: gateProjection,
                input: scratch.preFeedForward,
                output: scratch.gate,
                inputWidth: constants.width5376,
                outputWidth: constants.width21504
            )
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmv,
                projection: upProjection,
                input: scratch.preFeedForward,
                output: scratch.up,
                inputWidth: constants.width5376,
                outputWidth: constants.width21504
            )
            encoder.memoryBarrier(resources: [scratch.gate, scratch.up])

            encoder.setComputePipelineState(pipelines.activatedProduct)
            encoder.setBuffer(scratch.gate, offset: 0, index: 0)
            encoder.setBuffer(scratch.up, offset: 0, index: 1)
            encoder.setBuffer(scratch.activated, offset: 0, index: 2)
            Gemma4NativeLayerUtilities.dispatchLinear(
                encoder: encoder,
                pipeline: pipelines.activatedProduct,
                count: Self.intermediateSize
            )
        }
        encoder.memoryBarrier(resources: [scratch.activated])

        if usePromotedProjectionKernels {
            encodeIndexedProjection(
                encoder: encoder,
                pipeline: pipelines.indexedDown,
                packedPipeline: pipelines.packedIndexedDown,
                projection: indexedDownProjection,
                input: scratch.activated,
                output: scratch.down
            )
        } else {
            encodeQMV(
                encoder: encoder,
                pipeline: pipelines.qmvFast,
                projection: downProjection,
                input: scratch.activated,
                output: scratch.down,
                inputWidth: constants.width21504,
                outputWidth: constants.width5376
            )
        }
        encoder.memoryBarrier(resources: [scratch.down])

        encoder.setComputePipelineState(pipelines.mlpToNextBoundary)
        encoder.setBuffer(scratch.down, offset: 0, index: 0)
        encoder.setBuffer(scratch.attentionResidual, offset: 0, index: 1)
        encoder.setBuffer(norms.postFeedForward, offset: 0, index: 2)
        encoder.setBuffer(norms.layerScalar, offset: 0, index: 3)
        encoder.setBuffer(norms.nextInput, offset: 0, index: 4)
        encoder.setBuffer(scratch.output, offset: 0, index: 5)
        encoder.setBuffer(scratch.hiddenNorm, offset: 0, index: 6)
        dispatchBoundary(encoder: encoder, pipeline: pipelines.mlpToNextBoundary)
        encoder.memoryBarrier(resources: [scratch.output, scratch.hiddenNorm])
        if consumePosition {
            encodedCurrentPosition = true
        }

        return Gemma4NativeEncodedLayerState(
            output: scratch.output,
            nextInputNormalized: scratch.hiddenNorm,
            keyCache: scratch.keyCache,
            valueCache: scratch.valueCache,
            activeCacheLength: activeCacheLength,
            cacheCapacity: cacheCapacity
        )
    }

    func markReplayCompleted() throws {
        guard !encodedCurrentPosition else {
            throw ProbeError.invalidInput(
                "full replay cannot consume an already encoded position")
        }
        encodedCurrentPosition = true
    }

    private func encodeIndexedFullQK(
        encoder: any Gemma4NativeCommandEncoder,
        input: any MTLBuffer
    ) {
        encoder.setComputePipelineState(pipelines.indexedFullQK)
        encoder.setBuffer(indexedQueryProjection.weight, offset: 0, index: 0)
        encoder.setBuffer(indexedQueryProjection.indices, offset: 0, index: 1)
        encoder.setBuffer(indexedQueryProjection.lut, offset: 0, index: 2)
        encoder.setBuffer(indexedKeyProjection.weight, offset: 0, index: 3)
        encoder.setBuffer(indexedKeyProjection.indices, offset: 0, index: 4)
        encoder.setBuffer(indexedKeyProjection.lut, offset: 0, index: 5)
        encoder.setBuffer(input, offset: 0, index: 6)
        encoder.setBuffer(scratch.rawQuery, offset: 0, index: 7)
        encoder.setBuffer(scratch.rawKey, offset: 0, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 4_608, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 9, depth: 1)
        )
    }

    private func encodeIndexedGateUpActivation(
        encoder: any Gemma4NativeCommandEncoder,
        input: any MTLBuffer
    ) {
        encoder.setComputePipelineState(pipelines.indexedGateUpActivation)
        encoder.setBuffer(indexedGateProjection.weight, offset: 0, index: 0)
        encoder.setBuffer(indexedGateProjection.indices, offset: 0, index: 1)
        encoder.setBuffer(indexedGateProjection.lut, offset: 0, index: 2)
        encoder.setBuffer(indexedUpProjection.weight, offset: 0, index: 3)
        encoder.setBuffer(indexedUpProjection.indices, offset: 0, index: 4)
        encoder.setBuffer(indexedUpProjection.lut, offset: 0, index: 5)
        encoder.setBuffer(input, offset: 0, index: 6)
        encoder.setBuffer(scratch.activated, offset: 0, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 10_752, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
    }

    private func encodeIndexedProjection(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        packedPipeline: any MTLComputePipelineState,
        projection: Gemma4NativeIndexedProjectionBuffers,
        input: any MTLBuffer,
        output: any MTLBuffer
    ) {
        encoder.setComputePipelineState(
            projection.packedIndices == nil ? pipeline : packedPipeline)
        encoder.setBuffer(projection.weight, offset: 0, index: 0)
        encoder.setBuffer(
            projection.packedIndices ?? projection.indices,
            offset: 0,
            index: 1
        )
        encoder.setBuffer(projection.lut, offset: 0, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 1_344, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
    }

    private func encodeRMS(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        input: any MTLBuffer,
        weight: any MTLBuffer,
        output: any MTLBuffer,
        rows: Int,
        axis: any MTLBuffer,
        weightStride: any MTLBuffer,
        threads: Int
    ) {
        Gemma4NativeLayerUtilities.encodeRMS(
            encoder: encoder,
            pipeline: pipeline,
            input: input,
            weight: weight,
            output: output,
            epsilon: constants.epsilon,
            axis: axis,
            weightStride: weightStride,
            rows: rows,
            threads: threads
        )
    }

    private func encodeQMV(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        projection: Gemma4NativeProjectionBuffers,
        input: any MTLBuffer,
        output: any MTLBuffer,
        inputWidth: any MTLBuffer,
        outputWidth: any MTLBuffer
    ) {
        Gemma4NativeLayerUtilities.encodeQMV(
            encoder: encoder,
            pipeline: pipeline,
            weight: projection.weight,
            scales: projection.scales,
            biases: projection.biases,
            input: input,
            output: output,
            inputWidth: inputWidth,
            outputWidth: outputWidth,
            outputElements: projection.outputWidth
        )
    }

    private func encodeGEMV(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        matrix: any MTLBuffer,
        vector: any MTLBuffer,
        output: any MTLBuffer,
        inputSize: any MTLBuffer,
        outputSize: any MTLBuffer,
        matrixLeadingDimension: any MTLBuffer,
        vectorBatchStrides: any MTLBuffer,
        matrixBatchStrides: any MTLBuffer,
        threadgroups: MTLSize,
        threads: MTLSize
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(matrix, offset: 0, index: 0)
        encoder.setBuffer(vector, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(inputSize, offset: 0, index: 4)
        encoder.setBuffer(outputSize, offset: 0, index: 5)
        encoder.setBuffer(matrixLeadingDimension, offset: 0, index: 6)
        encoder.setBuffer(constants.gemvBatchDimensions, offset: 0, index: 9)
        encoder.setBuffer(constants.gemvBatchShape, offset: 0, index: 10)
        encoder.setBuffer(vectorBatchStrides, offset: 0, index: 11)
        encoder.setBuffer(matrixBatchStrides, offset: 0, index: 12)
        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threads
        )
    }

    private func dispatchBoundary(
        encoder: any Gemma4NativeCommandEncoder,
        pipeline: any MTLComputePipelineState
    ) {
        encoder.dispatchThreads(
            MTLSize(width: 1_024, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
        )
    }

    private static func validateNorm(
        _ array: MLXArray,
        size: Int,
        name: String
    ) throws {
        guard array.dtype == .bfloat16,
              array.ndim == 1,
              array.size == size
        else {
            throw ProbeError.invalidInput("full \(name) must be BF16 [\(size)]")
        }
    }

    private static func makeScratch(
        device: any MTLDevice,
        cacheCapacity: Int,
        usePrivateStorage: Bool,
        useSharedCacheStorage: Bool
    ) throws -> ScratchBuffers {
        let storageMode: MTLStorageMode = usePrivateStorage ? .private : .shared
        func buffer(_ elements: Int, _ name: String) throws -> any MTLBuffer {
            try Gemma4NativeLayerUtilities.scratchBuffer(
                elements: elements,
                name: "full.\(name)",
                device: device,
                storageMode: storageMode
            )
        }
        func cacheBuffer(_ elements: Int, _ name: String) throws -> any MTLBuffer {
            try Gemma4NativeLayerUtilities.scratchBuffer(
                elements: elements,
                name: "full.\(name)",
                device: device,
                storageMode: useSharedCacheStorage ? .shared : storageMode
            )
        }
        let cacheElements = keyValueHeads * cacheCapacity * headDimension
        let scoreElements = queryHeads * cacheCapacity
        return ScratchBuffers(
            hiddenNorm: try buffer(hiddenSize, "hiddenNorm"),
            rawQuery: try buffer(queryWidth, "rawQuery"),
            rawKey: try buffer(keyValueWidth, "rawKey"),
            normalizedValue: try buffer(keyValueWidth, "normalizedValue"),
            ropedQuery: try buffer(queryWidth, "ropedQuery"),
            ropedKey: try buffer(keyValueWidth, "ropedKey"),
            scaledQuery: try buffer(queryWidth, "scaledQuery"),
            scores: try buffer(scoreElements, "scores"),
            probabilities: try buffer(scoreElements, "probabilities"),
            attention: try buffer(queryWidth, "attention"),
            attentionProjection: try buffer(hiddenSize, "attentionProjection"),
            attentionResidual: try buffer(hiddenSize, "attentionResidual"),
            preFeedForward: try buffer(hiddenSize, "preFeedForward"),
            gate: try buffer(intermediateSize, "gate"),
            up: try buffer(intermediateSize, "up"),
            activated: try buffer(intermediateSize, "activated"),
            down: try buffer(hiddenSize, "down"),
            output: try buffer(hiddenSize, "output"),
            keyCache: try cacheBuffer(cacheElements, "keyCache"),
            valueCache: try cacheBuffer(cacheElements, "valueCache")
        )
    }

    private static func makeConstants(
        device: any MTLDevice,
        epsilon: Float,
        positionOffset: Int,
        priorLength: Int,
        activeLength: Int,
        cacheCapacity: Int
    ) throws -> ConstantBuffers {
        func scalar<T>(_ value: T, _ name: String) throws -> any MTLBuffer {
            try Gemma4NativeLayerUtilities.scalarBuffer(
                value,
                name: "full \(name)",
                device: device
            )
        }
        func array<T>(_ values: [T], _ name: String) throws -> any MTLBuffer {
            try Gemma4NativeLayerUtilities.arrayBuffer(
                values,
                name: "full \(name)",
                device: device
            )
        }
        let matrixHeadStride = Int64(cacheCapacity * headDimension)
        return ConstantBuffers(
            epsilon: try scalar(epsilon, "epsilon"),
            hiddenAxis: try scalar(UInt32(hiddenSize), "hidden axis"),
            weightStride: try scalar(UInt32(1), "weight stride"),
            width2048: try scalar(Int32(keyValueWidth), "width 2048"),
            width5376: try scalar(Int32(hiddenSize), "width 5376"),
            width16384: try scalar(Int32(queryWidth), "width 16384"),
            width21504: try scalar(Int32(intermediateSize), "width 21504"),
            ropeOffset: try scalar(Int32(positionOffset), "RoPE offset"),
            appendIndex: try scalar(UInt32(priorLength), "append index"),
            cacheCapacity: try scalar(UInt32(cacheCapacity), "cache capacity"),
            gemvBatchDimensions: try scalar(Int32(2), "GEMV batch dimensions"),
            gemvBatchShape: try array(
                [Int32(keyValueHeads), Int32(repeatCount)],
                "GEMV batch shape"
            ),
            queryKeyInputSize: try scalar(Int32(headDimension), "QK input size"),
            queryKeyOutputSize: try scalar(Int32(activeLength), "QK output size"),
            queryKeyMatrixLeadingDimension: try scalar(
                Int32(headDimension), "QK matrix leading dimension"),
            queryKeyVectorBatchStrides: try array(
                [Int64(repeatCount * headDimension), Int64(headDimension)],
                "QK vector batch strides"
            ),
            queryKeyMatrixBatchStrides: try array(
                [matrixHeadStride, Int64(0)],
                "QK matrix batch strides"
            ),
            probabilityValueInputSize: try scalar(
                Int32(activeLength), "PV input size"),
            probabilityValueOutputSize: try scalar(
                Int32(headDimension), "PV output size"),
            probabilityValueMatrixLeadingDimension: try scalar(
                Int32(headDimension), "PV matrix leading dimension"),
            probabilityValueVectorBatchStrides: try array(
                [Int64(repeatCount * activeLength), Int64(activeLength)],
                "PV vector batch strides"
            ),
            probabilityValueMatrixBatchStrides: try array(
                [matrixHeadStride, Int64(0)],
                "PV matrix batch strides"
            ),
            softmaxAxisSize: try scalar(Int32(activeLength), "softmax axis size")
        )
    }

    private static func makePipelines(
        device: any MTLDevice,
        trustedLibrary: any MTLLibrary,
        helperLibrary: any MTLLibrary
    ) throws -> Pipelines {
        func trusted(
            _ name: String,
            boolConstants: [Int: Bool] = [:]
        ) throws -> any MTLComputePipelineState {
            try Gemma4NativeLayerUtilities.trustedPipeline(
                named: name,
                library: trustedLibrary,
                device: device,
                boolConstants: boolConstants
            )
        }
        func helper(_ name: String) throws -> any MTLComputePipelineState {
            try Gemma4NativeLayerUtilities.helperPipeline(
                named: name,
                library: helperLibrary,
                device: device
            )
        }
        return Pipelines(
            rmsLooped: try trusted("rms_loopedbfloat16", boolConstants: [20: true]),
            qmv: try trusted("affine_qmv_bfloat16_t_gs_64_b_4_batch_0"),
            qmvFast: try trusted("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0"),
            indexedFullQK: try helper("mlxfast_indexed_full_qk_qmv_5376_v1"),
            indexedOutput: try helper("mlxfast_indexed_output_qmv_fast_16384_v1"),
            packedIndexedOutput: try helper(
                "mlxfast_packed12_indexed_output_qmv_fast_16384_v1"),
            indexedGateUpActivation: try helper(
                "mlxfast_indexed_fused_gate_up_activation_qmv_5376_v1"),
            indexedDown: try helper("mlxfast_indexed_down_qmv_21504_v1"),
            packedIndexedDown: try helper(
                "mlxfast_packed12_indexed_down_qmv_21504_v1"),
            fusedAttentionPreparation: try helper(
                "mlxfast_fused_full_attention_rms_rope_table_shared_kv_512_v5"
            ),
            fusedAttentionPreparationAndAppend: try helper(
                "mlxfast_native_fused_full_attention_prep_append_512_v1"
            ),
            queryKeyGEMV: try trusted(
                "gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc1_axpby0"),
            preciseSoftmax: try trusted("block_softmax_precise_bfloat16"),
            probabilityValueGEMV: try trusted(
                "gemv_t_bfloat16_bm1_bn4_sm8_sn4_tm4_tn4_nc1_axpby0"),
            cacheAppend: try helper("mlxfast_full_cache_append"),
            multiplyOne: try helper("mlxfast_bf16_multiply_one"),
            activatedProduct: try helper("mlxfast_bf16_activated_product"),
            attentionToMLPBoundary: try helper(
                "mlxfast_fused_attention_to_mlp_boundary_5376_v1"),
            mlpToNextBoundary: try helper(
                "mlxfast_fused_mlp_to_next_boundary_5376_v1")
        )
    }
}

/// Diagnostic 60-layer text-trunk executor. It intentionally retains each
/// layer probe's independent pipelines, constants, caches, and scratch while
/// proving the one-command-buffer control plane before any arena coloring or
/// pipeline deduplication.
final class Gemma4NativeTrunkProbe {
    typealias ProbeError = Gemma4NativeSlidingLayerProbe.ProbeError

    static let groupedLayerRanges: [Range<Int>] = [
        0..<1,
        1..<11,
        11..<21,
        21..<31,
        31..<41,
        41..<51,
        51..<60,
    ]

    struct CacheResult {
        let layerIndex: Int
        let layerType: Gemma4LayerType
        let keyCache: MLXArray
        let valueCache: MLXArray
        let activeCacheLength: Int
        let cacheCapacity: Int
    }

    struct Result {
        let hidden: MLXArray
        let finalNormalized: MLXArray
        /// Completed aliases for first-differing-layer diagnostics.
        let layerHidden: [MLXArray]
        let layerNextInputNormalized: [MLXArray]
        let caches: [CacheResult]
    }

    private enum LayerPlan {
        case sliding(Gemma4NativeSlidingLayerProbe)
        case full(Gemma4NativeFullLayerProbe)

        var registryID: UInt64 {
            switch self {
            case .sliding(let probe): probe.metalRegistryID
            case .full(let probe): probe.metalRegistryID
            }
        }

        var layerType: Gemma4LayerType {
            switch self {
            case .sliding: .sliding
            case .full: .full
            }
        }

        var cacheHeads: Int {
            switch self {
            case .sliding: 16
            case .full: 4
            }
        }

        var headDimension: Int {
            switch self {
            case .sliding: 256
            case .full: 512
            }
        }

        var fixedCacheCapacity: Int {
            switch self {
            case .sliding(let probe): probe.fixedCacheCapacity
            case .full(let probe): probe.fixedCacheCapacity
            }
        }

        func prefillCacheArrays() throws -> (MLXArray, MLXArray) {
            switch self {
            case .sliding(let probe):
                try probe.prefillCacheArrays()
            case .full(let probe):
                try probe.prefillCacheArrays()
            }
        }

        func resetCache(
            keys: MLXArray,
            values: MLXArray,
            position: Int
        ) throws {
            switch self {
            case .sliding(let probe):
                try probe.resetCache(
                    keys: keys,
                    values: values,
                    position: position
                )
            case .full(let probe):
                try probe.resetCache(
                    keys: keys,
                    values: values,
                    position: position
                )
            }
        }

        func encodePrivateCacheReset(
            keys: MLXArray,
            values: MLXArray,
            position: Int,
            blit: any MTLBlitCommandEncoder
        ) throws {
            switch self {
            case .sliding(let probe):
                try probe.encodePrivateCacheReset(
                    keys: keys,
                    values: values,
                    position: position,
                    blit: blit
                )
            case .full(let probe):
                try probe.encodePrivateCacheReset(
                    keys: keys,
                    values: values,
                    position: position,
                    blit: blit
                )
            }
        }

        func commitPrivateCacheReset(position: Int) throws {
            switch self {
            case .sliding(let probe):
                try probe.commitPrivateCacheReset(position: position)
            case .full(let probe):
                try probe.commitPrivateCacheReset(position: position)
            }
        }

        func adoptCacheBuffers(
            _ buffers: Gemma4NativeAdoptedCacheBuffers,
            position: Int,
            residencyCollector: (any Gemma4NativeResidencyCollecting)? = nil
        ) throws {
            switch self {
            case .sliding(let probe):
                try probe.adoptCacheBuffers(
                    buffers,
                    position: position,
                    residencyCollector: residencyCollector
                )
            case .full(let probe):
                try probe.adoptCacheBuffers(
                    buffers,
                    position: position,
                    residencyCollector: residencyCollector
                )
            }
        }

        func needsCacheGrowth(position: Int) -> Bool {
            switch self {
            case .sliding(let probe): probe.needsCacheGrowth(position: position)
            case .full(let probe): probe.needsCacheGrowth(position: position)
            }
        }

        func encodeCacheGrowth(
            position: Int,
            blit: any MTLBlitCommandEncoder
        ) throws {
            switch self {
            case .sliding(let probe):
                try probe.encodeCacheGrowth(position: position, blit: blit)
            case .full(let probe):
                try probe.encodeCacheGrowth(position: position, blit: blit)
            }
        }

        func commitCacheGrowth() throws {
            switch self {
            case .sliding(let probe): try probe.commitCacheGrowth()
            case .full(let probe): try probe.commitCacheGrowth()
            }
        }

        func prepare(position: Int) throws {
            switch self {
            case .sliding(let probe):
                try probe.prepare(position: position)
            case .full(let probe):
                try probe.prepare(position: position)
            }
        }

        func encode(
            into encoder: any Gemma4NativeCommandEncoder,
            inputBuffer: any MTLBuffer,
            normalizedInputBuffer: (any MTLBuffer)?,
            consumePosition: Bool = true
        ) throws -> Gemma4NativeEncodedLayerState {
            switch self {
            case .sliding(let probe):
                try probe.encode(
                    into: encoder,
                    inputBuffer: inputBuffer,
                    normalizedInputBuffer: normalizedInputBuffer,
                    consumePosition: consumePosition
                )
            case .full(let probe):
                try probe.encode(
                    into: encoder,
                    inputBuffer: inputBuffer,
                    normalizedInputBuffer: normalizedInputBuffer,
                    consumePosition: consumePosition
                )
            }
        }

        func markReplayCompleted() throws {
            switch self {
            case .sliding(let probe):
                try probe.markReplayCompleted()
            case .full(let probe):
                try probe.markReplayCompleted()
            }
        }
    }

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let plans: [LayerPlan]
    private let usePrivateStorage: Bool
    private var currentPosition: Int
    private var encodedCurrentPosition = false
    private var tokenEncodingBegun = false
    private var encodedLayerCount = 0
    private var lastEncodedLayers: [Gemma4NativeEncodedLayerState] = []
    private var adoptedCacheArrays: [MLXArray] = []

    init(
        residentWeights: Gemma4NativeResidentWeights,
        config: Gemma4Config,
        priorKeys: [MLXArray],
        priorValues: [MLXArray],
        positionOffset: Int,
        usePrivateStorage: Bool = false,
        useSharedCacheStorage: Bool = false
    ) throws {
        guard config.numHiddenLayers == 60,
              config.layerTypes.count == config.numHiddenLayers,
              residentWeights.layers.count == config.numHiddenLayers,
              priorKeys.count == config.numHiddenLayers,
              priorValues.count == config.numHiddenLayers,
              positionOffset >= 0,
              positionOffset + 1 < config.slidingWindow,
              config.slidingWindow == 1_024
        else {
            throw ProbeError.invalidInput(
                "trunk requires 60 caches at one pre-window position below 1023")
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              device.hasUnifiedMemory,
              let queue = device.makeCommandQueue()
        else {
            throw ProbeError.unavailable("trunk Metal device or command queue")
        }

        var plans: [LayerPlan] = []
        plans.reserveCapacity(config.numHiddenLayers)
        let allBlockWeights = residentWeights.layers.map(\.block)
        let finalNorm = residentWeights.model.finalNorm
        for layerIndex in 0..<config.numHiddenLayers {
            guard priorKeys[layerIndex].ndim == 4,
                  priorValues[layerIndex].shape == priorKeys[layerIndex].shape,
                  priorKeys[layerIndex].dim(2) == positionOffset
            else {
                throw ProbeError.invalidInput(
                    "layer \(layerIndex) cache does not share trunk position \(positionOffset)")
            }
            let blockWeights = allBlockWeights[layerIndex]
            let nextInputNormWeight = layerIndex + 1 < config.numHiddenLayers
                ? allBlockWeights[layerIndex + 1].inputLayerNorm
                : finalNorm
            let attentionWeights = residentWeights.layers[layerIndex].attention
            let mlpWeights = residentWeights.layers[layerIndex].mlp
            let plan: LayerPlan
            switch config.layerTypes[layerIndex] {
            case .sliding:
                plan = .sliding(try Gemma4NativeSlidingLayerProbe(
                    blockWeights: blockWeights,
                    attentionWeights: attentionWeights,
                    mlpWeights: mlpWeights,
                    priorKeys: priorKeys[layerIndex],
                    priorValues: priorValues[layerIndex],
                    positionOffset: positionOffset,
                    cacheCapacity: Gemma4ModelCache.nativeCacheHorizon,
                    rmsNormEps: Float(config.rmsNormEps),
                    ropeBase: Float(config.slidingRope.theta),
                    nextInputNormWeight: nextInputNormWeight,
                    metalDevice: device,
                    commandQueue: queue,
                    usePrivateStorage: usePrivateStorage,
                    useSharedCacheStorage: useSharedCacheStorage,
                    qMetadata: residentWeights.layers[layerIndex].qMetadata,
                    kMetadata: residentWeights.layers[layerIndex].kMetadata,
                    vMetadata: residentWeights.layers[layerIndex].vMetadata,
                    outputMetadata: residentWeights.layers[layerIndex].outputMetadata,
                    gateMetadata: residentWeights.layers[layerIndex].gateMetadata,
                    upMetadata: residentWeights.layers[layerIndex].upMetadata,
                    downMetadata: residentWeights.layers[layerIndex].downMetadata
                ))
            case .full:
                plan = .full(try Gemma4NativeFullLayerProbe(
                    blockWeights: blockWeights,
                    attentionWeights: attentionWeights,
                    mlpWeights: mlpWeights,
                    priorKeys: priorKeys[layerIndex],
                    priorValues: priorValues[layerIndex],
                    positionOffset: positionOffset,
                    cacheCapacity: 4_096,
                    rmsNormEps: Float(config.rmsNormEps),
                    ropeBase: Float(config.fullRope.theta),
                    partialRotaryFactor: Float(config.fullRope.partialRotaryFactor),
                    nextInputNormWeight: nextInputNormWeight,
                    metalDevice: device,
                    commandQueue: queue,
                    usePrivateStorage: usePrivateStorage,
                    useSharedCacheStorage: useSharedCacheStorage,
                    qMetadata: residentWeights.layers[layerIndex].qMetadata,
                    kMetadata: residentWeights.layers[layerIndex].kMetadata,
                    outputMetadata: residentWeights.layers[layerIndex].outputMetadata,
                    gateMetadata: residentWeights.layers[layerIndex].gateMetadata,
                    upMetadata: residentWeights.layers[layerIndex].upMetadata,
                    downMetadata: residentWeights.layers[layerIndex].downMetadata
                ))
            }
            guard plan.registryID == device.registryID else {
                throw ProbeError.invalidInput(
                    "layer \(layerIndex) Metal registry differs from trunk device")
            }
            plans.append(plan)
        }

        self.device = device
        self.queue = queue
        self.plans = plans
        self.usePrivateStorage = usePrivateStorage
        self.currentPosition = positionOffset
    }

    var metalRegistryID: UInt64 {
        device.registryID
    }

    var metalDevice: any MTLDevice {
        device
    }

    var commandQueue: any MTLCommandQueue {
        queue
    }

    func prefillCacheArrays() throws -> (keys: [MLXArray], values: [MLXArray]) {
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        keys.reserveCapacity(plans.count)
        values.reserveCapacity(plans.count)
        for plan in plans {
            let arrays = try plan.prefillCacheArrays()
            keys.append(arrays.0)
            values.append(arrays.1)
        }
        return (keys, values)
    }

    /// Replace every layer's logical prefix in its existing fixed-capacity
    /// cache. Caller must ensure no native command buffer is in flight.
    func resetCaches(
        priorKeys: [MLXArray],
        priorValues: [MLXArray],
        position: Int
    ) throws {
        guard !tokenEncodingBegun,
              priorKeys.count == plans.count,
              priorValues.count == plans.count,
              position >= 0,
              position < 1_024
        else {
            throw ProbeError.invalidInput(
                "trunk reset requires 60 caches at one pre-window position")
        }

        // Validate the complete import before mutating any layer cache.
        for layerIndex in plans.indices {
            let plan = plans[layerIndex]
            let expectedShape = [
                1,
                plan.cacheHeads,
                position,
                plan.headDimension,
            ]
            guard position < plan.fixedCacheCapacity,
                  priorKeys[layerIndex].dtype == .bfloat16,
                  priorKeys[layerIndex].shape == expectedShape,
                  priorValues[layerIndex].dtype == .bfloat16,
                  priorValues[layerIndex].shape == expectedShape
            else {
                throw ProbeError.invalidInput(
                    "layer \(layerIndex) reset cache shape/capacity mismatch")
            }
        }
        if usePrivateStorage {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder()
            else {
                throw ProbeError.unavailable(
                    "transactional private cache import command buffer")
            }
            commandBuffer.label = "Gemma4NativeTrunkProbe.transactionalCacheImport"
            blit.label = "Gemma4NativeTrunkProbe.transactionalCacheImport.blit"
            do {
                for layerIndex in plans.indices {
                    try plans[layerIndex].encodePrivateCacheReset(
                        keys: priorKeys[layerIndex],
                        values: priorValues[layerIndex],
                        position: position,
                        blit: blit
                    )
                }
                blit.endEncoding()
            } catch {
                blit.endEncoding()
                throw error
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            guard commandBuffer.status == .completed else {
                throw ProbeError.commandFailed(
                    commandBuffer.error?.localizedDescription
                        ?? "cache import status \(commandBuffer.status.rawValue)")
            }
            for plan in plans {
                try plan.commitPrivateCacheReset(position: position)
            }
        } else {
            for layerIndex in plans.indices {
                try plans[layerIndex].resetCache(
                    keys: priorKeys[layerIndex],
                    values: priorValues[layerIndex],
                    position: position
                )
            }
        }
        currentPosition = position
        encodedCurrentPosition = false
        tokenEncodingBegun = false
        encodedLayerCount = 0
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    func adoptCaches(
        keys: [MLXArray],
        values: [MLXArray],
        position: Int,
        residencyCollector: (any Gemma4NativeResidencyCollecting)? = nil
    ) throws {
        guard !tokenEncodingBegun,
              !encodedCurrentPosition,
              encodedLayerCount == 0,
              keys.count == plans.count,
              values.count == plans.count,
              position >= 0,
              position < 1_024
        else {
            throw ProbeError.invalidInput(
                "trunk cache adoption requires 60 capacity-shaped caches before execution")
        }
        var buffers: [Gemma4NativeAdoptedCacheBuffers] = []
        buffers.reserveCapacity(plans.count)
        for layerIndex in plans.indices {
            let plan = plans[layerIndex]
            guard keys[layerIndex].dtype == .bfloat16,
                  keys[layerIndex].ndim == 4,
                  keys[layerIndex].dim(0) == 1,
                  keys[layerIndex].dim(1) == plan.cacheHeads,
                  keys[layerIndex].dim(2) > position,
                  keys[layerIndex].dim(2) <= plan.fixedCacheCapacity,
                  keys[layerIndex].dim(3) == plan.headDimension,
                  values[layerIndex].dtype == .bfloat16,
                  values[layerIndex].shape == keys[layerIndex].shape,
                  let keyBuffer = keys[layerIndex].asMTLBuffer(
                      device: device,
                      noCopy: true
                  ),
                  let valueBuffer = values[layerIndex].asMTLBuffer(
                      device: device,
                      noCopy: true
                  )
            else {
                throw ProbeError.invalidInput(
                    "layer \(layerIndex) capacity cache cannot be adopted without copying")
            }
            buffers.append(Gemma4NativeAdoptedCacheBuffers(
                keys: keyBuffer,
                values: valueBuffer
            ))
        }
        for layerIndex in plans.indices {
            try plans[layerIndex].adoptCacheBuffers(
                buffers[layerIndex],
                position: position,
                residencyCollector: residencyCollector
            )
        }
        adoptedCacheArrays = keys + values
        currentPosition = position
        encodedCurrentPosition = false
        tokenEncodingBegun = false
        encodedLayerCount = 0
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    func adoptCombinedCaches(
        storage: [MLXArray],
        position: Int,
        residencyCollector: (any Gemma4NativeResidencyCollecting)? = nil
    ) throws {
        guard !tokenEncodingBegun,
              !encodedCurrentPosition,
              encodedLayerCount == 0,
              storage.count == plans.count,
              position >= 0,
              position < 1_024
        else {
            throw ProbeError.invalidInput(
                "trunk combined cache adoption requires 60 parents before execution")
        }
        var buffers: [Gemma4NativeAdoptedCacheBuffers] = []
        buffers.reserveCapacity(plans.count)
        for layerIndex in plans.indices {
            let plan = plans[layerIndex]
            let parent = storage[layerIndex]
            guard parent.dtype == .bfloat16,
                  parent.ndim == 5,
                  parent.dim(0) == 2,
                  parent.dim(1) == 1,
                  parent.dim(2) == plan.cacheHeads,
                  parent.dim(3) > position,
                  parent.dim(3) <= plan.fixedCacheCapacity,
                  parent.dim(4) == plan.headDimension,
                  let parentBuffer = parent.asMTLBuffer(
                    device: device,
                    noCopy: true
                  )
            else {
                throw ProbeError.invalidInput(
                    "layer \(layerIndex) combined cache parent cannot be adopted")
            }
            let slabBytes = plan.cacheHeads * parent.dim(3)
                * plan.headDimension * DType.bfloat16.size
            guard parentBuffer.length == 2 * slabBytes,
                  let keys = device.makeBuffer(
                    bytesNoCopy: parentBuffer.contents(),
                    length: slabBytes
                  ),
                  let values = device.makeBuffer(
                    bytesNoCopy: parentBuffer.contents().advanced(by: slabBytes),
                    length: slabBytes
                  )
            else {
                throw ProbeError.unavailable(
                    "layer \(layerIndex) combined cache slab aliases")
            }
            buffers.append(Gemma4NativeAdoptedCacheBuffers(
                keys: keys,
                values: values
            ))
            residencyCollector?.collect(keys)
            residencyCollector?.collect(values)
        }
        for layerIndex in plans.indices {
            try plans[layerIndex].adoptCacheBuffers(
                buffers[layerIndex],
                position: position,
                residencyCollector: residencyCollector
            )
        }
        adoptedCacheArrays = storage
        currentPosition = position
        encodedCurrentPosition = false
        tokenEncodingBegun = false
        encodedLayerCount = 0
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    func activatePrefillCaches(position: Int) throws {
        guard !tokenEncodingBegun,
              !encodedCurrentPosition,
              encodedLayerCount == 0,
              position >= 0,
              position < 1_024
        else {
            throw ProbeError.invalidInput(
                "prefill cache activation requires an unconsumed pre-window position")
        }
        for plan in plans {
            try plan.commitPrivateCacheReset(position: position)
        }
        currentPosition = position
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    func rewindAfterWarmup() throws {
        guard encodedCurrentPosition,
              !tokenEncodingBegun,
              encodedLayerCount == plans.count
        else {
            throw ProbeError.invalidInput(
                "trunk warmup rewind requires one completed position-zero token")
        }
        for plan in plans {
            try plan.commitPrivateCacheReset(position: 0)
        }
        currentPosition = 0
        encodedCurrentPosition = false
        tokenEncodingBegun = false
        encodedLayerCount = 0
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    /// Update shared dynamic constants only after the command buffer that used
    /// the current position has completed.
    func prepare(position: Int) throws {
        if position == currentPosition,
           !encodedCurrentPosition,
           !tokenEncodingBegun,
           encodedLayerCount == 0
        {
            return
        }
        guard encodedCurrentPosition,
              !tokenEncodingBegun,
              encodedLayerCount == plans.count,
              position == currentPosition + 1
        else {
            throw ProbeError.invalidInput(
                "trunk position must advance monotonically by one after encode")
        }
        let growingPlans = plans.filter { $0.needsCacheGrowth(position: position) }
        if !growingPlans.isEmpty {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder()
            else {
                throw ProbeError.unavailable("native cache growth command buffer")
            }
            do {
                for plan in growingPlans {
                    try plan.encodeCacheGrowth(position: position, blit: blit)
                }
                blit.endEncoding()
            } catch {
                blit.endEncoding()
                throw error
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            guard commandBuffer.status == .completed else {
                throw ProbeError.commandFailed(
                    commandBuffer.error?.localizedDescription
                        ?? "cache growth status \(commandBuffer.status.rawValue)")
            }
            for plan in growingPlans {
                try plan.commitCacheGrowth()
            }
            adoptedCacheArrays.removeAll(keepingCapacity: false)
        }
        for plan in plans {
            try plan.prepare(position: position)
        }
        currentPosition = position
        encodedCurrentPosition = false
        tokenEncodingBegun = false
        encodedLayerCount = 0
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    func beginTokenEncoding() throws {
        guard !encodedCurrentPosition,
              !tokenEncodingBegun,
              encodedLayerCount == 0
        else {
            throw ProbeError.invalidInput(
                "trunk token encoding is already begun or the position is consumed")
        }
        lastEncodedLayers.removeAll(keepingCapacity: true)
        tokenEncodingBegun = true
    }

    func encodeLayers(
        range: Range<Int>,
        into encoder: any Gemma4NativeCommandEncoder,
        inputBuffer: any MTLBuffer,
        normalizedInputBuffer: (any MTLBuffer)? = nil
    ) throws -> Gemma4NativeEncodedTrunkState {
        guard tokenEncodingBegun,
              !encodedCurrentPosition,
              !range.isEmpty,
              range.lowerBound == encodedLayerCount,
              range.lowerBound >= 0,
              range.upperBound <= plans.count,
              inputBuffer.device.registryID == device.registryID,
              inputBuffer.length >= 5_376 * 2,
              normalizedInputBuffer?.device.registryID == nil
                || normalizedInputBuffer?.device.registryID == device.registryID,
              normalizedInputBuffer?.length == nil
                || normalizedInputBuffer!.length >= 5_376 * 2
        else {
            throw ProbeError.invalidInput(
                "trunk layer range must be the next nonempty contiguous same-device range")
        }
        if encodedLayerCount > 0 {
            guard let expectedInput = lastEncodedLayers.last?.output,
                  let expectedNormalized = lastEncodedLayers.last?.nextInputNormalized,
                  let normalizedInputBuffer,
                  (inputBuffer as AnyObject) === (expectedInput as AnyObject),
                  (normalizedInputBuffer as AnyObject)
                    === (expectedNormalized as AnyObject)
            else {
                throw ProbeError.invalidInput(
                    "trunk grouped input pair does not chain from the previous range output")
            }
        } else if normalizedInputBuffer != nil {
            throw ProbeError.invalidInput(
                "trunk layer zero must compute its own input normalization")
        }

        var currentHidden = inputBuffer
        var currentNormalized = normalizedInputBuffer
        for layerIndex in range {
            let encoded = try plans[layerIndex].encode(
                into: encoder,
                inputBuffer: currentHidden,
                normalizedInputBuffer: currentNormalized
            )
            lastEncodedLayers.append(encoded)
            currentHidden = encoded.output
            currentNormalized = encoded.nextInputNormalized
        }
        encodedLayerCount = range.upperBound
        if encodedLayerCount == plans.count {
            encodedCurrentPosition = true
            tokenEncodingBegun = false
        }
        guard let currentNormalized else {
            throw ProbeError.invalidInput("trunk range produced no normalized sibling")
        }
        return Gemma4NativeEncodedTrunkState(
            hidden: currentHidden,
            nextInputNormalized: currentNormalized
        )
    }

    func encode(
        into encoder: any Gemma4NativeCommandEncoder,
        inputBuffer: any MTLBuffer
    ) throws -> Gemma4NativeEncodedTrunkState {
        try beginTokenEncoding()
        return try encodeLayers(
            range: 0..<plans.count,
            into: encoder,
            inputBuffer: inputBuffer
        )
    }

    func record(
        into encoder: any Gemma4NativeCommandEncoder,
        inputBuffer: any MTLBuffer
    ) throws -> Gemma4NativeEncodedTrunkState {
        try recordLayers(
            range: 0..<plans.count,
            into: encoder,
            inputBuffer: inputBuffer
        )
    }

    func recordLayers(
        range: Range<Int>,
        into encoder: any Gemma4NativeCommandEncoder,
        inputBuffer: any MTLBuffer,
        normalizedInputBuffer: (any MTLBuffer)? = nil
    ) throws -> Gemma4NativeEncodedTrunkState {
        guard !encodedCurrentPosition,
              !tokenEncodingBegun,
              encodedLayerCount == 0,
              !range.isEmpty,
              range.lowerBound >= 0,
              range.upperBound <= plans.count,
              inputBuffer.device.registryID == device.registryID,
              inputBuffer.length >= 5_376 * 2,
              normalizedInputBuffer?.device.registryID == nil
                || normalizedInputBuffer?.device.registryID == device.registryID,
              normalizedInputBuffer?.length == nil
                || normalizedInputBuffer!.length >= 5_376 * 2
        else {
            throw ProbeError.invalidInput(
                "trunk recording requires an unconsumed same-device layer range")
        }

        var currentHidden = inputBuffer
        var currentNormalized = normalizedInputBuffer
        for layerIndex in range {
            let encoded = try plans[layerIndex].encode(
                into: encoder,
                inputBuffer: currentHidden,
                normalizedInputBuffer: currentNormalized,
                consumePosition: false
            )
            currentHidden = encoded.output
            currentNormalized = encoded.nextInputNormalized
        }
        guard let currentNormalized else {
            throw ProbeError.invalidInput("trunk recording produced no normalized output")
        }
        return Gemma4NativeEncodedTrunkState(
            hidden: currentHidden,
            nextInputNormalized: currentNormalized
        )
    }

    func beginReplay() throws {
        guard !encodedCurrentPosition,
              !tokenEncodingBegun,
              encodedLayerCount == 0
        else {
            throw ProbeError.invalidInput(
                "trunk replay is already begun or the position is consumed")
        }
        tokenEncodingBegun = true
    }

    func completeReplay() throws {
        guard !encodedCurrentPosition,
              tokenEncodingBegun,
              encodedLayerCount == 0
        else {
            throw ProbeError.invalidInput("trunk replay state is inconsistent")
        }
        for plan in plans {
            try plan.markReplayCompleted()
        }
        encodedLayerCount = plans.count
        encodedCurrentPosition = true
        tokenEncodingBegun = false
        lastEncodedLayers.removeAll(keepingCapacity: true)
    }

    func run(hidden: MLXArray) throws -> Result {
        guard !usePrivateStorage else {
            throw ProbeError.invalidInput(
                "trunk run() cannot create MLX aliases for private storage; use runHiddenBits")
        }
        guard hidden.dtype == .bfloat16,
              hidden.shape == [1, 1, 5_376],
              let hiddenBuffer = hidden.asMTLBuffer(device: device, noCopy: true),
              hiddenBuffer.device.registryID == device.registryID,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.invalidInput(
                "trunk hidden must be contiguous same-device BF16 [1,1,5376]")
        }

        commandBuffer.label = "Gemma4NativeTrunkProbe"
        encoder.label = "Gemma4NativeTrunkProbe.layers0-59"
        _ = try encode(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            inputBuffer: hiddenBuffer
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "status \(commandBuffer.status.rawValue)")
        }

        let layerHidden = lastEncodedLayers.map { encoded in
            Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: encoded.output,
                shape: [1, 1, 5_376]
            )
        }
        let layerNextInputNormalized = lastEncodedLayers.map { encoded in
            Gemma4NativeLayerUtilities.makeMLXAlias(
                buffer: encoded.nextInputNormalized,
                shape: [1, 1, 5_376]
            )
        }
        guard let finalHidden = layerHidden.last else {
            throw ProbeError.invalidInput("trunk contains no layer outputs")
        }
        guard let finalNormalized = layerNextInputNormalized.last else {
            throw ProbeError.invalidInput("trunk contains no normalized layer outputs")
        }
        let caches = zip(plans, lastEncodedLayers).enumerated().map { entry in
            let layerIndex = entry.offset
            let (plan, encoded) = entry.element
            let shape = [
                1,
                plan.cacheHeads,
                encoded.cacheCapacity,
                plan.headDimension,
            ]
            return CacheResult(
                layerIndex: layerIndex,
                layerType: plan.layerType,
                keyCache: Gemma4NativeLayerUtilities.makeMLXAlias(
                    buffer: encoded.keyCache,
                    shape: shape
                ),
                valueCache: Gemma4NativeLayerUtilities.makeMLXAlias(
                    buffer: encoded.valueCache,
                    shape: shape
                ),
                activeCacheLength: encoded.activeCacheLength,
                cacheCapacity: encoded.cacheCapacity
            )
        }
        return Result(
            hidden: finalHidden,
            finalNormalized: finalNormalized,
            layerHidden: layerHidden,
            layerNextInputNormalized: layerNextInputNormalized,
            caches: caches
        )
    }

    /// Diagnostic execution without constructing MLX aliases over persistent
    /// layer/cache buffers. This matches production ownership across steps.
    func runHiddenBits(hidden: MLXArray) throws -> [UInt16] {
        guard hidden.dtype == .bfloat16,
              hidden.shape == [1, 1, 5_376],
              let hiddenBuffer = hidden.asMTLBuffer(device: device, noCopy: true),
              hiddenBuffer.device.registryID == device.registryID,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ProbeError.invalidInput(
                "trunk hidden must be contiguous same-device BF16 [1,1,5376]")
        }
        let finalBuffer = try encode(
            into: Gemma4NativeDirectCommandEncoder(encoder),
            inputBuffer: hiddenBuffer
        ).hidden
        encoder.endEncoding()
        let readableBuffer: any MTLBuffer
        if finalBuffer.storageMode == .private {
            guard let sharedOutput = device.makeBuffer(
                      length: 5_376 * 2,
                      options: .storageModeShared
                  ),
                  let blit = commandBuffer.makeBlitCommandEncoder()
            else {
                throw ProbeError.unavailable(
                    "private trunk diagnostic output blit")
            }
            sharedOutput.label = "Gemma4NativeTrunkProbe.diagnosticSharedOutput"
            blit.copy(
                from: finalBuffer,
                sourceOffset: 0,
                to: sharedOutput,
                destinationOffset: 0,
                size: 5_376 * 2
            )
            blit.endEncoding()
            readableBuffer = sharedOutput
        } else {
            readableBuffer = finalBuffer
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw ProbeError.commandFailed(
                commandBuffer.error?.localizedDescription
                    ?? "status \(commandBuffer.status.rawValue)")
        }
        let pointer = readableBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: pointer, count: 5_376))
    }

    /// Six ordered command buffers on one queue, committed without intermediate
    /// CPU waits. Queue order carries each 10-layer output dependency.
    func runHiddenBitsGrouped(hidden: MLXArray) throws -> [UInt16] {
        guard hidden.dtype == .bfloat16,
              hidden.shape == [1, 1, 5_376],
              let hiddenBuffer = hidden.asMTLBuffer(device: device, noCopy: true),
              hiddenBuffer.device.registryID == device.registryID
        else {
            throw ProbeError.invalidInput(
                "grouped trunk hidden must be contiguous same-device BF16 [1,1,5376]")
        }

        try beginTokenEncoding()
        var current = Gemma4NativeEncodedTrunkState(
            hidden: hiddenBuffer,
            nextInputNormalized: hiddenBuffer
        )
        var hasNormalizedInput = false
        var commandBuffers: [any MTLCommandBuffer] = []
        commandBuffers.reserveCapacity(Self.groupedLayerRanges.count)
        var readableBuffer: (any MTLBuffer)?

        for (groupIndex, range) in Self.groupedLayerRanges.enumerated() {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder()
            else {
                throw ProbeError.unavailable(
                    "grouped trunk command buffer \(groupIndex)")
            }
            commandBuffer.label = "Gemma4NativeTrunkProbe.group\(groupIndex)"
            encoder.label = "Gemma4NativeTrunkProbe.layers\(range.lowerBound)-\(range.upperBound - 1)"
            current = try encodeLayers(
                range: range,
                into: Gemma4NativeDirectCommandEncoder(encoder),
                inputBuffer: current.hidden,
                normalizedInputBuffer: hasNormalizedInput
                    ? current.nextInputNormalized
                    : nil
            )
            hasNormalizedInput = true
            encoder.endEncoding()

            if range.upperBound == plans.count {
                if current.hidden.storageMode == .private {
                    guard let sharedOutput = device.makeBuffer(
                              length: 5_376 * 2,
                              options: .storageModeShared
                          ),
                          let blit = commandBuffer.makeBlitCommandEncoder()
                    else {
                        throw ProbeError.unavailable(
                            "grouped private trunk diagnostic output blit")
                    }
                    sharedOutput.label =
                        "Gemma4NativeTrunkProbe.groupedDiagnosticSharedOutput"
                    blit.copy(
                        from: current.hidden,
                        sourceOffset: 0,
                        to: sharedOutput,
                        destinationOffset: 0,
                        size: 5_376 * 2
                    )
                    blit.endEncoding()
                    readableBuffer = sharedOutput
                } else {
                    readableBuffer = current.hidden
                }
            }

            commandBuffer.commit()
            commandBuffers.append(commandBuffer)
        }

        guard commandBuffers.count == Self.groupedLayerRanges.count,
              let finalCommandBuffer = commandBuffers.last,
              let readableBuffer
        else {
            throw ProbeError.invalidInput(
                "grouped trunk did not encode all six command buffers")
        }
        finalCommandBuffer.waitUntilCompleted()
        for (groupIndex, commandBuffer) in commandBuffers.enumerated() {
            guard commandBuffer.status == .completed else {
                throw ProbeError.commandFailed(
                    commandBuffer.error?.localizedDescription
                        ?? "group \(groupIndex) status \(commandBuffer.status.rawValue)")
            }
        }
        let pointer = readableBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: pointer, count: 5_376))
    }
}

#endif
