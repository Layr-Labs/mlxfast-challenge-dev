import Foundation
import MLX
import MLXFastCore

public struct DeepSeekCachedKV {
    public let kv: MLXArray
    public let keyOffset: Int

    public init(kv: MLXArray, keyOffset: Int) {
        self.kv = kv
        self.keyOffset = keyOffset
    }
}

public final class DeepSeekLocalKVCache {
    public let maxSize: Int
    public private(set) var offset: Int
    public private(set) var startPosition: Int
    private var kv: MLXArray?

    public init(maxSize: Int, offset: Int = 0, startPosition: Int = 0) {
        self.maxSize = maxSize
        self.offset = offset
        self.startPosition = startPosition
    }

    public func updateAndFetch(_ newKV: MLXArray) throws -> DeepSeekCachedKV {
        guard maxSize > 0 else {
            throw MLXFastError.invalidInput("local KV cache maxSize must be positive")
        }
        guard newKV.shape.count == 4 else {
            throw MLXFastError.invalidInput("local KV cache input must have shape [batch, heads, length, hidden]")
        }
        let incoming = newKV.shape[2]
        guard incoming > 0 else {
            if let kv {
                return DeepSeekCachedKV(kv: kv, keyOffset: startPosition)
            }
            return DeepSeekCachedKV(kv: newKV, keyOffset: offset)
        }

        let combinedStart = startPosition
        let combined = kv.map { concatenated([$0, newKV], axis: 2) } ?? newKV
        offset += incoming

        if combined.shape[2] > maxSize {
            let drop = combined.shape[2] - maxSize
            kv = combined[0..., 0..., drop..., 0...]
            startPosition = combinedStart + drop
        } else {
            kv = combined
            startPosition = combinedStart
        }

        return DeepSeekCachedKV(kv: combined, keyOffset: combinedStart)
    }

    func arraysForMaterialization() -> [MLXArray] {
        kv.map { [$0] } ?? []
    }
}

public struct DeepSeekPoolingWindow {
    public let kv: MLXArray
    public let gate: MLXArray
    public let baseOffset: Int
}

public final class DeepSeekPoolingCache {
    public let ratio: Int
    private var bufferedKV: MLXArray?
    private var bufferedGate: MLXArray?
    private var pooled: MLXArray?

    public init(ratio: Int) {
        self.ratio = ratio
    }

    public var pooledLength: Int {
        pooled?.shape[1] ?? 0
    }

    public func accumulateWindows(
        kv newKV: MLXArray,
        gate newGate: MLXArray,
        offset: Int
    ) throws -> DeepSeekPoolingWindow {
        guard ratio > 0 else {
            throw MLXFastError.invalidInput("pooling cache ratio must be positive")
        }
        guard newKV.shape.count == 3, newGate.shape.count == 3 else {
            throw MLXFastError.invalidInput("pooling cache inputs must have shape [batch, length, hidden]")
        }
        guard newKV.shape[0] == newGate.shape[0], newKV.shape[1] == newGate.shape[1] else {
            throw MLXFastError.invalidInput("pooling cache kv/gate batch and length must match")
        }

        let previousRemainder = bufferedKV?.shape[1] ?? 0
        let combinedKV = bufferedKV.map { concatenated([$0, newKV], axis: 1) } ?? newKV
        let combinedGate = bufferedGate.map { concatenated([$0, newGate], axis: 1) } ?? newGate
        let total = combinedKV.shape[1]
        let usable = (total / ratio) * ratio
        let remainder = total - usable

        let readyKV: MLXArray
        let readyGate: MLXArray
        if usable > 0 {
            readyKV = combinedKV[0..., 0..<usable, 0...]
            readyGate = combinedGate[0..., 0..<usable, 0...]
        } else {
            readyKV = zeros([newKV.shape[0], 0, newKV.shape[2]], dtype: newKV.dtype)
            readyGate = zeros([newGate.shape[0], 0, newGate.shape[2]], dtype: newGate.dtype)
        }

        if remainder > 0 {
            bufferedKV = combinedKV[0..., usable..., 0...]
            bufferedGate = combinedGate[0..., usable..., 0...]
        } else {
            bufferedKV = nil
            bufferedGate = nil
        }

        return DeepSeekPoolingWindow(
            kv: readyKV,
            gate: readyGate,
            baseOffset: offset - previousRemainder
        )
    }

    public func updateAndFetch(_ newPooled: MLXArray) -> MLXArray {
        guard newPooled.shape[1] > 0 else {
            if let pooled {
                return pooled
            }
            return zeros([newPooled.shape[0], 0, newPooled.shape[2]], dtype: newPooled.dtype)
        }
        if let existing = pooled {
            pooled = concatenated([existing, newPooled], axis: 1)
        } else {
            pooled = newPooled
        }
        return pooled!
    }

    public func makeMask(queryLength: Int, offset: Int) -> MLXArray? {
        guard let pooled, queryLength > 1 else {
            return nil
        }
        let poolIndex = arange(pooled.shape[1], dtype: .int32)
        let queryIndex = arange(offset + 1, offset + queryLength + 1, dtype: .int32)
            .expandedDimensions(axis: 1)
        return poolIndex .< queryIndex.floorDivide(Int32(ratio))
    }

    func arraysForMaterialization() -> [MLXArray] {
        [bufferedKV, bufferedGate, pooled].compactMap { $0 }
    }
}

public struct DeepSeekDeferredInputChunk {
    public let x: MLXArray
    public let offset: Int
}

public final class DeepSeekDeferredInputCache {
    private var chunks: [DeepSeekDeferredInputChunk] = []
    private var nextOffset: Int?
    private var batchSize: Int?
    private var hiddenSize: Int?
    private var dtype: DType?

    public init() {}

    public var isEmpty: Bool {
        chunks.isEmpty
    }

    public func append(_ x: MLXArray, offset: Int) throws {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("deferred input cache expects shape [batch, length, hidden]")
        }
        let length = x.shape[1]
        guard length > 0 else {
            return
        }
        if let batchSize {
            guard x.shape[0] == batchSize else {
                throw MLXFastError.invalidInput(
                    "deferred input cache received batch \(x.shape[0]); expected \(batchSize)"
                )
            }
        } else {
            batchSize = x.shape[0]
        }
        if let hiddenSize {
            guard x.shape[2] == hiddenSize else {
                throw MLXFastError.invalidInput(
                    "deferred input cache received hidden \(x.shape[2]); expected \(hiddenSize)"
                )
            }
        } else {
            hiddenSize = x.shape[2]
        }
        if let dtype {
            guard x.dtype == dtype else {
                throw MLXFastError.invalidInput(
                    "deferred input cache received dtype \(x.dtype); expected \(dtype)"
                )
            }
        } else {
            dtype = x.dtype
        }
        if let nextOffset {
            guard offset == nextOffset else {
                throw MLXFastError.invalidInput(
                    "deferred input cache received offset \(offset); expected \(nextOffset)"
                )
            }
        }
        chunks.append(DeepSeekDeferredInputChunk(x: x, offset: offset))
        nextOffset = offset + length
    }

    public func drain() -> [DeepSeekDeferredInputChunk] {
        let pending = chunks
        chunks.removeAll(keepingCapacity: true)
        nextOffset = nil
        batchSize = nil
        hiddenSize = nil
        dtype = nil
        return pending
    }

    public func drainMerged() -> DeepSeekDeferredInputChunk? {
        let pending = drain()
        guard let first = pending.first else {
            return nil
        }
        guard pending.count > 1 else {
            return first
        }
        return DeepSeekDeferredInputChunk(
            x: concatenated(pending.map(\.x), axis: 1),
            offset: first.offset
        )
    }

    func arraysForMaterialization() -> [MLXArray] {
        chunks.map(\.x)
    }
}

public final class DeepSeekLayerCache {
    public let local: DeepSeekLocalKVCache
    public let pooled: DeepSeekPoolingCache?
    public let indexPooled: DeepSeekPoolingCache?
    public let deferredIndexInput: DeepSeekDeferredInputCache?

    public init(
        local: DeepSeekLocalKVCache,
        pooled: DeepSeekPoolingCache?,
        indexPooled: DeepSeekPoolingCache?,
        deferredIndexInput: DeepSeekDeferredInputCache? = nil
    ) {
        self.local = local
        self.pooled = pooled
        self.indexPooled = indexPooled
        self.deferredIndexInput = deferredIndexInput
    }

    func arraysForMaterialization() -> [MLXArray] {
        local.arraysForMaterialization()
            + (pooled?.arraysForMaterialization() ?? [])
            + (indexPooled?.arraysForMaterialization() ?? [])
            + (deferredIndexInput?.arraysForMaterialization() ?? [])
    }
}

public final class DeepSeekModelCache {
    public let layers: [DeepSeekLayerCache]

    public init(config: DeepSeekConfig) {
        self.layers = config.compressRatios.map { ratio in
            DeepSeekLayerCache(
                local: DeepSeekLocalKVCache(maxSize: config.slidingWindow),
                pooled: ratio == 0 ? nil : DeepSeekPoolingCache(ratio: ratio),
                indexPooled: ratio == 4 ? DeepSeekPoolingCache(ratio: ratio) : nil,
                deferredIndexInput: ratio == 4 ? DeepSeekDeferredInputCache() : nil
            )
        }
    }

    func arraysForMaterialization() -> [MLXArray] {
        layers.flatMap { $0.arraysForMaterialization() }
    }

    public func materializeCachedState() {
        for array in arraysForMaterialization() {
            eval(array)
        }
    }
}
