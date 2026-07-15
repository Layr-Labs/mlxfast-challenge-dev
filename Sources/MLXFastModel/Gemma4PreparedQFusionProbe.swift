import Foundation
import MLX

/// Exact model-layer order expected from Gemma 4's five-sliding/one-full
/// attention pattern. Internal so inexpensive accounting tests can qualify the
/// audit state independently of the real transformed-model run.
let gemma4PreparedQExpectedSlidingLayerIndices = (0..<59).filter {
    $0 % 6 != 5
}

struct Gemma4PreparedQAuditAccounting {
    enum AccountingError: Error, Equatable {
        case unexpectedLayer(expected: Int, actual: Int)
        case wrongWordCount(expected: Int, actual: Int)
        case incomplete(expectedLayers: Int, actualLayers: Int)
    }

    private(set) var layerCount = 0
    private(set) var arrayCount = 0
    private(set) var valuesPerSide = 0

    mutating func record(layerIndex: Int, wordCount: Int) throws {
        guard layerCount < gemma4PreparedQExpectedSlidingLayerIndices.count else {
            throw AccountingError.unexpectedLayer(expected: -1, actual: layerIndex)
        }
        let expected = gemma4PreparedQExpectedSlidingLayerIndices[layerCount]
        guard layerIndex == expected else {
            throw AccountingError.unexpectedLayer(expected: expected, actual: layerIndex)
        }
        let expectedWords = 32 * 512 * 256
        guard wordCount == expectedWords else {
            throw AccountingError.wrongWordCount(
                expected: expectedWords, actual: wordCount)
        }
        layerCount += 1
        arrayCount += 2
        valuesPerSide += wordCount
    }

    func finish() throws {
        guard layerCount == gemma4PreparedQExpectedSlidingLayerIndices.count else {
            throw AccountingError.incomplete(
                expectedLayers: gemma4PreparedQExpectedSlidingLayerIndices.count,
                actualLayers: layerCount)
        }
    }
}

enum Gemma4IntegratedAuditMode: Equatable {
    case off
    case compare
    case terminalSentinel
}

enum Gemma4IntegratedAuditCheckpointMode: Equatable {
    case off
    case trace
}

enum Gemma4IntegratedAuditReturnedLogitsEvalMode: Equatable {
    case off
    case force
}

/// Host-only control-flow trace for qualification builds. This deliberately
/// accepts no model value and performs no MLX inspection or evaluation.
@inline(never)
func gemma4IntegratedAuditCheckpoint(_ name: StaticString) {
    guard gemma4IntegratedAuditCheckpointMode == .trace else { return }
    FileHandle.standardError.write(
        Data("integrated_q_sdpa_checkpoint=\(name)\n".utf8)
    )
}

struct Gemma4IntegratedFusionAuditAccounting {
    enum AccountingError: Error, Equatable {
        case unexpectedLayer(expected: Int, actual: Int)
        case incomplete(expectedLayers: Int, actualLayers: Int)
        case attentionTotalMismatch(expected: Int, actual: Int)
        case cacheTotalMismatch(expected: Int, actual: Int)
        case duplicateCompletion
    }

    private(set) var layerCount = 0
    private(set) var attentionWords = 0
    private(set) var cacheWords = 0
    private(set) var completionCount = 0

    mutating func record(
        layerIndex: Int, attentionWordCount: Int, cacheWordCount: Int
    ) throws {
        guard layerCount < gemma4PreparedQExpectedSlidingLayerIndices.count else {
            throw AccountingError.unexpectedLayer(expected: -1, actual: layerIndex)
        }
        let expectedLayer = gemma4PreparedQExpectedSlidingLayerIndices[layerCount]
        guard layerIndex == expectedLayer else {
            throw AccountingError.unexpectedLayer(
                expected: expectedLayer, actual: layerIndex)
        }
        layerCount += 1
        attentionWords += attentionWordCount
        cacheWords += cacheWordCount
    }

    mutating func finish() throws {
        guard completionCount == 0 else {
            throw AccountingError.duplicateCompletion
        }
        guard layerCount == gemma4PreparedQExpectedSlidingLayerIndices.count else {
            throw AccountingError.incomplete(
                expectedLayers: gemma4PreparedQExpectedSlidingLayerIndices.count,
                actualLayers: layerCount)
        }
        guard attentionWords == 209_715_200 else {
            throw AccountingError.attentionTotalMismatch(
                expected: 209_715_200, actual: attentionWords)
        }
        guard cacheWords == 314_572_800 else {
            throw AccountingError.cacheTotalMismatch(
                expected: 314_572_800, actual: cacheWords)
        }
        completionCount += 1
    }
}

/// Compares the two diagnostic boundaries synchronously. The lexical scope is
/// intentional: no MLX graph root or mismatch-only host storage escapes it.
/// Internal visibility permits the qualification tests to exercise the exact
/// lifecycle boundary; production callers use the singleton below.
@inline(never)
func gemma4CompareIntegratedFusionWords(
    layerIndex: Int,
    candidateAttention: MLXArray,
    referenceAttention: MLXArray,
    candidateCache: MLXArray,
    referenceCache: MLXArray
) -> (attentionWordCount: Int, cacheWordCount: Int) {
    autoreleasepool {
        let candidateAttentionBits = candidateAttention.view(dtype: .uint16)
        let referenceAttentionBits = referenceAttention.view(dtype: .uint16)
        let candidateCacheBits = candidateCache.view(dtype: .uint16)
        let referenceCacheBits = referenceCache.view(dtype: .uint16)
        let attentionEqual = arrayEqual(
            candidateAttentionBits, referenceAttentionBits)
        let cacheEqual = arrayEqual(candidateCacheBits, referenceCacheBits)
        eval(attentionEqual, cacheEqual)
        if !attentionEqual.item(Bool.self) {
            let actual = candidateAttentionBits.asArray(UInt16.self)
            let expected = referenceAttentionBits.asArray(UInt16.self)
            let index = expected.indices.first { expected[$0] != actual[$0] }!
            let dimension = index % 256
            let row = index / 256
            preconditionFailure(
                "integrated attention mismatch layer=\(layerIndex) "
                    + "head=\(row / 512) token=\(row % 512) "
                    + "dimension=\(dimension) flat=\(index) "
                    + "expected=\(expected[index]) actual=\(actual[index])")
        }
        if !cacheEqual.item(Bool.self) {
            let actual = candidateCacheBits.asArray(UInt16.self)
            let expected = referenceCacheBits.asArray(UInt16.self)
            let index = expected.indices.first { expected[$0] != actual[$0] }!
            let dimension = index % 256
            let row = index / 256
            let token = row % 768
            let slabHead = row / 768
            preconditionFailure(
                "integrated cache mismatch layer=\(layerIndex) "
                    + "slab=\(slabHead / 16) head=\(slabHead % 16) "
                    + "token=\(token) dimension=\(dimension) flat=\(index) "
                    + "expected=\(expected[index]) actual=\(actual[index])")
        }
        return (referenceAttentionBits.size, referenceCacheBits.size)
    }
}

/// Synchronous, reference-propagating authority for the integrated boundary.
final class Gemma4IntegratedFusionRealModelAudit: @unchecked Sendable {
    static let shared = Gemma4IntegratedFusionRealModelAudit()
    private let lock = NSLock()
    private var accounting = Gemma4IntegratedFusionAuditAccounting()
    private init() {}

    func audit(
        layerIndex: Int,
        candidateAttention: MLXArray,
        referenceAttention: MLXArray,
        candidateCache: MLXArray,
        referenceCache: MLXArray
    ) {
        let counts = gemma4CompareIntegratedFusionWords(
            layerIndex: layerIndex,
            candidateAttention: candidateAttention,
            referenceAttention: referenceAttention,
            candidateCache: candidateCache,
            referenceCache: referenceCache)

        lock.lock()
        if layerIndex == 0 { accounting = Gemma4IntegratedFusionAuditAccounting() }
        var successLine: String?
        do {
            try accounting.record(
                layerIndex: layerIndex,
                attentionWordCount: counts.attentionWordCount,
                cacheWordCount: counts.cacheWordCount)
            if accounting.layerCount == 50 {
                try accounting.finish()
                if gemma4IntegratedQSDPARealModelAuditMode == .terminalSentinel {
                    preconditionFailure(
                        "INTEGRATED_Q_SDPA_AUDIT_COMPLETE_50_209715200_314572800")
                }
                let attentionWords = accounting.attentionWords
                let cacheWords = accounting.cacheWords
                successLine = "integrated_q_sdpa_audit layers=50 attention_matched="
                    + "\(attentionWords)/209715200 cache_matched="
                    + "\(cacheWords)/314572800\n"
            }
        } catch {
            lock.unlock()
            preconditionFailure("integrated Q/SDPA accounting: \(error)")
        }
        lock.unlock()
        if let successLine {
            FileHandle.standardError.write(Data(successLine.utf8))
        }
    }
}

// Qualification-only kernel for the proposed sliding-prefill Q/SDPA fusion.
// This symbol is deliberately not referenced by Gemma4FastEngine (or any other
// production path).  Its grid and row ownership match the future SDPA
// consumer, while each row preserves the promoted producer's arithmetic.
private let gemma4ProbeSlidingPrefillQ16RowKernel = MLXFast.metalKernel(
    name: "gemma4_probe_sliding_prefill_q_16row_v1",
    inputNames: [
        "raw_q", "q_weight", "start_position", "rope_cosines", "rope_sines",
    ],
    outputNames: ["queries"],
    source: """
        constexpr uint kLength = 512;
        constexpr uint kHeadDim = 256;
        constexpr uint kQHeads = 32;
        constexpr uint kRowsPerGroup = 16;
        constexpr uint kRowThreads = 64;
        constexpr uint kReads = 4;
        constexpr uint kPairs = kHeadDim / 2;

        const uint group_thread = thread_position_in_threadgroup.x;
        const uint row_owner = group_thread / kRowThreads;
        const uint row_thread = group_thread % kRowThreads;
        const uint row_simd = row_thread / 32;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint query_start = threadgroup_position_in_grid.z * kRowsPerGroup;
        const int64_t dimension_stride = raw_q_strides[2];

        threadgroup float local_sums[2][32];
        threadgroup float inverse_mean[2];

        // Two independent 64-thread row owners process eight row pairs.  Full
        // barriers make reuse of the pair-local scratch unambiguous.
        for (uint row_pair = 0; row_pair < kRowsPerGroup; row_pair += 2) {
            const uint token = query_start + row_pair + row_owner;
            const device bfloat* row_input = raw_q
                + static_cast<int64_t>(token) * raw_q_strides[1]
                + static_cast<int64_t>(query_head * kHeadDim)
                    * dimension_stride;
            const device bfloat* input = row_input
                + static_cast<int64_t>(row_thread * kReads) * dimension_stride;

            float accumulator = 0.0f;
            for (uint index = 0; index < kReads; ++index) {
                const float value = input[
                    static_cast<int64_t>(index) * dimension_stride];
                accumulator += value * value;
            }
            accumulator = simd_sum(accumulator);

            if (row_simd == 0) {
                local_sums[row_owner][thread_index_in_simdgroup] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_index_in_simdgroup == 0) {
                local_sums[row_owner][row_simd] = accumulator;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row_simd == 0) {
                accumulator = simd_sum(
                    local_sums[row_owner][thread_index_in_simdgroup]);
                if (thread_index_in_simdgroup == 0) {
                    inverse_mean[row_owner] = metal::precise::rsqrt(
                        accumulator / kHeadDim + 1.0e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            device bfloat* output = queries
                + (static_cast<int64_t>(query_head) * kLength + token)
                    * kHeadDim;
            for (uint pair = row_thread; pair < kPairs; pair += kRowThreads) {
                const bfloat normalized_left = static_cast<bfloat>(
                    row_input[static_cast<int64_t>(pair) * dimension_stride]
                        * inverse_mean[row_owner]);
                const bfloat normalized_right = static_cast<bfloat>(
                    row_input[static_cast<int64_t>(pair + kPairs)
                        * dimension_stride] * inverse_mean[row_owner]);
                const bfloat weighted_left = q_weight[pair] * normalized_left;
                const bfloat weighted_right =
                    q_weight[pair + kPairs] * normalized_right;
                const uint rope_index =
                    (static_cast<uint>(start_position) + token) * kPairs + pair;
                const float cosine = rope_cosines[rope_index];
                const float sine = rope_sines[rope_index];
                const float left = static_cast<float>(weighted_left);
                const float right = static_cast<float>(weighted_right);
                output[pair] = static_cast<bfloat>(
                    left * cosine - right * sine);
                output[pair + kPairs] = static_cast<bfloat>(
                    left * sine + right * cosine);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """,
    header: "using namespace metal;",
    ensureRowContiguous: false
)

/// Reconstructs the production sliding-L512 prepared-Q tensor for raw-bit
/// differential qualification.  This has no production callers.
func gemma4ProbeSlidingPreparedQ(
    rawQueries: MLXArray,
    qNormWeight: MLXArray,
    startPosition: Int,
    ropeCosines: MLXArray,
    ropeSines: MLXArray
) -> MLXArray {
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawQueries.shape == [1, 512, 8192])
    precondition(qNormWeight.dtype == .bfloat16)
    precondition(qNormWeight.shape == [256])
    precondition(startPosition == 0)
    precondition(ropeCosines.dtype == .float32)
    precondition(ropeSines.dtype == .float32)
    precondition(ropeCosines.shape.count == 2)
    precondition(ropeSines.shape.count == 2)
    precondition(ropeCosines.shape[0] >= 512 && ropeCosines.shape[1] == 128)
    precondition(ropeSines.shape[0] >= 512 && ropeSines.shape[1] == 128)

    return gemma4ProbeSlidingPrefillQ16RowKernel(
        [
            rawQueries, qNormWeight, MLXArray(Int32(startPosition)),
            ropeCosines, ropeSines,
        ],
        grid: (128, 32, 32),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 32, 512, 256]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Qualification-only, synchronously evaluated comparison over real projected
/// tensors. The candidate is never returned; only the promoted direct producer
/// is allowed to propagate to attention.
final class Gemma4PreparedQRealModelAudit: @unchecked Sendable {
    static let shared = Gemma4PreparedQRealModelAudit()

    private let lock = NSLock()
    private var accounting = Gemma4PreparedQAuditAccounting()

    private init() {}

    func audit(
        layerIndex: Int,
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray?,
        preparation: FusedAttentionRMSPreparation,
        capacity: Int
    ) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }

        if layerIndex == 0 {
            accounting = Gemma4PreparedQAuditAccounting()
        }
        precondition(layerIndex % 6 != 5, "prepared-Q audit received full layer")
        precondition(rawQueries.dtype == .bfloat16)
        precondition(rawQueries.shape == [1, 512, 8192])
        precondition(rawKeys.dtype == .bfloat16)
        precondition(rawKeys.shape == [1, 512, 4096])
        if let rawValues {
            precondition(rawValues.dtype == .bfloat16)
            precondition(rawValues.shape == [1, 512, 4096])
        }
        precondition(preparation.isSliding && preparation.headDim == 256)
        precondition(capacity >= 512)

        let candidate = gemma4ProbeSlidingPreparedQ(
            rawQueries: rawQueries,
            qNormWeight: preparation.qNormWeight,
            startPosition: 0,
            ropeCosines: preparation.ropeCosines,
            ropeSines: preparation.ropeSines
        )
        let reference = preparation
            .callDirectCombinedQKVPrefillForPreparedQAudit(
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: 0,
                length: 512,
                capacity: capacity
            )
        let candidateBits = candidate.view(dtype: .uint16)
        let referenceBits = reference.view(dtype: .uint16)
        let equal = arrayEqual(candidateBits, referenceBits)
        eval(equal)
        guard equal.item(Bool.self) else {
            let actual = candidateBits.asArray(UInt16.self)
            let expected = referenceBits.asArray(UInt16.self)
            let index = expected.indices.first { expected[$0] != actual[$0] }!
            let dimension = index % 256
            let row = index / 256
            let token = row % 512
            let head = row / 512
            preconditionFailure(
                "prepared-Q real-model mismatch layer=\(layerIndex) "
                    + "head=\(head) token=\(token) dimension=\(dimension) "
                    + "expected=\(expected[index]) actual=\(actual[index])"
            )
        }

        do {
            try accounting.record(
                layerIndex: layerIndex, wordCount: referenceBits.size)
            if accounting.layerCount
                == gemma4PreparedQExpectedSlidingLayerIndices.count
            {
                try accounting.finish()
                FileHandle.standardError.write(Data((
                    "layers=\(accounting.layerCount) "
                        + "arrays=\(accounting.arrayCount) "
                        + "values_per_side=\(accounting.valuesPerSide) "
                        + "matched=\(accounting.valuesPerSide)\n"
                ).utf8))
            }
        } catch {
            preconditionFailure("prepared-Q real-model accounting: \(error)")
        }
        return reference
    }
}
