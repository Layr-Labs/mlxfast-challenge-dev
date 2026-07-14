import Foundation
@testable import MLXFastModel
import Testing

@Test
func startupMemoryPolicyProtectsDocumented36GiBLocalMachine() {
    let policy = Gemma4StartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )

    #expect(policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 6 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 128)
    #expect(policy.maxOperationsPerCommandBuffer == 64)
    #expect(policy.clearAllocatorCacheAfterWarmup)
    #expect(policy.environmentOverrides == [
        "MLX_COMPILED_DECODE": "0",
        "MLX_COMPILED_MLP_TAIL": "0",
        "DARKBLOOM_COMPILED_DECODE": "0",
        "MLXFAST_COMBINED_ATTENTION_PREFILL": "0",
        "MLXFAST_VERIFY_COMBINED_ATTENTION_PREFILL": "0",
        "MLXFAST_FUSED_QKV": "0",
        "MLXFAST_FUSED_FULL_QK": "0",
        "MLXFAST_FUSED_GATE_UP": "0",
        "MLXFAST_INDEXED_DOWN": "0",
        "MLXFAST_INDEXED_OUTPUT_FAST": "0",
        "DARKBLOOM_PACKED_GATE_UP_INDICES": "0",
        "DARKBLOOM_VERIFY_PACKED_GATE_UP_BITS": "0",
        "DARKBLOOM_GATE_UP_COTILED_FIXED12": "0",
        "DARKBLOOM_VERIFY_GATE_UP_COTILED_FIXED12_BITS": "0",
        "MLXFAST_PACKED_DOWN_INDICES": "0",
        "MLXFAST_VERIFY_PACKED_DOWN_BITS": "0",
        "DARKBLOOM_DOWN_COTILED_FIXED12": "0",
        "DARKBLOOM_VERIFY_DOWN_COTILED_FIXED12_BITS": "0",
        "MLXFAST_PACKED_OUTPUT_INDICES": "0",
        "MLXFAST_VERIFY_PACKED_OUTPUT_BITS": "0",
        "DARKBLOOM_OUTPUT_COTILED": "0",
        "DARKBLOOM_OUTPUT_COTILED_SLIDING": "0",
        "DARKBLOOM_OUTPUT_COTILED_FULL": "0",
        "DARKBLOOM_VERIFY_OUTPUT_COTILED_BITS": "0",
        "DARKBLOOM_VERIFY_OUTPUT_STOCK_BITS": "0",
        "DARKBLOOM_TIED_HEAD_QMV": "0",
        "DARKBLOOM_VERIFY_TIED_HEAD_BITS": "0",
    ])
}

@Test
func startupMemoryPolicyKeepsRanked128GiBProfile() {
    let policy = Gemma4StartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(128) << 30
    )

    #expect(!policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 32 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 320)
    #expect(policy.maxOperationsPerCommandBuffer == 128)
    #expect(!policy.clearAllocatorCacheAfterWarmup)
    #expect(policy.environmentOverrides.isEmpty)
}

@Test
func startupMemoryPolicyUses64GiBAsFullProfileBoundary() {
    #expect(
        Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: (UInt64(64) << 30) - 1
        ).isLowMemory
    )
    #expect(
        !Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(64) << 30
        ).isLowMemory
    )
}

@Test
func startupMemoryPolicyIsAppliedBeforeModelLoadAndCleansWarmupCache() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastModel/Gemma4RuntimeWeights.swift",
        encoding: .utf8
    )
    let apply = try #require(source.range(of: "policy.apply()"))
    let load = try #require(source.range(of: "loadLibraryModel("))
    #expect(apply.lowerBound < load.lowerBound)

    let warmup = try #require(source.range(of: "Self.warmLibraryModel(model)"))
    let cleanup = try #require(
        source.range(
            of: "if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true"
        )
    )
    let clear = try #require(
        source.range(of: "Memory.clearCache()", range: cleanup.lowerBound..<source.endIndex)
    )
    #expect(warmup.lowerBound < cleanup.lowerBound)
    #expect(cleanup.lowerBound < clear.lowerBound)
}
