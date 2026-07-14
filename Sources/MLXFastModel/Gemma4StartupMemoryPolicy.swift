import Darwin
import Foundation
import MLX

/// Selects a model-startup profile from the machine's physical-memory budget.
///
/// The optimized Gemma runtime can retain roughly 14.5 GiB of alternate
/// co-tiled/combined weight layouts in addition to the ~16.9 GiB checkpoint.
/// That full profile is appropriate for the 128 GiB ranked runner, but leaves
/// too little headroom for macOS, warmup activations, KV state, and Metal
/// buffers on the documented 36 GiB local minimum.
public struct Gemma4StartupMemoryPolicy: Equatable, Sendable {
    public static let fullProfileMinimumPhysicalMemoryBytes = UInt64(64) << 30

    public let isLowMemory: Bool
    public let cacheLimitBytes: Int
    public let maxMegabytesPerCommandBuffer: Int
    public let maxOperationsPerCommandBuffer: Int
    public let clearAllocatorCacheAfterWarmup: Bool
    public let environmentOverrides: [String: String]

    public static func resolve(physicalMemoryBytes: UInt64) -> Gemma4StartupMemoryPolicy {
        if physicalMemoryBytes < fullProfileMinimumPhysicalMemoryBytes {
            return Gemma4StartupMemoryPolicy(
                isLowMemory: true,
                cacheLimitBytes: 6 << 30,
                maxMegabytesPerCommandBuffer: 128,
                maxOperationsPerCommandBuffer: 64,
                clearAllocatorCacheAfterWarmup: true,
                environmentOverrides: [
                    // Avoid retaining compiled closures and the largest alternate
                    // weight layouts in addition to the source model.
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
                ]
            )
        }

        return Gemma4StartupMemoryPolicy(
            isLowMemory: false,
            cacheLimitBytes: 32 << 30,
            maxMegabytesPerCommandBuffer: 320,
            maxOperationsPerCommandBuffer: 128,
            clearAllocatorCacheAfterWarmup: false,
            environmentOverrides: [:]
        )
    }

    func apply() {
        setenv(
            "MLX_MAX_MB_PER_BUFFER",
            String(maxMegabytesPerCommandBuffer),
            1
        )
        setenv(
            "MLX_MAX_OPS_PER_BUFFER",
            String(maxOperationsPerCommandBuffer),
            1
        )
        for (name, value) in environmentOverrides {
            setenv(name, value, 1)
        }
        Memory.cacheLimit = cacheLimitBytes
    }
}
