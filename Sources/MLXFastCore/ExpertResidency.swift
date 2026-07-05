import Foundation

/// Policy for keeping ALL routed-expert tensors RAM-resident instead of
/// streaming them from SSD during forwards.
///
/// The official benchmark contract targets an Apple M3 Ultra runner with at
/// least 256 GB of unified memory, where the full 4-bit expert set (~141 GiB)
/// plus dense weights and runtime buffers fit comfortably in RAM. On that
/// hardware the baseline loads every expert tensor once at (untimed) loader
/// initialization and serves all forwards from memory — SSD streaming no
/// longer appears in any scored window.
///
/// Machines below the threshold (developer laptops, small CI runners) fall
/// back to the previous SSD-streaming runtime automatically so local
/// iteration keeps working; those timings are directional only, like all
/// local numbers.
///
/// The decision is a pure function of the environment and physical memory so
/// the trusted harness parent and the sandboxed worker independently compute
/// the SAME answer for the same machine: the parent uses it to pick the
/// right diagnostics/gates, and submitted code inside the worker cannot lie
/// to the parent about which mode was sanctioned.
public enum ExpertResidencyPolicy {
    /// Below this the full expert set cannot be RAM-resident next to the
    /// dense weights and runtime buffers. The official runner has >= 256 GB;
    /// the margin down to 192 GiB admits other large-memory Macs without
    /// letting a 128 GiB machine thrash.
    public static let minimumPhysicalMemoryBytes: UInt64 = 192 << 30

    public static let environmentKey = "MLXFAST_EXPERT_RESIDENT"

    /// Diagnostic `bandwidth_source` label for runs whose scored windows read
    /// no expert bytes because the full expert set is RAM-resident.
    public static let residentBandwidthSource = "ram_resident_experts"

    /// True when this process should load all expert tensors RAM-resident.
    /// `MLXFAST_EXPERT_RESIDENT=1/0` forces the mode either way (fixtures,
    /// debugging); unset means auto by physical memory.
    public static func fullResidencyEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Bool {
        switch environment[environmentKey]?.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return physicalMemory >= minimumPhysicalMemoryBytes
        }
    }
}
