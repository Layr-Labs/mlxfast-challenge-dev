import Foundation
import MLX

enum Gemma4PromptLookupPendingResolution: Equatable {
    case none
    case accepted
    case mismatch
}

struct Gemma4PromptLookupDraft: Equatable {
    let tokens: [Int32]

    init(tokens: [Int32]) {
        precondition((1...2).contains(tokens.count))
        self.tokens = tokens
    }
}

/// Request-local token history for generic prompt lookup. Drafts do not enter
/// committed history until the caller presents them on later API invocations.
struct Gemma4PromptLookupState: Equatable {
    private(set) var tokens: [Int32] = []
    private(set) var pendingDrafts: [Int32] = []

    mutating func reset(tokens: [Int32] = []) {
        self.tokens = tokens
        pendingDrafts.removeAll(keepingCapacity: true)
    }

    mutating func recordInput(_ token: Int32) {
        precondition(pendingDrafts.isEmpty)
        tokens.append(token)
    }

    func draft() -> Gemma4PromptLookupDraft? {
        gemma4PromptLookupDraft(tokens: tokens)
    }

    func draft(appending token: Int32) -> Gemma4PromptLookupDraft? {
        gemma4PromptLookupDraft(tokens: tokens + [token])
    }

    mutating func setPendingDrafts(_ tokens: [Int32]) {
        precondition(pendingDrafts.isEmpty)
        precondition((1...2).contains(tokens.count))
        pendingDrafts = tokens
    }

    mutating func resolvePending(actualInput: Int32) -> Gemma4PromptLookupPendingResolution {
        guard let pending = pendingDrafts.first else { return .none }
        guard actualInput == pending else {
            pendingDrafts.removeAll(keepingCapacity: true)
            return .mismatch
        }
        pendingDrafts.removeFirst()
        tokens.append(actualInput)
        return .accepted
    }

    mutating func cancelPending() {
        pendingDrafts.removeAll(keepingCapacity: true)
    }
}

/// Fixed generic N=6 lookup. The first draft policy is exactly the promoted
/// longest-first/newest-first k=1 policy. Up to one additional continuation is
/// taken only from that same occurrence and may not overlap the current suffix.
func gemma4PromptLookupDraft(tokens: [Int32]) -> Gemma4PromptLookupDraft? {
    guard tokens.count >= 2 else { return nil }
    let maximumDraftCount = gemma4PromptLookupK2Enabled ? 2 : 1
    for length in stride(from: min(6, tokens.count), through: 1, by: -1) {
        let suffixStart = tokens.count - length
        guard suffixStart > 0 else { continue }
        for start in stride(from: suffixStart - 1, through: 0, by: -1) {
            var matches = true
            for index in 0..<length where tokens[start + index] != tokens[suffixStart + index] {
                matches = false
                break
            }
            if matches {
                let available = suffixStart - (start + length)
                let count = min(maximumDraftCount, available)
                guard count > 0 else { continue }
                return Gemma4PromptLookupDraft(
                    tokens: Array(tokens[(start + length)..<(start + length + count)]))
            }
        }
    }
    return nil
}

/// Promoted scalar lookup retained for tests and the process-start rollback.
func gemma4PromptLookupScalarDraft(tokens: [Int32]) -> Int32? {
    guard tokens.count >= 2 else { return nil }
    for length in stride(from: min(6, tokens.count), through: 1, by: -1) {
        let suffixStart = tokens.count - length
        guard suffixStart > 0 else { continue }
        for start in stride(from: suffixStart - 1, through: 0, by: -1) {
            var matches = true
            for index in 0..<length where tokens[start + index] != tokens[suffixStart + index] {
                matches = false
                break
            }
            if matches { return tokens[start + length] }
        }
    }
    return nil
}

private let gemma4PromptLookupK2Enabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PROMPT_LOOKUP_K2"] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

@inline(__always)
func gemma4PromptLookupEnvironmentEnabled(_ environment: [String: String]) -> Bool {
    guard let raw = environment["DARKBLOOM_PROMPT_LOOKUP"] else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

@inline(__always)
func gemma4PromptLookupVerificationEnabled(_ environment: [String: String]) -> Bool {
    guard let raw = environment["DARKBLOOM_VERIFY_PROMPT_LOOKUP_BITS"] else { return false }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

func gemma4PromptLookupInputTokens(_ input: MLXArray) -> [Int32]? {
    guard input.dtype == .int32, input.ndim == 2, input.dim(0) == 1, input.dim(1) > 0 else {
        return nil
    }
    return input.asArray(Int32.self)
}

func gemma4VerifyPromptLookupRows(
    _ candidate: MLXArray,
    references: [MLXArray],
    candidateOffsets: [Int],
    referenceOffsets: [Int]
) {
    precondition(candidate.dtype == .float32)
    precondition((2...3).contains(candidate.dim(0)) && candidate.dim(1) == 262_144)
    precondition(references.count == candidate.dim(0))
    var comparisons: [MLXArray] = []
    for row in references.indices {
        precondition(references[row].dtype == .float32)
        precondition(references[row].shape == [1, 1, 262_144])
        comparisons.append(arrayEqual(
            candidate[row..<(row + 1), 0...].reshaped(1, 1, 262_144).view(dtype: .uint32),
            references[row].view(dtype: .uint32)))
    }
    eval(comparisons)
    precondition(comparisons.allSatisfy { $0.item(Bool.self) },
                 "exact prompt-lookup rows differ from serialized decode")
    precondition(!candidateOffsets.isEmpty)
    precondition(candidateOffsets.allSatisfy { $0 == candidateOffsets[0] })
    precondition(referenceOffsets == candidateOffsets)
}

func gemma4VerifyPromptLookupPair(
    _ candidate: MLXArray,
    reference0: MLXArray,
    reference1: MLXArray,
    candidateOffsets: [Int],
    referenceOffsets: [Int]
) {
    gemma4VerifyPromptLookupRows(candidate, references: [reference0, reference1],
                                 candidateOffsets: candidateOffsets,
                                 referenceOffsets: referenceOffsets)
}
