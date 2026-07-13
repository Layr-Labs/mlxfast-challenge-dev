import Foundation
import MLX

enum Gemma4PromptLookupPendingResolution: Equatable {
    case none
    case accepted
    case mismatch
}

/// Request-local token history for generic one-token prompt lookup.
///
/// A draft is the token following the most recent prior occurrence of the
/// longest suffix up to six tokens. Pending drafts are not added to history
/// until the caller presents that token on the next model invocation.
struct Gemma4PromptLookupState: Equatable {
    private(set) var tokens: [Int32] = []
    private(set) var pendingDrafts: [Int32] = []
    private(set) var acceptedPrefixCount = 0

    // Compatibility view used by the k=1 rollback path.
    var pendingDraft: Int32? { pendingDrafts.first }

    mutating func reset(tokens: [Int32] = []) {
        self.tokens = tokens
        pendingDrafts.removeAll(keepingCapacity: true)
        acceptedPrefixCount = 0
    }

    mutating func recordInput(_ token: Int32) {
        precondition(pendingDrafts.isEmpty)
        tokens.append(token)
    }

    func draft() -> Int32? { gemma4PromptLookupDraft(tokens: tokens) }

    func draft(appending token: Int32) -> Int32? {
        gemma4PromptLookupDraft(tokens: tokens + [token])
    }

    func drafts(appending token: Int32) -> [Int32] {
        guard let first = gemma4PromptLookupDraft(tokens: tokens + [token]) else {
            return []
        }
        guard let second = gemma4PromptLookupDraft(tokens: tokens + [token, first]) else {
            return [first]
        }
        return [first, second]
    }

    mutating func setPendingDraft(_ token: Int32) {
        setPendingDrafts([token])
    }

    mutating func setPendingDrafts(_ drafts: [Int32]) {
        precondition(pendingDrafts.isEmpty && (1...2).contains(drafts.count))
        pendingDrafts = drafts
        acceptedPrefixCount = 0
    }

    mutating func resolvePending(actualInput: Int32) -> Gemma4PromptLookupPendingResolution {
        guard let pending = pendingDrafts.first else { return .none }
        guard actualInput == pending else {
            pendingDrafts.removeAll(keepingCapacity: true)
            acceptedPrefixCount = 0
            return .mismatch
        }
        pendingDrafts.removeFirst()
        tokens.append(actualInput)
        acceptedPrefixCount += 1
        return .accepted
    }

    mutating func cancelPending() {
        pendingDrafts.removeAll(keepingCapacity: true)
        acceptedPrefixCount = 0
    }
}

/// Fixed generic N=6, k=1 lookup. Suffix lengths are tried longest-first, then
/// prior starts newest-first, exactly matching the offline policy evaluator.
func gemma4PromptLookupDraft(tokens: [Int32]) -> Int32? {
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
            if matches {
                return tokens[start + length]
            }
        }
    }
    return nil
}

@inline(__always)
func gemma4PromptLookupEnvironmentEnabled(_ environment: [String: String]) -> Bool {
    guard let raw = environment["DARKBLOOM_PROMPT_LOOKUP"] else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

@inline(__always)
func gemma4PromptLookupK2Enabled(_ environment: [String: String]) -> Bool {
    guard let raw = environment["DARKBLOOM_PROMPT_LOOKUP_K2"] else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

@inline(__always)
func gemma4PromptLookupVerificationEnabled(_ environment: [String: String]) -> Bool {
    guard let raw = environment["DARKBLOOM_VERIFY_PROMPT_LOOKUP_BITS"] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

func gemma4PromptLookupInputTokens(_ input: MLXArray) -> [Int32]? {
    guard input.dtype == .int32,
          input.ndim == 2,
          input.dim(0) == 1,
          input.dim(1) > 0
    else { return nil }
    return input.asArray(Int32.self)
}

func gemma4VerifyPromptLookupPair(
    _ candidate: MLXArray,
    reference0: MLXArray,
    reference1: MLXArray,
    candidateOffsets: [Int],
    referenceOffsets: [Int]
) {
    precondition(candidate.dtype == .float32 && candidate.shape == [2, 262_144])
    precondition(
        reference0.dtype == .float32 && reference0.shape == [1, 1, 262_144])
    precondition(
        reference1.dtype == .float32 && reference1.shape == [1, 1, 262_144])
    let firstMatches = arrayEqual(
        candidate[0..<1, 0...].reshaped(1, 1, 262_144).view(dtype: .uint32),
        reference0.view(dtype: .uint32)
    )
    let secondMatches = arrayEqual(
        candidate[1..<2, 0...].reshaped(1, 1, 262_144).view(dtype: .uint32),
        reference1.view(dtype: .uint32)
    )
    eval(firstMatches, secondMatches)
    precondition(
        firstMatches.item(Bool.self) && secondMatches.item(Bool.self),
        "exact prompt-lookup pair differs from serialized decode"
    )
    precondition(!candidateOffsets.isEmpty)
    precondition(candidateOffsets.allSatisfy { $0 == candidateOffsets[0] })
    precondition(referenceOffsets == candidateOffsets)
}
