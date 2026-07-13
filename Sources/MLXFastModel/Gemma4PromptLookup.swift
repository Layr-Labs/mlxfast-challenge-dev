import Foundation
import MLX

enum Gemma4PromptLookupPendingResolution: Equatable {
    case none
    case accepted
    case mismatch
}

/// Request-local token history for generic prompt lookup with up to two drafts.
///
/// Drafts are the tokens following the most recent prior occurrence of the
/// longest suffix up to six tokens. Pending drafts are not added to history
/// until the caller presents each token on a subsequent model invocation.
struct Gemma4PromptLookupState: Equatable {
    private(set) var tokens: [Int32] = []
    private(set) var pendingDrafts: [Int32] = []

    /// Compatibility view for the promoted one-draft model integration.
    var pendingDraft: Int32? {
        pendingDrafts.first
    }

    mutating func reset(tokens: [Int32] = []) {
        self.tokens = tokens
        pendingDrafts.removeAll(keepingCapacity: true)
    }

    mutating func recordInput(_ token: Int32) {
        precondition(pendingDrafts.isEmpty)
        tokens.append(token)
    }

    func drafts(maxCount: Int = 2) -> [Int32] {
        gemma4PromptLookupDrafts(tokens: tokens, maxCount: maxCount)
    }

    func drafts(appending token: Int32, maxCount: Int = 2) -> [Int32] {
        gemma4PromptLookupDrafts(tokens: tokens + [token], maxCount: maxCount)
    }

    /// Compatibility wrappers for the promoted one-draft model integration.
    func draft() -> Int32? {
        drafts(maxCount: 1).first
    }

    func draft(appending token: Int32) -> Int32? {
        drafts(appending: token, maxCount: 1).first
    }

    mutating func setPendingDrafts(_ tokens: [Int32]) {
        precondition(pendingDrafts.isEmpty)
        precondition((1...2).contains(tokens.count))
        pendingDrafts = tokens
    }

    mutating func setPendingDraft(_ token: Int32) {
        setPendingDrafts([token])
    }

    mutating func resolvePending(actualInput: Int32) -> Gemma4PromptLookupPendingResolution {
        guard let expected = pendingDrafts.first else { return .none }
        guard actualInput == expected else {
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

/// Fixed generic N=6 lookup returning up to `maxCount` contiguous continuation
/// tokens from the selected prior match. Suffix lengths are tried longest-first,
/// then prior starts newest-first, exactly matching the promoted k=1 policy.
/// Consequently the first returned draft is bit-for-bit policy-compatible with
/// `gemma4PromptLookupDraft(tokens:)`.
func gemma4PromptLookupDrafts(tokens: [Int32], maxCount: Int = 2) -> [Int32] {
    precondition(maxCount > 0)
    guard tokens.count >= 2 else { return [] }
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
                let continuationStart = start + length
                let continuationCount = min(
                    maxCount,
                    tokens.count - continuationStart
                )
                let continuationEnd = continuationStart + continuationCount
                return Array(tokens[continuationStart..<continuationEnd])
            }
        }
    }
    return []
}

/// Compatibility wrapper for the promoted one-draft model integration.
func gemma4PromptLookupDraft(tokens: [Int32]) -> Int32? {
    gemma4PromptLookupDrafts(tokens: tokens, maxCount: 1).first
}

@inline(__always)
func gemma4PromptLookupEnvironmentEnabled(_ environment: [String: String]) -> Bool {
    guard let raw = environment["DARKBLOOM_PROMPT_LOOKUP"] else { return true }
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

func gemma4VerifyPromptLookupTriple(
    _ candidate: MLXArray,
    references: [MLXArray],
    candidateOffsets: [Int],
    referenceOffsets: [Int]
) {
    precondition(candidate.dtype == .float32 && candidate.shape == [3, 262_144])
    precondition(references.count == 3)
    var matches: [MLXArray] = []
    for index in 0..<3 {
        precondition(
            references[index].dtype == .float32
                && references[index].shape == [1, 1, 262_144])
        matches.append(arrayEqual(
            candidate[index..<(index + 1), 0...]
                .reshaped(1, 1, 262_144)
                .view(dtype: .uint32),
            references[index].view(dtype: .uint32)
        ))
    }
    eval(matches)
    precondition(
        matches.allSatisfy { $0.item(Bool.self) },
        "exact two-draft prompt lookup differs from serialized decode"
    )
    precondition(!candidateOffsets.isEmpty)
    precondition(candidateOffsets.allSatisfy { $0 == candidateOffsets[0] })
    precondition(referenceOffsets == candidateOffsets)
}
