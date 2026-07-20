import Foundation

/// Shared wire contract for the operator-private GPQA reference fixture.
///
/// Both the archived serial attachment/generation tools and the default MTP
/// semantic-only capture decode this exact shape. The MTP path intentionally
/// ignores `acceptedTokenSequences` for verdict purposes, but decoding the
/// field here still fails closed on a malformed fixture instead of maintaining
/// a second, drifting schema.
public struct GPQAReferenceDocument: Decodable, Equatable, Sendable {
    public static let jsonKeys: Set<String> = ["cases"]
    public static let maximumCaseCount = 32

    public let cases: [GPQAReferenceCase]

    private enum CodingKeys: String, CodingKey, CaseIterable {
        case cases
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownGPQAKeys(
            decoder,
            allowedKeys: Set(CodingKeys.allCases.map(\.rawValue)),
            description: "GPQA reference document"
        )
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedCases = try container.decode(
            [GPQAReferenceCase].self,
            forKey: .cases
        )
        guard !decodedCases.isEmpty,
              decodedCases.count <= Self.maximumCaseCount
        else {
            throw gpqaDataCorrupted(
                decoder,
                "GPQA reference document must contain 1..."
                    + "\(Self.maximumCaseCount) cases"
            )
        }
        var identifiers = Set<String>()
        for testCase in decodedCases {
            if let identifier = testCase.id,
               !identifiers.insert(
                   identifier.trimmingCharacters(in: .whitespacesAndNewlines)
               ).inserted
            {
                throw gpqaDataCorrupted(
                    decoder,
                    "GPQA reference document contains duplicate case IDs"
                )
            }
        }
        cases = decodedCases
    }
}

public struct GPQAReferenceCase: Decodable, Equatable, Sendable {
    public static let jsonKeys: Set<String> = [
        "id",
        "prompt",
        "expected_response",
        "answer_key",
        "accepted_token_sequences",
        "accepted_responses",
        "domain",
        "subdomain",
    ]
    public static let maximumIdentifierByteCount = 256
    public static let maximumPromptByteCount = 64 * 1024
    public static let maximumAnswerByteCount = 8 * 1024
    public static let maximumMetadataByteCount = 256
    public static let maximumAcceptedValueCount = 32
    public static let maximumAcceptedTokenCount = 64

    public let id: String?
    public let prompt: String
    public let expectedResponse: String?
    public let answerKey: String?
    public let acceptedTokenSequences: [[Int]]?
    public let acceptedResponses: [String]?
    public let domain: String?
    public let subdomain: String?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id
        case prompt
        case expectedResponse = "expected_response"
        case answerKey = "answer_key"
        case acceptedTokenSequences = "accepted_token_sequences"
        case acceptedResponses = "accepted_responses"
        case domain
        case subdomain
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownGPQAKeys(
            decoder,
            allowedKeys: Set(CodingKeys.allCases.map(\.rawValue)),
            description: "GPQA reference case"
        )
        let container = try decoder.container(keyedBy: CodingKeys.self)

        id = try decodeOptionalBoundedGPQAString(
            from: container,
            forKey: .id,
            maximumByteCount: Self.maximumIdentifierByteCount,
            decoder: decoder
        )
        prompt = try decodeRequiredBoundedGPQAString(
            from: container,
            forKey: .prompt,
            maximumByteCount: Self.maximumPromptByteCount,
            decoder: decoder
        )
        expectedResponse = try decodeOptionalBoundedGPQAString(
            from: container,
            forKey: .expectedResponse,
            maximumByteCount: Self.maximumAnswerByteCount,
            decoder: decoder
        )
        answerKey = try decodeOptionalBoundedGPQAString(
            from: container,
            forKey: .answerKey,
            maximumByteCount: Self.maximumMetadataByteCount,
            decoder: decoder
        )
        domain = try decodeOptionalBoundedGPQAString(
            from: container,
            forKey: .domain,
            maximumByteCount: Self.maximumMetadataByteCount,
            decoder: decoder
        )
        subdomain = try decodeOptionalBoundedGPQAString(
            from: container,
            forKey: .subdomain,
            maximumByteCount: Self.maximumMetadataByteCount,
            decoder: decoder
        )

        if container.contains(.acceptedResponses) {
            let values = try container.decodeIfPresent(
                [String].self,
                forKey: .acceptedResponses
            )
            if let values {
                guard !values.isEmpty,
                      values.count <= Self.maximumAcceptedValueCount
                else {
                    throw gpqaDataCorrupted(
                        decoder,
                        "accepted_responses must contain 1..."
                            + "\(Self.maximumAcceptedValueCount) values"
                    )
                }
                var uniqueValues = Set<String>()
                acceptedResponses = try values.map { value in
                    let validated = try validateGPQAString(
                        value,
                        maximumByteCount: Self.maximumAnswerByteCount,
                        decoder: decoder,
                        description: "accepted_responses value"
                    )
                    let normalized = validated.trimmingCharacters(
                        in: .whitespacesAndNewlines
                    )
                    guard uniqueValues.insert(normalized).inserted else {
                        throw gpqaDataCorrupted(
                            decoder,
                            "accepted_responses contains duplicate values"
                        )
                    }
                    return validated
                }
            } else {
                acceptedResponses = nil
            }
        } else {
            acceptedResponses = nil
        }

        if container.contains(.acceptedTokenSequences) {
            let sequences = try container.decodeIfPresent(
                [[Int]].self,
                forKey: .acceptedTokenSequences
            )
            if let sequences {
                guard !sequences.isEmpty,
                      sequences.count <= Self.maximumAcceptedValueCount
                else {
                    throw gpqaDataCorrupted(
                        decoder,
                        "accepted_token_sequences must contain 1..."
                            + "\(Self.maximumAcceptedValueCount) sequences"
                    )
                }
                var uniqueSequences = Set<[Int]>()
                for sequence in sequences {
                    guard !sequence.isEmpty,
                          sequence.count <= Self.maximumAcceptedTokenCount,
                          sequence.allSatisfy({
                              $0 >= 0 && $0 < MLXFastConstants.vocabSize
                          }),
                          uniqueSequences.insert(sequence).inserted
                    else {
                        throw gpqaDataCorrupted(
                            decoder,
                            "accepted_token_sequences contains an empty, "
                                + "oversized, duplicate, or invalid sequence"
                        )
                    }
                }
                acceptedTokenSequences = sequences
            } else {
                acceptedTokenSequences = nil
            }
        } else {
            acceptedTokenSequences = nil
        }

        guard expectedResponse != nil
                || answerKey != nil
                || acceptedResponses != nil
                || acceptedTokenSequences != nil
        else {
            throw gpqaDataCorrupted(
                decoder,
                "GPQA reference case has no answer or accepted-token contract"
            )
        }
    }

    public var identifier: String {
        trimmed(id) ?? "gpqa-private"
    }

    public var semanticReferenceAnswer: String {
        if let expected = trimmed(expectedResponse) {
            return expected
        }
        if let accepted = acceptedResponses?.compactMap(trimmed),
           !accepted.isEmpty
        {
            return accepted.joined(separator: "\n")
        }
        if let answerKey = trimmed(answerKey) {
            if let answerText = multipleChoiceAnswerText(
                in: prompt,
                answerKey: answerKey
            ) {
                return "\(answerKey). \(answerText)"
            }
            return "Correct option: \(answerKey)"
        }
        return ""
    }

    private func trimmed(_ value: String?) -> String? {
        let value = value?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ) ?? ""
        return value.isEmpty ? nil : value
    }

    private func multipleChoiceAnswerText(
        in prompt: String,
        answerKey: String
    ) -> String? {
        let normalizedKey = answerKey
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .uppercased()
        guard normalizedKey.count == 1 else {
            return nil
        }
        for rawLine in prompt.split(
            separator: "\n",
            omittingEmptySubsequences: false
        ) {
            let line = rawLine.trimmingCharacters(
                in: .whitespacesAndNewlines
            )
            for marker in [
                "\(normalizedKey).",
                "\(normalizedKey):",
                "\(normalizedKey))",
            ] where line.hasPrefix(marker) {
                let start = line.index(
                    line.startIndex,
                    offsetBy: marker.count
                )
                let value = line[start...].trimmingCharacters(
                    in: .whitespacesAndNewlines
                )
                if !value.isEmpty {
                    return value
                }
            }
        }
        return nil
    }
}

private struct GPQAReferenceWireCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

private func rejectUnknownGPQAKeys(
    _ decoder: Decoder,
    allowedKeys: Set<String>,
    description: String
) throws {
    let container = try decoder.container(
        keyedBy: GPQAReferenceWireCodingKey.self
    )
    if let unknownKey = container.allKeys.first(
        where: { !allowedKeys.contains($0.stringValue) }
    ) {
        throw DecodingError.dataCorrupted(
            DecodingError.Context(
                codingPath: decoder.codingPath + [unknownKey],
                debugDescription:
                    "\(description) contains unknown field "
                    + unknownKey.stringValue
            )
        )
    }
}

private func decodeRequiredBoundedGPQAString<Key: CodingKey>(
    from container: KeyedDecodingContainer<Key>,
    forKey key: Key,
    maximumByteCount: Int,
    decoder: Decoder
) throws -> String {
    try validateGPQAString(
        container.decode(String.self, forKey: key),
        maximumByteCount: maximumByteCount,
        decoder: decoder,
        description: key.stringValue
    )
}

private func decodeOptionalBoundedGPQAString<Key: CodingKey>(
    from container: KeyedDecodingContainer<Key>,
    forKey key: Key,
    maximumByteCount: Int,
    decoder: Decoder
) throws -> String? {
    guard let value = try container.decodeIfPresent(
        String.self,
        forKey: key
    ) else {
        return nil
    }
    return try validateGPQAString(
        value,
        maximumByteCount: maximumByteCount,
        decoder: decoder,
        description: key.stringValue
    )
}

private func validateGPQAString(
    _ value: String,
    maximumByteCount: Int,
    decoder: Decoder,
    description: String
) throws -> String {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
          trimmed.utf8.count <= maximumByteCount
    else {
        throw gpqaDataCorrupted(
            decoder,
            "\(description) must be nonempty and no larger than "
                + "\(maximumByteCount) UTF-8 bytes"
        )
    }
    return value
}

private func gpqaDataCorrupted(
    _ decoder: Decoder,
    _ description: String
) -> DecodingError {
    DecodingError.dataCorrupted(
        DecodingError.Context(
            codingPath: decoder.codingPath,
            debugDescription: description
        )
    )
}
