import Foundation
@testable import MLXFastCore

func syntheticGemmaGoldenData() throws -> Data {
    try JSONSerialization.data(
        withJSONObject: [
            "version": 1,
            "cases": [
                [
                    "name": "synthetic-gemma",
                    "prompt_tokens": Array(
                        repeating: 1,
                        count: MLXFastConstants.correctnessPromptTokens
                    ),
                    "expected_tokens": Array(
                        repeating: 3,
                        count: MLXFastConstants.correctnessSteps
                    ),
                ]
            ],
        ],
        options: [.sortedKeys]
    )
}

func syntheticGemmaGoldenJSON() throws -> String {
    String(
        decoding: try syntheticGemmaGoldenData(),
        as: UTF8.self
    )
}
