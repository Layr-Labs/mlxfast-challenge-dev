// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastModel", targets: ["MLXFastModel"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
    ],
    dependencies: [
        // Vanilla upstream mlx-swift-lm carries the Gemma 4 text tower this
        // benchmark's reference is built on. It (tools-version 6.1) and mlx-swift
        // 0.31.4 build on the tenki runners' Swift 6.1 -- unlike the Layr-Labs
        // fork, which requires 6.3 (see PR #365). mlx-swift-lm 3.31.4 asks for
        // mlx-swift .upToNextMinor(from 0.31.4); pin it to exactly 0.31.4 so the
        // graph never resolves 0.31.6 (which is 6.3-only).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.4"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        .target(
            name: "MLXFastModel",
            dependencies: [
                "MLXFastCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                "MLXFastHarness",
            ]
        ),
    ]
)
