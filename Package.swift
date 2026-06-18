// swift-tools-version: 6.0
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
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
        .library(name: "MLXFastDeepSeek", targets: ["MLXFastDeepSeek"]),
        .library(name: "MLXFastSubmission", targets: ["MLXFastSubmission"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.4"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
            ]
        ),
        .target(
            name: "MLXFastSubmission",
            dependencies: ["MLXFastCore"]
        ),
        .target(
            name: "MLXFastDeepSeek",
            dependencies: [
                "MLXFastCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                "MLXFastDeepSeek",
                "MLXFastSubmission",
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                "MLXFastDeepSeek",
                "MLXFastSubmission",
            ]
        ),
    ]
)
