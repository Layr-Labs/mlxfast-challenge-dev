#if canImport(Metal)
import Foundation
import Metal

enum Gemma4NativeKernelLibrary {
    static func load(device: any MTLDevice) -> (any MTLLibrary)? {
        for url in candidateURLs() where FileManager.default.fileExists(atPath: url.path) {
            if let library = try? device.makeLibrary(URL: url) {
                return library
            }
        }
        return nil
    }

    static func pipeline(
        library: any MTLLibrary,
        device: any MTLDevice,
        name: String,
        boolConstants: [Int: Bool] = [:],
        intConstants: [Int: Int32] = [:]
    ) -> (any MTLComputePipelineState)? {
        let function: (any MTLFunction)?
        if boolConstants.isEmpty && intConstants.isEmpty {
            function = library.makeFunction(name: name)
        } else {
            let values = MTLFunctionConstantValues()
            for (index, value) in boolConstants {
                var value = value
                values.setConstantValue(&value, type: .bool, index: index)
            }
            for (index, value) in intConstants {
                var value = value
                values.setConstantValue(&value, type: .int, index: index)
            }
            function = try? library.makeFunction(name: name, constantValues: values)
        }
        guard let function else { return nil }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.supportIndirectCommandBuffers = true
        var reflection: MTLAutoreleasedComputePipelineReflection?
        return try? device.makeComputePipelineState(
            descriptor: descriptor,
            options: [],
            reflection: &reflection
        )
    }

    private static func candidateURLs() -> [URL] {
        var result: [URL] = []
        if let explicit = ProcessInfo.processInfo.environment["MLXFAST_MLX_METALLIB"] {
            result.append(URL(fileURLWithPath: explicit))
        }
        for bundle in Bundle.allBundles + Bundle.allFrameworks {
            if let executable = bundle.executableURL {
                result.append(executable.deletingLastPathComponent()
                    .appendingPathComponent("mlx.metallib"))
            }
            result.append(bundle.bundleURL.appendingPathComponent("mlx.metallib"))
        }
        for argument in CommandLine.arguments {
            let url = URL(fileURLWithPath: argument)
            result.append(url.deletingLastPathComponent()
                .appendingPathComponent("mlx.metallib"))
        }
        result.append(URL(fileURLWithPath: ".build/release/mlx.metallib"))
        result.append(URL(fileURLWithPath: ".build/arm64-apple-macosx/release/mlx.metallib"))
        var seen: Set<String> = []
        return result.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }
}
#endif
