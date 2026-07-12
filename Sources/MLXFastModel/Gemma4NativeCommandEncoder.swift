#if canImport(Metal)
import Metal

protocol Gemma4NativeResidencyCollecting: AnyObject {
    func collect(_ buffer: any MTLBuffer)
    func remove(_ buffer: any MTLBuffer)
    func activate(on queue: any MTLCommandQueue)
}

@available(macOS 15.0, *)
final class Gemma4NativeResidencyCollector: Gemma4NativeResidencyCollecting {
    private let residencySet: any MTLResidencySet
    private var resourceIDs: Set<ObjectIdentifier> = []
    private var attached = false
    private var hasPendingChanges = false

    init?(device: any MTLDevice) {
        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 4_096
        guard let residencySet = try? device.makeResidencySet(descriptor: descriptor) else {
            return nil
        }
        self.residencySet = residencySet
    }

    func collect(_ buffer: any MTLBuffer) {
        let identifier = ObjectIdentifier(buffer as AnyObject)
        if resourceIDs.insert(identifier).inserted {
            residencySet.addAllocation(buffer)
            hasPendingChanges = true
        }
    }

    func remove(_ buffer: any MTLBuffer) {
        let identifier = ObjectIdentifier(buffer as AnyObject)
        if resourceIDs.remove(identifier) != nil {
            residencySet.removeAllocation(buffer)
            hasPendingChanges = true
        }
    }

    func activate(on queue: any MTLCommandQueue) {
        if hasPendingChanges {
            residencySet.commit()
            residencySet.requestResidency()
            hasPendingChanges = false
        }
        if !attached {
            queue.addResidencySet(residencySet)
            attached = true
        }
    }
}

protocol Gemma4NativeCommandEncoder: AnyObject {
    var device: any MTLDevice { get }

    func setComputePipelineState(_ state: any MTLComputePipelineState)
    func setBuffer(_ buffer: (any MTLBuffer)?, offset: Int, index: Int)
    func dispatchThreads(
        _ threadsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    )
    func dispatchThreadgroups(
        _ threadgroupsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    )
    func memoryBarrier(resources: [any MTLResource])
}

final class Gemma4NativeDirectCommandEncoder: Gemma4NativeCommandEncoder {
    let base: any MTLComputeCommandEncoder
    private let residencyCollector: (any Gemma4NativeResidencyCollecting)?

    var device: any MTLDevice { base.device }

    init(
        _ base: any MTLComputeCommandEncoder,
        residencyCollector: (any Gemma4NativeResidencyCollecting)? = nil
    ) {
        self.base = base
        self.residencyCollector = residencyCollector
    }

    func setComputePipelineState(_ state: any MTLComputePipelineState) {
        base.setComputePipelineState(state)
    }

    func setBuffer(_ buffer: (any MTLBuffer)?, offset: Int, index: Int) {
        if let buffer {
            residencyCollector?.collect(buffer)
        }
        base.setBuffer(buffer, offset: offset, index: index)
    }

    func dispatchThreads(
        _ threadsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        base.dispatchThreads(
            threadsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func dispatchThreadgroups(
        _ threadgroupsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        base.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func memoryBarrier(resources: [any MTLResource]) {
        base.memoryBarrier(resources: resources)
    }
}

final class Gemma4NativeIndirectCommandRecorder: Gemma4NativeCommandEncoder {
    enum RecorderError: Error, CustomStringConvertible {
        case unavailable(String)
        case invalidDispatch(String)
        case commandCapacityExceeded(Int)

        var description: String {
            switch self {
            case .unavailable(let message):
                return "native ICB resource unavailable: \(message)"
            case .invalidDispatch(let message):
                return "invalid native ICB dispatch: \(message)"
            case .commandCapacityExceeded(let capacity):
                return "native ICB command count exceeds capacity \(capacity)"
            }
        }
    }

    static let defaultCommandCapacity = 768
    static let maxBufferBindings = 16

    let indirectCommandBuffer: any MTLIndirectCommandBuffer
    let device: any MTLDevice
    private(set) var commandCount = 0
    private(set) var recordingError: Error?

    private let commandCapacity: Int
    private let inheritedBufferBindings: [Int: ObjectIdentifier]
    private var pipeline: (any MTLComputePipelineState)?
    private var buffers = [(buffer: any MTLBuffer, offset: Int)?](
        repeating: nil,
        count: maxBufferBindings
    )
    private var barrierBeforeNextDispatch = false
    private var resources: [any MTLResource] = []
    private var resourceIDs: Set<ObjectIdentifier> = []

    init(
        device: any MTLDevice,
        commandCapacity: Int = defaultCommandCapacity,
        inheritedBufferBindings: [Int: any MTLBuffer] = [:]
    ) throws {
        guard commandCapacity > 0 else {
            throw RecorderError.unavailable("positive command capacity")
        }
        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = .concurrentDispatch
        descriptor.inheritBuffers = !inheritedBufferBindings.isEmpty
        descriptor.inheritPipelineState = false
        descriptor.maxKernelBufferBindCount = Self.maxBufferBindings
        guard let indirectCommandBuffer = device.makeIndirectCommandBuffer(
            descriptor: descriptor,
            maxCommandCount: commandCapacity,
            options: []
        ) else {
            throw RecorderError.unavailable("indirect command buffer")
        }
        indirectCommandBuffer.label = "Gemma4NativeWholeToken.ICB"
        self.device = device
        self.indirectCommandBuffer = indirectCommandBuffer
        self.commandCapacity = commandCapacity
        self.inheritedBufferBindings = inheritedBufferBindings.mapValues {
            ObjectIdentifier($0 as AnyObject)
        }
    }

    func setComputePipelineState(_ state: any MTLComputePipelineState) {
        pipeline = state
    }

    func setBuffer(_ buffer: (any MTLBuffer)?, offset: Int, index: Int) {
        guard recordingError == nil else { return }
        guard buffers.indices.contains(index), offset >= 0 else {
            recordingError = RecorderError.unavailable(
                "buffer binding \(index) outside 0..<\(buffers.count)")
            return
        }
        if let buffer {
            if inheritedBufferBindings[index] == ObjectIdentifier(buffer as AnyObject) {
                buffers[index] = nil
            } else {
                buffers[index] = (buffer, offset)
                retainResource(buffer)
            }
        } else {
            buffers[index] = nil
        }
    }

    func dispatchThreads(
        _ threadsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        guard recordingError == nil else { return }
        guard Self.dividesExactly(threadsPerGrid.width, by: threadsPerThreadgroup.width),
              Self.dividesExactly(threadsPerGrid.height, by: threadsPerThreadgroup.height),
              Self.dividesExactly(threadsPerGrid.depth, by: threadsPerThreadgroup.depth)
        else {
            recordingError = RecorderError.invalidDispatch(
                "nonuniform grid \(threadsPerGrid) cannot be represented exactly")
            return
        }
        recordDispatch(
            threadgroups: MTLSize(
                width: threadsPerGrid.width / threadsPerThreadgroup.width,
                height: threadsPerGrid.height / threadsPerThreadgroup.height,
                depth: threadsPerGrid.depth / threadsPerThreadgroup.depth
            ),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func dispatchThreadgroups(
        _ threadgroupsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        guard recordingError == nil else { return }
        guard threadgroupsPerGrid.width > 0,
              threadgroupsPerGrid.height > 0,
              threadgroupsPerGrid.depth > 0,
              threadsPerThreadgroup.width > 0,
              threadsPerThreadgroup.height > 0,
              threadsPerThreadgroup.depth > 0
        else {
            recordingError = RecorderError.invalidDispatch(
                "threadgroup dimensions must be positive")
            return
        }
        recordDispatch(
            threadgroups: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
    }

    func memoryBarrier(resources: [any MTLResource]) {
        for resource in resources {
            retainResource(resource)
        }
        barrierBeforeNextDispatch = true
    }

    func finish() throws -> Gemma4NativeIndirectCommandStream {
        if let recordingError {
            throw recordingError
        }
        guard commandCount > 0 else {
            throw RecorderError.unavailable("recorded command stream is empty")
        }
        return Gemma4NativeIndirectCommandStream(
            indirectCommandBuffer: indirectCommandBuffer,
            commandCount: commandCount,
            resources: resources
        )
    }

    private func recordDispatch(
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        guard let pipeline else {
            recordingError = RecorderError.unavailable(
                "dispatch \(commandCount) has no pipeline")
            return
        }
        guard commandCount < commandCapacity else {
            recordingError = RecorderError.commandCapacityExceeded(commandCapacity)
            return
        }
        let command = indirectCommandBuffer.indirectComputeCommandAt(commandCount)
        if barrierBeforeNextDispatch {
            command.setBarrier()
        }
        command.setComputePipelineState(pipeline)
        for (index, binding) in buffers.enumerated() {
            guard let binding else { continue }
            command.setKernelBuffer(
                binding.buffer,
                offset: binding.offset,
                at: index
            )
        }
        command.concurrentDispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        commandCount += 1
        barrierBeforeNextDispatch = false
    }

    private func retainResource(_ resource: any MTLResource) {
        let identifier = ObjectIdentifier(resource as AnyObject)
        if resourceIDs.insert(identifier).inserted {
            resources.append(resource)
        }
    }

    private static func dividesExactly(_ value: Int, by divisor: Int) -> Bool {
        value > 0 && divisor > 0 && value.isMultiple(of: divisor)
    }
}

struct Gemma4NativeIndirectCommandStream {
    let indirectCommandBuffer: any MTLIndirectCommandBuffer
    let commandCount: Int
    let resources: [any MTLResource]

    func execute(
        into encoder: any MTLComputeCommandEncoder,
        inheritedBuffers: [Int: any MTLBuffer] = [:]
    ) {
        for (index, buffer) in inheritedBuffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        for resource in resources {
            encoder.useResource(resource, usage: [.read, .write])
        }
        for buffer in inheritedBuffers.values {
            encoder.useResource(buffer, usage: [.read, .write])
        }
        encoder.executeCommandsInBuffer(
            indirectCommandBuffer,
            range: 0..<commandCount
        )
    }
}
#endif
