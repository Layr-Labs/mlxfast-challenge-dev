import Foundation
import MLX
import MLXFastCore

/// A [experts, rows, columns] batch of routed-expert projection weights for
/// one decode token, feeding a single gatherQuantizedMM instead of one
/// quantizedMM per expert. MLX's gather quantized-matvec kernel runs the same
/// inner routine and fp32 accumulation as the per-expert kernel at M=1, so
/// outputs are bit-identical while 18 dispatches per layer become 3.
public struct DeepSeekExpertSlab {
    public let weights: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
}

extension DeepSeekWeightLoader {
    /// Builds the slab for one projection across the routed experts of one
    /// token, in routing order. Bytes come from the same per-slice bank reads
    /// (byte cache and metrics behave exactly as the per-expert path) and the
    /// resident scales store, memcpy'd once into page-aligned buffers that
    /// MLXArray wraps zero-copy. Returns nil whenever the layer's records do
    /// not match the stacked quantized layout — callers keep the per-expert
    /// path, which reproduces today's behavior exactly.
    public func expertSlab(
        layerIndex: Int,
        expertIndices: [Int],
        projection: DeepSeekExpertProjection,
        expectedShape: [Int]
    ) throws -> DeepSeekExpertSlab? {
        guard !expertIndices.isEmpty, expectedShape.count == 2 else {
            return nil
        }
        let candidates = DeepSeekWeightNames.routedExpert(
            layerIndex: layerIndex,
            expertIndex: expertIndices[0],
            projection: projection
        )
        // The slab must consume the same record expertLinearWeight would pick:
        // only engage when the first resolving candidate is the stacked one.
        guard
            let candidate = candidates.first(where: { expertBank.record(named: $0) != nil }),
            let record = expertBank.record(named: candidate),
            record.dtype == "U32",
            record.shape.count == 3,
            let stackCount = record.shape.first,
            stackCount > 0,
            record.byteLength % stackCount == 0,
            expertIndices.allSatisfy({ $0 >= 0 && $0 < stackCount })
        else {
            return nil
        }

        let rows = expectedShape[0]
        let inputColumns = expectedShape[1]
        let packedColumns = record.shape[2]
        guard
            record.shape[1] == rows,
            packedColumns > 0,
            (packedColumns * 32) % inputColumns == 0,
            [2, 4, 8].contains(packedColumns * 32 / inputColumns)
        else {
            return nil
        }
        let bits = packedColumns * 32 / inputColumns

        let scalesName = expertScalesCompanionName(for: candidate)
        guard
            let scalesRecord = expertBank.record(named: scalesName),
            scalesRecord.shape.count == 3,
            scalesRecord.shape[0] == stackCount,
            scalesRecord.shape[1] == rows,
            let scaleGroups = scalesRecord.shape.last,
            scaleGroups > 0,
            inputColumns % scaleGroups == 0,
            scalesRecord.byteLength % stackCount == 0
        else {
            return nil
        }

        let biasesName = expertBiasesCompanionName(for: candidate)
        let biasesRecord = expertBank.record(named: biasesName)
        if let biasesRecord {
            guard
                biasesRecord.shape.count == 3,
                biasesRecord.shape[0] == stackCount,
                biasesRecord.shape[1] == rows,
                biasesRecord.shape.last == scaleGroups,
                biasesRecord.byteLength % stackCount == 0
            else {
                return nil
            }
        }

        guard let scalesDType = try? TensorDType.parse(scalesRecord.dtype) else {
            return nil
        }
        let weightsFill: (Int, UnsafeMutableRawBufferPointer) throws -> Void = { slot, destination in
            if let pinned = pinnedExpertCodes?.materializedTensor(
                named: candidate,
                firstAxisIndex: expertIndices[slot]
            ) {
                copyBytes(pinned.bytes, to: destination)
                return
            }
            let tensor = try expertBank.materializedTensor(
                named: candidate,
                firstAxisIndex: expertIndices[slot]
            )
            copyBytes(tensor.bytes, to: destination)
        }
        let scalesFill: (Int, UnsafeMutableRawBufferPointer) throws -> Void = { slot, destination in
            if let resident = residentExpertScales?.materializedTensor(
                named: scalesName,
                firstAxisIndex: expertIndices[slot]
            ) {
                copyBytes(resident.bytes, to: destination)
            } else {
                let tensor = try expertBank.materializedTensor(
                    named: scalesName,
                    firstAxisIndex: expertIndices[slot]
                )
                copyBytes(tensor.bytes, to: destination)
            }
        }
        guard
            let weightsArray = try slabArray(
                shape: [expertIndices.count, rows, packedColumns],
                dtype: .uint32,
                sliceByteLength: record.byteLength / stackCount,
                fill: weightsFill
            ),
            let scalesArray = try slabArray(
                shape: [expertIndices.count, rows, scaleGroups],
                dtype: MLXArrayTensorBridge.mlxDType(for: scalesDType),
                sliceByteLength: scalesRecord.byteLength / stackCount,
                fill: scalesFill
            )
        else {
            return nil
        }

        var biasesArray: MLXArray?
        if let biasesRecord {
            let biasesFill: (Int, UnsafeMutableRawBufferPointer) throws -> Void = { slot, destination in
                let tensor = try expertBank.materializedTensor(
                    named: biasesName,
                    firstAxisIndex: expertIndices[slot]
                )
                copyBytes(tensor.bytes, to: destination)
            }
            guard
                let biasesDType = try? TensorDType.parse(biasesRecord.dtype),
                let array = try slabArray(
                    shape: [expertIndices.count, rows, scaleGroups],
                    dtype: MLXArrayTensorBridge.mlxDType(for: biasesDType),
                    sliceByteLength: biasesRecord.byteLength / stackCount,
                    fill: biasesFill
                )
            else {
                return nil
            }
            biasesArray = array
        }

        let mode: QuantizationMode = biasesRecord == nil && scalesDType == .u8 ? .mxfp4 : .affine
        return DeepSeekExpertSlab(
            weights: weightsArray,
            scales: scalesArray,
            biases: biasesArray,
            groupSize: inputColumns / scaleGroups,
            bits: bits,
            mode: mode
        )
    }

    /// Allocates one 16 KiB-aligned buffer for the whole slab (alignment is
    /// required for MLXArray's zero-copy rawPointer wrap), fills it slice by
    /// slice for each expert in routing order, and hands ownership to the
    /// array's finalizer.
    private func slabArray(
        shape: [Int],
        dtype: DType,
        sliceByteLength: Int,
        fill: (_ slot: Int, _ destination: UnsafeMutableRawBufferPointer) throws -> Void
    ) throws -> MLXArray? {
        guard sliceByteLength > 0, let expertCountShape = shape.first else {
            return nil
        }
        let totalBytes = expertCountShape * sliceByteLength
        var rawPointer: UnsafeMutableRawPointer?
        guard posix_memalign(&rawPointer, 16384, totalBytes) == 0, let base = rawPointer else {
            return nil
        }
        var filled = false
        defer {
            if !filled {
                free(base)
            }
        }
        for slot in 0..<expertCountShape {
            let destination = UnsafeMutableRawBufferPointer(
                start: base + slot * sliceByteLength,
                count: sliceByteLength
            )
            try fill(slot, destination)
        }
        filled = true
        return MLXArray(rawPointer: base, shape, dtype: dtype) {
            free(base)
        }
    }

    private func copyBytes(_ source: Data, to destination: UnsafeMutableRawBufferPointer) {
        precondition(source.count == destination.count, "expert slab slice size mismatch")
        source.withUnsafeBytes { raw in
            destination.copyMemory(from: raw)
        }
    }

    private func expertScalesCompanionName(for weightName: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".scales"
        }
        return "\(weightName).scales"
    }

    private func expertBiasesCompanionName(for weightName: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".biases"
        }
        return "\(weightName).biases"
    }
}
