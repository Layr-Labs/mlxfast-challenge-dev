import MLX

/// Compile-time feasibility probe for an in-kernel packed-Q4 -> bounded BF16
/// MPP tile. This deliberately is not a production projection: qualification
/// must first establish raw-word parity with MLX's generated NAX kernel.
private let gemma4CooperativeQ4MPPFeasibilityKernel = MLXFast.metalKernel(
    name: "gemma4_cooperative_q4_mpp_feasibility_v1",
    inputNames: ["x", "packed", "scales", "biases"],
    outputNames: ["output"],
    source: """
        const uint lane = thread_index_in_simdgroup;
        threadgroup bfloat decoded[16 * 32];

        // A single SIMDgroup reconstructs a bounded 32-output x 16-input
        // affine-Q4 tile. No persistent dequantized device tensor is used.
        for (uint index = lane; index < 16u * 32u; index += 32u) {
            const uint output_column = index / 16u;
            const uint input_column = index % 16u;
            const uint packed_index = output_column * 2u + input_column / 8u;
            const uint word = packed[packed_index];
            const uint nibble = (word >> ((input_column & 7u) * 4u)) & 15u;
            decoded[index] = static_cast<bfloat>(
                static_cast<bfloat>(nibble) * scales[output_column]
                    + biases[output_column]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 16, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            descriptor, metal::execution_simdgroup> operation;
        device bfloat* mutable_x = const_cast<device bfloat*>(x);
        auto lhs = metal::tensor(
            mutable_x,
            metal::dextents<int, 2>{16, 16},
            metal::array<int64_t, 2>{1, 16});
        auto rhs = metal::tensor(
            decoded,
            metal::dextents<int, 2>{16, 32},
            metal::array<int, 2>{1, 16});
        auto destination = metal::tensor(
            output,
            metal::dextents<int, 2>{32, 16},
            metal::array<int, 2>{1, 32});
        operation.run(lhs, rhs, destination);
        """,
    header: """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """,
    ensureRowContiguous: false
)

/// Runs the public-API feasibility probe. Kept internal so tests can force
/// Metal compilation without making an unqualified route reachable by the
/// production engine.
func gemma4CooperativeQ4MPPFeasibility(
    x: MLXArray,
    packed: MLXArray,
    scales: MLXArray,
    biases: MLXArray
) -> MLXArray {
    precondition(x.dtype == .bfloat16 && x.shape == [16, 16])
    precondition(packed.dtype == .uint32 && packed.shape == [32, 2])
    precondition(scales.dtype == .bfloat16 && scales.shape == [32])
    precondition(biases.dtype == .bfloat16 && biases.shape == [32])
    return gemma4CooperativeQ4MPPFeasibilityKernel(
        [x, packed, scales, biases],
        grid: (32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[16, 32]],
        outputDTypes: [.bfloat16]
    )[0]
}
