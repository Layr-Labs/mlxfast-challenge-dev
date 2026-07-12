import MLX

private let gemma4IndexedUpQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_up_qmv_5376",
    inputNames: ["weight", "indices", "lut", "x"], outputNames: ["output"],
    source: """
        constexpr int kGroups = 84, kRowBytes = 2688;
        const int first = threadgroup_position_in_grid.y * 4;
        const int lane = thread_index_in_simdgroup;
        const device uchar* w = reinterpret_cast<const device uchar*>(weight) + first*kRowBytes + lane*4;
        const device ushort* ix = indices + first*kGroups + lane/8;
        const device bfloat* input = x + lane*8;
        float result[4] = {0};
        for (int block=0; block<21; ++block) {
            float v[8]; float sum = load_values(input, v);
            for (int row=0; row<4; ++row) {
                uint pair=lut[ix[row*kGroups]];
                result[row] += qdot(w+row*kRowBytes, v, float(as_type<bfloat>(ushort(pair))),
                    float(as_type<bfloat>(ushort(pair>>16))), sum);
            }
            w += 128; ix += 4; input += 256;
        }
        for (int row=0; row<4; ++row) { result[row]=simd_sum(result[row]); if(lane==0) output[first+row]=bfloat(result[row]); }
    """,
    header: """
        using namespace metal;
        inline float load_values(const device bfloat* x, thread float* v) { float s=0; for(int i=0;i<8;i+=4){s+=x[i]+x[i+1]+x[i+2]+x[i+3];v[i]=x[i];v[i+1]=x[i+1]/16.0f;v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;}return s;}
        inline float qdot(const device uchar* w,const thread float* v,float scale,float bias,float sum){const device ushort* p=reinterpret_cast<const device ushort*>(w);float a=0;for(int i=0;i<2;++i)a+=v[4*i]*(p[i]&15)+v[4*i+1]*(p[i]&240)+v[4*i+2]*(p[i]&3840)+v[4*i+3]*(p[i]&61440);return scale*a+sum*bias;}
    """, ensureRowContiguous: true
)

func gemma4IndexedUp(_ projection: FastQuantizedProjection, _ metadata: IndexedAffineMetadata, _ input: MLXArray) -> MLXArray {
    gemma4IndexedUpQMV([projection.weight, metadata.indices, metadata.lut, input], grid: (32, 21_504 / 4, 1), threadGroup: (32, 1, 1), outputShapes: [[1,1,21_504]], outputDTypes: [.bfloat16])[0]
}
