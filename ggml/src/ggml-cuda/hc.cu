#include "common.cuh"
#include "hc.cuh"

static __device__ __forceinline__ float hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// dst[i, t] = mean_c x[i, c, t]*sigmoid(gate[i, c, t])
static __global__ void hc_mix_f32(
        const float * x,
        const float * gate,
        float * dst,
        const uint3 n_embd_fdv,
        int64_t hc,
        int64_t n_tokens,
        float inv_hc,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sg0, int64_t sg1, int64_t sg2,
        int64_t sd0, int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = (int64_t) n_embd_fdv.z * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const uint2   ti = fast_div_modulo((uint32_t) ir, n_embd_fdv);
    const int64_t it = ti.x;
    const int64_t i0 = ti.y;

    // the same operation order as the elementwise graph: gate, product, running sum, scale
    float sum = 0.0f;
    for (int64_t ic = 0; ic < hc; ++ic) {
        const float xv = x   [i0*sx0 + ic*sx1 + it*sx2];
        const float gv = gate[i0*sg0 + ic*sg1 + it*sg2];
        sum = __fadd_rn(sum, __fmul_rn(xv, hc_sigmoid(gv)));
    }

    dst[i0*sd0 + it*sd1] = __fmul_rn(sum, inv_hc);
}

// dst[i, c, t] = residual[i, c, t] + x[i, t]*2*sigmoid(scale*gate[c, t])
static __global__ void hc_combine_f32(
        const float * x,
        const float * residual,
        const float * gate,
        float * dst,
        const uint3 n_embd_fdv,
        const uint3 hc_fdv,
        int64_t n_tokens,
        float scale,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sg0, int64_t sg1,
        int64_t sd0, int64_t sd1, int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = (int64_t) n_embd_fdv.z * hc_fdv.z * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const uint2   ri = fast_div_modulo((uint32_t) ir, n_embd_fdv);
    const uint2   tc = fast_div_modulo(ri.x, hc_fdv);
    const int64_t i0 = ri.y;
    const int64_t ic = tc.y;
    const int64_t it = tc.x;

    const float xv = x       [i0*sx0 + it*sx1];
    const float rv = residual[i0*sr0 + ic*sr1 + it*sr2];
    const float gv = gate    [ic*sg0 + it*sg1];

    // the same operation order as the elementwise graph: scale, gate, scale, product, add
    const float w = __fmul_rn(hc_sigmoid(__fmul_rn(gv, scale)), 2.0f);

    dst[i0*sd0 + ic*sd1 + it*sd2] = __fadd_rn(rv, __fmul_rn(xv, w));
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * gate = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,    nb);
    GGML_TENSOR_LOCALS(size_t, nbg, gate, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,  nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    GGML_ASSERT(nr <= std::numeric_limits<uint32_t>::max());
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(hc_mix_f32, launch_params,
            (const float *) x->data, (const float *) gate->data, (float *) dst->data,
            init_fastdiv_values(n_embd), hc, n_tokens, 1.0f / (float) hc,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbg0 / sizeof(float), nbg1 / sizeof(float), nbg2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * gate     = dst->src[2];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbg, gate,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const float scale = ggml_get_op_params_f32(dst, 0);

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    GGML_ASSERT(nr <= std::numeric_limits<uint32_t>::max());
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(hc_combine_f32, launch_params,
            (const float *) x->data, (const float *) residual->data, (const float *) gate->data, (float *) dst->data,
            init_fastdiv_values(n_embd), init_fastdiv_values(hc), n_tokens, scale,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbg0 / sizeof(float), nbg1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
