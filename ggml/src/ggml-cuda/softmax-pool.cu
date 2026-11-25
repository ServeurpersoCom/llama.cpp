#include "softmax-pool.cuh"

// one thread per channel and pool, the threads of a warp read consecutive channels
// the softmax over the members is computed online, so each score and value is read once
static __global__ void softmax_pool_f32(
        const char * v, const char * s, const float * b, float * dst,
        const int64_t C, const int64_t M, const int64_t P,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbs1, const int64_t nbs2) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= C*P) {
        return;
    }

    const int64_t c = i % C;
    const int64_t p = i / C;

    const char * vp = v + p*nbv2 + c*sizeof(float);
    const char * sp = s + p*nbs2 + c*sizeof(float);

    float mx  = -INFINITY;
    float den = 0.0f;
    float acc = 0.0f;

    for (int64_t m = 0; m < M; ++m) {
        const float x  = *(const float *) (sp + m*nbs1) + (b ? b[m*C + c] : 0.0f);
        const float vm = *(const float *) (vp + m*nbv1);
        if (x > mx) {
            const float r = expf(mx - x);
            den *= r;
            acc *= r;
            mx   = x;
        }
        const float e = expf(x - mx);
        den += e;
        acc += e*vm;
    }

    dst[i] = acc/den;
}

void ggml_cuda_op_softmax_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[0];
    const ggml_tensor * s = dst->src[1];
    const ggml_tensor * b = dst->src[2];

    GGML_ASSERT(v->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t C = v->ne[0];
    const int64_t M = v->ne[1];
    const int64_t P = v->ne[2];

    const int     block_size = 256;
    const int64_t n_blocks   = (C*P + block_size - 1) / block_size;
    GGML_ASSERT(n_blocks <= INT_MAX);

    softmax_pool_f32<<<n_blocks, block_size, 0, ctx.stream()>>>(
        (const char *) v->data, (const char *) s->data, b ? (const float *) b->data : nullptr, (float *) dst->data,
        C, M, P, v->nb[1], v->nb[2], s->nb[1], s->nb[2]);
}
