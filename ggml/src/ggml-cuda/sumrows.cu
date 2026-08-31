#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

// one thread per dst element, i0 is the fast index so the reads along a row coalesce,
// the dst coords address the k == 0 slice of src0 and the reduction walks dim with its stride
static __global__ void sum_axis_f32(
        const char * GGML_CUDA_RESTRICT src, float * GGML_CUDA_RESTRICT dst, const int64_t nelem,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t nek, const int64_t nbk,
        const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    const int64_t t = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();

    if (t >= nelem) {
        return;
    }

    const int64_t i0 = t % ne0;
    const int64_t r  = t / ne0;
    const int64_t i1 = r % ne1;
    const int64_t i2 = (r / ne1) % ne2;
    const int64_t i3 =  r / (ne1*ne2);

    const char * src_row = src + i1*nb1 + i2*nb2 + i3*nb3;

    // partial sums keep the error growth of the reduction close to the row kernel
    constexpr int n_acc = 4;
    float acc[n_acc] = { 0.0f };

    int64_t k = 0;
    for (; k + n_acc <= nek; k += n_acc) {
        for (int j = 0; j < n_acc; j++) {
            acc[j] += ((const float *) (src_row + (k + j)*nbk))[i0];
        }
    }
    for (; k < nek; k++) {
        acc[0] += ((const float *) (src_row + k*nbk))[i0];
    }

    dst[t] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int dim = ggml_get_op_params_i32(dst, 0);

    if (dim != 0) {
        GGML_ASSERT(src0->nb[0] == sizeof(float));

        const int64_t nelem = ggml_nelements(dst);

        const dim3 block_dims(256, 1, 1);
        const dim3 block_nums((nelem + block_dims.x - 1)/block_dims.x, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(sum_axis_f32, launch_params,
                (const char *) src0->data, dst_d, nelem,
                dst->ne[0], dst->ne[1], dst->ne[2],
                src0->ne[dim], src0->nb[dim],
                src0->nb[1], src0->nb[2], src0->nb[3]);

        return;
    }

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    }
}
