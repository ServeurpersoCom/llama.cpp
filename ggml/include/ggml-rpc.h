#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    1

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 97, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// fetch multiple tensor regions from the same RPC buffer in a single round trip;
// also exposed via ggml_backend_reg_get_proc_address("ggml_backend_rpc_get_tensor_batch")
GGML_BACKEND_API bool ggml_backend_rpc_get_tensor_batch(
        ggml_backend_buffer_t buffer,
        size_t n_tensors,
        const struct ggml_tensor ** tensors,
        const size_t * offsets,
        const size_t * sizes,
        void ** dsts);

// request the same regions ahead of time without waiting for the data: the response
// streams into a client-side cache (drained lazily before the next command that reads
// from the socket) and later ggml_backend_rpc_get_tensor_batch calls are served from
// the cache, so the transfer overlaps whatever the server does next (e.g. computing
// the following chunk of a prompt);
// also exposed via ggml_backend_reg_get_proc_address("ggml_backend_rpc_prefetch_tensor_batch")
GGML_BACKEND_API bool ggml_backend_rpc_prefetch_tensor_batch(
        ggml_backend_buffer_t buffer,
        size_t n_tensors,
        const struct ggml_tensor ** tensors,
        const size_t * offsets,
        const size_t * sizes);

#ifdef  __cplusplus
}
#endif
