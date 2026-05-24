#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE lib_t;
#else
#include <dlfcn.h>
typedef void * lib_t;
#endif // defined(_WIN32)

static double to_mib(size_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static const char * dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "UNKNOWN";
}

// open the first library that loads from a candidate list
static lib_t lib_open(const char * const * names) {
    for (int i = 0; names[i]; i++) {
#if defined(_WIN32)
        lib_t h = LoadLibraryA(names[i]);
#else
        lib_t h = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
#endif // defined(_WIN32)
        if (h) {
            return h;
        }
    }
    return nullptr;
}

static void * lib_sym(lib_t h, const char * name) {
#if defined(_WIN32)
    return (void *) GetProcAddress(h, name);
#else
    return dlsym(h, name);
#endif // defined(_WIN32)
}

// stage 1: every backend answers free/total with its own native query
static void stage1_backend_agnostic(void) {
    ggml_backend_load_all();
    size_t n = ggml_backend_dev_count();
    printf("backend agnostic view: %zu device(s)\n", n);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  [%zu] %-5s %-22s free %8.1f MiB / total %8.1f MiB  (%s)\n",
               i,
               dev_type_str(ggml_backend_dev_type(dev)),
               ggml_backend_dev_name(dev),
               to_mib(free),
               to_mib(total),
               ggml_backend_dev_description(dev));
    }
}

typedef int (*cuda_count_fn)(int *);
typedef int (*cuda_pci_fn)(char *, int, int);
typedef int (*cuda_set_fn)(int);
typedef int (*cuda_mem_fn)(size_t *, size_t *);
typedef int (*nvml_init_fn)(void);
typedef int (*nvml_handle_fn)(const char *, void **);
typedef int (*nvml_mem_fn)(void *, void *);

// stage 2: only CUDA owns two modes, the contextless NVML read and the
// cudaMemGetInfo read that materializes the costly primary context
static void stage2_cuda_two_modes(void) {
#if defined(_WIN32)
    const char * cudart_names[] = { "cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", nullptr };
    const char * nvml_names[]   = { "nvml.dll", nullptr };
#else
    const char * cudart_names[] = { "libcudart.so", "libcudart.so.13", "libcudart.so.12", nullptr };
    const char * nvml_names[]   = { "libnvidia-ml.so.1", "libnvidia-ml.so", nullptr };
#endif // defined(_WIN32)
    lib_t cudart = lib_open(cudart_names);
    lib_t nvml   = lib_open(nvml_names);
    if (!cudart || !nvml) {
        printf("cuda two modes: skipped (cudart or nvml not present)\n");
        return;
    }
    auto cuda_count = (cuda_count_fn)  lib_sym(cudart, "cudaGetDeviceCount");
    auto cuda_pci   = (cuda_pci_fn)    lib_sym(cudart, "cudaDeviceGetPCIBusId");
    auto cuda_set   = (cuda_set_fn)    lib_sym(cudart, "cudaSetDevice");
    auto cuda_mem   = (cuda_mem_fn)    lib_sym(cudart, "cudaMemGetInfo");
    auto nvml_init  = (nvml_init_fn)   lib_sym(nvml, "nvmlInit_v2");
    auto nvml_hpci  = (nvml_handle_fn) lib_sym(nvml, "nvmlDeviceGetHandleByPciBusId_v2");
    auto nvml_mem   = (nvml_mem_fn)    lib_sym(nvml, "nvmlDeviceGetMemoryInfo");
    if (!cuda_count || !cuda_pci || !cuda_set || !cuda_mem || !nvml_init || !nvml_hpci || !nvml_mem) {
        printf("cuda two modes: skipped (missing symbols)\n");
        return;
    }
    if (nvml_init() != 0) {
        printf("cuda two modes: skipped (nvmlInit failed)\n");
        return;
    }
    int n = 0;
    if (cuda_count(&n) != 0 || n <= 0) {
        printf("cuda two modes: skipped (no cuda device)\n");
        return;
    }
    printf("cuda two modes: %d device(s)\n", n);
    for (int d = 0; d < n; d++) {
        char pci[32] = { 0 };
        // property query, does not create the primary context
        if (cuda_pci(pci, sizeof(pci), d) != 0) {
            printf("  [%d] cudaDeviceGetPCIBusId failed\n", d);
            continue;
        }
        void * h = nullptr;
        if (nvml_hpci(pci, &h) != 0) {
            printf("  [%d] %s nvml handle failed\n", d, pci);
            continue;
        }
        struct { uint64_t total; uint64_t free; uint64_t used; } m0 = {};
        struct { uint64_t total; uint64_t free; uint64_t used; } m1 = {};
        // nvml free read while no cuda context exists yet
        nvml_mem(h, &m0);
        size_t cfree  = 0;
        size_t ctotal = 0;
        cuda_set(d);
        // this call materializes the primary context on the device
        cuda_mem(&cfree, &ctotal);
        // nvml free read once the context is alive
        nvml_mem(h, &m1);
        long long delta = (long long) m0.free - (long long) m1.free;
        printf("  [%d] %s\n", d, pci);
        printf("       nvml free before : %8.1f MiB\n", to_mib((size_t) m0.free));
        printf("       cudaMemGetInfo   : %8.1f MiB free / %8.1f MiB total\n", to_mib(cfree), to_mib(ctotal));
        printf("       nvml free after  : %8.1f MiB\n", to_mib((size_t) m1.free));
        printf("       context cost     : %8.1f MiB\n", (double) delta / (1024.0 * 1024.0));
    }
}

int main(void) {
    stage1_backend_agnostic();
    printf("\n");
    stage2_cuda_two_modes();
    return 0;
}
