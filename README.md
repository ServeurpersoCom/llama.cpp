# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

> **This fork:** branch `moe-stream-master` is upstream `master` plus SSD streaming of
> Mixture-of-Experts weights, so MoE models much larger than RAM run with a bounded
> memory footprint. See [Running MoE models that do not fit in RAM](#running-moe-models-that-do-not-fit-in-ram).
> Everything else is unchanged; without the `--moe-stream*` flags it behaves like stock llama.cpp.

## Running MoE models that do not fit in RAM

In a Mixture-of-Experts model most of the weights are routed experts, and each token
uses only a few of them. GLM-5.3-Flash, for example, is 320B parameters, but 304B of
those are experts and a token touches 8 of the 288 experts in each of 42 layers. Stock
llama.cpp still has to hold every expert in memory (or let `mmap` page them through the
file cache, which is slow and uncontrolled).

With `--moe-stream`, the routed-expert tensors are never loaded. Each layer gets a cache
of expert *slots* on the compute device; after the router picks the experts for a token,
missing ones are read from the GGUF file by a pool of I/O threads and evicted by a
decaying hotness score. Routing decisions are unchanged, so output is bit-identical to a
fully resident run. Only latency changes. Everything that is not a routed expert
(attention, routers, shared experts, embeddings, output head, KV cache) stays resident
as usual, so that is the memory floor.

### Flags

| Flag | Meaning |
| --- | --- |
| `--moe-stream` | enable expert streaming |
| `--moe-stream-cache <N\|Ns>` | cache budget in GiB (`40`), or exact slots per layer (`64s`); implies `--moe-stream` |
| `--moe-stream-direct` | bypass the OS page cache (O_DIRECT on Linux, F_NOCACHE on macOS); use when the model far exceeds RAM, or to benchmark a smaller machine honestly |
| `--moe-stream-io-threads N` | reader threads (default: auto) |
| `--moe-stream-ram N` | pinned host mirror in GiB for discrete GPUs; set `0` on unified-memory machines such as Apple Silicon |
| `LLAMA_MOE_STREAM_DYN=N` (env) | size of the hotness-ranked pool. Slots beyond it are pinned by expert ID; set it equal to the slot count to make the whole cache hotness-ranked, which measured better |

The dynamic pool must hold at least `3 * n_expert_used` slots (24 for an 8-expert
model) for chunked prefill, so use at least that many slots.

### Example: GLM-5.3-Flash on a Mac

```sh
LLAMA_MOE_STREAM_DYN=24 llama-server \
    -m GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
    --moe-stream-cache 24s --moe-stream-ram 0 --moe-stream-direct -c 8192
```

Measured on an M5 Max with the 157 GB `UD-IQ4_XS` GGUF, 480-token prompt, 256
generated tokens, `temp 0`, Metal, page cache bypassed except where noted:

| Slots per layer | Expert cache | Peak footprint | Hit rate | Prefill | Decode | Fits a |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 (buffered) | 11.4 GB | 22.8 GB | 50% | 24.5 tok/s | 3.1 tok/s | 32 GB Mac |
| 24 | 11.4 GB | 22.7 GB | 50% | 26.2 tok/s | 2.7 tok/s | 32 GB Mac |
| 48 | 22.8 GB | 34.8 GB | 62% | 35.2 tok/s | 3.7 tok/s | 48 GB Mac |
| 96 | 45.7 GB | 58.8 GB | 76% | 44.4 tok/s | 4.9 tok/s | 64 GB Mac |
| 160 | 76.1 GB | 90.1 GB | 85% | 45.2 tok/s | 7.5 tok/s | 128 GB Mac |

The resident part was 9 GB (core weights) plus about 0.5 GB of KV and compute buffers at
8K context. All five runs produced identical text. Decode is bound by SSD bandwidth
times miss rate: roughly 6 GB/s from the internal SSD here, with 14 MB per expert read.

### Sizing a cache

1. Find the model's fixed cost: load once with a small cache and read the
   `model buffer size`, `KV buffer size` and `compute buffer size` lines (run with `-lv 4`).
2. Subtract that and about 4 GB for the OS from your RAM; the rest is the expert cache.
3. Slot size is bytes per expert at your quantization; the loader prints
   `expert cache size = ... (N slots per layer)` so you can check the arithmetic.
4. More slots raise the hit rate roughly with the logarithm of cache size. Pass
   `--moe-stream-direct` when the model is much larger than RAM, otherwise leave it off
   and let the page cache help.

Higher-precision quantizations cost more per slot and per miss but nothing extra on
disk, so streaming lets a quality-first setup keep experts at 6- or 8-bit at the price
of tokens per second rather than of RAM.

### Notes

- Works for any MoE architecture, since it operates on the generic `ffn_*_exps` tensors.
  Dense models are unaffected by the flags.
- `glm5next` GGUFs (converted with Unsloth's pre-merge PR) load as `glm5-next` on this
  branch.
- One context per streamed model: concurrent decodes on the same model can evict each
  other's experts.
- Upstream tracking: this is [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)
  and [ServeurpersoCom's partition rebase](https://github.com/ServeurpersoCom/llama.cpp/tree/moe-stream-partition),
  rebased onto master with macOS fixes.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
