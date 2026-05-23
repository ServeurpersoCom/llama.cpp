// Qwen3-Omni code2wav streaming detokenizer, self contained.
// Consumes 16 group RVQ talker codes and produces a 24kHz mono waveform.
// No dependency on mtmd or llama : mtmd includes this and wraps the endpoint,
// the dependency arrow points one way so coupling stays low.

#pragma once

#include "mtmd-api.h"

#include <cstdint>
#include <vector>

struct code2wav_context;

// decode params, also the streaming latency knob.
// chunk_size frames per window, left_context frames of lookback trimmed after.
struct code2wav_params {
    int  chunk_size   = 300;   // HF default, lower it to cut streaming latency
    int  left_context = 25;    // overlap frames, trimmed from each window front
    bool use_gpu      = true;  // falls back to cpu when no gpu backend is found
};

// load weights and hparams from a code2wav gguf and pick a backend.
// returns nullptr on failure.
MTMD_API code2wav_context * code2wav_init(const char * gguf_path, code2wav_params params);
MTMD_API void               code2wav_free(code2wav_context * ctx);

// model facts the caller needs to wire the stream.
MTMD_API uint32_t code2wav_sample_rate   (const code2wav_context * ctx);  // 24000
MTMD_API int      code2wav_n_quantizers  (const code2wav_context * ctx);  // 16
MTMD_API int      code2wav_total_upsample(const code2wav_context * ctx);  // 1920

// one shot decode of all frames. codes are frame major, n_quantizers per frame,
// raw values 0..codebook_size, the module folds the per group offset itself.
// this is the harness path, it runs the same chunked logic as the stream.
MTMD_API std::vector<float> code2wav_decode(code2wav_context * ctx, const int32_t * codes, int n_frames);

// streaming path : feed frames as the LM samples them, pull pcm as windows fill.
// reset clears the frame history.
// push appends n_frames, decodes every window that is now complete, and appends
// the new samples to out (out is never cleared by the module).
// flush decodes the remaining tail once the LM stops.
// push calls followed by flush produce byte for byte the same pcm as decode.
MTMD_API void code2wav_stream_reset(code2wav_context * ctx);
MTMD_API void code2wav_stream_push (code2wav_context * ctx, const int32_t * codes, int n_frames, std::vector<float> & out);
MTMD_API void code2wav_stream_flush(code2wav_context * ctx, std::vector<float> & out);
