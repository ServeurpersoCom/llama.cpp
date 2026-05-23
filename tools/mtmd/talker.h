// Qwen3-Omni talker, self contained.
// Runs the MoE talker body plus the dense MTP code predictor autoregressively
// and produces 16 group RVQ codes per frame, the input that code2wav vocodes.
// No dependency on mtmd or llama : mtmd includes this and wraps the endpoint,
// the dependency arrow points one way so coupling stays low.

#pragma once

#include <cstdint>
#include <vector>

struct talker_context;

// generation params.
// max_frames caps the run, the loop also stops on the codec eos code.
// the body samples cb0 greedily, the predictor samples the 15 residuals greedily,
// which matches the reference forward up to bf16 tie flips on near equal logits.
struct talker_params {
    int  max_frames = 4096;   // hard cap, the eos code stops earlier in practice
    bool use_gpu    = true;   // falls back to cpu when no gpu backend is found
};

// the per frame conditioning the thinker hands the talker. all hidden vectors are
// n_embd wide. the talker reads these and never depends on the thinker runtime.
//   input_embed       prefill embeds assembled by the thinker, [n_embd, n_prefill]
//   trailing_text     thinker text hidden after the first token, [n_embd, n_trail]
//   tts_pad           the tts pad embed used once trailing is exhausted, [n_embd]
//   prefill_pos       3d mrope positions for the prefill, [3, n_prefill] (i32)
// the three rope rows are equal in the tts case, the talker uses a scalar counter.
struct talker_cond {
    const float *   input_embed;
    int             n_prefill;
    const float *   trailing_text;
    int             n_trail;
    const float *   tts_pad;
    const int32_t * prefill_pos;
};

// load weights and hparams from a talker gguf and pick a backend.
// the gguf carries the body blocks blk.0..N, the mtp blocks blk.N.mtp.*, the
// codec head output.weight, the codec embed codec_embd.weight and mtp.* tensors.
// returns nullptr on failure.
talker_context * talker_init(const char * gguf_path, talker_params params);
void             talker_free(talker_context * ctx);

// model facts the caller needs.
int talker_n_embd       (const talker_context * ctx);  // 1024
int talker_n_codebooks  (const talker_context * ctx);  // 16, cb0 plus 15 residuals
int talker_codec_eos    (const talker_context * ctx);  // 4198

// one shot generation. runs the body and the predictor frame by frame with kv
// cache until the eos code or max_frames, fills codes frame major (n_codebooks
// per frame, cb0 first) and returns the frame count. codes is resized by the call.
// this is the harness and endpoint path, both share the same loop.
int talker_generate(talker_context * ctx, const talker_cond & cond, std::vector<int32_t> & codes);
