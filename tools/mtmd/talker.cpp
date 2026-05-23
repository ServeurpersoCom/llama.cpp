// Qwen3-Omni talker implementation, pure ggml.
// The body is a Qwen3 MoE decoder, the code predictor (MTP) is 5 dense layers
// with 15 stacked heads and 15 stacked codec embeddings. Each frame the body
// emits cb0 through the codec head, the predictor emits the 15 residual codes,
// then the next conditioning embed is rebuilt from those codes plus the trailing
// text hidden (or the tts pad once trailing runs out) and fed back.

#include "talker.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define TK_ARCH "qwen3omni-talker"

// body hparams plus the mtp config that the gguf does not key separately.
struct tk_hparams {
    int   n_layer;        // body blocks
    int   n_embd;
    int   n_head;
    int   n_head_kv;      // body kv heads
    int   head_dim;
    int   n_ff_exp;
    int   n_expert;
    int   n_expert_used;
    int   n_ff_shexp;
    float rms_eps;
    float rope_theta;
    int   mtp_layer;      // count of dense mtp blocks
    int   mtp_head_kv;    // mtp kv heads, differ from the body value
    int   mtp_vocab;      // residual codebook size
    int   n_codebooks;    // cb0 plus residuals
    int   codec_eos;
};

struct tk_body_layer {
    ggml_tensor * attn_norm;
    ggml_tensor * wq, * wk, * wv, * wo;
    ggml_tensor * q_norm, * k_norm;
    ggml_tensor * ffn_norm;
    ggml_tensor * gate_inp;
    ggml_tensor * gate_exps, * up_exps, * down_exps;
    ggml_tensor * gate_inp_shexp;
    ggml_tensor * gate_shexp, * up_shexp, * down_shexp;
};

struct tk_mtp_layer {
    ggml_tensor * attn_norm, * wq, * wk, * wv, * wo, * q_norm, * k_norm;
    ggml_tensor * ffn_norm, * gate, * up, * down;
};

struct talker_context {
    talker_params params;
    ggml_backend_t backend;
    ggml_context * ctx_data;
    gguf_context * gguf;
    tk_hparams hp;

    std::vector<tk_body_layer> body;
    std::vector<tk_mtp_layer>  mtp;
    ggml_tensor * out_norm;     // body final norm
    ggml_tensor * codec_head;   // cb0 head, output.weight
    ggml_tensor * codec_embd;   // cb0 embed table, codec_embd.weight
    ggml_tensor * mtp_norm;     // mtp.output_norm.weight
    ggml_tensor * mtp_head;     // mtp.lm_head.weight, stacked [n_embd, vocab, 15]
    ggml_tensor * mtp_embd;     // mtp.codec_embd.weight, stacked [n_embd, vocab, 15]
};

static ggml_tensor * tk_get(ggml_context * ctx, const std::string & n) {
    ggml_tensor * t = ggml_get_tensor(ctx, n.c_str());
    if (!t) { fprintf(stderr, "talker: missing tensor %s\n", n.c_str()); return nullptr; }
    return t;
}

static uint32_t tk_u32(gguf_context * g, const char * k, uint32_t def) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? def : gguf_get_val_u32(g, id);
}

static float tk_f32(gguf_context * g, const char * k, float def) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? def : gguf_get_val_f32(g, id);
}

static int tk_argmax(const float * v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}

// RMSNorm over ne0 then scale by w.
static ggml_tensor * tk_rmsnorm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// one Qwen3 MoE body layer with kv cache. x is the new token slice [n_embd, n_new],
// past k/v hold the prior positions. attends over past then new with a causal mask.
static ggml_tensor * tk_body_layer_kv(ggml_context * ctx, const tk_hparams & hp, const tk_body_layer & l,
                                      ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask,
                                      ggml_tensor * pastk, ggml_tensor * pastv,
                                      ggml_tensor ** newk, ggml_tensor ** newv) {
    const int n_new = (int) x->ne[1];
    const int hd    = hp.head_dim;
    const int kvd   = hd * hp.n_head_kv;
    const int T     = (pastk ? (int) pastk->ne[1] : 0) + n_new;

    ggml_tensor * cur = tk_rmsnorm(ctx, x, l.attn_norm, hp.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wq, cur), hd, hp.n_head,    n_new);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wk, cur), hd, hp.n_head_kv, n_new);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wv, cur), hd, hp.n_head_kv, n_new);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.rms_eps), l.q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.rms_eps), l.k_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // expose the new k/v for cache writeback, flattened [kv_dim, n_new]
    ggml_tensor * kf = ggml_reshape_2d(ctx, ggml_cont(ctx, k), kvd, n_new);
    ggml_tensor * vf = ggml_reshape_2d(ctx, ggml_cont(ctx, v), kvd, n_new);
    *newk = kf; *newv = vf;

    ggml_tensor * kall = pastk ? ggml_concat(ctx, pastk, kf, 1) : kf;
    ggml_tensor * vall = pastv ? ggml_concat(ctx, pastv, vf, 1) : vf;
    kall = ggml_reshape_3d(ctx, kall, hd, hp.n_head_kv, T);
    vall = ggml_reshape_3d(ctx, vall, hd, hp.n_head_kv, T);

    ggml_tensor * qh = ggml_permute(ctx, q,    0, 2, 1, 3);
    ggml_tensor * kh = ggml_permute(ctx, kall, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);
    ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, vall, 1, 2, 0, 3));
    ggml_tensor * kqv = ggml_permute(ctx, ggml_mul_mat(ctx, vh, kq), 0, 2, 1, 3);
    cur = ggml_mul_mat(ctx, l.wo, ggml_cont_2d(ctx, kqv, hd * hp.n_head, n_new));

    ggml_tensor * ffn_in = ggml_add(ctx, x, cur);

    // moe ffn : softmax over experts, top_k, renorm, plus a sigmoid gated shared expert
    cur = tk_rmsnorm(ctx, ffn_in, l.ffn_norm, hp.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, l.gate_inp, cur);
    ggml_tensor * probs  = ggml_soft_max(ctx, logits);
    ggml_tensor * sel    = ggml_top_k(ctx, probs, hp.n_expert_used);
    ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, hp.n_expert, n_new), sel);
    w = ggml_reshape_2d(ctx, w, hp.n_expert_used, n_new);
    ggml_tensor * wsum = ggml_sum_rows(ctx, w);
    w = ggml_div(ctx, w, wsum);
    w = ggml_reshape_3d(ctx, w, 1, hp.n_expert_used, n_new);

    ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, n_new);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, l.up_exps,   cur3, sel);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, l.gate_exps, cur3, sel);
    gate = ggml_silu(ctx, gate);
    ggml_tensor * g = ggml_mul(ctx, gate, up);
    ggml_tensor * exp_out = ggml_mul_mat_id(ctx, l.down_exps, g, sel);
    exp_out = ggml_mul(ctx, exp_out, w);

    ggml_tensor * moe = ggml_view_2d(ctx, exp_out, hp.n_embd, n_new, exp_out->nb[2], 0);
    for (int e = 1; e < hp.n_expert_used; e++) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, exp_out, hp.n_embd, n_new, exp_out->nb[2], e * exp_out->nb[1]));
    }

    ggml_tensor * sg = ggml_silu(ctx, ggml_mul_mat(ctx, l.gate_shexp, cur));
    ggml_tensor * su = ggml_mul_mat(ctx, l.up_shexp, cur);
    ggml_tensor * sh = ggml_mul_mat(ctx, l.down_shexp, ggml_mul(ctx, sg, su));
    ggml_tensor * sgate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, l.gate_inp_shexp, cur));
    sh = ggml_mul(ctx, sh, sgate);

    cur = ggml_add(ctx, moe, sh);
    return ggml_add(ctx, ffn_in, cur);
}

// one dense MTP layer with kv cache, gqa 16/8, dense SwiGLU.
static ggml_tensor * tk_mtp_layer_kv(ggml_context * ctx, const tk_hparams & hp, const tk_mtp_layer & l,
                                     ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask,
                                     ggml_tensor * pastk, ggml_tensor * pastv,
                                     ggml_tensor ** newk, ggml_tensor ** newv) {
    const int n_new = (int) x->ne[1];
    const int hd    = hp.head_dim;
    const int kvd   = hd * hp.mtp_head_kv;
    const int T     = (pastk ? (int) pastk->ne[1] : 0) + n_new;

    ggml_tensor * cur = tk_rmsnorm(ctx, x, l.attn_norm, hp.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wq, cur), hd, hp.n_head,     n_new);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wk, cur), hd, hp.mtp_head_kv, n_new);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wv, cur), hd, hp.mtp_head_kv, n_new);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.rms_eps), l.q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.rms_eps), l.k_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kf = ggml_reshape_2d(ctx, ggml_cont(ctx, k), kvd, n_new);
    ggml_tensor * vf = ggml_reshape_2d(ctx, ggml_cont(ctx, v), kvd, n_new);
    *newk = kf; *newv = vf;

    ggml_tensor * kall = pastk ? ggml_concat(ctx, pastk, kf, 1) : kf;
    ggml_tensor * vall = pastv ? ggml_concat(ctx, pastv, vf, 1) : vf;
    kall = ggml_reshape_3d(ctx, kall, hd, hp.mtp_head_kv, T);
    vall = ggml_reshape_3d(ctx, vall, hd, hp.mtp_head_kv, T);

    ggml_tensor * qh = ggml_permute(ctx, q,    0, 2, 1, 3);
    ggml_tensor * kh = ggml_permute(ctx, kall, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);
    ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, vall, 1, 2, 0, 3));
    ggml_tensor * kqv = ggml_permute(ctx, ggml_mul_mat(ctx, vh, kq), 0, 2, 1, 3);
    cur = ggml_mul_mat(ctx, l.wo, ggml_cont_2d(ctx, kqv, hd * hp.n_head, n_new));

    ggml_tensor * ffn_in = ggml_add(ctx, x, cur);

    cur = tk_rmsnorm(ctx, ffn_in, l.ffn_norm, hp.rms_eps);
    ggml_tensor * gg = ggml_silu(ctx, ggml_mul_mat(ctx, l.gate, cur));
    ggml_tensor * u  = ggml_mul_mat(ctx, l.up, cur);
    cur = ggml_mul_mat(ctx, l.down, ggml_mul(ctx, gg, u));
    return ggml_add(ctx, ffn_in, cur);
}

// run the body over a full prefix of embeds [n_embd, T] with a causal mask and a
// scalar position counter starting at pos0. recompute prefix, byte exact against
// the reference. returns the last position post norm hidden and the cb0 logits.
static void tk_body_recompute(talker_context * ctx, const std::vector<float> & embeds, int T, int pos0,
                              std::vector<float> & hidden_out, std::vector<float> & cb0_logits) {
    const tk_hparams & hp = ctx->hp;
    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 16 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 16, false);
    ggml_init_params ip; ip.mem_size = meta; ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, GGML_DEFAULT_GRAPH_SIZE * 16, false);

    ggml_tensor * inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.n_embd, T); ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, T); ggml_set_input(mask);

    ggml_tensor * cur = inp;
    for (int i = 0; i < hp.n_layer; i++) {
        ggml_tensor * nk, * nv;
        cur = tk_body_layer_kv(c, hp, ctx->body[i], cur, pos, mask, nullptr, nullptr, &nk, &nv);
    }
    ggml_tensor * normed = tk_rmsnorm(c, cur, ctx->out_norm, hp.rms_eps); ggml_set_output(normed);
    ggml_tensor * logits = ggml_mul_mat(c, ctx->codec_head, normed); ggml_set_output(logits);
    ggml_build_forward_expand(gf, normed);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(inp, embeds.data(), 0, (size_t) hp.n_embd * T * sizeof(float));
    std::vector<int32_t> posd(T);
    for (int i = 0; i < T; i++) posd[i] = pos0 + i;
    ggml_backend_tensor_set(pos, posd.data(), 0, posd.size() * sizeof(int32_t));
    std::vector<float> md((size_t) T * T);
    for (int q = 0; q < T; q++) {
        for (int kk = 0; kk < T; kk++) md[(size_t) q * T + kk] = kk <= q ? 0.0f : -INFINITY;
    }
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    const int vocab = (int) logits->ne[0];
    hidden_out.resize(hp.n_embd);
    cb0_logits.resize(vocab);
    ggml_backend_tensor_get(normed, hidden_out.data(), (size_t)(T - 1) * hp.n_embd * sizeof(float), hp.n_embd * sizeof(float));
    ggml_backend_tensor_get(logits, cb0_logits.data(), (size_t)(T - 1) * vocab * sizeof(float), vocab * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(c);
}

// embed a code through a stacked table [n_embd, vocab, n] slice g, or a plain 2d
// table when n is 1. returns the [n_embd] row.
static std::vector<float> tk_embed(talker_context * ctx, ggml_tensor * tbl, int g, int code) {
    ggml_init_params ip; ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead(); ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(c);

    ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1); ggml_set_input(ids);
    ggml_tensor * t2 = tbl->ne[2] > 1
        ? ggml_view_2d(c, tbl, ctx->hp.n_embd, tbl->ne[1], tbl->nb[1], (size_t) g * tbl->nb[2])
        : tbl;
    ggml_tensor * row = ggml_get_rows(c, t2, ids); ggml_set_output(row);
    ggml_build_forward_expand(gf, row);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    int32_t id = code;
    ggml_backend_tensor_set(ids, &id, 0, sizeof(int32_t));
    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> out(ctx->hp.n_embd);
    ggml_backend_tensor_get(row, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(c);
    return out;
}

// run the dense mtp over a full prefix of embeds [n_embd, T] with a causal mask,
// return the last position logits for head. recompute prefix, the mtp is tiny so
// this is cheap and byte exact against the reference.
static std::vector<float> tk_mtp_recompute(talker_context * ctx, const std::vector<float> & embeds, int T, int head) {
    const tk_hparams & hp = ctx->hp;
    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_init_params ip; ip.mem_size = meta; ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, GGML_DEFAULT_GRAPH_SIZE * 8, false);

    ggml_tensor * inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.n_embd, T); ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, T); ggml_set_input(mask);

    ggml_tensor * cur = inp;
    for (int i = 0; i < hp.mtp_layer; i++) {
        ggml_tensor * nk, * nv;
        cur = tk_mtp_layer_kv(c, hp, ctx->mtp[i], cur, pos, mask, nullptr, nullptr, &nk, &nv);
    }
    ggml_tensor * normed = tk_rmsnorm(c, cur, ctx->mtp_norm, hp.rms_eps);
    ggml_tensor * w = ggml_view_2d(c, ctx->mtp_head, hp.n_embd, hp.mtp_vocab, ctx->mtp_head->nb[1], (size_t) head * ctx->mtp_head->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(c, w, normed); ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(inp, embeds.data(), 0, (size_t) hp.n_embd * T * sizeof(float));
    std::vector<int32_t> posd(T);
    for (int i = 0; i < T; i++) posd[i] = i;
    ggml_backend_tensor_set(pos, posd.data(), 0, posd.size() * sizeof(int32_t));
    std::vector<float> md((size_t) T * T);
    for (int q = 0; q < T; q++) {
        for (int kk = 0; kk < T; kk++) md[(size_t) q * T + kk] = kk <= q ? 0.0f : -INFINITY;
    }
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    const int vocab = (int) logits->ne[0];
    std::vector<float> out(vocab);
    ggml_backend_tensor_get(logits, out.data(), (size_t)(T - 1) * vocab * sizeof(float), vocab * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(c);
    return out;
}

// generate the 15 residual codes for one frame : prefill [body_hidden, embed(cb0)]
// then 14 ar steps, recompute prefix each step.
static void tk_predict_residuals(talker_context * ctx, const std::vector<float> & body_hidden, int cb0, int * res) {
    const tk_hparams & hp = ctx->hp;
    std::vector<float> embeds = body_hidden;
    std::vector<float> e0 = tk_embed(ctx, ctx->codec_embd, 0, cb0);
    embeds.insert(embeds.end(), e0.begin(), e0.end());
    int T = 2;

    std::vector<float> lg = tk_mtp_recompute(ctx, embeds, T, 0);
    res[0] = tk_argmax(lg.data(), hp.mtp_vocab);
    for (int gi = 1; gi < hp.n_codebooks - 1; gi++) {
        std::vector<float> e = tk_embed(ctx, ctx->mtp_embd, gi - 1, res[gi - 1]);
        embeds.insert(embeds.end(), e.begin(), e.end());
        T++;
        std::vector<float> l = tk_mtp_recompute(ctx, embeds, T, gi);
        res[gi] = tk_argmax(l.data(), hp.mtp_vocab);
    }
}

talker_context * talker_init(const char * gguf_path, talker_params params) {
    talker_context * ctx = new talker_context();
    ctx->params = params;

    ctx->backend = params.use_gpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : nullptr;
    if (!ctx->backend) ctx->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ctx->ctx_data = nullptr;
    gguf_init_params gp; gp.no_alloc = true; gp.ctx = &ctx->ctx_data;
    ctx->gguf = gguf_init_from_file(gguf_path, gp);
    if (!ctx->gguf) { fprintf(stderr, "talker: cannot load %s\n", gguf_path); talker_free(ctx); return nullptr; }
    gguf_context * g = ctx->gguf;

    tk_hparams & hp = ctx->hp;
    const int block_count = (int) tk_u32(g, TK_ARCH ".block_count", 25);
    hp.mtp_layer    = 5;
    hp.n_layer      = block_count - hp.mtp_layer;
    hp.n_embd       = (int) tk_u32(g, TK_ARCH ".embedding_length", 1024);
    hp.n_head       = (int) tk_u32(g, TK_ARCH ".attention.head_count", 16);
    hp.n_head_kv    = (int) tk_u32(g, TK_ARCH ".attention.head_count_kv", 2);
    hp.head_dim     = (int) tk_u32(g, TK_ARCH ".attention.key_length", 128);
    hp.n_ff_exp     = (int) tk_u32(g, TK_ARCH ".expert_feed_forward_length", 384);
    hp.n_expert     = (int) tk_u32(g, TK_ARCH ".expert_count", 128);
    hp.n_expert_used = (int) tk_u32(g, TK_ARCH ".expert_used_count", 6);
    hp.n_ff_shexp   = (int) tk_u32(g, TK_ARCH ".expert_shared_feed_forward_length", 768);
    hp.rms_eps      = tk_f32(g, TK_ARCH ".attention.layer_norm_rms_epsilon", 1e-6f);
    hp.rope_theta   = tk_f32(g, TK_ARCH ".rope.freq_base", 1000000.0f);
    hp.mtp_head_kv  = 8;
    hp.mtp_vocab    = 2048;
    hp.n_codebooks  = 16;
    hp.codec_eos    = 4198;

    ggml_backend_alloc_ctx_tensors(ctx->ctx_data, ctx->backend);
    FILE * wf = fopen(gguf_path, "rb");
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<char> tmp;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx->ctx_data); t; t = ggml_get_next_tensor(ctx->ctx_data, t)) {
        const int64_t tid = gguf_find_tensor(g, t->name);
        if (tid < 0) continue;
        const size_t off = data_off + gguf_get_tensor_offset(g, tid);
        const size_t nb = ggml_nbytes(t);
        tmp.resize(nb);
        fseek(wf, (long) off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, wf) != nb) { fclose(wf); talker_free(ctx); return nullptr; }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
    }
    fclose(wf);

    ctx->body.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        tk_body_layer & l = ctx->body[i];
        l.attn_norm      = tk_get(ctx->ctx_data, p + "attn_norm.weight");
        l.wq             = tk_get(ctx->ctx_data, p + "attn_q.weight");
        l.wk             = tk_get(ctx->ctx_data, p + "attn_k.weight");
        l.wv             = tk_get(ctx->ctx_data, p + "attn_v.weight");
        l.wo             = tk_get(ctx->ctx_data, p + "attn_output.weight");
        l.q_norm         = tk_get(ctx->ctx_data, p + "attn_q_norm.weight");
        l.k_norm         = tk_get(ctx->ctx_data, p + "attn_k_norm.weight");
        l.ffn_norm       = tk_get(ctx->ctx_data, p + "ffn_norm.weight");
        l.gate_inp       = tk_get(ctx->ctx_data, p + "ffn_gate_inp.weight");
        l.gate_exps      = tk_get(ctx->ctx_data, p + "ffn_gate_exps.weight");
        l.up_exps        = tk_get(ctx->ctx_data, p + "ffn_up_exps.weight");
        l.down_exps      = tk_get(ctx->ctx_data, p + "ffn_down_exps.weight");
        l.gate_inp_shexp = tk_get(ctx->ctx_data, p + "ffn_gate_inp_shexp.weight");
        l.gate_shexp     = tk_get(ctx->ctx_data, p + "ffn_gate_shexp.weight");
        l.up_shexp       = tk_get(ctx->ctx_data, p + "ffn_up_shexp.weight");
        l.down_shexp     = tk_get(ctx->ctx_data, p + "ffn_down_shexp.weight");
    }

    ctx->mtp.resize(hp.mtp_layer);
    for (int i = 0; i < hp.mtp_layer; i++) {
        const std::string p = "blk." + std::to_string(hp.n_layer + i) + ".mtp.";
        tk_mtp_layer & l = ctx->mtp[i];
        l.attn_norm = tk_get(ctx->ctx_data, p + "attn_norm.weight");
        l.wq        = tk_get(ctx->ctx_data, p + "attn_q.weight");
        l.wk        = tk_get(ctx->ctx_data, p + "attn_k.weight");
        l.wv        = tk_get(ctx->ctx_data, p + "attn_v.weight");
        l.wo        = tk_get(ctx->ctx_data, p + "attn_output.weight");
        l.q_norm    = tk_get(ctx->ctx_data, p + "attn_q_norm.weight");
        l.k_norm    = tk_get(ctx->ctx_data, p + "attn_k_norm.weight");
        l.ffn_norm  = tk_get(ctx->ctx_data, p + "ffn_norm.weight");
        l.gate      = tk_get(ctx->ctx_data, p + "ffn_gate.weight");
        l.up        = tk_get(ctx->ctx_data, p + "ffn_up.weight");
        l.down      = tk_get(ctx->ctx_data, p + "ffn_down.weight");
    }

    ctx->out_norm   = tk_get(ctx->ctx_data, "output_norm.weight");
    ctx->codec_head = tk_get(ctx->ctx_data, "output.weight");
    ctx->codec_embd = tk_get(ctx->ctx_data, "codec_embd.weight");
    ctx->mtp_norm   = tk_get(ctx->ctx_data, "mtp.output_norm.weight");
    ctx->mtp_head   = tk_get(ctx->ctx_data, "mtp.lm_head.weight");
    ctx->mtp_embd   = tk_get(ctx->ctx_data, "mtp.codec_embd.weight");

    return ctx;
}

void talker_free(talker_context * ctx) {
    if (!ctx) return;
    if (ctx->gguf) gguf_free(ctx->gguf);
    if (ctx->ctx_data) ggml_free(ctx->ctx_data);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int talker_n_embd     (const talker_context * ctx) { return ctx->hp.n_embd; }
int talker_n_codebooks(const talker_context * ctx) { return ctx->hp.n_codebooks; }
int talker_codec_eos  (const talker_context * ctx) { return ctx->hp.codec_eos; }

int talker_generate(talker_context * ctx, const talker_cond & cond, std::vector<int32_t> & codes) {
    const tk_hparams & hp = ctx->hp;
    codes.clear();

    // scalar position start from the last prefill mrope column (rows are equal here)
    int pos0 = cond.prefill_pos
        ? cond.prefill_pos[(size_t) 2 * cond.n_prefill + (cond.n_prefill - 1)] - (cond.n_prefill - 1)
        : 0;

    // the growing embed prefix, starts as the assembled prefill
    std::vector<float> prefix(cond.input_embed, cond.input_embed + (size_t) cond.n_prefill * hp.n_embd);
    int T = cond.n_prefill;

    std::vector<float> hidden, cb0_logits;
    tk_body_recompute(ctx, prefix, T, pos0, hidden, cb0_logits);

    int step = 0;
    while (step < ctx->params.max_frames) {
        int cb0 = tk_argmax(cb0_logits.data(), (int) cb0_logits.size());
        if (cb0 == hp.codec_eos) break;

        int res[15];
        tk_predict_residuals(ctx, hidden, cb0, res);

        codes.push_back(cb0);
        for (int k = 0; k < hp.n_codebooks - 1; k++) codes.push_back(res[k]);

        // rebuild the next conditioning embed : codec_embd(cb0) + sum of the 15
        // residual embeds through their stacked tables, plus the text conditioning
        std::vector<float> cond_e = tk_embed(ctx, ctx->codec_embd, 0, cb0);
        for (int k = 1; k <= hp.n_codebooks - 1; k++) {
            std::vector<float> e = tk_embed(ctx, ctx->mtp_embd, k - 1, res[k - 1]);
            for (int i = 0; i < hp.n_embd; i++) cond_e[i] += e[i];
        }
        const float * txt = step < cond.n_trail ? cond.trailing_text + (size_t) step * hp.n_embd : cond.tts_pad;
        for (int i = 0; i < hp.n_embd; i++) cond_e[i] += txt[i];

        // append the new frame and recompute the body over the grown prefix
        prefix.insert(prefix.end(), cond_e.begin(), cond_e.end());
        T++;
        tk_body_recompute(ctx, prefix, T, pos0, hidden, cb0_logits);
        step++;
    }

    return (int) (codes.size() / hp.n_codebooks);
}
