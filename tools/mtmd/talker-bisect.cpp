// Qwen3-Omni talker bisect harness.
// Builds the 20 layer Qwen3-MoE talker body from the gguf, feeds the HF prefill
// input embeds, and dumps every layer output so a python script can cosine each
// stage against the HF reference. teacher forced, deterministic, no sampling.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define TK_ARCH "qwen3omni-talker"

struct tk_hparams {
    int   n_layer;      // main talker blocks
    int   n_embd;
    int   n_head;
    int   n_head_kv;
    int   head_dim;
    int   n_ff_exp;
    int   n_expert;
    int   n_expert_used;
    int   n_ff_shexp;
    float rms_eps;
    float rope_theta;
};

struct tk_layer {
    ggml_tensor * attn_norm;
    ggml_tensor * wq, * wk, * wv, * wo;
    ggml_tensor * q_norm, * k_norm;
    ggml_tensor * ffn_norm;
    ggml_tensor * gate_inp;
    ggml_tensor * gate_exps, * up_exps, * down_exps;
    ggml_tensor * gate_inp_shexp;
    ggml_tensor * gate_shexp, * up_shexp, * down_shexp;
};

static ggml_tensor * tk_get(ggml_context * ctx, const std::string & n) {
    ggml_tensor * t = ggml_get_tensor(ctx, n.c_str());
    if (!t) { fprintf(stderr, "talker: missing tensor %s\n", n.c_str()); exit(1); }
    return t;
}

static uint32_t tk_u32(gguf_context * g, const char * k) {
    int64_t id = gguf_find_key(g, k);
    if (id < 0) { fprintf(stderr, "talker: missing key %s\n", k); exit(1); }
    return gguf_get_val_u32(g, id);
}

static float tk_f32(gguf_context * g, const char * k) {
    int64_t id = gguf_find_key(g, k);
    if (id < 0) { fprintf(stderr, "talker: missing key %s\n", k); exit(1); }
    return gguf_get_val_f32(g, id);
}

// RMSNorm over ne0 then scale by w.
static ggml_tensor * tk_rmsnorm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// one Qwen3-MoE decoder layer, hidden is [n_embd, T].
static ggml_tensor * tk_build_layer(ggml_context * ctx, const tk_hparams & hp, const tk_layer & l,
                                     ggml_tensor * inpL, ggml_tensor * pos, ggml_tensor * mask) {
    const int T  = (int) inpL->ne[1];
    const int hd = hp.head_dim;

    // attention
    ggml_tensor * cur = tk_rmsnorm(ctx, inpL, l.attn_norm, hp.rms_eps);

    ggml_tensor * q = ggml_mul_mat(ctx, l.wq, cur);
    ggml_tensor * k = ggml_mul_mat(ctx, l.wk, cur);
    ggml_tensor * v = ggml_mul_mat(ctx, l.wv, cur);

    q = ggml_reshape_3d(ctx, q, hd, hp.n_head,    T);
    k = ggml_reshape_3d(ctx, k, hd, hp.n_head_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, hp.n_head_kv, T);

    // per head RMSNorm on q and k (q_norm/k_norm are [head_dim])
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.rms_eps), l.q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.rms_eps), l.k_norm);

    // NEOX rope
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // GQA attention, ggml_mul_mat broadcasts the kv heads over the q heads
    q = ggml_permute(ctx, q, 0, 2, 1, 3);                 // [hd, T, n_head]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);                 // [hd, T, n_head_kv]
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);           // [T, T, n_head]
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);

    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [T, hd, n_head_kv]
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);         // [hd, T, n_head]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);             // [hd, n_head, T]
    cur = ggml_cont_2d(ctx, kqv, hd * hp.n_head, T);
    cur = ggml_mul_mat(ctx, l.wo, cur);

    ggml_tensor * ffn_in = ggml_add(ctx, inpL, cur);

    // moe ffn
    cur = tk_rmsnorm(ctx, ffn_in, l.ffn_norm, hp.rms_eps);

    ggml_tensor * logits = ggml_mul_mat(ctx, l.gate_inp, cur);          // [n_expert, T]
    ggml_tensor * probs  = ggml_soft_max(ctx, logits);                 // [n_expert, T]
    ggml_tensor * sel    = ggml_top_k(ctx, probs, hp.n_expert_used);   // [n_used, T]

    ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, hp.n_expert, T), sel); // [1, n_used, T]
    w = ggml_reshape_2d(ctx, w, hp.n_expert_used, T);
    // norm_topk_prob : renormalize the kept weights
    ggml_tensor * wsum = ggml_sum_rows(ctx, w);                        // [1, T]
    w = ggml_div(ctx, w, wsum);
    w = ggml_reshape_3d(ctx, w, 1, hp.n_expert_used, T);

    ggml_tensor * cur3 = ggml_reshape_3d(ctx, cur, hp.n_embd, 1, T);   // broadcast over experts
    ggml_tensor * up   = ggml_mul_mat_id(ctx, l.up_exps,   cur3, sel); // [n_ff_exp, n_used, T]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, l.gate_exps, cur3, sel);
    gate = ggml_silu(ctx, gate);
    ggml_tensor * g = ggml_mul(ctx, gate, up);
    ggml_tensor * exp_out = ggml_mul_mat_id(ctx, l.down_exps, g, sel); // [n_embd, n_used, T]
    exp_out = ggml_mul(ctx, exp_out, w);

    // sum the n_used expert outputs
    ggml_tensor * moe = ggml_view_2d(ctx, exp_out, hp.n_embd, T, exp_out->nb[2], 0);
    for (int e = 1; e < hp.n_expert_used; e++) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, exp_out, hp.n_embd, T, exp_out->nb[2], e * exp_out->nb[1]));
    }

    // shared expert with its sigmoid gate
    ggml_tensor * sg = ggml_silu(ctx, ggml_mul_mat(ctx, l.gate_shexp, cur));
    ggml_tensor * su = ggml_mul_mat(ctx, l.up_shexp, cur);
    ggml_tensor * sh = ggml_mul_mat(ctx, l.down_shexp, ggml_mul(ctx, sg, su));
    ggml_tensor * sgate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, l.gate_inp_shexp, cur)); // [1, T]
    sh = ggml_mul(ctx, sh, sgate);

    cur = ggml_add(ctx, moe, sh);
    return ggml_add(ctx, ffn_in, cur);
}

static std::vector<float> read_bin(const char * path, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "talker: cannot open %s\n", path); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "talker: short read %s\n", path); exit(1); }
    fclose(f);
    return v;
}

static void write_bin(const char * path, const std::vector<float> & v) {
    FILE * f = fopen(path, "wb");
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

// one dense code predictor (MTP) layer, GQA 16/8 heads, dense SwiGLU ffn.
struct tk_mtp_layer {
    ggml_tensor * attn_norm, * wq, * wk, * wv, * wo, * q_norm, * k_norm;
    ggml_tensor * ffn_norm, * gate, * up, * down;
};

static ggml_tensor * tk_build_mtp_layer(ggml_context * ctx, int n_head, int n_head_kv,
                                         int hd, float eps, float theta, const tk_mtp_layer & l,
                                         ggml_tensor * inpL, ggml_tensor * pos, ggml_tensor * mask) {
    const int T = (int) inpL->ne[1];

    ggml_tensor * cur = tk_rmsnorm(ctx, inpL, l.attn_norm, eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wq, cur), hd, n_head,    T);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wk, cur), hd, n_head_kv, T);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, l.wv, cur), hd, n_head_kv, T);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), l.q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), l.k_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * kqv = ggml_permute(ctx, ggml_mul_mat(ctx, v, kq), 0, 2, 1, 3);
    cur = ggml_mul_mat(ctx, l.wo, ggml_cont_2d(ctx, kqv, hd * n_head, T));

    ggml_tensor * ffn_in = ggml_add(ctx, inpL, cur);

    // dense SwiGLU
    cur = tk_rmsnorm(ctx, ffn_in, l.ffn_norm, eps);
    ggml_tensor * g = ggml_silu(ctx, ggml_mul_mat(ctx, l.gate, cur));
    ggml_tensor * u = ggml_mul_mat(ctx, l.up, cur);
    cur = ggml_mul_mat(ctx, l.down, ggml_mul(ctx, g, u));
    return ggml_add(ctx, ffn_in, cur);
}

// loaded mtp weights and the dense layer config.
struct tk_mtp {
    int n_embd, n_head, n_head_kv, hd, vocab;
    float eps, theta;
    std::vector<tk_mtp_layer> L;
    ggml_tensor * onorm, * lm_head, * codec;
};

// argmax over a flat logit vector.
static int tk_argmax(const std::vector<float> & v) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < (int) v.size(); i++) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}

// run the 5 dense mtp layers over embeds [n_embd, T] then output_norm + lm_head[head].
// returns the last position logits. dumps cpp_mtp_layer_N.bin when dump_layers is set.
static std::vector<float> tk_mtp_logits(ggml_backend_t backend, const tk_mtp & m,
                                        const std::vector<float> & embeds, int T, int head,
                                        bool dump_layers, const std::string & ref_dir) {
    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_init_params ip; ip.mem_size = meta; ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.n_embd, T);
    ggml_set_name(inp, "mtp_inp"); ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "mtp_pos"); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "mtp_mask"); ggml_set_input(mask);

    std::vector<ggml_tensor *> louts;
    ggml_tensor * cur = inp;
    for (int i = 0; i < (int) m.L.size(); i++) {
        cur = tk_build_mtp_layer(ctx, m.n_head, m.n_head_kv, m.hd, m.eps, m.theta, m.L[i], cur, pos, mask);
        ggml_set_output(cur);
        louts.push_back(cur);
    }

    ggml_tensor * normed = tk_rmsnorm(ctx, cur, m.onorm, m.eps);

    // lm_head[head] is the 2d slice [n_embd, vocab] of the stacked [n_embd, vocab, 15] tensor
    ggml_tensor * w = ggml_view_2d(ctx, m.lm_head, m.n_embd, m.vocab, m.lm_head->nb[1], (size_t) head * m.lm_head->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(ctx, w, normed);
    ggml_set_output(logits);

    for (auto * o : louts) ggml_build_forward_expand(gf, o);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(inp, embeds.data(), 0, (size_t) m.n_embd * T * sizeof(float));
    std::vector<int32_t> posd(T);
    for (int i = 0; i < T; i++) posd[i] = i;
    ggml_backend_tensor_set(pos, posd.data(), 0, posd.size() * sizeof(int32_t));
    std::vector<float> md((size_t) T * T);
    for (int q = 0; q < T; q++) {
        for (int kk = 0; kk < T; kk++) md[(size_t) q * T + kk] = kk <= q ? 0.0f : -INFINITY;
    }
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    if (dump_layers) {
        for (int i = 0; i < (int) louts.size(); i++) {
            std::vector<float> out((size_t) m.n_embd * T);
            ggml_backend_tensor_get(louts[i], out.data(), 0, out.size() * sizeof(float));
            char path[512];
            snprintf(path, sizeof(path), "%s/cpp_mtp_layer_%d.bin", ref_dir.c_str(), i);
            write_bin(path, out);
        }
    }

    const int vocab = (int) logits->ne[0];
    std::vector<float> lv((size_t) vocab * T);
    ggml_backend_tensor_get(logits, lv.data(), 0, lv.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    return std::vector<float>(lv.begin() + (size_t)(T - 1) * vocab, lv.end());
}

// embed code through codec_embd[g], return its [n_embd] row as f32.
static std::vector<float> tk_mtp_embed(ggml_backend_t backend, const tk_mtp & m, int g, int code) {
    ggml_init_params ip; ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead(); ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(ids);

    // codec_embd[g] is the 2d slice [n_embd, vocab] of the stacked [n_embd, vocab, 15] tensor
    ggml_tensor * tbl = ggml_view_2d(ctx, m.codec, m.n_embd, m.vocab, m.codec->nb[1], (size_t) g * m.codec->nb[2]);
    ggml_tensor * row = ggml_get_rows(ctx, tbl, ids);
    ggml_set_output(row);
    ggml_build_forward_expand(gf, row);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    int32_t id = code;
    ggml_backend_tensor_set(ids, &id, 0, sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(m.n_embd);
    ggml_backend_tensor_get(row, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    return out;
}

// run the moe body over embeds [n_embd, T] with a scalar position counter and a
// causal mask, return the last position raw hidden and the cb0 logits.
// the body weights and the codec head come from the loaded layer set.
static void tk_body_step(ggml_backend_t backend, const tk_hparams & hp, const std::vector<tk_layer> & L,
                         ggml_tensor * onorm, ggml_tensor * outw,
                         const std::vector<float> & embeds, int T,
                         std::vector<float> & hidden_out, std::vector<float> & cb0_logits) {
    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_init_params ip; ip.mem_size = meta; ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, T);
    ggml_set_name(inp, "ar_inp"); ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "ar_pos"); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "ar_mask"); ggml_set_input(mask);

    ggml_tensor * cur = inp;
    for (int i = 0; i < hp.n_layer; i++) cur = tk_build_layer(ctx, hp, L[i], cur, pos, mask);
    ggml_set_output(cur);

    ggml_tensor * normed = tk_rmsnorm(ctx, cur, onorm, hp.rms_eps);
    ggml_set_output(normed);
    ggml_tensor * logits = ggml_mul_mat(ctx, outw, normed);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, normed);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
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

    ggml_backend_graph_compute(backend, gf);

    // post output_norm hidden at the last position, matches HF hidden_states[-1]
    hidden_out.resize(hp.n_embd);
    ggml_backend_tensor_get(normed, hidden_out.data(), (size_t)(T - 1) * hp.n_embd * sizeof(float), hp.n_embd * sizeof(float));

    // cb0 logits at the last position
    const int vocab = (int) logits->ne[0];
    cb0_logits.resize(vocab);
    ggml_backend_tensor_get(logits, cb0_logits.data(), (size_t)(T - 1) * vocab * sizeof(float), vocab * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

// embed code through a plain 2d table [n_embd, vocab], return its [n_embd] row.
static std::vector<float> tk_row(ggml_backend_t backend, ggml_tensor * tbl, int n_embd, int code) {
    ggml_init_params ip; ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead(); ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(ids);
    ggml_tensor * row = ggml_get_rows(ctx, tbl, ids);
    ggml_set_output(row);
    ggml_build_forward_expand(gf, row);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    int32_t id = code;
    ggml_backend_tensor_set(ids, &id, 0, sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(n_embd);
    ggml_backend_tensor_get(row, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    return out;
}

// generate the 15 residual codes for one frame given the body hidden and cb0.
// builds the mtp prefill [body_hidden, codec_main(cb0)] then the 14 step ar loop.
static void tk_mtp_frame(ggml_backend_t backend, const tk_mtp & m, ggml_tensor * codec_main,
                         const std::vector<float> & body_hidden, int cb0, int res_out[15]) {
    std::vector<float> embeds = body_hidden;
    std::vector<float> e0 = tk_row(backend, codec_main, m.n_embd, cb0);
    embeds.insert(embeds.end(), e0.begin(), e0.end());
    int Tg = 2;

    std::vector<float> lg0 = tk_mtp_logits(backend, m, embeds, Tg, 0, false, "");
    res_out[0] = tk_argmax(lg0);
    for (int gi = 1; gi < 15; gi++) {
        std::vector<float> e = tk_mtp_embed(backend, m, gi - 1, res_out[gi - 1]);
        embeds.insert(embeds.end(), e.begin(), e.end());
        Tg++;
        std::vector<float> lg = tk_mtp_logits(backend, m, embeds, Tg, gi, false, "");
        res_out[gi] = tk_argmax(lg);
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <talker.gguf> <talker_ref_dir>\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir   = argv[2];

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ggml_context * ctx_data = nullptr;
    gguf_init_params gp; gp.no_alloc = true; gp.ctx = &ctx_data;
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) { fprintf(stderr, "talker: cannot load %s\n", gguf_path.c_str()); return 2; }

    tk_hparams hp;
    hp.n_layer       = 20; // main blocks blk.0..19, the 5 mtp blocks come later
    hp.n_embd        = (int) tk_u32(g, TK_ARCH ".embedding_length");
    hp.n_head        = (int) tk_u32(g, TK_ARCH ".attention.head_count");
    hp.n_head_kv     = (int) tk_u32(g, TK_ARCH ".attention.head_count_kv");
    hp.head_dim      = (int) tk_u32(g, TK_ARCH ".attention.key_length");
    hp.n_ff_exp      = (int) tk_u32(g, TK_ARCH ".expert_feed_forward_length");
    hp.n_expert      = (int) tk_u32(g, TK_ARCH ".expert_count");
    hp.n_expert_used = (int) tk_u32(g, TK_ARCH ".expert_used_count");
    hp.n_ff_shexp    = (int) tk_u32(g, TK_ARCH ".expert_shared_feed_forward_length");
    hp.rms_eps       = tk_f32(g, TK_ARCH ".attention.layer_norm_rms_epsilon");
    hp.rope_theta    = tk_f32(g, TK_ARCH ".rope.freq_base");

    ggml_backend_alloc_ctx_tensors(ctx_data, backend);
    FILE * wf = fopen(gguf_path.c_str(), "rb");
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<char> tmp;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_data); t; t = ggml_get_next_tensor(ctx_data, t)) {
        const int64_t tid = gguf_find_tensor(g, t->name);
        if (tid < 0) continue;
        const size_t off = data_off + gguf_get_tensor_offset(g, tid);
        const size_t nb = ggml_nbytes(t);
        tmp.resize(nb);
        fseek(wf, (long) off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, wf) != nb) { return 3; }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
    }
    fclose(wf);

    std::vector<tk_layer> L(hp.n_layer);
    for (int i = 0; i < hp.n_layer; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        tk_layer & l = L[i];
        l.attn_norm      = tk_get(ctx_data, p + "attn_norm.weight");
        l.wq             = tk_get(ctx_data, p + "attn_q.weight");
        l.wk             = tk_get(ctx_data, p + "attn_k.weight");
        l.wv             = tk_get(ctx_data, p + "attn_v.weight");
        l.wo             = tk_get(ctx_data, p + "attn_output.weight");
        l.q_norm         = tk_get(ctx_data, p + "attn_q_norm.weight");
        l.k_norm         = tk_get(ctx_data, p + "attn_k_norm.weight");
        l.ffn_norm       = tk_get(ctx_data, p + "ffn_norm.weight");
        l.gate_inp       = tk_get(ctx_data, p + "ffn_gate_inp.weight");
        l.gate_exps      = tk_get(ctx_data, p + "ffn_gate_exps.weight");
        l.up_exps        = tk_get(ctx_data, p + "ffn_up_exps.weight");
        l.down_exps      = tk_get(ctx_data, p + "ffn_down_exps.weight");
        l.gate_inp_shexp = tk_get(ctx_data, p + "ffn_gate_inp_shexp.weight");
        l.gate_shexp     = tk_get(ctx_data, p + "ffn_gate_shexp.weight");
        l.up_shexp       = tk_get(ctx_data, p + "ffn_up_shexp.weight");
        l.down_shexp     = tk_get(ctx_data, p + "ffn_down_shexp.weight");
    }

    // read the HF prefill input embeds, file is [T, n_embd] row major = ggml [n_embd, T]
    int T = 0;
    {
        FILE * sf = fopen((ref_dir + "/prefill_inputs_embeds.shape").c_str(), "r");
        int b = 0, t = 0, d = 0; fscanf(sf, "%d %d %d", &b, &t, &d); fclose(sf);
        T = t;
        fprintf(stderr, "talker: prefill T=%d n_embd=%d\n", T, d);
    }
    std::vector<float> emb = read_bin((ref_dir + "/prefill_inputs_embeds.bin").c_str(), (size_t) T * hp.n_embd);

    // graph
    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_init_params ip; ip.mem_size = meta; ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE * 8, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.n_embd, T);
    ggml_set_name(inp, "inp"); ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    std::vector<ggml_tensor *> outs;
    ggml_tensor * cur = inp;
    for (int i = 0; i < hp.n_layer; i++) {
        cur = tk_build_layer(ctx0, hp, L[i], cur, pos, mask);
        ggml_set_name(cur, ("layer_" + std::to_string(i)).c_str());
        ggml_set_output(cur);
        outs.push_back(cur);
    }
    for (auto * o : outs) ggml_build_forward_expand(gf, o);

    // final norm then the main codec head (codebook 0)
    ggml_tensor * onorm  = tk_get(ctx_data, "output_norm.weight");
    ggml_tensor * outw   = tk_get(ctx_data, "output.weight");
    ggml_tensor * normed = tk_rmsnorm(ctx0, cur, onorm, hp.rms_eps);
    ggml_set_name(normed, "normed"); ggml_set_output(normed);
    ggml_tensor * logits = ggml_mul_mat(ctx0, outw, normed); // [vocab, T]
    ggml_set_name(logits, "logits"); ggml_set_output(logits);
    ggml_build_forward_expand(gf, normed);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(inp, emb.data(), 0, emb.size() * sizeof(float));
    std::vector<int32_t> posd(T); for (int i = 0; i < T; i++) posd[i] = i;
    ggml_backend_tensor_set(pos, posd.data(), 0, posd.size() * sizeof(int32_t));
    std::vector<float> md((size_t) T * T);
    for (int q = 0; q < T; q++) for (int kk = 0; kk < T; kk++) md[(size_t) q * T + kk] = kk <= q ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    for (int i = 0; i < hp.n_layer; i++) {
        std::vector<float> out((size_t) hp.n_embd * T);
        ggml_backend_tensor_get(outs[i], out.data(), 0, out.size() * sizeof(float));
        char path[512];
        snprintf(path, sizeof(path), "%s/cpp_layer_%02d.bin", ref_dir.c_str(), i);
        write_bin(path, out);
    }
    fprintf(stderr, "talker: dumped %d cpp layer outputs to %s/cpp_layer_NN.bin\n", hp.n_layer, ref_dir.c_str());

    // final norm + logits
    {
        std::vector<float> nv((size_t) hp.n_embd * T);
        ggml_backend_tensor_get(normed, nv.data(), 0, nv.size() * sizeof(float));
        write_bin((ref_dir + "/cpp_norm.bin").c_str(), nv);
        const int vocab = (int) logits->ne[0];
        std::vector<float> lv((size_t) vocab * T);
        ggml_backend_tensor_get(logits, lv.data(), 0, lv.size() * sizeof(float));
        write_bin((ref_dir + "/cpp_logits.bin").c_str(), lv);
        fprintf(stderr, "talker: dumped cpp_norm (%d,%d) and cpp_logits (%d,%d)\n", hp.n_embd, T, vocab, T);
    }

    // code predictor (MTP) : 5 dense layers blk.20..24 predict the 15 residual codebooks
    // for frame 0. attention is gqa 16/8 here, head_count_kv differs from the body value 2.
    tk_mtp m;
    m.n_embd    = hp.n_embd;
    m.n_head    = hp.n_head;
    m.n_head_kv = 8;
    m.hd        = hp.head_dim;
    m.vocab     = 2048;
    m.eps       = hp.rms_eps;
    m.theta     = hp.rope_theta;
    m.L.resize(5);
    for (int i = 0; i < 5; i++) {
        const std::string p = "blk." + std::to_string(20 + i) + ".mtp.";
        tk_mtp_layer & l = m.L[i];
        l.attn_norm = tk_get(ctx_data, p + "attn_norm.weight");
        l.wq        = tk_get(ctx_data, p + "attn_q.weight");
        l.wk        = tk_get(ctx_data, p + "attn_k.weight");
        l.wv        = tk_get(ctx_data, p + "attn_v.weight");
        l.wo        = tk_get(ctx_data, p + "attn_output.weight");
        l.q_norm    = tk_get(ctx_data, p + "attn_q_norm.weight");
        l.k_norm    = tk_get(ctx_data, p + "attn_k_norm.weight");
        l.ffn_norm  = tk_get(ctx_data, p + "ffn_norm.weight");
        l.gate      = tk_get(ctx_data, p + "ffn_gate.weight");
        l.up        = tk_get(ctx_data, p + "ffn_up.weight");
        l.down      = tk_get(ctx_data, p + "ffn_down.weight");
    }
    m.onorm   = tk_get(ctx_data, "mtp.output_norm.weight");
    m.lm_head = tk_get(ctx_data, "mtp.lm_head.weight");
    m.codec   = tk_get(ctx_data, "mtp.codec_embd.weight");


    // mtp prefill embeds [Tm, n_embd], Tm is 2 : [talker_hidden, embed(code0)]
    int Tm = 0;
    {
        FILE * sf = fopen((ref_dir + "/mtp_inputs_embeds.shape").c_str(), "r");
        int b = 0, t = 0, d = 0; fscanf(sf, "%d %d %d", &b, &t, &d); fclose(sf);
        Tm = t;
    }
    std::vector<float> memb = read_bin((ref_dir + "/mtp_inputs_embeds.bin").c_str(), (size_t) Tm * hp.n_embd);

    // reference residual codes, stored on disk as f32 ids
    std::vector<float> rc = read_bin((ref_dir + "/mtp_codes.bin").c_str(), 15);
    int ref_codes[15];
    for (int i = 0; i < 15; i++) ref_codes[i] = (int) lroundf(rc[i]);

    // prefill bisect : dump the 5 layer outputs for cosine, then lm_head[0] gives residual 1
    std::vector<float> lg0 = tk_mtp_logits(backend, m, memb, Tm, 0, true, ref_dir);
    int gen[15];
    gen[0] = tk_argmax(lg0);
    int matches = gen[0] == ref_codes[0] ? 1 : 0;
    fprintf(stderr, "talker mtp: dumped cpp_mtp_layer_0..4 to %s\n", ref_dir.c_str());
    fprintf(stderr, "talker mtp: residual 1 argmax=%d ref=%d %s\n", gen[0], ref_codes[0], gen[0] == ref_codes[0] ? "OK" : "MISMATCH");

    // autoregressive : feed each produced residual back through codec_embd[g-1], 14 steps
    std::vector<float> embeds = memb;
    int Tg = Tm;
    for (int gi = 1; gi < 15; gi++) {
        std::vector<float> e = tk_mtp_embed(backend, m, gi - 1, gen[gi - 1]);
        embeds.insert(embeds.end(), e.begin(), e.end());
        Tg++;
        std::vector<float> lg = tk_mtp_logits(backend, m, embeds, Tg, gi, false, ref_dir);
        gen[gi] = tk_argmax(lg);
        if (gen[gi] == ref_codes[gi]) matches++;
        fprintf(stderr, "talker mtp: residual %d argmax=%d ref=%d %s\n", gi + 1, gen[gi], ref_codes[gi], gen[gi] == ref_codes[gi] ? "OK" : "diff");
    }
    fprintf(stderr, "talker mtp: %d/15 residual codes match the HF reference\n", matches);

    // talker body autoregressive bisect, stage 1 : conditioning embeds are given
    // (the dumped ar_cond_NN), recompute prefix each frame, scalar position counter.
    // validate the body hidden cosine and the codec head cb0 argmax per frame.
    auto cosine = [](const std::vector<float> & a, const std::vector<float> & b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < a.size(); i++) { dot += (double) a[i] * b[i]; na += (double) a[i] * a[i]; nb += (double) b[i] * b[i]; }
        return dot / (sqrt(na) * sqrt(nb));
    };
    auto load_ref = [&](const std::string & name, int n) {
        FILE * sf = fopen((ref_dir + "/" + name + ".shape").c_str(), "r");
        if (!sf) return std::vector<float>();
        std::vector<int> dims; int d;
        while (fscanf(sf, "%d", &d) == 1) dims.push_back(d);
        fclose(sf);
        size_t cnt = 1; for (int x : dims) cnt *= (size_t) x;
        (void) n;
        return read_bin((ref_dir + "/" + name + ".bin").c_str(), cnt);
    };

    // count the available frames from ar_codes [K, 16]
    int K = 0;
    {
        FILE * sf = fopen((ref_dir + "/ar_codes.shape").c_str(), "r");
        if (sf) { int k = 0, w = 0; fscanf(sf, "%d %d", &k, &w); fclose(sf); K = k; }
    }
    if (K > 0) {
        std::vector<float> ar_codes = load_ref("ar_codes", 0);          // [K,16] as f32 ids
        // prefill embeds, frame 0 conditioning is the full Tp token block
        std::vector<float> cond0 = load_ref("ar_cond_00", 0);
        int Tp = (int) (cond0.size() / hp.n_embd);
        fprintf(stderr, "talker ar: prefill Tp=%d frames K=%d\n", Tp, K);

        std::vector<float> prefix = cond0;
        int T = Tp;
        int ar_match = 0;
        for (int n = 0; n < K; n++) {
            std::vector<float> hid, lg;
            tk_body_step(backend, hp, L, tk_get(ctx_data, "output_norm.weight"), tk_get(ctx_data, "output.weight"), prefix, T, hid, lg);

            std::vector<float> ref_hid = load_ref("ar_hidden_" + std::string(n < 10 ? "0" : "") + std::to_string(n), 0);
            double ch = cosine(hid, ref_hid);
            int cb0 = tk_argmax(lg);
            int ref_cb0 = (int) lroundf(ar_codes[(size_t) n * 16 + 0]);
            if (cb0 == ref_cb0) ar_match++;
            fprintf(stderr, "talker ar: frame %d hidden_cos=%.8f cb0=%d ref=%d %s\n",
                    n, ch, cb0, ref_cb0, cb0 == ref_cb0 ? "OK" : "MISMATCH");

            // append the next frame given conditioning embed for the recompute prefix
            if (n + 1 < K) {
                std::vector<float> condn = load_ref("ar_cond_" + std::string(n + 1 < 10 ? "0" : "") + std::to_string(n + 1), 0);
                prefix.insert(prefix.end(), condn.begin(), condn.end());
                T++;
            }
        }
        fprintf(stderr, "talker ar: %d/%d cb0 match the HF reference\n", ar_match, K);

        // stage 2 : reconstruct the conditioning embed from the reference codes and
        // cosine it against the dumped ar_cond_NN. cond_{m} is built from frame m-1.
        ggml_tensor * codec_main = tk_get(ctx_data, "codec_embd.weight");
        std::vector<float> trailing = load_ref("ar_trailing", 0);   // [Ttrail, n_embd]
        std::vector<float> tts_pad  = load_ref("ar_tts_pad", 0);    // [n_embd]
        int Ttrail = (int) (trailing.size() / hp.n_embd);
        for (int mfr = 1; mfr < K; mfr++) {
            const int src = mfr - 1;
            const int cb0 = (int) lroundf(ar_codes[(size_t) src * 16 + 0]);

            // cb0 embed through the top level codec table
            std::vector<float> cond = tk_row(backend, codec_main, hp.n_embd, cb0);

            // add the 15 residual embeds through the stacked mtp tables 0..14
            for (int k = 1; k <= 15; k++) {
                const int res = (int) lroundf(ar_codes[(size_t) src * 16 + k]);
                std::vector<float> e = tk_mtp_embed(backend, m, k - 1, res);
                for (int i = 0; i < hp.n_embd; i++) cond[i] += e[i];
            }

            // add the text conditioning : trailing[step] while in range, else tts_pad
            const int step = src;
            const float * txt = step < Ttrail ? trailing.data() + (size_t) step * hp.n_embd : tts_pad.data();
            for (int i = 0; i < hp.n_embd; i++) cond[i] += txt[i];

            std::vector<float> ref = load_ref("ar_cond_" + std::string(mfr < 10 ? "0" : "") + std::to_string(mfr), 0);
            double cc = cosine(cond, ref);
            fprintf(stderr, "talker ar: cond %d recon_cos=%.8f %s\n", mfr, cc, cc > 0.999 ? "OK" : "CHECK");
        }

        // stage 3 : autonomous closed loop. start from the prefill, the body emits cb0,
        // the mtp emits the 15 residuals, reconstruct the conditioning, reinject. nothing
        // from the reference re enters the loop, only the final 16 code check per frame.
        ggml_tensor * codec_main2 = tk_get(ctx_data, "codec_embd.weight");
        std::vector<float> trail = load_ref("ar_trailing", 0);
        std::vector<float> pad   = load_ref("ar_tts_pad", 0);
        int Ttr = (int) (trail.size() / hp.n_embd);

        std::vector<float> aprefix = cond0;
        int Ta = Tp;
        int frame_ok = 0, code_ok = 0, code_tot = 0;
        for (int n = 0; n < K; n++) {
            std::vector<float> hid, lg;
            tk_body_step(backend, hp, L, tk_get(ctx_data, "output_norm.weight"), tk_get(ctx_data, "output.weight"), aprefix, Ta, hid, lg);
            int cb0 = tk_argmax(lg);
            int res[15];
            tk_mtp_frame(backend, m, codec_main2, hid, cb0, res);


            // score the 16 codes of this frame against the reference
            int match = cb0 == (int) lroundf(ar_codes[(size_t) n * 16 + 0]) ? 1 : 0;
            for (int k = 0; k < 15; k++) match += res[k] == (int) lroundf(ar_codes[(size_t) n * 16 + 1 + k]) ? 1 : 0;
            code_ok += match; code_tot += 16;
            if (match == 16) frame_ok++;
            fprintf(stderr, "talker ar3: frame %d cb0=%d codes %d/16 %s\n", n, cb0, match, match == 16 ? "OK" : "drift");

            // reconstruct the next conditioning and reinject
            if (n + 1 < K) {
                std::vector<float> cond = tk_row(backend, codec_main2, hp.n_embd, cb0);
                for (int k = 1; k <= 15; k++) {
                    std::vector<float> e = tk_mtp_embed(backend, m, k - 1, res[k - 1]);
                    for (int i = 0; i < hp.n_embd; i++) cond[i] += e[i];
                }
                const float * txt = n < Ttr ? trail.data() + (size_t) n * hp.n_embd : pad.data();
                for (int i = 0; i < hp.n_embd; i++) cond[i] += txt[i];
                aprefix.insert(aprefix.end(), cond.begin(), cond.end());
                Ta++;
            }
        }
        fprintf(stderr, "talker ar3: autonomous %d/%d frames exact, %d/%d codes match HF\n", frame_ok, K, code_ok, code_tot);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);
    ggml_free(ctx_data);
    gguf_free(g);
    ggml_backend_free(backend);
    return 0;
}
