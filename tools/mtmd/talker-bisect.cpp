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

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);
    ggml_free(ctx_data);
    gguf_free(g);
    ggml_backend_free(backend);
    return 0;
}
