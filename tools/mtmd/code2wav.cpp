// Qwen3-Omni code2wav decoder graph builder.
// Standalone vocoder graph, decoupled from clip and the llama arch framework.
// Consumes 16 group RVQ codes and produces a 24kHz mono waveform.
//
// Layout convention in this file:
//   transformer tensors are [C, T]  (ne0 = channels, ne1 = time)
//   conv stack tensors are  [T, C]  (ne0 = time,     ne1 = channels)
// the single transpose between the two lives at the end of the pre_transformer.

#include "code2wav.h"

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

// hparams read from the GGUF, sizes the whole graph.
struct code2wav_hparams {
    int   hidden_size;
    int   n_layer;
    int   n_head;
    int   n_head_kv;
    int   n_ff;
    float rms_eps;
    float rope_theta;
    int   sliding_window;
    int   decoder_dim;
    int   num_quantizers;
    int   codebook_size;
    int   total_upsample;
    int   sample_rate;
    std::vector<int> upsample_rates;
    std::vector<int> upsampling_ratios;
};

// one SnakeBeta activation, params already baked at conversion to a and inv_b.
struct code2wav_snake {
    ggml_tensor * a;
    ggml_tensor * inv_b;
};

// one pre_transformer layer, MHA causal sliding window plus SwiGLU plus layer scale.
struct code2wav_tf_layer {
    ggml_tensor * attn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * attn_scale;
    ggml_tensor * ffn_norm;
    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * down;
    ggml_tensor * ffn_scale;
};

// ConvNeXt block used by the two upsample stages, GELU and a learnt gamma.
struct code2wav_convnext {
    ggml_tensor * dwconv_w;
    ggml_tensor * dwconv_b;
    ggml_tensor * norm_w;
    ggml_tensor * norm_b;
    ggml_tensor * pwconv1_w;
    ggml_tensor * pwconv1_b;
    ggml_tensor * pwconv2_w;
    ggml_tensor * pwconv2_b;
    ggml_tensor * gamma;
};

// one upsample stage, a transposed conv that doubles length then a ConvNeXt block.
struct code2wav_upsample {
    ggml_tensor *      convt_w;
    ggml_tensor *      convt_b;
    code2wav_convnext  cnx;
};

// one BigVGAN residual unit, snake conv7 snake conv1 with a residual add.
struct code2wav_resunit {
    code2wav_snake act1;
    ggml_tensor *  conv1_w;
    ggml_tensor *  conv1_b;
    code2wav_snake act2;
    ggml_tensor *  conv2_w;
    ggml_tensor *  conv2_b;
    int            dilation;
};

// one decoder stage, snake then upsample transconv then three residual units.
struct code2wav_decstage {
    code2wav_snake snake_in;
    ggml_tensor *  convt_w;
    ggml_tensor *  convt_b;
    code2wav_resunit res[3];
    int            upsample_rate;
};

// the full decoder weight set.
struct code2wav_weights {
    ggml_tensor * code_embedding;
    ggml_tensor * code_offset;

    std::vector<code2wav_tf_layer> tf;
    ggml_tensor * tf_norm;

    std::vector<code2wav_upsample> upsample;

    ggml_tensor *                  conv_pre_w;
    ggml_tensor *                  conv_pre_b;
    std::vector<code2wav_decstage> dec;
    code2wav_snake                 snake_out;
    ggml_tensor *                  conv_post_w;
    ggml_tensor *                  conv_post_b;
};

// SnakeBeta naive, y = x + sin(a * x)^2 * inv_b, x is [T, C], a and inv_b are [1, C].
// the CUDA fusion pass recollapses these ops plus the following add into one kernel.
static ggml_tensor * c2w_snake(ggml_context * ctx0, ggml_tensor * x, const code2wav_snake & s) {
    // a and inv_b are pre shaped to [1, C] at load so no reshape node lands between the
    // five ops. that keeps MUL SIN SQR MUL ADD strictly consecutive in the graph, which
    // is what the cuda snake fusion pass matches on.
    ggml_tensor * ax = ggml_mul(ctx0, x, s.a);
    ggml_tensor * sn = ggml_sin(ctx0, ax);
    ggml_tensor * s2 = ggml_sqr(ctx0, sn);
    ggml_tensor * t  = ggml_mul(ctx0, s2, s.inv_b);
    return ggml_add(ctx0, x, t);
}

// reshape a loaded [C] snake weight to [1, C] in place, same memory layout, no graph node.
static void c2w_snake_reshape(code2wav_snake & s) {
    s.a->ne[1] = s.a->ne[0];     s.a->ne[0] = 1;
    s.a->nb[1] = s.a->nb[0] * 1; // nb stays element size based, ne0 = 1
    s.inv_b->ne[1] = s.inv_b->ne[0]; s.inv_b->ne[0] = 1;
    s.inv_b->nb[1] = s.inv_b->nb[0] * 1;
}

// causal Conv1d, left pad eff_k minus stride then conv at p0 zero.
// w is ggml [K, IC, OC], x is [T, IC], bias is [1, OC] for the broadcast.
static ggml_tensor * c2w_causal_conv1d(ggml_context * ctx0, ggml_tensor * w, ggml_tensor * b,
                                       ggml_tensor * x, int stride, int dilation) {
    const int k     = (int) w->ne[0];
    const int eff_k = (k - 1) * dilation + 1;
    const int left  = eff_k - stride;

    // all padding on the left for the causal alignment, conv then runs at p0 zero.
    ggml_tensor * xp  = ggml_pad_ext(ctx0, x, left, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * cur = ggml_conv_1d(ctx0, w, xp, stride, 0, dilation);
    if (b) {
        // bias is per channel [OC], reshape to [1, OC] to broadcast over length.
        cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return cur;
}

// transposed Conv1d via col2im, the upsampler of each stage.
// w is ggml [IC, OC*K] in oc*K+k column order, x is [T, IC], p0 crops K minus stride.
static ggml_tensor * c2w_transconv1d(ggml_context * ctx0, ggml_tensor * w, ggml_tensor * b,
                                     ggml_tensor * x, int stride, int oc) {
    const int K  = (int) (w->ne[1] / oc);
    const int p0 = K - stride;

    // mul_mat contracts over IC, so x rides as [IC, T] for the column build.
    ggml_tensor * xt   = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * cols = ggml_mul_mat(ctx0, w, xt);
    ggml_tensor * cur  = ggml_col2im_1d(ctx0, cols, stride, oc, p0);
    if (b) {
        // bias is per channel [OC], reshape to [1, OC] to broadcast over length.
        cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return cur;
}

// causal depthwise Conv1d, the ConvNeXt token mixer (groups equal channels).
static ggml_tensor * c2w_causal_dwconv1d(ggml_context * ctx0, ggml_tensor * w, ggml_tensor * b,
                                         ggml_tensor * x, int dilation) {
    const int k    = (int) w->ne[0];
    const int left = (k - 1) * dilation + 1 - 1;

    ggml_tensor * xp  = ggml_pad_ext(ctx0, x, left, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * cur = ggml_conv_1d_dw(ctx0, w, xp, 1, 0, dilation);
    if (b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return cur;
}

// ConvNeXt block, depthwise conv then channel mlp with GELU then learnt gamma then residual.
static ggml_tensor * c2w_convnext(ggml_context * ctx0, const code2wav_convnext & c,
                                  ggml_tensor * x, float ln_eps) {
    ggml_tensor * inp = x;

    const int dim = (int) x->ne[1];
    ggml_tensor * cur = c2w_causal_dwconv1d(ctx0, c.dwconv_w, c.dwconv_b, x, 1);

    // the channel mlp runs per time step, move channels to ne0 for the matmuls.
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, c.norm_w), c.norm_b);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, c.pwconv1_w, cur), c.pwconv1_b);
    cur = ggml_gelu(ctx0, cur);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, c.pwconv2_w, cur), c.pwconv2_b);
    cur = ggml_mul(ctx0, cur, c.gamma);

    // back to [T, C] and add the residual.
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    GGML_ASSERT(dim == (int) cur->ne[1]);
    return ggml_add(ctx0, inp, cur);
}

// BigVGAN residual unit, x + conv2(act2(conv1(act1(x)))).
static ggml_tensor * c2w_resunit(ggml_context * ctx0, const code2wav_resunit & r, ggml_tensor * x) {
    ggml_tensor * residual = x;

    ggml_tensor * cur = c2w_snake(ctx0, x, r.act1);
    cur = c2w_causal_conv1d(ctx0, r.conv1_w, r.conv1_b, cur, 1, r.dilation);
    cur = c2w_snake(ctx0, cur, r.act2);
    cur = c2w_causal_conv1d(ctx0, r.conv2_w, r.conv2_b, cur, 1, 1);
    return ggml_add(ctx0, cur, residual);
}

// one decoder stage, snake then transposed conv upsample then three residual units.
static ggml_tensor * c2w_decstage(ggml_context * ctx0, const code2wav_decstage & d, ggml_tensor * x) {
    const int oc = (int) (d.convt_w->ne[1] / (2 * d.upsample_rate));

    ggml_tensor * cur = c2w_snake(ctx0, x, d.snake_in);
    cur = c2w_transconv1d(ctx0, d.convt_w, d.convt_b, cur, d.upsample_rate, oc);
    for (int i = 0; i < 3; i++) {
        cur = c2w_resunit(ctx0, d.res[i], cur);
    }
    return cur;
}

// pre_transformer, eight layers of causal sliding window MHA plus SwiGLU plus layer scale.
// hidden is [C, T], pos is [T] int32, mask is [T_kv, T_q] f32 with the causal sliding window.
static ggml_tensor * c2w_pre_transformer(ggml_context * ctx0, const code2wav_hparams & hp,
                                         const code2wav_weights & w, ggml_tensor * hidden,
                                         ggml_tensor * pos, ggml_tensor * mask) {
    const int   n_head    = hp.n_head;
    const int   n_head_kv = hp.n_head_kv;
    const int   head_dim  = hp.hidden_size / n_head;
    const int   T         = (int) hidden->ne[1];
    const float scale     = 1.0f / sqrtf((float) head_dim);

    for (const auto & l : w.tf) {
        ggml_tensor * residual = hidden;

        // attention, pre norm then qkv then rope then scaled dot product.
        ggml_tensor * cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, hidden, hp.rms_eps), l.attn_norm);
        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.wq, cur), head_dim, n_head,    T);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.wk, cur), head_dim, n_head_kv, T);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.wv, cur), head_dim, n_head_kv, T);

        q = ggml_rope_ext(ctx0, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_ext(ctx0, kq, mask, scale, 0.0f);

        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, kqv, head_dim * n_head, T);
        cur = ggml_mul_mat(ctx0, l.wo, cur);

        // layer scale on the attention output then residual.
        cur = ggml_mul(ctx0, cur, l.attn_scale);
        hidden = ggml_add(ctx0, residual, cur);

        // SwiGLU feed forward, pre norm then gate silu times up then down.
        residual = hidden;
        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, hidden, hp.rms_eps), l.ffn_norm);
        ggml_tensor * g = ggml_silu(ctx0, ggml_mul_mat(ctx0, l.gate, cur));
        ggml_tensor * u = ggml_mul_mat(ctx0, l.up, cur);
        cur = ggml_mul_mat(ctx0, l.down, ggml_mul(ctx0, g, u));

        // layer scale on the feed forward output then residual.
        cur = ggml_mul(ctx0, cur, l.ffn_scale);
        hidden = ggml_add(ctx0, residual, cur);
    }

    // final norm closes the transformer.
    return ggml_mul(ctx0, ggml_rms_norm(ctx0, hidden, hp.rms_eps), w.tf_norm);
}

// top level forward, codes [num_quantizers, T] int32 -> wav [T_out] f32.
static ggml_tensor * code2wav_build_graph(ggml_context * ctx0, const code2wav_hparams & hp,
                                          const code2wav_weights & w, ggml_tensor * codes) {
    const int Q = hp.num_quantizers;
    const int T = (int) codes->ne[1];

    // embed every group then average over the 16 quantizers, result [C, T].
    // the per group offset (q * codebook_size) is folded into the codes by the runner.
    ggml_tensor * ids = ggml_reshape_1d(ctx0, codes, (int64_t) Q * T);
    ggml_tensor * emb = ggml_get_rows(ctx0, w.code_embedding, ids);
    emb = ggml_reshape_3d(ctx0, emb, hp.hidden_size, Q, T);

    // ggml_mean reduces ne0, so move the 16 groups onto ne0 then average them away.
    emb = ggml_cont(ctx0, ggml_permute(ctx0, emb, 1, 0, 2, 3));
    ggml_tensor * hidden = ggml_mean(ctx0, emb);
    hidden = ggml_reshape_2d(ctx0, hidden, hp.hidden_size, T);

    // rope positions and the causal sliding window mask, filled by the runner.
    // mask[k, q] is 0 when k <= q and q - k < sliding_window, else negative infinity.
    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "kq_mask");
    ggml_set_input(mask);

    hidden = c2w_pre_transformer(ctx0, hp, w, hidden, pos, mask);

    // hand off to the conv stack in [T, C] layout.
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, hidden));

    for (const auto & up : w.upsample) {
        cur = c2w_transconv1d(ctx0, up.convt_w, up.convt_b, cur, 2, hp.hidden_size);
        cur = c2w_convnext(ctx0, up.cnx, cur, 1e-6f);
    }

    cur = c2w_causal_conv1d(ctx0, w.conv_pre_w, w.conv_pre_b, cur, 1, 1);
    for (const auto & d : w.dec) {
        cur = c2w_decstage(ctx0, d, cur);
    }
    cur = c2w_snake(ctx0, cur, w.snake_out);
    cur = c2w_causal_conv1d(ctx0, w.conv_post_w, w.conv_post_b, cur, 1, 1);

    // clamp to the valid pcm range, output is a single channel.
    return ggml_clamp(ctx0, cur, -1.0f, 1.0f);
}

// gguf loader, maps metadata and tensors into the hparams and weight structs.

#define C2W_ARCH "qwen3omni-code2wav"

// fetch a loaded tensor by name, abort if missing.
static ggml_tensor * c2w_get(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "code2wav: missing tensor %s\n", name.c_str());
        exit(1);
    }
    return t;
}

static uint32_t c2w_u32(gguf_context * g, const char * key) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { fprintf(stderr, "code2wav: missing key %s\n", key); exit(1); }
    return gguf_get_val_u32(g, id);
}

static float c2w_f32(gguf_context * g, const char * key) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { fprintf(stderr, "code2wav: missing key %s\n", key); exit(1); }
    return gguf_get_val_f32(g, id);
}

static std::vector<int> c2w_arr_i32(gguf_context * g, const char * key) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { fprintf(stderr, "code2wav: missing key %s\n", key); exit(1); }
    const size_t n = gguf_get_arr_n(g, id);
    const int32_t * d = (const int32_t *) gguf_get_arr_data(g, id);
    return std::vector<int>(d, d + n);
}

// reads the hparams from the gguf metadata.
static code2wav_hparams c2w_load_hparams(gguf_context * g) {
    code2wav_hparams hp;
    hp.hidden_size       = (int) c2w_u32(g, C2W_ARCH ".hidden_size");
    hp.n_layer           = (int) c2w_u32(g, C2W_ARCH ".block_count");
    hp.n_head            = (int) c2w_u32(g, C2W_ARCH ".head_count");
    hp.n_head_kv         = (int) c2w_u32(g, C2W_ARCH ".head_count_kv");
    hp.n_ff              = (int) c2w_u32(g, C2W_ARCH ".feed_forward_length");
    hp.rms_eps           =       c2w_f32(g, C2W_ARCH ".rms_eps");
    hp.rope_theta        =       c2w_f32(g, C2W_ARCH ".rope_theta");
    hp.sliding_window    = (int) c2w_u32(g, C2W_ARCH ".sliding_window");
    hp.decoder_dim       = (int) c2w_u32(g, C2W_ARCH ".decoder_dim");
    hp.num_quantizers    = (int) c2w_u32(g, C2W_ARCH ".num_quantizers");
    hp.codebook_size     = (int) c2w_u32(g, C2W_ARCH ".codebook_size");
    hp.total_upsample    = (int) c2w_u32(g, C2W_ARCH ".total_upsample");
    hp.sample_rate       = (int) c2w_u32(g, C2W_ARCH ".sample_rate");
    hp.upsample_rates    = c2w_arr_i32(g, C2W_ARCH ".upsample_rates");
    hp.upsampling_ratios = c2w_arr_i32(g, C2W_ARCH ".upsampling_ratios");
    return hp;
}

// maps the loaded tensors into the weight struct by reconstructing the names.
static code2wav_weights c2w_load_weights(ggml_context * ctx, const code2wav_hparams & hp) {
    code2wav_weights w;

    w.code_embedding = c2w_get(ctx, "code_embedding.weight");

    w.tf.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; i++) {
        const std::string p = "pre_transformer.layers." + std::to_string(i) + ".";
        auto & l = w.tf[i];
        l.attn_norm  = c2w_get(ctx, p + "input_layernorm.weight");
        l.wq         = c2w_get(ctx, p + "self_attn.q_proj.weight");
        l.wk         = c2w_get(ctx, p + "self_attn.k_proj.weight");
        l.wv         = c2w_get(ctx, p + "self_attn.v_proj.weight");
        l.wo         = c2w_get(ctx, p + "self_attn.o_proj.weight");
        l.attn_scale = c2w_get(ctx, p + "self_attn_layer_scale.scale");
        l.ffn_norm   = c2w_get(ctx, p + "post_attention_layernorm.weight");
        l.gate       = c2w_get(ctx, p + "mlp.gate_proj.weight");
        l.up         = c2w_get(ctx, p + "mlp.up_proj.weight");
        l.down       = c2w_get(ctx, p + "mlp.down_proj.weight");
        l.ffn_scale  = c2w_get(ctx, p + "mlp_layer_scale.scale");
    }
    w.tf_norm = c2w_get(ctx, "pre_transformer.norm.weight");

    w.upsample.resize(hp.upsampling_ratios.size());
    for (size_t j = 0; j < hp.upsampling_ratios.size(); j++) {
        const std::string p = "upsample." + std::to_string(j) + ".";
        auto & u = w.upsample[j];
        u.convt_w        = c2w_get(ctx, p + "0.conv.weight");
        u.convt_b        = c2w_get(ctx, p + "0.conv.bias");
        u.cnx.dwconv_w   = c2w_get(ctx, p + "1.dwconv.conv.weight");
        u.cnx.dwconv_b   = c2w_get(ctx, p + "1.dwconv.conv.bias");
        u.cnx.norm_w     = c2w_get(ctx, p + "1.norm.weight");
        u.cnx.norm_b     = c2w_get(ctx, p + "1.norm.bias");
        u.cnx.pwconv1_w  = c2w_get(ctx, p + "1.pwconv1.weight");
        u.cnx.pwconv1_b  = c2w_get(ctx, p + "1.pwconv1.bias");
        u.cnx.pwconv2_w  = c2w_get(ctx, p + "1.pwconv2.weight");
        u.cnx.pwconv2_b  = c2w_get(ctx, p + "1.pwconv2.bias");
        u.cnx.gamma      = c2w_get(ctx, p + "1.gamma");
    }

    w.conv_pre_w = c2w_get(ctx, "decoder.0.conv.weight");
    w.conv_pre_b = c2w_get(ctx, "decoder.0.conv.bias");

    const int dilations[3] = { 1, 3, 9 };
    w.dec.resize(hp.upsample_rates.size());
    for (size_t n = 0; n < hp.upsample_rates.size(); n++) {
        const std::string p = "decoder." + std::to_string(n + 1) + ".block.";
        auto & d = w.dec[n];
        d.upsample_rate = hp.upsample_rates[n];
        d.snake_in.a     = c2w_get(ctx, p + "0.snake_a");
        d.snake_in.inv_b = c2w_get(ctx, p + "0.snake_inv_b");
        d.convt_w        = c2w_get(ctx, p + "1.conv.weight");
        d.convt_b        = c2w_get(ctx, p + "1.conv.bias");
        for (int b = 0; b < 3; b++) {
            const std::string q = p + std::to_string(b + 2) + ".";
            auto & r = d.res[b];
            r.dilation     = dilations[b];
            r.act1.a       = c2w_get(ctx, q + "act1.snake_a");
            r.act1.inv_b   = c2w_get(ctx, q + "act1.snake_inv_b");
            r.conv1_w      = c2w_get(ctx, q + "conv1.conv.weight");
            r.conv1_b      = c2w_get(ctx, q + "conv1.conv.bias");
            r.act2.a       = c2w_get(ctx, q + "act2.snake_a");
            r.act2.inv_b   = c2w_get(ctx, q + "act2.snake_inv_b");
            r.conv2_w      = c2w_get(ctx, q + "conv2.conv.weight");
            r.conv2_b      = c2w_get(ctx, q + "conv2.conv.bias");
        }
    }

    w.snake_out.a     = c2w_get(ctx, "decoder.5.snake_a");
    w.snake_out.inv_b = c2w_get(ctx, "decoder.5.snake_inv_b");

    // pre shape every snake weight to [1, C] so the cuda fusion sees consecutive ops
    c2w_snake_reshape(w.snake_out);
    for (auto & d : w.dec) {
        c2w_snake_reshape(d.snake_in);
        for (auto & r : d.res) {
            c2w_snake_reshape(r.act1);
            c2w_snake_reshape(r.act2);
        }
    }
    w.conv_post_w     = c2w_get(ctx, "decoder.6.conv.weight");
    w.conv_post_b     = c2w_get(ctx, "decoder.6.conv.bias");

    // code_offset is a constant buffer in HF, not a saved weight, so it stays null
    // here and the runner fills its own input tensor.
    w.code_offset = nullptr;
    return w;
}


// decode one window of frames, codes_slice is offset folded and frame major.
// builds a fresh graph sized to this window, runs it, returns the pcm window.
static std::vector<float> c2w_decode_window(ggml_backend_t backend, const code2wav_hparams & hp,
                                            const code2wav_weights & w,
                                            const int32_t * codes_slice, int T) {
    const int Q = hp.num_quantizers;

    const size_t meta = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ggml_init_params ip;
    ip.mem_size = meta;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, Q, T);
    ggml_set_name(codes, "codes");
    ggml_set_input(codes);

    ggml_tensor * wav = code2wav_build_graph(ctx0, hp, w, codes);
    ggml_set_output(wav);
    ggml_build_forward_expand(gf, wav);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(codes, codes_slice, 0, (size_t) Q * T * sizeof(int32_t));

    // positions are local to the window, the pre_transformer runs per window.
    ggml_tensor * pos = ggml_graph_get_tensor(gf, "pos");
    std::vector<int32_t> pos_data(T);
    for (int i = 0; i < T; i++) pos_data[i] = i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));

    // causal sliding window mask over the window.
    ggml_tensor * mask = ggml_graph_get_tensor(gf, "kq_mask");
    std::vector<float> mask_data((size_t) T * T);
    for (int q = 0; q < T; q++) {
        for (int k = 0; k < T; k++) {
            const bool ok = k <= q && q - k < hp.sliding_window;
            mask_data[(size_t) q * T + k] = ok ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    const int n_out = (int) ggml_nelements(wav);
    std::vector<float> out(n_out);
    ggml_backend_tensor_get(wav, out.data(), 0, n_out * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);
    return out;
}

// public context, owns the backend, the loaded weights and the stream history.
struct code2wav_context {
    ggml_backend_t   backend;
    gguf_context *   gguf;
    ggml_context *   ctx_data;
    code2wav_hparams hp;
    code2wav_weights w;
    code2wav_params  params;

    // streaming history, frames are offset folded and frame major.
    std::vector<int32_t> frames;
    int                  start;
};

code2wav_context * code2wav_init(const char * gguf_path, code2wav_params params) {
    code2wav_context * ctx = new code2wav_context();
    ctx->params = params;
    ctx->start = 0;

    ctx->backend = params.use_gpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : nullptr;
    if (!ctx->backend) {
        ctx->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    ggml_context * ctx_data = nullptr;
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = &ctx_data;
    ctx->gguf = gguf_init_from_file(gguf_path, gp);
    if (!ctx->gguf) {
        fprintf(stderr, "code2wav: cannot load %s\n", gguf_path);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_data = ctx_data;
    ctx->hp = c2w_load_hparams(ctx->gguf);

    // back the weight tensors and copy their bytes from the file.
    ggml_backend_alloc_ctx_tensors(ctx_data, ctx->backend);
    FILE * wf = fopen(gguf_path, "rb");
    const size_t data_off = gguf_get_data_offset(ctx->gguf);
    std::vector<char> tmp;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_data); t; t = ggml_get_next_tensor(ctx_data, t)) {
        const int64_t tid = gguf_find_tensor(ctx->gguf, t->name);
        const size_t off = data_off + gguf_get_tensor_offset(ctx->gguf, tid);
        const size_t nb = ggml_nbytes(t);
        tmp.resize(nb);
        fseek(wf, (long) off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, wf) != nb) { fclose(wf); code2wav_free(ctx); return nullptr; }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
    }
    fclose(wf);

    ctx->w = c2w_load_weights(ctx_data, ctx->hp);
    return ctx;
}

void code2wav_free(code2wav_context * ctx) {
    if (!ctx) return;
    if (ctx->ctx_data) ggml_free(ctx->ctx_data);
    if (ctx->gguf)     gguf_free(ctx->gguf);
    if (ctx->backend)  ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t code2wav_sample_rate   (const code2wav_context * ctx) { return (uint32_t) ctx->hp.sample_rate; }
int      code2wav_n_quantizers  (const code2wav_context * ctx) { return ctx->hp.num_quantizers; }
int      code2wav_total_upsample(const code2wav_context * ctx) { return ctx->hp.total_upsample; }

// fold the per group offset into a frame major buffer in place.
// q is the fastest index so the offset is (i mod Q) * codebook_size.
static void c2w_fold_offset(const code2wav_hparams & hp, int32_t * codes, size_t n) {
    const int Q = hp.num_quantizers;
    for (size_t i = 0; i < n; i++) {
        codes[i] += (int32_t) ((i % Q) * hp.codebook_size);
    }
}

// decode the frame range [a, end) from an offset folded buffer, trim the lookback.
// appends the kept samples to out.
static void c2w_emit_chunk(code2wav_context * ctx, const int32_t * folded, int a, int end, int ctxn,
                           std::vector<float> & out) {
    const int Q  = ctx->hp.num_quantizers;
    const int Tw = end - a;
    std::vector<float> win = c2w_decode_window(ctx->backend, ctx->hp, ctx->w, folded + (size_t) a * Q, Tw);
    const int trim = ctxn * ctx->hp.total_upsample;
    out.insert(out.end(), win.begin() + trim, win.end());
}

std::vector<float> code2wav_decode(code2wav_context * ctx, const int32_t * codes, int n_frames) {
    const int Q = ctx->hp.num_quantizers;
    std::vector<int32_t> folded(codes, codes + (size_t) n_frames * Q);
    c2w_fold_offset(ctx->hp, folded.data(), folded.size());

    std::vector<float> pcm;
    int start = 0;
    while (start < n_frames) {
        const int end  = start + ctx->params.chunk_size < n_frames ? start + ctx->params.chunk_size : n_frames;
        const int ctxn = start - ctx->params.left_context > 0 ? ctx->params.left_context : start;
        c2w_emit_chunk(ctx, folded.data(), start - ctxn, end, ctxn, pcm);
        start = end;
    }
    return pcm;
}

void code2wav_stream_reset(code2wav_context * ctx) {
    ctx->frames.clear();
    ctx->start = 0;
}

void code2wav_stream_push(code2wav_context * ctx, const int32_t * codes, int n_frames, std::vector<float> & out) {
    const int Q = ctx->hp.num_quantizers;

    // append the new frames and fold their offset, base is frame aligned so i mod Q holds.
    const size_t base = ctx->frames.size();
    ctx->frames.insert(ctx->frames.end(), codes, codes + (size_t) n_frames * Q);
    c2w_fold_offset(ctx->hp, ctx->frames.data() + base, (size_t) n_frames * Q);

    // decode every window that is now fully available.
    const int total = (int) (ctx->frames.size() / Q);
    while (total - ctx->start >= ctx->params.chunk_size) {
        const int end  = ctx->start + ctx->params.chunk_size;
        const int ctxn = ctx->start - ctx->params.left_context > 0 ? ctx->params.left_context : ctx->start;
        c2w_emit_chunk(ctx, ctx->frames.data(), ctx->start - ctxn, end, ctxn, out);
        ctx->start = end;
    }
}

void code2wav_stream_flush(code2wav_context * ctx, std::vector<float> & out) {
    const int Q = ctx->hp.num_quantizers;
    const int total = (int) (ctx->frames.size() / Q);
    if (ctx->start < total) {
        const int ctxn = ctx->start - ctx->params.left_context > 0 ? ctx->params.left_context : ctx->start;
        c2w_emit_chunk(ctx, ctx->frames.data(), ctx->start - ctxn, total, ctxn, out);
        ctx->start = total;
    }
}
