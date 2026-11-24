#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    for (ggml_op op : ops_all) {
        if (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    return fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        // MUL/ADD are commutative, the previous node can be either operand
        const ggml_tensor * src = nodes[j]->src[0] == nodes[j - 1] ? nodes[j]->src[1] : nodes[j]->src[0];

        // the weight/bias must have the same row width as the norm input
        if (src->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(src)) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

// situ(g, u) = b*tanh(g/b)*sigmoid(g) * lb*tanh(u/lb) on the gate and up views of the expert output,
// optionally after the per-expert weight scale (GET_ROWS of the scales by the expert ids + MUL)
static bool ggml_metal_fusion_check_situ(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const bool scaled = fusion->ops[0] == GGML_OP_GET_ROWS;
    const int  off    = scaled ? 2 : 0;
    const ggml_tensor * const * n = nodes + off;

    const ggml_tensor * g = n[0]->src[0];
    const ggml_tensor * u = n[5]->src[0];

    auto is_unary = [](const ggml_tensor * t, ggml_unary_op op, const ggml_tensor * src) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op && t->src[0] == src;
    };
    auto is_scale = [](const ggml_tensor * t, const ggml_tensor * src) {
        return t->op == GGML_OP_SCALE && t->src[0] == src && ggml_get_op_params_f32(t, 1) == 0.0f;
    };
    auto is_mul = [](const ggml_tensor * t, const ggml_tensor * a, const ggml_tensor * b) {
        return t->op == GGML_OP_MUL && ((t->src[0] == a && t->src[1] == b) || (t->src[0] == b && t->src[1] == a));
    };

    if (!is_scale(n[0], g) || !is_unary(n[1], GGML_UNARY_OP_TANH, n[0]) || !is_scale(n[2], n[1]) ||
        !is_unary(n[3], GGML_UNARY_OP_SIGMOID, g) || !is_mul(n[4], n[2], n[3]) ||
        !is_scale(n[5], u) || !is_unary(n[6], GGML_UNARY_OP_TANH, n[5]) || !is_scale(n[7], n[6]) ||
        !is_mul(n[8], n[4], n[7])) {
        return false;
    }

    // every operand is F32, the output is contiguous and the views read whole rows
    for (int j = 0; j < 9; ++j) {
        if (n[j]->type != GGML_TYPE_F32 || !ggml_are_same_shape(n[j], g)) {
            return false;
        }
    }
    if (g->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32 || !ggml_are_same_shape(g, u) ||
        g->nb[0] != sizeof(float) || u->nb[0] != sizeof(float) || g->ne[3] != 1 || !ggml_is_contiguous(n[8])) {
        return false;
    }

    // the intermediates are read inside the group only
    for (int j = 0; j < 8; ++j) {
        if ((n[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + off + j]) != 1) {
            return false;
        }
    }

    if (scaled) {
        const ggml_tensor * rows = nodes[0];
        const ggml_tensor * mul  = nodes[1];
        const ggml_tensor * gu   = mul->src[0] == rows ? mul->src[1] : mul->src[0];
        const ggml_tensor * tab  = rows->src[0];
        const ggml_tensor * ids  = rows->src[1];

        // the gate and up views read the scaled expert output, which nothing else reads
        if (!is_mul(mul, gu, rows) || g->view_src != mul || u->view_src != mul ||
            (mul->flags & GGML_TENSOR_FLAG_OUTPUT) || (rows->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(gf, node_idxs[idx + 0]) != 1 || ggml_node_get_use_count(gf, node_idxs[idx + 1]) != 2) {
            return false;
        }
        // one scale per expert, gathered by the expert ids [n_used, n_tokens]
        if (tab->type != GGML_TYPE_F32 || tab->ne[0] != 1 || tab->ne[3] != 1 || ids->type != GGML_TYPE_I32 ||
            rows->ne[0] != 1 || rows->ne[1] != mul->ne[1] || rows->ne[2] != mul->ne[2] ||
            ids->ne[0] != mul->ne[1] || ids->ne[1] != mul->ne[2] || (tab->ne[2] != 1 && tab->ne[2] != mul->ne[2]) ||
            gu->type != GGML_TYPE_F32 || !ggml_are_same_shape(gu, mul) || !ggml_is_contiguous(gu) || !ggml_is_contiguous(mul)) {
            return false;
        }
    }

    return true;
}

// x - tanh(f)*x + tanh(g)*m, the gated mix of a message m into x
static bool ggml_metal_fusion_check_tanh_gate(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);

    auto is_tanh = [](const ggml_tensor * t) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_TANH;
    };
    auto other = [](const ggml_tensor * t, const ggml_tensor * a) {
        return t->src[0] == a ? t->src[1] : t->src[1] == a ? t->src[0] : nullptr;
    };

    if (!is_tanh(n[0]) || !is_tanh(n[3])) {
        return false;
    }

    const ggml_tensor * x = other(n[1], n[0]);
    const ggml_tensor * m = other(n[4], n[3]);
    if (!x || !m || n[2]->src[0] != x || n[2]->src[1] != n[1] || other(n[5], n[2]) != n[4]) {
        return false;
    }

    const ggml_tensor * f = n[0]->src[0];
    const ggml_tensor * g = n[3]->src[0];
    const ggml_tensor * ins[4] = { x, m, f, g };
    for (const ggml_tensor * t : ins) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || !ggml_are_same_shape(t, n[5]) || t->ne[2] != 1 || t->ne[3] != 1) {
            return false;
        }
    }
    for (int j = 0; j < 6; ++j) {
        if (n[j]->type != GGML_TYPE_F32 || !ggml_are_same_shape(n[j], n[5])) {
            return false;
        }
    }
    if (!ggml_is_contiguous(n[5])) {
        return false;
    }

    // the intermediates are read inside the group only
    for (int j = 0; j < 5; ++j) {
        if ((n[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + j]) != 1) {
            return false;
        }
    }

    return true;
}

// ADD + SCALE [+ ADD]: every operand is a same-shape F32 tensor with contiguous rows
static bool ggml_metal_fusion_check_add_scale(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const int n_ops = (int) fusion->ops.size();

    const ggml_tensor * ins[3] = { n[0]->src[0], n[0]->src[1], n_ops == 3 ? (n[2]->src[0] == n[1] ? n[2]->src[1] : n[2]->src[0]) : nullptr };
    for (const ggml_tensor * t : ins) {
        if (t && (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || !ggml_are_same_shape(t, n[0]) || t->ne[3] != 1)) {
            return false;
        }
    }
    for (int j = 0; j < n_ops; ++j) {
        if (n[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return n[1]->src[0] == n[0] && ggml_is_contiguous(n[n_ops - 1]);
}

// mask[kv, t] + table[pos[kv, t], h, t] - sink[h, t] in the [n_kv, n_tokens/n_stream, n_head, n_stream] F16 layout
// of the flash attention mask, from the projected bias table [n_bins, n_head, n_tokens] and the distances [n_kv, n_tokens]
static bool ggml_metal_fusion_check_rel_mask(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);

    const ggml_tensor * sub  = n[0];
    const ggml_tensor * c0   = n[1];
    const ggml_tensor * rows = n[2];
    const ggml_tensor * c1   = n[3];
    const ggml_tensor * add  = n[4];
    const ggml_tensor * cpy  = n[5];

    const ggml_tensor * tab  = sub->src[0];
    const ggml_tensor * sink = sub->src[1];
    const ggml_tensor * pos  = rows->src[1];
    const ggml_tensor * mask = add->src[1];

    if (!c0->src[0] || c0->src[0]->op != GGML_OP_PERMUTE || c0->src[0]->view_src != sub || rows->src[0] != c0 ||
        !c1->src[0] || c1->src[0]->op != GGML_OP_PERMUTE || !c1->src[0]->src[0] || c1->src[0]->src[0]->view_src != rows ||
        add->src[0] != c1 || cpy->src[0] != add) {
        return false;
    }

    const int64_t n_kv  = pos->ne[0];
    const int64_t n_tok = pos->ne[1];
    const int64_t n_hd  = tab->ne[1];

    if (tab->type != GGML_TYPE_F32 || sink->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32 || mask->type != GGML_TYPE_F32 ||
        cpy->type != GGML_TYPE_F16 || !ggml_is_contiguous(tab) || !ggml_is_contiguous(pos) || !ggml_is_contiguous(cpy) ||
        sink->nb[0] != sizeof(float) || sink->ne[0] != 1 || sink->ne[1] != n_hd || sink->ne[2] != n_tok || tab->ne[2] != n_tok ||
        mask->nb[0] != sizeof(float) || mask->ne[0] != n_kv || mask->ne[2] != 1 ||
        c1->ne[0] != n_kv || c1->ne[2] != n_hd || c1->ne[1]*c1->ne[3] != n_tok || mask->ne[1] != c1->ne[1] || mask->ne[3] != c1->ne[3]) {
        return false;
    }

    for (int j = 0; j < 5; ++j) {
        if ((n[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + j]) != 1) {
            return false;
        }
    }

    return true;
}

// [x*w] / max(|x*w|, eps) * s + t per row, w a per-channel weight broadcast over the rows
static bool ggml_metal_fusion_check_l2_clip(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const bool has_w = fusion->ops.size() == 3;
    const ggml_tensor * l2 = n[has_w ? 1 : 0];
    const ggml_tensor * sc = n[has_w ? 2 : 1];
    const ggml_tensor * x  = has_w ? (ggml_are_same_shape(n[0]->src[0], n[0]) ? n[0]->src[0] : n[0]->src[1]) : l2->src[0];

    if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[3] != 1 || x->ne[0] > 1024 ||
        l2->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc)) {
        return false;
    }
    if (has_w) {
        const ggml_tensor * w = n[0]->src[0] == x ? n[0]->src[1] : n[0]->src[0];
        if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != x->ne[0] || n[0]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MV_FUSION_MAX_TOKENS 8

// W.concat(a, b) with F32 W [ne_a + ne_b, n_out] and F32 a, b with contiguous rows
static bool ggml_metal_fusion_check_cat_mv(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);

    const ggml_tensor * cat = n[0];
    const ggml_tensor * mm  = n[1];
    const ggml_tensor * a   = cat->src[0];
    const ggml_tensor * b   = cat->src[1];
    const ggml_tensor * w   = mm->src[0];

    // the matrix rows are read once per token: a batch is faster with the matrix-matrix kernels
    if (cat->ne[1] > GGML_METAL_MV_FUSION_MAX_TOKENS) {
        return false;
    }

    if (ggml_get_op_params_i32(cat, 0) != 0 || mm->src[1] != cat || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) ||
        a->ne[1] != b->ne[1] || a->ne[2] != 1 || b->ne[2] != 1 || cat->ne[2] != 1 || w->ne[0] != cat->ne[0] || w->ne[2] != 1 ||
        mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm)) {
        return false;
    }

    return !(cat->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(gf, node_idxs[idx]) == 1;
}

// x - tanh(Wf.d)*x + tanh(Wg.d)*m with two F32 projections of the same d
static bool ggml_metal_fusion_check_gate_mv(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);

    const ggml_tensor * m0 = n[0];
    const ggml_tensor * m1 = n[1];

    if (m0->op != GGML_OP_MUL_MAT || m1->op != GGML_OP_MUL_MAT || m0->src[1] != m1->src[1] || m0->ne[1] > GGML_METAL_MV_FUSION_MAX_TOKENS) {
        return false;
    }
    if (!ggml_metal_fusion_check_tanh_gate(fusion, n + 2, gf, node_idxs, idx + 2, mode)) {
        return false;
    }
    // the gate reads the two projections
    const ggml_tensor * f = n[2]->src[0];
    const ggml_tensor * g = n[5]->src[0];
    if (!((f == m0 && g == m1) || (f == m1 && g == m0))) {
        return false;
    }

    const ggml_tensor * d = m0->src[1];
    for (const ggml_tensor * w : { m0->src[0], m1->src[0] }) {
        if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != d->ne[0] || w->ne[2] != 1) {
            return false;
        }
    }
    if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || d->ne[0] > 256 || d->ne[2] != 1) {
        return false;
    }

    for (int j = 0; j < 2; ++j) {
        if ((n[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + j]) != 1) {
            return false;
        }
    }

    return true;
}

// a*sigmoid(g), F32 with contiguous rows
static bool ggml_metal_fusion_check_sigmoid_mul(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * sg  = n[0];
    const ggml_tensor * mul = n[1];
    const ggml_tensor * g   = sg->src[0];
    const ggml_tensor * a   = mul->src[0] == sg ? mul->src[1] : mul->src[0];

    if (ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID || !a) {
        return false;
    }
    for (const ggml_tensor * t : { g, a }) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || !ggml_are_same_shape(t, mul) || t->ne[3] != 1) {
            return false;
        }
    }

    return sg->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul);
}

// x*tab[ids], one factor per expert: GET_ROWS of the factors [1, n_expert(, n_tokens)] by the ids [n_used, n_tokens]
static bool ggml_metal_fusion_check_scale_id(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);

    const ggml_tensor * rows = n[0];
    const ggml_tensor * mul  = n[1];
    const ggml_tensor * tab  = rows->src[0];
    const ggml_tensor * ids  = rows->src[1];
    const ggml_tensor * x    = mul->src[0] == rows ? mul->src[1] : mul->src[0];

    if ((mul->src[0] != rows && mul->src[1] != rows) || !x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        !ggml_are_same_shape(x, mul) || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || x->ne[3] != 1 ||
        tab->type != GGML_TYPE_F32 || tab->ne[0] != 1 || tab->ne[3] != 1 || (tab->ne[2] != 1 && tab->ne[2] != x->ne[2]) ||
        ids->type != GGML_TYPE_I32 || ids->ne[0] != x->ne[1] || ids->ne[1] != x->ne[2] ||
        rows->ne[0] != 1 || rows->ne[1] != x->ne[1] || rows->ne[2] != x->ne[2]) {
        return false;
    }

    return !(rows->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(gf, node_idxs[idx]) == 1;
}

#define GGML_METAL_CONV_STEP_MAX_CONV   8
#define GGML_METAL_CONV_STEP_MAX_TOKENS 8

// the convolution step shared by the variants: x = concat(past, transpose(cur)), the store of its last n_conv - 1
// columns into the state and ssm_conv(x, w); x is read by n_uses nodes, the store and the convolution among them
static bool ggml_metal_fusion_check_conv_step_base(
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              int                       n_uses) {
    const ggml_tensor * cat  = n[0];
    const ggml_tensor * cpy  = n[1];
    const ggml_tensor * conv = n[2];
    const ggml_tensor * past = cat->src[0];
    const ggml_tensor * tr   = cat->src[1];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * last = cpy->src[0];

    if (ggml_get_op_params_i32(cat, 0) != 0 || tr->op != GGML_OP_TRANSPOSE || !tr->view_src ||
        conv->src[0] != cat || !last || last->view_src != cat) {
        return false;
    }

    // the transposed view [n_t, ch, n_s] of the contiguous input [ch, n_t*n_s]
    const ggml_tensor * cur = tr->view_src;

    const int64_t n_cols = past->ne[0];
    const int64_t ch     = tr->ne[1];
    const int64_t n_t    = tr->ne[0];
    const int64_t n_s    = tr->ne[2];

    // one thread per channel walks the tokens: a batch is faster with the batched convolution kernel
    if (n_t > GGML_METAL_CONV_STEP_MAX_TOKENS) {
        return false;
    }

    if (past->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || conv->type != GGML_TYPE_F32 ||
        tr->view_offs != 0 || tr->nb[1] != sizeof(float) || tr->nb[0] != (size_t) ch*sizeof(float) || tr->nb[2] != (size_t) (ch*n_t)*sizeof(float) ||
        ggml_nelements(cur) != ch*n_t*n_s || !ggml_is_contiguous(cur) ||
        cpy->type != GGML_TYPE_F32 || past->nb[0] != sizeof(float) || !ggml_is_contiguous(w) ||
        past->ne[1] != ch || past->ne[2] != n_s || w->ne[0] != n_cols + 1 || w->ne[1] != ch || n_cols + 1 > GGML_METAL_CONV_STEP_MAX_CONV ||
        last->ne[0] != n_cols || last->ne[1] != ch || last->ne[2] != n_s || last->view_offs != (size_t) n_t*sizeof(float) ||
        cpy->ne[0] != n_cols*ch || cpy->ne[1] != n_s || cpy->nb[0] != sizeof(float) || !ggml_is_contiguous(conv)) {
        return false;
    }

    return !(cat->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(gf, node_idxs[idx + 0]) == n_uses;
}

// x = concat(past, transpose(cur)), the state store of the last n_conv - 1 columns of x, out = ssm_conv(x, w),
// optionally (cur + out)*s + t; x is read by the store and the convolution only
static bool ggml_metal_fusion_check_conv_step(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    if (!ggml_metal_fusion_check_conv_step_base(n, gf, node_idxs, idx, 2)) {
        return false;
    }

    const ggml_tensor * conv = n[2];
    const ggml_tensor * cur  = n[0]->src[1]->view_src;
    const int64_t       ch   = n[0]->src[1]->ne[1];
    const int64_t       n_t  = n[0]->src[1]->ne[0];
    const int64_t       n_s  = n[0]->src[1]->ne[2];

    if (fusion->ops.size() >= 5) {
        const ggml_tensor * add = n[3];
        const ggml_tensor * sc  = n[4];
        const ggml_tensor * rs  = add->src[0] == cur ? add->src[1] : add->src[0];

        if ((add->src[0] != cur && add->src[1] != cur) || !ggml_are_same_shape(add, cur) || !rs || rs->view_src != conv ||
            sc->src[0] != add || !ggml_is_contiguous(sc) || ggml_nelements(sc) != ch*n_t*n_s || add->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 ||
            (conv->flags & GGML_TENSOR_FLAG_OUTPUT) || (add->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(gf, node_idxs[idx + 2]) != 1 || ggml_node_get_use_count(gf, node_idxs[idx + 3]) != 1) {
            return false;
        }
    }

    // the residual added to the scaled output
    if (fusion->ops.size() == 6) {
        const ggml_tensor * sc  = n[4];
        const ggml_tensor * res = n[5];
        const ggml_tensor * c   = res->src[0] == sc ? res->src[1] : res->src[0];

        if ((res->src[0] != sc && res->src[1] != sc) || !c || c->type != GGML_TYPE_F32 || !ggml_is_contiguous(c) ||
            !ggml_are_same_shape(c, sc) || res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) ||
            (sc->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + 4]) != 1) {
            return false;
        }
    }

    return true;
}

// the convolution step with input-dependent taps per head: out = ssm_conv(x, w) + sum_lag window(x, lag)*tap(lag),
// then (cur + out)*s + t, with the n_conv windows of x as transposed copies and the taps as views of one projection
static bool ggml_metal_fusion_check_conv_step_dyn(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const int n_lag = ((int) fusion->ops.size() - 5)/3;

    // x feeds the state store, the convolution and the windows
    if (!ggml_metal_fusion_check_conv_step_base(n, gf, node_idxs, idx, 2 + n_lag)) {
        return false;
    }

    const ggml_tensor * cat    = n[0];
    const ggml_tensor * conv   = n[2];
    const ggml_tensor * cur    = cat->src[1]->view_src;
    const int64_t       n_cols = cat->src[0]->ne[0];
    const int64_t       ch     = cat->src[1]->ne[1];
    const int64_t       n_t    = cat->src[1]->ne[0];
    const int64_t       n_s    = cat->src[1]->ne[2];

    if (n_lag != n_cols + 1) {
        return false;
    }

    const ggml_tensor * dyn  = nullptr;
    const ggml_tensor * prev = conv;
    int64_t n_heads = 0;

    for (int l = 0; l < n_lag; ++l) {
        const ggml_tensor * cont = n[3 + 3*l];
        const ggml_tensor * mul  = n[4 + 3*l];
        const ggml_tensor * add  = n[5 + 3*l];

        // the window of lag l: x[t + n_cols - l] for each token, as a contiguous [n_t, ch] copy
        const ggml_tensor * wt = cont->src[0];
        const ggml_tensor * wv = wt ? wt->src[0] : nullptr;
        if (!wt || wt->op != GGML_OP_TRANSPOSE || !wv || wv->op != GGML_OP_VIEW || wv->src[0] != cat ||
            wv->view_offs != (size_t) (n_cols - l)*sizeof(float) || wv->ne[0] != n_t || wv->ne[1] != ch || wv->ne[2] != n_s ||
            wv->nb[1] != cat->nb[1] || wv->nb[2] != cat->nb[2] || !ggml_is_contiguous(cont) || cont->type != GGML_TYPE_F32) {
            return false;
        }

        // window (as [ch/n_heads, n_heads, n_t*n_s]) times the tap of lag l (as [1, n_heads, n_t*n_s])
        const ggml_tensor * a   = mul->src[0];
        const ggml_tensor * tap = mul->src[1];
        if (!a || !tap || a->view_src != cont || tap->op != GGML_OP_VIEW || !tap->view_src || tap->ne[0] != 1 ||
            mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || ggml_nelements(mul) != ch*n_t*n_s ||
            mul->ne[2] != n_t*n_s || tap->ne[1] != mul->ne[1] || tap->ne[2] != mul->ne[2]) {
            return false;
        }

        const ggml_tensor * d = tap->view_src;
        if (l == 0) {
            dyn     = d;
            n_heads = mul->ne[1];
            if (dyn->type != GGML_TYPE_F32 || !ggml_is_contiguous(dyn) || dyn->ne[0] != n_lag*n_heads || ggml_nrows(dyn) != n_t*n_s ||
                ch % n_heads != 0) {
                return false;
            }
        }
        if (d != dyn || tap->view_offs != (size_t) l*sizeof(float) || tap->nb[1] != (size_t) n_lag*sizeof(float) ||
            tap->nb[2] != dyn->nb[1]) {
            return false;
        }

        const ggml_tensor * rs = add->src[1] && add->src[1]->view_src == mul ? add->src[1] : add->src[0];
        const ggml_tensor * ps = rs == add->src[1] ? add->src[0] : add->src[1];
        if (!rs || rs->view_src != mul || !ps || (ps != prev && ps->view_src != prev) || add->type != GGML_TYPE_F32 ||
            ggml_nelements(add) != ch*n_t*n_s || !ggml_is_contiguous(add)) {
            return false;
        }

        if ((cont->flags | mul->flags | add->flags | prev->flags) & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        if (ggml_node_get_use_count(gf, node_idxs[idx + 3 + 3*l]) != 1 || ggml_node_get_use_count(gf, node_idxs[idx + 4 + 3*l]) != 1 ||
            ggml_node_get_use_count(gf, node_idxs[idx + 5 + 3*l]) != 1) {
            return false;
        }

        prev = add;
    }

    if (ggml_node_get_use_count(gf, node_idxs[idx + 2]) != 1) {
        return false;
    }

    // (cur + out)*s + t
    const ggml_tensor * add = n[3 + 3*n_lag];
    const ggml_tensor * sc  = n[4 + 3*n_lag];
    const ggml_tensor * rs  = add->src[0] == cur ? add->src[1] : add->src[0];

    if ((add->src[0] != cur && add->src[1] != cur) || !ggml_are_same_shape(add, cur) || rs != prev ||
        sc->src[0] != add || !ggml_is_contiguous(sc) || ggml_nelements(sc) != ch*n_t*n_s || add->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 ||
        (add->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + 3 + 3*n_lag]) != 1) {
        return false;
    }

    return true;
}

#define GGML_METAL_DEPTH_MIX_MAX_STATES 64

// softmax attention of learned queries over stacked states: the earlier states, already normalized, and the
// rms-normalized current state; the output is the score-weighted sum of the states
static bool ggml_metal_fusion_check_depth_mix(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const bool repeat = n[3]->op == GGML_OP_REPEAT;

    // optionally followed by the rms norm of the output [times a weight], written as a second output
    const int n_base = repeat ? 7 : 6;
    const int n_post = (int) fusion->ops.size() - n_base;

    const ggml_tensor * norm   = n[0];
    const ggml_tensor * cat    = n[1];
    const ggml_tensor * cont   = n[2];
    const ggml_tensor * mm0    = n[repeat ? 4 : 3];
    const ggml_tensor * sm     = n[repeat ? 5 : 4];
    const ggml_tensor * mm1    = n[repeat ? 6 : 5];
    const ggml_tensor * cur    = norm->src[0];
    const ggml_tensor * stack  = cat->src[0];
    const ggml_tensor * query  = repeat ? n[3]->src[0] : mm0->src[1];

    if (cat->src[1] != norm || ggml_get_op_params_i32(cat, 0) != 1 ||
        !cont->src[0] || cont->src[0]->op != GGML_OP_TRANSPOSE || cont->src[0]->view_src != cat ||
        mm0->src[0] != cat || mm0->src[1] != (repeat ? n[3] : query) ||
        sm->src[0] != mm0 || sm->src[1] || sm->src[2] || ggml_get_op_params_f32(sm, 1) != 0.0f ||
        mm1->src[0] != cont || mm1->src[1] != sm) {
        return false;
    }

    const int64_t n_embd = cur->ne[0];
    const int64_t n_tok  = cur->ne[2];
    const int64_t nq     = query->ne[1];

    if (cur->type != GGML_TYPE_F32 || stack->type != GGML_TYPE_F32 || query->type != GGML_TYPE_F32 || mm1->type != GGML_TYPE_F32 ||
        cur->nb[0] != sizeof(float) || cur->ne[1] != 1 || cur->ne[3] != 1 ||
        !ggml_is_contiguous(stack) || stack->ne[0] != n_embd || stack->ne[2] != n_tok || stack->ne[3] != 1 ||
        stack->ne[1] + 1 > GGML_METAL_DEPTH_MIX_MAX_STATES ||
        !ggml_is_contiguous(query) || query->ne[0] != n_embd || query->ne[2] != 1 || query->ne[3] != 1 ||
        (repeat && (n[3]->ne[1] != nq || n[3]->ne[2] != n_tok)) || (!repeat && n_tok != 1) ||
        !ggml_is_contiguous(mm1) || mm1->ne[0] != n_embd || mm1->ne[1] != nq || mm1->ne[2] != n_tok) {
        return false;
    }

    // the intermediates are read inside the group only, the concat also by the transpose view
    for (int j = 0; j < n_base - 1; ++j) {
        const int uses = n[j] == cat ? 2 : 1;
        if ((n[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(gf, node_idxs[idx + j]) != uses) {
            return false;
        }
    }

    if (n_post > 0) {
        // the mixed state stays an output; one query and one token, whole-row norm of it
        const ggml_tensor * rn = n[n_base];
        const ggml_tensor * x  = rn->src[0];
        if (repeat || n_tok != 1 || !x || (x != mm1 && x->view_src != mm1) || x->view_offs != 0 ||
            ggml_nelements(x) != n_embd*nq || x->ne[0] != n_embd || !ggml_is_contiguous(x) || !ggml_is_contiguous(rn) ||
            rn->type != GGML_TYPE_F32) {
            return false;
        }
        if (n_post == 2) {
            const ggml_tensor * mul = n[n_base + 1];
            const ggml_tensor * w   = mul->src[1];
            if (mul->src[0] != rn || !w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != n_embd ||
                !ggml_is_contiguous(mul) ||
                !ggml_are_same_shape(mul, rn) || (rn->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                ggml_node_get_use_count(gf, node_idxs[idx + n_base]) != 1) {
                return false;
            }
        }
    }

    return true;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

static const std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sp = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sp_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sig = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sig_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    const int n_ops = (int) fusion->ops.size();

    bool with_norm = false;
    for (int j = 0; j < n_ops; ++j) {
        with_norm |= fusion->ops[j] == GGML_OP_SUM_ROWS;
    }
    const bool with_scale = fusion->ops[n_ops - 1] == GGML_OP_SCALE;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<ggml_op> & ops_all = fusion->ops_all;

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) ops_all.size()) {
        return false;
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * first = gf->nodes[raw_start];
    const ggml_tensor * out   = gf->nodes[raw_end];

    // the unary chains start at the sigmoid or the softplus node and rank biased scores
    const bool with_unary = first->op == GGML_OP_UNARY;
    const bool with_sp    = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS;
    const bool with_sig   = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID;

    if (with_unary && !with_sp && !with_sig) {
        return false;
    }

    const ggml_tensor * probs;          // the scoring output
    const ggml_tensor * scores;         // the node the argsort ranks (adds the bias for the unary chains)
    const ggml_tensor * probs_reshaped;
    const ggml_tensor * argsort;
    const ggml_tensor * ids;
    const ggml_tensor * get_rows;
    const ggml_tensor * logits = first->src[0];
    int ids_off;

    if (with_unary) {
        // the softplus chains add a SQRT node, the sigmoid chains start at the sigmoid node
        const int base = with_sp ? 1 : 0;

        probs          = gf->nodes[raw_start + base + 0];
        probs_reshaped = gf->nodes[raw_start + base + 1];
        scores         = gf->nodes[raw_start + base + 2];
        argsort        = gf->nodes[raw_start + base + 3];
        ids            = gf->nodes[raw_start + base + 4];
        get_rows       = gf->nodes[raw_start + base + 5];
        ids_off        = base + 4;

        if (scores->op != GGML_OP_ADD || scores->src[0] != probs ||
            (with_sp && (probs->op != GGML_OP_SQRT || probs->src[0] != first))) {
            return false;
        }

        // the selection bias is a per-expert vector shared by all rows
        const ggml_tensor * bias = scores->src[1];
        if (bias->type != GGML_TYPE_F32 || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            bias->ne[0] != probs->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
            return false;
        }
    } else {
        probs          = first;
        scores         = first;
        probs_reshaped = gf->nodes[raw_start + 1];
        argsort        = gf->nodes[raw_start + 2];
        ids            = gf->nodes[raw_start + 3];
        get_rows       = gf->nodes[raw_start + 4];
        ids_off        = 3;

        // the fused kernel implements plain softmax only
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    ((const int32_t *) first->op_params) + 0, sizeof(scale));
        memcpy(&max_bias, ((const int32_t *) first->op_params) + 1, sizeof(max_bias));
        if (scale != 1.0f || max_bias != 0.0f || first->src[1] || first->src[2]) {
            return false;
        }
    }

    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        return false;
    }

    // the top-k gathers the unbiased scores, the argsort ranks the biased ones
    if (probs_reshaped->src[0] != probs || argsort->src[0] != scores ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const int norm_off = with_unary ? 6 + (with_sp ? 1 : 0) : 5;

        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + norm_off + 0];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + norm_off + 1];
        const ggml_tensor * denom            = gf->nodes[raw_start + norm_off + 2];
        const ggml_tensor * div              = gf->nodes[raw_start + norm_off + 3];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + norm_off + 4];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            denom->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != denom ||
            out_reshaped->src[0] != div) {
            return false;
        }

        // the kernel applies the sum adjustment as a lower clamp or an additive bias, never a scale
        if (denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) != 1.0f) {
            return false;
        }

        if (with_scale && out->src[0] != out_reshaped) {
            return false;
        }
    } else if (with_scale && out->src[0] != get_rows) {
        return false;
    }

    const int outputs[2] = { raw_start + ids_off, raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

struct ggml_metal_moe_reduce_match {
    const ggml_tensor * experts;
    const ggml_tensor * weights;
    const ggml_tensor * dst;
    int node_count;
};

static bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + (int) ops_all.size() > gf->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const int raw_count     = (int) ops_all.size();
    const int n_expert_used = raw_count / 2;

    if (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        return false;
    }

    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views != n_expert_used) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            return false;
        }
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        if (gf->nodes[node_idx + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    if (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        return false;
    }

    if ((int) fusion->ops.size() != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + (int) fusion->ops.size() - 1] != raw_end) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const struct ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_operand(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    if (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_residual(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    return ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    if (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

static const std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

static const std::vector<ggml_op> ops_depth_mix = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_depth_mix_view = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_VIEW, GGML_OP_MUL_MAT
};
// the decode variants with the norm of the mixed state: through a reshape or a view of it, with or without the weight
static const std::vector<ggml_op> ops_depth_mix_norm = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_RMS_NORM
};
static const std::vector<ggml_op> ops_depth_mix_reshape_norm = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM
};
static const std::vector<ggml_op> ops_depth_mix_view_norm = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_RMS_NORM
};
static const std::vector<ggml_op> ops_depth_mix_reshape_norm_mul = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_depth_mix_view_norm_mul = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_RMS_NORM, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_depth_mix_repeat = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_REPEAT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT
};
static const std::vector<ggml_op> ops_depth_mix_repeat_view = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_REPEAT, GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT
};

static const std::vector<ggml_op> ops_add_scale     = { GGML_OP_ADD, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_add_scale_add = { GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD };

static const std::vector<ggml_op> ops_conv_step = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV
};
static const std::vector<ggml_op> ops_conv_step_add_scale = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_conv_step_add_scale_res = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD
};

// the taps of a 4-wide convolution: the tap reshape is expanded before the concat
static const std::vector<ggml_op> ops_conv_step_dyn4 = {
    GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_ADD, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_rel_mask = {
    GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_CPY
};
// the raw order of a batch can carry the cast of the mask before the add, which the optimizer then moves out of
// the group: this entry never matches the encoded order, it only keeps the inputs allocated up to the last node
static const std::vector<ggml_op> ops_rel_mask_cast = {
    GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_CPY, GGML_OP_ADD, GGML_OP_CPY
};

// the raw variants carry the views met between the nodes, so that the inputs stay allocated up to the last node
static const std::vector<ggml_op> ops_l2_clip   = { GGML_OP_L2_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_l2_clip_w = { GGML_OP_MUL, GGML_OP_L2_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_cat_mv = { GGML_OP_CONCAT, GGML_OP_MUL_MAT };
static const std::vector<ggml_op> ops_sigmoid_mul = { GGML_OP_UNARY, GGML_OP_MUL };
static const std::vector<ggml_op> ops_scale_id    = { GGML_OP_GET_ROWS, GGML_OP_MUL };
static const std::vector<ggml_op> ops_gate_mv = {
    GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_gate_mv_view = {
    GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_tanh_gate = {
    GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_tanh_gate_view = {
    GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_ADD
};

// the raw sequences carry the up view (and the gate view after the scale) so that the inputs stay allocated
static const std::vector<ggml_op> ops_situ = {
    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_situ_scaled = {
    GGML_OP_GET_ROWS, GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL
};

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   {}, false, ggml_metal_fusion_check_norm },

    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_2, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_3, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_4, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_5, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_6, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_7, {}, false, ggml_metal_fusion_check_add_chain },

    { GGML_METAL_FUSION_SNAKE, ops_snake, {}, false, ggml_metal_fusion_check_snake },

    { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },

    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe,                     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_scale,               {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm,                {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale,          {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_bias,           {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale_bias,     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp,                  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_scale,            {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm,             {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale,       {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_bias,        {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale_bias,  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig,                 {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_scale,           {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm,            {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale,      {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_bias,       {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale_bias, {2}, true, ggml_metal_fusion_check_topk_moe },

    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, {}, true, ggml_metal_fusion_check_moe_reduce },

    { GGML_METAL_FUSION_SSM_CONV_SILU, ops_ssm_conv_silu, {}, false, ggml_metal_fusion_check_ssm_conv_silu },

    { GGML_METAL_FUSION_MUL_MAT_ADD, ops_mul_mat_add, {}, false, ggml_metal_fusion_check_mul_mat_add },

    { GGML_METAL_FUSION_SITU, ops_situ,        {}, true, ggml_metal_fusion_check_situ },
    { GGML_METAL_FUSION_SITU, ops_situ_scaled, {}, true, ggml_metal_fusion_check_situ },

    { GGML_METAL_FUSION_TANH_GATE, ops_tanh_gate,      {}, true, ggml_metal_fusion_check_tanh_gate },
    { GGML_METAL_FUSION_TANH_GATE, ops_tanh_gate_view, {}, true, ggml_metal_fusion_check_tanh_gate },

    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix,        {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_view,   {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_norm,             {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_reshape_norm,     {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_view_norm,        {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_reshape_norm_mul, {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_view_norm_mul,    {}, true, ggml_metal_fusion_check_depth_mix },

    { GGML_METAL_FUSION_ADD_SCALE, ops_add_scale,     {}, false, ggml_metal_fusion_check_add_scale },

    { GGML_METAL_FUSION_CONV_STEP, ops_conv_step,           {}, true, ggml_metal_fusion_check_conv_step },
    { GGML_METAL_FUSION_CONV_STEP, ops_conv_step_add_scale_res, {}, true, ggml_metal_fusion_check_conv_step },
    { GGML_METAL_FUSION_CONV_STEP, ops_conv_step_dyn4,          {}, true, ggml_metal_fusion_check_conv_step_dyn },

    { GGML_METAL_FUSION_REL_MASK, ops_rel_mask,      {}, true, ggml_metal_fusion_check_rel_mask },

    { GGML_METAL_FUSION_L2_CLIP, ops_l2_clip,   {}, false, ggml_metal_fusion_check_l2_clip },

    { GGML_METAL_FUSION_CAT_MV,  ops_cat_mv,       {}, true, ggml_metal_fusion_check_cat_mv },

    { GGML_METAL_FUSION_SIGMOID_MUL, ops_sigmoid_mul, {}, false, ggml_metal_fusion_check_sigmoid_mul },
    { GGML_METAL_FUSION_SCALE_ID,    ops_scale_id,    {}, true,  ggml_metal_fusion_check_scale_id },
    { GGML_METAL_FUSION_GATE_MV, ops_gate_mv,      {}, true, ggml_metal_fusion_check_gate_mv },
    { GGML_METAL_FUSION_GATE_MV, ops_gate_mv_view, {}, true, ggml_metal_fusion_check_gate_mv },
    { GGML_METAL_FUSION_L2_CLIP, ops_l2_clip_w, {}, false, ggml_metal_fusion_check_l2_clip },
    { GGML_METAL_FUSION_REL_MASK, ops_rel_mask_cast, {}, true, ggml_metal_fusion_check_rel_mask },
    { GGML_METAL_FUSION_CONV_STEP, ops_conv_step_add_scale, {}, true, ggml_metal_fusion_check_conv_step },
    { GGML_METAL_FUSION_ADD_SCALE, ops_add_scale_add, {}, false, ggml_metal_fusion_check_add_scale },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_repeat,      {}, true, ggml_metal_fusion_check_depth_mix },
    { GGML_METAL_FUSION_DEPTH_MIX, ops_depth_mix_repeat_view, {}, true, ggml_metal_fusion_check_depth_mix },
};

// ---- alloc deps -----------------------------------------------------------

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < (int) fusion->ops.size(); j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, const ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    const ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    if (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: number of nodes starting at idx (a raw sequential graph index) that the longest
// pattern matching there fuses. matching runs on the same filtered (view transparent) node
// sequence that the compute phase uses, so the returned count is the raw index span from idx to
// the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // a view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    // one pattern per group: back-to-back groups stay separate nodes for the reorder, which can then run
    // independent chains (e.g. the controls of the Q, K and V inputs of a layer) side by side
    int len = 1;
    const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, 0, GGML_METAL_FUSION_STRUCTURAL, &len);
    if (!fusion) {
        return 1;
    }

    const int i_f = len;

    // map the matched non-empty nodes back to the raw index span (views are included, so it can exceed the op count)
    return idxs[i_f - 1] - idx + 1;
}
