#include "server-scope.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <regex>
#include <unordered_map>
#include <vector>

using json = common_json;

// frames pending delivery beyond this shed the oldest one, seq exposes the gap
static constexpr size_t SCOPE_MAX_PENDING = 32;

// single pass accumulator over the float values of one tensor
// adding a statistic: one field, one line in add(), one line in to_json()
struct scope_stats {
    int64_t n = 0;
    int64_t n_nan = 0;
    int64_t n_inf = 0;
    int64_t n_zero = 0;
    float vmin = INFINITY;
    float vmax = -INFINITY;
    double sum = 0.0;
    double sum_sq = 0.0;

    void add(float v) {
        n++;
        if (std::isnan(v)) { n_nan++; return; }
        if (std::isinf(v)) { n_inf++; return; }
        if (v == 0.0f) n_zero++;
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
        sum += v;
        sum_sq += (double) v*v;
    }

    json to_json() const {
        json stats = {
            {"n",      n},
            {"n_nan",  n_nan},
            {"n_inf",  n_inf},
            {"n_zero", n_zero},
        };
        const int64_t n_finite = n - n_nan - n_inf;
        if (n_finite > 0) {
            stats["min"]  = vmin;
            stats["max"]  = vmax;
            stats["mean"] = sum / n_finite;
            stats["rms"]  = std::sqrt(sum_sq / n_finite);
        }
        return stats;
    }
};

// concurrency: every mutable field is guarded by mutex except the lock free fast path
// reads of armed and has_filter in the ask phase, armed is published with release
// semantics after the stream state is prepared under the mutex, so an acquire load
// observing armed as true also observes the matching filter and cleared buffers
struct scope_state {
    std::mutex mutex;
    std::condition_variable cv;

    std::atomic<bool> armed{false};
    bool open = false;

    std::regex filter;
    std::atomic<bool> has_filter{false};

    // number of ask phase calls in the current frame, nonzero proves a graph executed
    std::atomic<uint64_t> n_asked{0};

    // frame index since the stream opened, monotonic even across shed frames
    uint64_t seq = 0;

    // single id space for compute nodes and leafs, src entries are positions in nodes
    json nodes = json::array();
    std::unordered_map<const ggml_tensor *, size_t> index;
    size_t n_leafs = 0;

    int64_t t_start_us = 0;

    std::deque<json> frames;

    // host staging buffer for tensors living on device memory, released on close
    std::vector<uint8_t> staging;
};

static scope_state g_scope;

static float scope_get_f32(const uint8_t * data, ggml_type type, const size_t * nb, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const size_t i = i3*nb[3] + i2*nb[2] + i1*nb[1] + i0*nb[0];
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) &data[i];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
        case GGML_TYPE_I64:  return (float) *(const int64_t *) &data[i];
        case GGML_TYPE_I32:  return (float) *(const int32_t *) &data[i];
        case GGML_TYPE_I16:  return (float) *(const int16_t *) &data[i];
        case GGML_TYPE_I8:   return (float) *(const int8_t  *) &data[i];
        default:             return 0.0f;
    }
}

static bool scope_type_readable(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I64:
        case GGML_TYPE_I32:
        case GGML_TYPE_I16:
        case GGML_TYPE_I8:
            return true;
        default:
            return false;
    }
}

// zero or trivial compute ops, the frontend contracts these into edges
static bool scope_is_plumbing(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_GET_ROWS:
            return true;
        default:
            return false;
    }
}

// kind of a src tensor never seen as a captured compute node
static const char * scope_leaf_kind(const ggml_tensor * t) {
    if (t->op != GGML_OP_NONE) {
        return "uncaptured";
    }
    if (t->flags & GGML_TENSOR_FLAG_INPUT) {
        return "input";
    }
    if (strncmp(t->name, "cache_", 6) == 0 || (t->view_src && strncmp(t->view_src->name, "cache_", 6) == 0)) {
        return "kv";
    }
    return "weight";
}

static json scope_node_base(const ggml_tensor * t) {
    return {
        {"name",   t->name},
        {"op",     ggml_op_desc(t)},
        {"type",   ggml_type_name(t->type)},
        {"shape",  json::array({t->ne[0], t->ne[1], t->ne[2], t->ne[3]})},
        {"buffer", t->buffer ? ggml_backend_buffer_name(t->buffer) : "none"},
    };
}

// resolves a src tensor to its node position, registers a leaf entry on first sight
static size_t scope_src_index(const ggml_tensor * s) {
    auto it = g_scope.index.find(s);
    if (it != g_scope.index.end()) {
        return it->second;
    }
    const size_t i = g_scope.nodes.size();
    json leaf = scope_node_base(s);
    leaf["leaf"] = scope_leaf_kind(s);
    g_scope.nodes.push_back(std::move(leaf));
    g_scope.index.emplace(s, i);
    g_scope.n_leafs++;
    return i;
}

static json scope_tensor_stats(const ggml_tensor * t, const uint8_t * data) {
    scope_stats stats;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    stats.add(scope_get_f32(data, t->type, t->nb, i0, i1, i2, i3));
                }
            }
        }
    }
    return stats.to_json();
}

static bool scope_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    if (ask) {
        if (!g_scope.armed.load(std::memory_order_acquire)) {
            return false;
        }
        if (g_scope.n_asked.fetch_add(1, std::memory_order_relaxed) == 0) {
            g_scope.t_start_us = ggml_time_us();
        }
        if (!g_scope.has_filter.load(std::memory_order_relaxed)) {
            return true;
        }
        std::lock_guard<std::mutex> lock(g_scope.mutex);
        return std::regex_search(t->name, g_scope.filter);
    }

    std::lock_guard<std::mutex> lock(g_scope.mutex);
    if (!g_scope.armed.load(std::memory_order_acquire)) {
        return true;
    }

    json src = json::array();
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        src.push_back(scope_src_index(t->src[i]));
    }

    json node = scope_node_base(t);
    node["src"] = std::move(src);
    if (scope_is_plumbing(t)) {
        node["plumbing"] = true;
    }

    if (scope_type_readable(t->type)) {
        const uint8_t * data;
        if (ggml_backend_buffer_is_host(t->buffer)) {
            data = (const uint8_t *) t->data;
        } else {
            const size_t n_bytes = ggml_nbytes(t);
            g_scope.staging.resize(n_bytes);
            ggml_backend_tensor_get(t, g_scope.staging.data(), 0, n_bytes);
            data = g_scope.staging.data();
        }
        node["stats"] = scope_tensor_stats(t, data);
    }

    const size_t i = g_scope.nodes.size();
    g_scope.nodes.push_back(std::move(node));
    g_scope.index[t] = i;
    return true;
}

// clears the frame accumulator, the next decode starts a fresh frame
static void scope_frame_reset_locked() {
    g_scope.n_asked.store(0, std::memory_order_relaxed);
    g_scope.nodes = json::array();
    g_scope.index.clear();
    g_scope.n_leafs = 0;
}

void server_scope_install(common_params & params) {
    if (!params.endpoint_scope) {
        return;
    }
    params.cb_eval = scope_cb_eval;
    params.cb_eval_user_data = nullptr;
}

void server_scope_on_decode_done() {
    if (!g_scope.armed.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_scope.mutex);
    if (!g_scope.open || g_scope.n_asked.load(std::memory_order_relaxed) == 0) {
        return;
    }
    json frame = {
        {"seq",          g_scope.seq++},
        {"n_nodes",      g_scope.nodes.size()},
        {"n_leafs",      g_scope.n_leafs},
        {"t_capture_ms", (ggml_time_us() - g_scope.t_start_us) / 1000.0},
        {"nodes",        std::move(g_scope.nodes)},
    };
    scope_frame_reset_locked();
    if (g_scope.frames.size() >= SCOPE_MAX_PENDING) {
        g_scope.frames.pop_front();
    }
    g_scope.frames.push_back(std::move(frame));
    g_scope.cv.notify_all();
}

json server_scope_open(const std::string & filter) {
    std::lock_guard<std::mutex> lock(g_scope.mutex);
    if (g_scope.open) {
        return {{"error", "a scope stream is already open"}, {"error_type", "unavailable"}};
    }
    g_scope.has_filter.store(!filter.empty(), std::memory_order_relaxed);
    if (!filter.empty()) {
        try {
            g_scope.filter = std::regex(filter, std::regex::optimize);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid filter regex: ") + e.what()}, {"error_type", "invalid_request"}};
        }
    }
    scope_frame_reset_locked();
    g_scope.frames.clear();
    g_scope.seq = 0;
    g_scope.open = true;
    g_scope.armed.store(true, std::memory_order_release);
    return json();
}

bool server_scope_next_frame(std::string & out_json, int timeout_ms) {
    std::unique_lock<std::mutex> lock(g_scope.mutex);
    g_scope.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [] { return !g_scope.frames.empty(); });
    if (g_scope.frames.empty()) {
        return false;
    }
    out_json = g_scope.frames.front().dump();
    g_scope.frames.pop_front();
    return true;
}

void server_scope_close() {
    std::lock_guard<std::mutex> lock(g_scope.mutex);
    g_scope.armed.store(false, std::memory_order_release);
    g_scope.open = false;
    g_scope.frames.clear();
    scope_frame_reset_locked();
    g_scope.staging.clear();
    g_scope.staging.shrink_to_fit();
}
