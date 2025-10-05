#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "common/log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstring>

#include "index.html.hpp"

namespace fs = std::filesystem;

namespace {

using json = nlohmann::json;

struct tensor_layout {
    size_t width  = 1;
    size_t height = 1;
    size_t depth  = 1;
};

struct tensor_entry {
    std::string              name;
    ggml_tensor *            tensor = nullptr;
    int64_t                  tensor_index = -1;
    size_t                   offset = 0;
    size_t                   n_elements = 0;
    size_t                   n_bytes = 0;
    std::vector<int64_t>     shape;
    tensor_layout            layout;
};

struct tokenizer_info {
    int64_t total_tokens = 0;
    int64_t key_tokens   = -1;
    int64_t key_scores   = -1;
    int64_t key_types    = -1;
};

struct viewer_state {
    std::string              model_path;
    size_t                   file_size = 0;
    gguf_context_ptr         gguf;
    ggml_context_ptr         tensor_ctx;
    size_t                   data_offset = 0;
    size_t                   alignment   = 0;
    std::vector<tensor_entry> tensors;
    std::unordered_map<std::string, size_t> tensor_index_by_name;
    tokenizer_info           tokenizer;
};

std::optional<int64_t> parse_i64(const std::string & value) {
    if (value.empty()) {
        return std::nullopt;
    }
    int64_t result = 0;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (ec != std::errc() || ptr != value.data() + value.size()) {
        return std::nullopt;
    }
    return result;
}

std::optional<size_t> parse_size_t(const std::string & value) {
    auto v = parse_i64(value);
    if (!v || *v < 0) {
        return std::nullopt;
    }
    return static_cast<size_t>(*v);
}

std::string url_decode(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        char c = value[i];
        if (c == '+') {
            result.push_back(' ');
        } else if (c == '%' && i + 2 < value.size()) {
            int hi = std::toupper(static_cast<unsigned char>(value[i + 1]));
            int lo = std::toupper(static_cast<unsigned char>(value[i + 2]));
            auto hex = [](int ch) -> int {
                if (ch >= '0' && ch <= '9') return ch - '0';
                if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
                return -1;
            };
            int high = hex(hi);
            int low  = hex(lo);
            if (high >= 0 && low >= 0) {
                result.push_back(static_cast<char>((high << 4) | low));
                i += 2;
            } else {
                result.push_back(c);
            }
        } else {
            result.push_back(c);
        }
    }
    return result;
}

void set_json_response(httplib::Response & res, const json & body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

json make_error(const std::string & message) {
    json body;
    body["error"] = message;
    return body;
}

json kv_scalar_to_json(const gguf_context * ctx, int64_t key_id, gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, key_id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, key_id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, key_id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, key_id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, key_id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, key_id);
        case GGUF_TYPE_FLOAT32:return gguf_get_val_f32(ctx, key_id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, key_id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, key_id);
        case GGUF_TYPE_FLOAT64:return gguf_get_val_f64(ctx, key_id);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(ctx, key_id);
        case GGUF_TYPE_STRING: return gguf_get_val_str(ctx, key_id);
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
        default:
            break;
    }
    return nullptr;
}

json kv_array_preview(const gguf_context * ctx, int64_t key_id, size_t limit, bool & truncated) {
    const gguf_type arr_type = gguf_get_arr_type(ctx, key_id);
    const size_t n = gguf_get_arr_n(ctx, key_id);
    truncated = n > limit;

    json out = json::array();
    const size_t to_take = std::min(n, limit);

    switch (arr_type) {
        case GGUF_TYPE_UINT8: {
            const auto * data = static_cast<const uint8_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_INT8: {
            const auto * data = static_cast<const int8_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_UINT16: {
            const auto * data = static_cast<const uint16_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_INT16: {
            const auto * data = static_cast<const int16_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_UINT32: {
            const auto * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_INT32: {
            const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_FLOAT32: {
            const auto * data = static_cast<const float *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_UINT64: {
            const auto * data = static_cast<const uint64_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_INT64: {
            const auto * data = static_cast<const int64_t *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_FLOAT64: {
            const auto * data = static_cast<const double *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_BOOL: {
            const auto * data = static_cast<const bool *>(gguf_get_arr_data(ctx, key_id));
            for (size_t i = 0; i < to_take; ++i) out.push_back(data[i]);
        } break;
        case GGUF_TYPE_STRING: {
            for (size_t i = 0; i < to_take; ++i) out.push_back(gguf_get_arr_str(ctx, key_id, i));
        } break;
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
        default:
            break;
    }

    return out;
}

json describe_kv(const gguf_context * ctx, int64_t key_id, size_t preview_limit) {
    const char * key_name = gguf_get_key(ctx, key_id);
    const gguf_type type  = gguf_get_kv_type(ctx, key_id);

    json node;
    node["key"] = key_name;
    node["type"] = gguf_type_name(type);

    if (type == GGUF_TYPE_ARRAY) {
        const size_t n = gguf_get_arr_n(ctx, key_id);
        node["length"] = n;
        const gguf_type arr_type = gguf_get_arr_type(ctx, key_id);
        node["arrayType"] = gguf_type_name(arr_type);
        bool truncated = false;
        node["preview"] = kv_array_preview(ctx, key_id, preview_limit, truncated);
        node["previewTruncated"] = truncated;
    } else {
        node["value"] = kv_scalar_to_json(ctx, key_id, type);
    }

    return node;
}

std::vector<int64_t> tensor_shape(const ggml_tensor * tensor) {
    const int nd = ggml_n_dims(tensor);
    std::vector<int64_t> shape;
    shape.reserve(nd);
    for (int i = 0; i < nd; ++i) {
        shape.push_back(tensor->ne[i]);
    }
    return shape;
}

tensor_layout compute_tensor_layout(const std::vector<int64_t> & shape, size_t n_elements) {
    constexpr size_t kMaxLine = 1024;

    tensor_layout layout;
    layout.width  = 1;
    layout.height = 1;
    layout.depth  = 1;

    if (shape.empty()) {
        layout.width = std::max<size_t>(1, n_elements);
        layout.height = 1;
    } else if (shape.size() == 1) {
        const size_t total = std::max<size_t>(1, n_elements);
        size_t width = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(total))));
        if (width == 0) {
            width = 1;
        }
        width = std::min(width, kMaxLine);
        if (width == 0) {
            width = 1;
        }
        size_t height = (total + width - 1) / width;
        if (height == 0) {
            height = 1;
        }
        layout.width = width;
        layout.height = height;
    } else if (shape.size() == 2) {
        layout.width = static_cast<size_t>(std::max<int64_t>(1, shape.back()));
        layout.height = static_cast<size_t>(std::max<int64_t>(1, shape.front()));
    } else {
        const size_t depth = static_cast<size_t>(std::max<int64_t>(1, shape.back()));
        size_t width = static_cast<size_t>(std::max<int64_t>(1, shape[shape.size() - 2]));
        if (width == 0) {
            width = 1;
        }

        size_t plane = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            plane *= static_cast<size_t>(std::max<int64_t>(1, shape[i]));
        }
        if (plane == 0) {
            plane = width;
        }

        size_t height = plane / width;
        if (height * width < plane) {
            ++height;
        }
        if (height == 0) {
            height = 1;
        }

        layout.width = width;
        layout.height = height;
        layout.depth = depth;
    }

    if (layout.width == 0) {
        layout.width = 1;
    }
    if (layout.height == 0) {
        layout.height = (layout.width > 0) ? (n_elements + layout.width - 1) / layout.width : 1;
        if (layout.height == 0) {
            layout.height = 1;
        }
    }
    if (layout.depth == 0) {
        layout.depth = 1;
    }

    const size_t slice = layout.width * layout.height;
    if (slice > 0) {
        size_t required_depth = (n_elements + slice - 1) / slice;
        if (required_depth == 0) {
            required_depth = 1;
        }
        layout.depth = std::max(layout.depth, required_depth);
    }

    return layout;
}

json tensor_to_json(const tensor_entry & entry, size_t data_offset) {
    json node;
    node["name"]       = entry.name;
    node["type"]       = ggml_type_name(entry.tensor->type);
    node["nElements"]  = entry.n_elements;
    node["nBytes"]     = entry.n_bytes;
    node["offset"]     = entry.offset;
    node["fileOffset"] = data_offset + entry.offset;
    node["shape"]      = entry.shape;
    node["ndim"]       = ggml_n_dims(entry.tensor);
    node["layout"]     = {
        {"width",  entry.layout.width},
        {"height", entry.layout.height},
        {"depth",  entry.layout.depth},
    };
    return node;
}

struct tensor_window_result {
    size_t offset = 0;
    size_t count  = 0;
    size_t total  = 0;
    float min     = 0.0f;
    float max     = 0.0f;
    std::vector<float> values;
};

struct tensor_tile_result {
    size_t x      = 0;
    size_t y      = 0;
    size_t slice  = 0;
    size_t width  = 0;
    size_t height = 0;
    size_t stride = 0;
    size_t offset = 0;
    size_t valid  = 0;
    float min     = 0.0f;
    float max     = 0.0f;
    size_t slice_valid = 0;
    float slice_min    = 0.0f;
    float slice_max    = 0.0f;
    std::vector<float> values;
    std::vector<uint8_t> mask;
};

bool tensor_window_values(
        const viewer_state & state,
        const tensor_entry & entry,
        size_t offset,
        size_t count,
        tensor_window_result & out,
        std::string & error) {
    out = {};
    out.total = entry.n_elements;

    if (count == 0 || entry.n_elements == 0) {
        return true;
    }

    if (offset >= entry.n_elements) {
        out.offset = entry.n_elements;
        return true;
    }

    const auto * traits = ggml_get_type_traits(entry.tensor->type);
    if (!traits) {
        error = "Unknown tensor type";
        return false;
    }

    const size_t block_size = traits->blck_size > 0 ? traits->blck_size : 1;
    const size_t type_size  = traits->type_size;

    if (type_size == 0) {
        error = "Invalid tensor type size";
        return false;
    }

    const size_t start_block = offset / block_size;
    const size_t block_offset = start_block * block_size;
    const size_t tensor_bytes = entry.n_bytes;
    const size_t start_byte_offset = start_block * type_size;

    if (start_byte_offset > tensor_bytes) {
        error = "Tensor offset outside data range";
        return false;
    }

    const size_t end_index = std::min(entry.n_elements, offset + count);
    const size_t end_block = (end_index + block_size - 1) / block_size;
    const size_t requested_blocks = end_block > start_block ? (end_block - start_block) : 1;

    const size_t available_bytes = tensor_bytes - start_byte_offset;
    const size_t requested_bytes = requested_blocks * type_size;
    const size_t bytes_to_read = std::min(requested_bytes, available_bytes);
    const size_t blocks_to_read = bytes_to_read / type_size;

    if (blocks_to_read == 0) {
        return true;
    }

    const size_t elements_to_convert = blocks_to_read * block_size;

    std::ifstream file(state.model_path, std::ios::binary);
    if (!file) {
        error = "Failed to open model file";
        return false;
    }

    const size_t absolute_offset = state.data_offset + entry.offset + start_byte_offset;
    if (absolute_offset + bytes_to_read > state.file_size) {
        error = "Tensor offset outside file bounds";
        return false;
    }

    file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    std::vector<uint8_t> raw(bytes_to_read, 0);
    file.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(bytes_to_read));
    if (file.gcount() < static_cast<std::streamsize>(bytes_to_read)) {
        error = "Failed to read tensor data";
        return false;
    }

    std::vector<float> converted(elements_to_convert, 0.0f);

    if (entry.tensor->type == GGML_TYPE_F32) {
        const size_t elements_read = bytes_to_read / sizeof(float);
        const float * src = reinterpret_cast<const float *>(raw.data());
        converted.assign(src, src + elements_read);
    } else if (traits->to_float == nullptr) {
        converted.resize(bytes_to_read);
        for (size_t i = 0; i < bytes_to_read; ++i) {
            converted[i] = static_cast<float>(raw[i]);
        }
    } else {
        traits->to_float(raw.data(), converted.data(), static_cast<int64_t>(elements_to_convert));
    }

    const size_t start_index_in_block = offset - block_offset;
    size_t available_values = converted.size() > start_index_in_block
        ? converted.size() - start_index_in_block
        : 0;

    size_t take = std::min(end_index - offset, available_values);

    out.offset = offset;
    out.count = take;

    if (take == 0) {
        return true;
    }

    out.values.resize(take);
    std::memcpy(out.values.data(), converted.data() + start_index_in_block, take * sizeof(float));

    auto [min_it, max_it] = std::minmax_element(out.values.begin(), out.values.end());
    out.min = *min_it;
    out.max = *max_it;
    return true;
}

bool tensor_tile_values(
        const viewer_state & state,
        const tensor_entry & entry,
        size_t slice_index,
        size_t x,
        size_t y,
        size_t width,
        size_t height,
        tensor_tile_result & out,
        std::string & error) {
    out = {};

    const tensor_layout & layout = entry.layout;
    if (layout.width == 0 || layout.height == 0) {
        return true;
    }

    const size_t slice_size = layout.width * layout.height;
    if (slice_size == 0) {
        return true;
    }

    if (layout.depth == 0) {
        return true;
    }

    if (slice_index >= layout.depth) {
        slice_index = layout.depth - 1;
    }

    if (x >= layout.width) {
        x = layout.width - 1;
    }
    if (y >= layout.height) {
        y = layout.height - 1;
    }

    if (width == 0 || height == 0) {
        return true;
    }

    width = std::min(width, layout.width - x);
    height = std::min(height, layout.height - y);

    out.x = x;
    out.y = y;
    out.slice = slice_index;
    out.width = width;
    out.height = height;
    out.stride = layout.width;

    if (width == 0 || height == 0) {
        return true;
    }

    const size_t base_offset = slice_index * slice_size;
    if (base_offset >= entry.n_elements) {
        return true;
    }

    const size_t slice_offset = base_offset;
    const size_t slice_available = entry.n_elements - slice_offset;
    if (slice_available > 0) {
        const size_t slice_count = std::min(slice_size, slice_available);
        if (slice_count > 0) {
            tensor_window_result slice_window;
            if (!tensor_window_values(state, entry, slice_offset, slice_count, slice_window, error)) {
                return false;
            }
            if (slice_window.count > 0) {
                out.slice_min = slice_window.min;
                out.slice_max = slice_window.max;
                out.slice_valid = slice_window.count;
            }
        }
    }

    const size_t start_offset = base_offset + y * layout.width + x;
    if (start_offset >= entry.n_elements) {
        return true;
    }

    const size_t max_available = entry.n_elements - start_offset;
    if (max_available == 0) {
        return true;
    }

    const size_t row_stride = layout.width;
    const size_t fetch_span = height == 0 ? 0 : (height - 1) * row_stride + width;
    const size_t fetch_count = std::min(max_available, fetch_span);

    if (fetch_count == 0) {
        return true;
    }

    tensor_window_result window;
    if (!tensor_window_values(state, entry, start_offset, fetch_count, window, error)) {
        return false;
    }

    out.offset = start_offset;
    out.values.assign(width * height, 0.0f);
    out.mask.assign(width * height, 0);
    out.valid = 0;

    if (window.count == 0 || window.values.empty()) {
        return true;
    }

    size_t src_index = 0;
    const size_t values_size = window.values.size();

    for (size_t row = 0; row < height && src_index < values_size; ++row) {
        const size_t dest_row_start = row * width;
        const size_t available_in_row = row_stride - x;
        const size_t remaining_src = values_size - src_index;
        const size_t take = std::min({width, available_in_row, remaining_src});

        if (take == 0) {
            break;
        }

        for (size_t col = 0; col < take; ++col) {
            const float value = window.values[src_index + col];
            const size_t dest_index = dest_row_start + col;
            out.values[dest_index] = value;
            out.mask[dest_index] = 1;
            if (out.valid == 0) {
                out.min = value;
                out.max = value;
            } else {
                out.min = std::min(out.min, value);
                out.max = std::max(out.max, value);
            }
            ++out.valid;
        }

        src_index += take;

        if (row + 1 >= height || src_index >= values_size) {
            break;
        }

        const size_t skip = row_stride - take;
        if (skip == 0) {
            continue;
        }
        if (skip > values_size - src_index) {
            break;
        }
        src_index += skip;
    }

    return true;
}

void build_tokenizer_info(viewer_state & state) {
    tokenizer_info info;
    info.key_tokens = gguf_find_key(state.gguf.get(), "tokenizer.ggml.tokens");
    if (info.key_tokens >= 0) {
        info.total_tokens = gguf_get_arr_n(state.gguf.get(), info.key_tokens);
    }
    info.key_scores = gguf_find_key(state.gguf.get(), "tokenizer.ggml.scores");
    info.key_types  = gguf_find_key(state.gguf.get(), "tokenizer.ggml.token_type");
    state.tokenizer = std::move(info);
}

bool ensure_count_in_range(size_t & offset, size_t & limit, size_t total) {
    if (offset > total) {
        offset = total;
    }
    if (limit == 0) {
        limit = 1;
    }
    if (offset + limit > total) {
        limit = total - offset;
    }
    return true;
}

void setup_routes(httplib::Server & server, std::shared_ptr<viewer_state> state) {
    server.Get("/", [state](const httplib::Request &, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char *>(index_html), index_html_len, "text/html; charset=utf-8");
    });

    server.Get("/api/info", [state](const httplib::Request &, httplib::Response & res) {
        json body;
        body["modelPath"] = state->model_path;
        body["fileSize"] = state->file_size;
        body["nKv"] = gguf_get_n_kv(state->gguf.get());
        body["nTensors"] = gguf_get_n_tensors(state->gguf.get());
        body["ggufVersion"] = gguf_get_version(state->gguf.get());
        body["alignment"] = state->alignment;
        body["dataOffset"] = state->data_offset;
        body["tokenizer"] = {
            {"hasTokens", state->tokenizer.key_tokens >= 0},
            {"totalTokens", state->tokenizer.total_tokens}
        };
        set_json_response(res, body);
    });

    server.Get("/api/kv", [state](const httplib::Request & req, httplib::Response & res) {
        size_t limit = 8;
        if (auto value = req.get_param_value("preview"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                limit = *parsed;
            }
        }

        json kvs = json::array();
        const int64_t total = gguf_get_n_kv(state->gguf.get());
        for (int64_t i = 0; i < total; ++i) {
            kvs.push_back(describe_kv(state->gguf.get(), i, limit));
        }
        set_json_response(res, kvs);
    });

    server.Get(R"(/api/tensors$)", [state](const httplib::Request &, httplib::Response & res) {
        json tensors_json = json::array();
        for (const auto & entry : state->tensors) {
            tensors_json.push_back(tensor_to_json(entry, state->data_offset));
        }
        set_json_response(res, tensors_json);
    });

    server.Get(R"(/api/tensors/(.+)/raw)", [state](const httplib::Request & req, httplib::Response & res) {
        const std::string name = url_decode(req.matches[1]);
        auto it = state->tensor_index_by_name.find(name);
        if (it == state->tensor_index_by_name.end()) {
            set_json_response(res, make_error("tensor not found"), 404);
            return;
        }
        const tensor_entry & entry = state->tensors[it->second];

        size_t x = 0;
        size_t y = 0;
        size_t width = 1024;
        size_t height = 1024;
        size_t slice = 0;

        if (auto value = req.get_param_value("x"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                x = *parsed;
            }
        }
        if (auto value = req.get_param_value("y"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                y = *parsed;
            }
        }
        if (auto value = req.get_param_value("width"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                width = *parsed;
            }
        }
        if (auto value = req.get_param_value("height"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                height = *parsed;
            }
        }
        if (auto value = req.get_param_value("slice"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                slice = *parsed;
            }
        }

        if (width == 0) {
            width = 1;
        }
        if (height == 0) {
            height = 1;
        }

        tensor_tile_result tile;
        std::string error;
        if (!tensor_tile_values(*state, entry, slice, x, y, width, height, tile, error)) {
            set_json_response(res, make_error(error.empty() ? "failed to read tensor" : error), 500);
            return;
        }

        json body;
        body["name"] = entry.name;
        body["type"] = ggml_type_name(entry.tensor->type);
        body["shape"] = entry.shape;
        body["total"] = entry.n_elements;
        body["layout"] = {
            {"width", entry.layout.width},
            {"height", entry.layout.height},
            {"depth", entry.layout.depth},
        };
        body["origin"] = {
            {"x", tile.x},
            {"y", tile.y},
            {"slice", tile.slice},
        };
        body["viewport"] = {
            {"width", tile.width},
            {"height", tile.height},
        };
        body["offset"] = tile.offset;
        body["count"] = tile.width * tile.height;

        if (tile.slice_valid > 0) {
            body["sliceMin"] = tile.slice_min;
            body["sliceMax"] = tile.slice_max;
        } else {
            body["sliceMin"] = nullptr;
            body["sliceMax"] = nullptr;
        }

        if (tile.valid > 0) {
            body["min"] = tile.min;
            body["max"] = tile.max;
        } else {
            body["min"] = nullptr;
            body["max"] = nullptr;
        }

        json values = json::array();
        for (size_t i = 0; i < tile.values.size(); ++i) {
            if (i < tile.mask.size() && tile.mask[i]) {
                values.push_back(tile.values[i]);
            } else {
                values.push_back(nullptr);
            }
        }
        body["values"] = std::move(values);
        body["validCount"] = tile.valid;
        set_json_response(res, body);
    });

    server.Get(R"(/api/tokenizer$)", [state](const httplib::Request & req, httplib::Response & res) {
        json body;
        if (state->tokenizer.key_tokens < 0) {
            body["hasTokenizer"] = false;
            set_json_response(res, body);
            return;
        }

        size_t offset = 0;
        size_t limit = 100;
        if (auto value = req.get_param_value("offset"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                offset = *parsed;
            }
        }
        if (auto value = req.get_param_value("limit"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                limit = *parsed;
            }
        }

        const size_t total = static_cast<size_t>(state->tokenizer.total_tokens);
        ensure_count_in_range(offset, limit, total);

        body["hasTokenizer"] = true;
        body["total"] = total;
        body["offset"] = offset;
        body["limit"] = limit;
        json items = json::array();
        const size_t to = offset + limit;
        const float * scores_ptr = nullptr;
        const int32_t * types_ptr = nullptr;
        if (state->tokenizer.key_scores >= 0) {
            scores_ptr = static_cast<const float *>(gguf_get_arr_data(state->gguf.get(), state->tokenizer.key_scores));
        }
        if (state->tokenizer.key_types >= 0) {
            types_ptr = static_cast<const int32_t *>(gguf_get_arr_data(state->gguf.get(), state->tokenizer.key_types));
        }

        for (size_t i = offset; i < to; ++i) {
            json item;
            item["index"] = i;
            item["token"] = gguf_get_arr_str(state->gguf.get(), state->tokenizer.key_tokens, i);
            if (scores_ptr) {
                item["score"] = scores_ptr[i];
            }
            if (types_ptr) {
                item["tokenType"] = types_ptr[i];
            }
            items.push_back(item);
        }
        body["items"] = std::move(items);
        set_json_response(res, body);
    });
}

viewer_state load_state(const std::string & model_path) {
    viewer_state state;
    state.model_path = model_path;
    state.file_size = static_cast<size_t>(fs::file_size(model_path));

    struct ggml_context * tensor_ctx_raw = nullptr;
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &tensor_ctx_raw;

    state.gguf.reset(gguf_init_from_file(model_path.c_str(), params));
    if (!state.gguf) {
        throw std::runtime_error("failed to load GGUF metadata");
    }

    state.tensor_ctx.reset(tensor_ctx_raw);
    state.data_offset = gguf_get_data_offset(state.gguf.get());
    state.alignment   = gguf_get_alignment(state.gguf.get());

    // iterate tensors
    for (ggml_tensor * cur = ggml_get_first_tensor(state.tensor_ctx.get()); cur != nullptr; cur = ggml_get_next_tensor(state.tensor_ctx.get(), cur)) {
        tensor_entry entry;
        entry.tensor = cur;
        entry.name = ggml_get_name(cur);
        entry.n_elements = ggml_nelements(cur);
        entry.n_bytes = ggml_nbytes(cur);
        entry.shape = tensor_shape(cur);
        entry.layout = compute_tensor_layout(entry.shape, entry.n_elements);
        entry.tensor_index = gguf_find_tensor(state.gguf.get(), entry.name.c_str());
        if (entry.tensor_index >= 0) {
            entry.offset = gguf_get_tensor_offset(state.gguf.get(), entry.tensor_index);
        } else {
            entry.offset = 0;
        }
        state.tensor_index_by_name.emplace(entry.name, state.tensors.size());
        state.tensors.push_back(std::move(entry));
    }

    build_tokenizer_info(state);
    return state;
}

void print_usage(const char * argv0) {
    fprintf(stderr, "Usage: %s --model <path> [--host 127.0.0.1] [--port 8080]\n", argv0);
}

struct cli_params {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
};

enum class cli_status {
    ok,
    help,
    error,
};

cli_status parse_cli(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--model" && i + 1 < argc) {
            params.model = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            params.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            auto parsed = parse_i64(argv[++i]);
            if (!parsed) {
                fprintf(stderr, "Invalid value for --port\n");
                return cli_status::error;
            }
            params.port = static_cast<int>(*parsed);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return cli_status::help;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return cli_status::error;
        }
    }

    if (params.model.empty()) {
        fprintf(stderr, "Missing required --model argument\n");
        print_usage(argv[0]);
        return cli_status::error;
    }

    if (params.port <= 0 || params.port > 65535) {
        fprintf(stderr, "Port must be between 1 and 65535\n");
        return cli_status::error;
    }

    return cli_status::ok;
}

} // namespace

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(0);

    cli_params params;
    switch (parse_cli(argc, argv, params)) {
        case cli_status::ok:
            break;
        case cli_status::help:
            return 0;
        case cli_status::error:
        default:
            return 1;
    }

    try {
        auto state_ptr = std::make_shared<viewer_state>(load_state(params.model));
        LOG_INF("Loaded %s with %zu tensors and %lld key/value pairs\n", params.model.c_str(), state_ptr->tensors.size(), (long long)gguf_get_n_kv(state_ptr->gguf.get()));

        httplib::Server server;
        server.set_default_headers({{"Access-Control-Allow-Origin", "*"}});
        setup_routes(server, state_ptr);

        LOG_INF("Serving llama-gguf-viewer at http://%s:%d\n", params.host.c_str(), params.port);
        server.listen(params.host.c_str(), params.port);
    } catch (const std::exception & ex) {
        LOG_ERR("Fatal error: %s\n", ex.what());
        return 1;
    }

    return 0;
}
