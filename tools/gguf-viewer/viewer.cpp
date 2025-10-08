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
#include <memory>
#include <mutex>
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
    size_t                   state_index = 0;
    size_t                   offset = 0;
    size_t                   n_elements = 0;
    size_t                   n_bytes = 0;
    std::vector<int64_t>     shape;
    tensor_layout            layout;
};

struct tensor_slice_stats {
    bool   computed = false;
    size_t valid    = 0;
    float  min      = 0.0f;
    float  max      = 0.0f;
};

struct tokenizer_info {
    int64_t total_tokens = 0;
    int64_t key_tokens   = -1;
    int64_t key_scores   = -1;
    int64_t key_types    = -1;
};

struct viewer_state {
    std::string                                             model_path;
    std::string                                             relative_path;
    size_t                                                  file_size = 0;
    gguf_context_ptr                                        gguf;
    ggml_context_ptr                                        tensor_ctx;
    size_t                                                  data_offset = 0;
    size_t                                                  alignment   = 0;
    std::vector<tensor_entry>                               tensors;
    std::unordered_map<std::string, size_t>                 tensor_index_by_name;
    tokenizer_info                                          tokenizer;
    mutable std::mutex                                      slice_cache_mutex;
    mutable std::vector<std::vector<tensor_slice_stats>>    slice_stats_cache;
};

struct server_state {
    fs::path                                                       root;
    std::unordered_map<std::string, std::shared_ptr<viewer_state>> viewers;
    std::mutex                                                     mutex;
};

struct model_descriptor {
    std::string relative;
    std::string name;
    uintmax_t   size = 0;
};

bool has_gguf_extension(const fs::path & path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".gguf";
}

std::optional<fs::path> safe_relative(const fs::path & target, const fs::path & base) {
    std::error_code ec;
    fs::path relative = fs::relative(target, base, ec);
    if (ec) {
        return std::nullopt;
    }
    for (const auto & part : relative) {
        if (part == "..") {
            return std::nullopt;
        }
    }
    return relative;
}

std::vector<model_descriptor> scan_models(const fs::path & root) {
    std::vector<model_descriptor> models;
    std::error_code ec;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        return models;
    }
    for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            break;
        }
        const auto & entry = *it;
        if (!entry.is_regular_file(ec)) {
            if (ec) {
                ec.clear();
            }
            continue;
        }
        if (!has_gguf_extension(entry.path())) {
            continue;
        }
        auto rel = safe_relative(entry.path(), root);
        if (!rel) {
            continue;
        }
        model_descriptor desc;
        desc.relative = rel->generic_string();
        desc.name = entry.path().filename().string();
        desc.size = entry.file_size(ec);
        if (ec) {
            desc.size = 0;
            ec.clear();
        }
        models.push_back(std::move(desc));
    }
    std::sort(models.begin(), models.end(), [](const model_descriptor & a, const model_descriptor & b) {
        return a.relative < b.relative;
    });
    return models;
}

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

std::shared_ptr<viewer_state> load_state(const std::string & model_path);

json make_error(const std::string & message) {
    json body;
    body["error"] = message;
    return body;
}

std::optional<std::string> get_model_parameter(const httplib::Request & req) {
    if (auto value = req.get_param_value("model"); !value.empty()) {
        return value;
    }
    return std::nullopt;
}

std::optional<std::string> normalize_model_key(const fs::path & root, const std::string & value) {
    if (value.empty()) {
        return std::nullopt;
    }

    std::error_code ec;
    fs::path candidate(value);
    fs::path resolved = candidate.is_absolute()
        ? fs::weakly_canonical(candidate, ec)
        : fs::weakly_canonical(root / candidate, ec);
    if (ec) {
        return std::nullopt;
    }

    auto relative = safe_relative(resolved, root);
    if (!relative) {
        return std::nullopt;
    }

    return relative->generic_string();
}

template <typename Fn>
bool with_viewer_state(const std::shared_ptr<server_state> & state, const httplib::Request & req, httplib::Response & res, Fn && fn) {
    auto param = get_model_parameter(req);
    if (!param) {
        set_json_response(res, make_error("missing model parameter"), 400);
        return false;
    }

    const std::string decoded = url_decode(*param);
    std::vector<std::string> candidate_keys;
    if (auto normalized = normalize_model_key(state->root, decoded)) {
        candidate_keys.push_back(*normalized);
    }
    candidate_keys.push_back(decoded);

    std::shared_ptr<viewer_state> viewer;
    std::string relative;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        for (const auto & key : candidate_keys) {
            auto it = state->viewers.find(key);
            if (it != state->viewers.end()) {
                viewer = it->second;
                if (viewer) {
                    relative = viewer->relative_path.empty() ? key : viewer->relative_path;
                }
                break;
            }
        }
    }

    if (!viewer) {
        set_json_response(res, make_error("model not loaded"), 409);
        return false;
    }

    fn(viewer, relative);
    return true;
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
        layout.width = static_cast<size_t>(std::max<int64_t>(1, shape.front()));
        layout.height = static_cast<size_t>(std::max<int64_t>(1, shape.back()));
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
    size_t block_size = 1;
    if (const auto * traits = ggml_get_type_traits(entry.tensor->type)) {
        if (traits->blck_size > 0) {
            block_size = traits->blck_size;
        }
    }
    node["blockSize"] = block_size;
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

struct tensor_histogram_result {
    size_t slice      = 0;
    uint64_t total    = 0;
    uint64_t max_bin  = 0;
    float range_min   = 0.0f;
    float range_max   = 0.0f;
    std::vector<uint64_t> bins;
};

struct tensor_value_details {
    size_t element_index      = 0;
    size_t element_count      = 0;
    size_t block_index        = 0;
    size_t index_in_block     = 0;
    size_t tensor_byte_offset = 0;
    size_t file_byte_offset   = 0;
    float  value              = 0.0f;
    bool   value_valid        = false;
};

bool tensor_element_details(
        const viewer_state & state,
        const tensor_entry & entry,
        size_t slice_index,
        size_t x,
        size_t y,
        tensor_value_details & out,
        std::string & error) {
    out = {};

    const tensor_layout & layout = entry.layout;
    if (layout.width == 0 || layout.height == 0 || layout.depth == 0) {
        error = "invalid tensor layout";
        return false;
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

    const size_t slice_size = layout.width * layout.height;
    if (slice_size == 0) {
        error = "invalid slice size";
        return false;
    }

    const size_t element_index = slice_index * slice_size + y * layout.width + x;
    if (element_index >= entry.n_elements) {
        error = "element outside tensor";
        return false;
    }

    const auto * traits = ggml_get_type_traits(entry.tensor->type);
    if (!traits) {
        error = "unknown tensor type";
        return false;
    }

    const size_t block_size = traits->blck_size > 0 ? static_cast<size_t>(traits->blck_size) : 1;
    const size_t type_size  = traits->type_size;
    if (type_size == 0) {
        error = "invalid tensor type size";
        return false;
    }

    const size_t block_index = element_index / block_size;
    const size_t index_in_block = element_index % block_size;
    const size_t block_offset_bytes = block_index * type_size;

    const size_t absolute_offset = state.data_offset + entry.offset + block_offset_bytes;
    if (absolute_offset + type_size > state.file_size) {
        error = "tensor data outside file";
        return false;
    }

    std::ifstream file(state.model_path, std::ios::binary);
    if (!file) {
        error = "failed to open model file";
        return false;
    }

    file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    std::vector<uint8_t> raw_block(type_size);
    file.read(reinterpret_cast<char *>(raw_block.data()), static_cast<std::streamsize>(type_size));
    if (file.gcount() < static_cast<std::streamsize>(type_size)) {
        error = "failed to read tensor block";
        return false;
    }

    const size_t elements_in_block = std::min(block_size, entry.n_elements - block_index * block_size);
    std::vector<float> converted(block_size, 0.0f);

    if (entry.tensor->type == GGML_TYPE_F32) {
        const auto * src = reinterpret_cast<const float *>(raw_block.data());
        std::copy(src, src + block_size, converted.begin());
    } else if (entry.tensor->type == GGML_TYPE_F16) {
        const auto * src = reinterpret_cast<const ggml_fp16_t *>(raw_block.data());
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (entry.tensor->type == GGML_TYPE_BF16) {
        const auto * src = reinterpret_cast<const ggml_bf16_t *>(raw_block.data());
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else if (entry.tensor->type == GGML_TYPE_I8) {
        const auto * src = reinterpret_cast<const int8_t *>(raw_block.data());
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = static_cast<float>(src[i]);
        }
    } else if (entry.tensor->type == GGML_TYPE_I16) {
        const auto * src = reinterpret_cast<const int16_t *>(raw_block.data());
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = static_cast<float>(src[i]);
        }
    } else if (entry.tensor->type == GGML_TYPE_I32) {
        const auto * src = reinterpret_cast<const int32_t *>(raw_block.data());
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = static_cast<float>(src[i]);
        }
    } else if (entry.tensor->type == GGML_TYPE_I64) {
        const auto * src = reinterpret_cast<const int64_t *>(raw_block.data());
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = static_cast<float>(src[i]);
        }
    } else if (traits->to_float) {
        traits->to_float(raw_block.data(), converted.data(), static_cast<int64_t>(block_size));
    } else {
        for (size_t i = 0; i < block_size; ++i) {
            converted[i] = static_cast<float>(raw_block[i % raw_block.size()]);
        }
    }

    const float value = converted[index_in_block];

    out.element_index = element_index;
    out.element_count = entry.n_elements;
    out.block_index = block_index;
    out.index_in_block = index_in_block;
    out.tensor_byte_offset = entry.offset + block_offset_bytes;
    out.file_byte_offset = absolute_offset;
    out.value = value;
    out.value_valid = index_in_block < elements_in_block;

    return true;
}

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

bool tensor_slice_statistics(
        const viewer_state & state,
        const tensor_entry & entry,
        size_t slice_index,
        tensor_slice_stats & out,
        std::string & error) {
    out = {};

    const tensor_layout & layout = entry.layout;
    if (layout.depth == 0) {
        return true;
    }

    if (slice_index >= layout.depth) {
        slice_index = layout.depth - 1;
    }

    {
        std::lock_guard<std::mutex> lock(state.slice_cache_mutex);
        if (entry.state_index < state.slice_stats_cache.size()) {
            const auto & tensor_cache = state.slice_stats_cache[entry.state_index];
            if (slice_index < tensor_cache.size() && tensor_cache[slice_index].computed) {
                out = tensor_cache[slice_index];
                return true;
            }
        }
    }

    const size_t slice_size = layout.width * layout.height;
    if (slice_size == 0) {
        return true;
    }

    const size_t base_offset = slice_index * slice_size;
    if (base_offset >= entry.n_elements) {
        return true;
    }

    const size_t slice_available = entry.n_elements - base_offset;
    if (slice_available == 0) {
        return true;
    }

    const size_t slice_count = std::min(slice_size, slice_available);
    if (slice_count == 0) {
        return true;
    }

    tensor_window_result slice_window;
    if (!tensor_window_values(state, entry, base_offset, slice_count, slice_window, error)) {
        return false;
    }

    tensor_slice_stats computed;
    computed.computed = true;
    computed.valid = slice_window.count;
    if (slice_window.count > 0) {
        computed.min = slice_window.min;
        computed.max = slice_window.max;
    }

    {
        std::lock_guard<std::mutex> lock(state.slice_cache_mutex);
        if (state.slice_stats_cache.size() < state.tensors.size()) {
            state.slice_stats_cache.resize(state.tensors.size());
        }
        if (entry.state_index >= state.slice_stats_cache.size()) {
            state.slice_stats_cache.resize(entry.state_index + 1);
        }
        auto & tensor_cache = state.slice_stats_cache[entry.state_index];
        if (tensor_cache.size() < layout.depth) {
            tensor_cache.resize(layout.depth);
        }
        tensor_cache[slice_index] = computed;
    }

    out = computed;
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

    tensor_slice_stats slice_stats;
    if (!tensor_slice_statistics(state, entry, slice_index, slice_stats, error)) {
        return false;
    }
    if (slice_stats.valid > 0) {
        out.slice_min = slice_stats.min;
        out.slice_max = slice_stats.max;
        out.slice_valid = slice_stats.valid;
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

bool tensor_slice_histogram(
        const viewer_state & state,
        const tensor_entry & entry,
        size_t slice_index,
        size_t bin_count,
        tensor_histogram_result & out,
        std::string & error) {
    out = {};

    if (bin_count == 0) {
        return true;
    }

    const tensor_layout & layout = entry.layout;
    if (layout.depth == 0) {
        return true;
    }

    const size_t slice_size = layout.width * layout.height;
    if (slice_size == 0) {
        return true;
    }

    if (slice_index >= layout.depth) {
        slice_index = layout.depth - 1;
    }

    const size_t base_offset = slice_index * slice_size;
    if (base_offset >= entry.n_elements) {
        return true;
    }

    const size_t slice_available = entry.n_elements - base_offset;
    if (slice_available == 0) {
        return true;
    }

    const size_t slice_count = std::min(slice_size, slice_available);
    if (slice_count == 0) {
        return true;
    }

    tensor_window_result window;
    if (!tensor_window_values(state, entry, base_offset, slice_count, window, error)) {
        return false;
    }

    if (window.count == 0 || window.values.empty()) {
        return true;
    }

    out.slice = slice_index;
    out.bins.assign(bin_count, 0);
    out.range_min = window.min;
    out.range_max = window.max;

    if (!std::isfinite(out.range_min) || !std::isfinite(out.range_max)) {
        return true;
    }

    if (out.range_max < out.range_min) {
        std::swap(out.range_max, out.range_min);
    }

    if (out.range_max == out.range_min) {
        if (bin_count == 0) {
            return true;
        }

        uint64_t finite_count = 0;
        for (float value : window.values) {
            if (std::isfinite(value)) {
                ++finite_count;
            }
        }

        if (finite_count == 0) {
            return true;
        }

        const size_t index = std::min(bin_count - 1, bin_count / 2);
        out.bins[index] = finite_count;
        out.max_bin = finite_count;
        out.total = finite_count;
        return true;
    }

    const float span = out.range_max - out.range_min;
    if (!std::isfinite(span) || span <= 0.0f) {
        return true;
    }

    const float bin_scale = static_cast<float>(bin_count) / span;

    for (float value : window.values) {
        if (!std::isfinite(value)) {
            continue;
        }

        const float relative = (value - out.range_min) * bin_scale;
        size_t index = relative < 0.0f ? 0 : static_cast<size_t>(relative);
        if (index >= bin_count) {
            index = bin_count - 1;
        }

        uint64_t & bin = out.bins[index];
        bin += 1;
        out.total += 1;
        if (bin > out.max_bin) {
            out.max_bin = bin;
        }
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

void setup_routes(httplib::Server & server, std::shared_ptr<server_state> state) {
    server.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char *>(index_html), index_html_len, "text/html; charset=utf-8");
    });

    server.Get("/api/models", [state](const httplib::Request & req, httplib::Response & res) {
        json body;
        body["root"] = state->root.generic_string();
        json items = json::array();
        std::string selected;
        if (auto param = get_model_parameter(req)) {
            const std::string decoded = url_decode(*param);
            if (auto normalized = normalize_model_key(state->root, decoded)) {
                selected = *normalized;
            } else {
                selected = decoded;
            }
        }
        auto models = scan_models(state->root);
        for (const auto & item : models) {
            json entry;
            entry["path"] = item.relative;
            entry["name"] = item.name;
            entry["size"] = item.size;
            if (!selected.empty() && item.relative == selected) {
                entry["selected"] = true;
            }
            items.push_back(std::move(entry));
        }
        body["items"] = std::move(items);
        if (!selected.empty()) {
            body["selected"] = selected;
        }
        set_json_response(res, body);
    });

    server.Post("/api/models/select", [state](const httplib::Request & req, httplib::Response & res) {
        auto param = get_model_parameter(req);
        if (!param) {
            set_json_response(res, make_error("missing model parameter"), 400);
            return;
        }

        const std::string decoded = url_decode(*param);
        fs::path candidate(decoded);
        std::error_code ec;
        fs::path resolved = candidate.is_absolute()
            ? fs::weakly_canonical(candidate, ec)
            : fs::weakly_canonical(state->root / candidate, ec);
        if (ec) {
            set_json_response(res, make_error("failed to resolve model path"), 400);
            return;
        }

        auto relative = safe_relative(resolved, state->root);
        if (!relative) {
            set_json_response(res, make_error("model is outside of root"), 400);
            return;
        }

        std::error_code status_ec;
        if (!fs::exists(resolved, status_ec) || status_ec) {
            set_json_response(res, make_error("model not found"), 404);
            return;
        }
        if (!fs::is_regular_file(resolved, status_ec) || status_ec) {
            set_json_response(res, make_error("model is not a regular file"), 400);
            return;
        }
        if (!has_gguf_extension(resolved)) {
            set_json_response(res, make_error("model must have .gguf extension"), 400);
            return;
        }

        const std::string key = relative->generic_string();
        std::shared_ptr<viewer_state> viewer;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            auto it = state->viewers.find(key);
            if (it != state->viewers.end()) {
                viewer = it->second;
            }
        }

        if (!viewer) {
            std::shared_ptr<viewer_state> loaded;
            try {
                loaded = load_state(resolved.string());
            } catch (const std::exception & ex) {
                set_json_response(res, make_error(ex.what()), 500);
                return;
            }

            loaded->relative_path = key;

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                auto & slot = state->viewers[key];
                if (!slot) {
                    slot = loaded;
                }
                viewer = slot;
            }

            LOG_INF("Loaded %s with %zu tensors and %lld key/value pairs\n",
                resolved.string().c_str(),
                viewer->tensors.size(),
                (long long)gguf_get_n_kv(viewer->gguf.get()));
        }

        json body;
        body["selected"] = key;
        set_json_response(res, body);
    });

    server.Get("/api/info", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string & relative) {
            json body;
            body["modelPath"] = viewer->model_path;
            body["relativePath"] = relative;
            body["fileSize"] = viewer->file_size;
            body["nKv"] = gguf_get_n_kv(viewer->gguf.get());
            body["nTensors"] = gguf_get_n_tensors(viewer->gguf.get());
            body["ggufVersion"] = gguf_get_version(viewer->gguf.get());
            body["alignment"] = viewer->alignment;
            body["dataOffset"] = viewer->data_offset;
            body["tokenizer"] = {
                {"hasTokens", viewer->tokenizer.key_tokens >= 0},
                {"totalTokens", viewer->tokenizer.total_tokens}
            };
            set_json_response(res, body);
        });
    });

    server.Get("/api/kv", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string &) {
            size_t limit = 8;
            if (auto value = req.get_param_value("preview"); !value.empty()) {
                if (auto parsed = parse_size_t(value)) {
                    limit = *parsed;
                }
            }

            json kvs = json::array();
            const int64_t total = gguf_get_n_kv(viewer->gguf.get());
            for (int64_t i = 0; i < total; ++i) {
                kvs.push_back(describe_kv(viewer->gguf.get(), i, limit));
            }
            set_json_response(res, kvs);
        });
    });

    server.Get(R"(/api/tensors$)", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string &) {
            json tensors_json = json::array();
            for (const auto & entry : viewer->tensors) {
                tensors_json.push_back(tensor_to_json(entry, viewer->data_offset));
            }
            set_json_response(res, tensors_json);
        });
    });

    server.Get(R"(/api/tensors/(.+)/raw)", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string &) {
            const std::string name = url_decode(req.matches[1]);
            auto it = viewer->tensor_index_by_name.find(name);
            if (it == viewer->tensor_index_by_name.end()) {
                set_json_response(res, make_error("tensor not found"), 404);
                return;
            }
            const tensor_entry & entry = viewer->tensors[it->second];

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

            const tensor_layout layout = compute_tensor_layout(entry.shape, entry.n_elements);
            if (x >= layout.width || y >= layout.height) {
                set_json_response(res, make_error("window outside of tensor"), 400);
                return;
            }
            if (width == 0 || height == 0) {
                set_json_response(res, make_error("invalid viewport size"), 400);
                return;
            }

            width = std::min(width, layout.width - x);
            height = std::min(height, layout.height - y);
            if (slice >= layout.depth) {
                slice = layout.depth - 1;
            }

            std::string error;
            tensor_tile_result tile;
            if (!tensor_tile_values(*viewer, entry, slice, x, y, width, height, tile, error)) {
                set_json_response(res, make_error(error.empty() ? "failed to read tensor window" : error), 500);
                return;
            }

            json origin = {
                {"x", tile.x},
                {"y", tile.y},
            };
            if (layout.depth > 1) {
                origin["slice"] = tile.slice;
            }

            json viewport = {
                {"width", tile.width},
                {"height", tile.height},
            };

            json values = json::array();
            for (size_t i = 0; i < tile.values.size(); ++i) {
                if (i < tile.mask.size() && tile.mask[i]) {
                    values.push_back(tile.values[i]);
                } else {
                    values.push_back(nullptr);
                }
            }

            json body;
            body["layout"] = {
                {"width", layout.width},
                {"height", layout.height},
                {"depth", layout.depth},
            };
            body["origin"] = std::move(origin);
            body["viewport"] = std::move(viewport);
            body["values"] = std::move(values);
            body["offset"] = entry.offset + viewer->data_offset;
            if (tile.valid > 0) {
                body["min"] = tile.min;
                body["max"] = tile.max;
            } else {
                body["min"] = nullptr;
                body["max"] = nullptr;
            }
            if (layout.depth > 1) {
                if (tile.slice_valid > 0) {
                    body["sliceMin"] = tile.slice_min;
                    body["sliceMax"] = tile.slice_max;
                } else {
                    body["sliceMin"] = nullptr;
                    body["sliceMax"] = nullptr;
                }
            }

            set_json_response(res, body);
        });
    });

    server.Get(R"(/api/tensors/(.+)/value)", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string &) {
            const std::string name = url_decode(req.matches[1]);
            auto it = viewer->tensor_index_by_name.find(name);
            if (it == viewer->tensor_index_by_name.end()) {
                set_json_response(res, make_error("tensor not found"), 404);
                return;
            }

            const tensor_entry & entry = viewer->tensors[it->second];

            size_t x = 0;
            size_t y = 0;
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
            if (auto value = req.get_param_value("slice"); !value.empty()) {
                if (auto parsed = parse_size_t(value)) {
                    slice = *parsed;
                }
            }

            tensor_value_details details;
            std::string error;
            if (!tensor_element_details(*viewer, entry, slice, x, y, details, error)) {
                if (error.empty()) {
                    error = "failed to resolve tensor element";
                }
                set_json_response(res, make_error(error), 400);
                return;
            }

            const auto * traits = ggml_get_type_traits(entry.tensor->type);
            const size_t block_size = traits && traits->blck_size > 0 ? static_cast<size_t>(traits->blck_size) : 1;

            json body;
            body["tensor"] = entry.name;
            body["type"] = ggml_type_name(entry.tensor->type);
            body["coordinate"] = {
                {"x", x},
                {"y", y},
                {"slice", slice},
            };
            body["index"] = details.element_index;
            body["count"] = details.element_count;
            body["block"] = {
                {"index", details.block_index},
                {"offset", details.index_in_block},
                {"size", block_size},
            };
            body["tensorOffset"] = details.tensor_byte_offset;
            body["fileOffset"] = details.file_byte_offset;
            if (details.value_valid) {
                body["value"] = details.value;
            } else {
                body["value"] = nullptr;
            }
            set_json_response(res, body);
        });
    });

    server.Get(R"(/api/tensors/(.+)/histogram)", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string &) {
            const std::string name = url_decode(req.matches[1]);
            auto it = viewer->tensor_index_by_name.find(name);
            if (it == viewer->tensor_index_by_name.end()) {
                set_json_response(res, make_error("tensor not found"), 404);
                return;
            }

            const tensor_entry & entry = viewer->tensors[it->second];

            size_t width = 1024;
            size_t height = 512;
            size_t slice = 0;

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
                set_json_response(res, make_error("width must be greater than zero"), 400);
                return;
            }

            if (height == 0) {
                height = 1;
            }

            tensor_histogram_result histogram;
            std::string error;
            if (!tensor_slice_histogram(*viewer, entry, slice, width, histogram, error)) {
                set_json_response(res, make_error(error.empty() ? "failed to compute histogram" : error), 500);
                return;
            }

            json bins = json::array();
            for (uint64_t count : histogram.bins) {
                bins.push_back(count);
            }

            json body;
            body["width"] = width;
            body["height"] = height;
            body["slice"] = histogram.slice;
            body["range"] = { {"min", histogram.range_min}, {"max", histogram.range_max} };
            body["maxCount"] = histogram.max_bin;
            body["total"] = histogram.total;
            body["min"] = histogram.range_min;
            body["max"] = histogram.range_max;
            body["bins"] = std::move(bins);

            set_json_response(res, body);
        });
    });

    server.Get(R"(/api/tokenizer$)", [state](const httplib::Request & req, httplib::Response & res) {
        with_viewer_state(state, req, res, [&](const std::shared_ptr<viewer_state> & viewer, const std::string &) {
            json body;
            if (viewer->tokenizer.key_tokens < 0) {
                body["hasTokenizer"] = false;
                body["total"] = 0;
                body["offset"] = 0;
                body["limit"] = 0;
                body["items"] = json::array();
                set_json_response(res, body);
                return;
            }

            size_t offset = 0;
            size_t limit = 256;
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

            const size_t total = static_cast<size_t>(viewer->tokenizer.total_tokens);
            ensure_count_in_range(offset, limit, total);

            body["hasTokenizer"] = true;
            body["total"] = total;
            body["offset"] = offset;
            body["limit"] = limit;
            json items = json::array();
            const size_t to = offset + limit;
            const float * scores_ptr = nullptr;
            const int32_t * types_ptr = nullptr;
            if (viewer->tokenizer.key_scores >= 0) {
                scores_ptr = static_cast<const float *>(gguf_get_arr_data(viewer->gguf.get(), viewer->tokenizer.key_scores));
            }
            if (viewer->tokenizer.key_types >= 0) {
                types_ptr = static_cast<const int32_t *>(gguf_get_arr_data(viewer->gguf.get(), viewer->tokenizer.key_types));
            }

            for (size_t i = offset; i < to; ++i) {
                json item;
                item["index"] = i;
                item["token"] = gguf_get_arr_str(viewer->gguf.get(), viewer->tokenizer.key_tokens, i);
                if (scores_ptr) {
                    item["score"] = scores_ptr[i];
                }
                if (types_ptr) {
                    item["tokenType"] = types_ptr[i];
                }
                items.push_back(std::move(item));
            }
            body["items"] = std::move(items);
            set_json_response(res, body);
        });
    });
}

std::shared_ptr<viewer_state> load_state(const std::string & model_path) {
    auto state = std::make_shared<viewer_state>();
    state->model_path = model_path;
    state->file_size = static_cast<size_t>(fs::file_size(model_path));

    struct ggml_context * tensor_ctx_raw = nullptr;
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &tensor_ctx_raw;

    state->gguf.reset(gguf_init_from_file(model_path.c_str(), params));
    if (!state->gguf) {
        throw std::runtime_error("failed to load GGUF metadata");
    }

    state->tensor_ctx.reset(tensor_ctx_raw);
    state->data_offset = gguf_get_data_offset(state->gguf.get());
    state->alignment   = gguf_get_alignment(state->gguf.get());

    // iterate tensors
    for (ggml_tensor * cur = ggml_get_first_tensor(state->tensor_ctx.get()); cur != nullptr; cur = ggml_get_next_tensor(state->tensor_ctx.get(), cur)) {
        tensor_entry entry;
        entry.tensor = cur;
        entry.name = ggml_get_name(cur);
        entry.n_elements = ggml_nelements(cur);
        entry.n_bytes = ggml_nbytes(cur);
        entry.shape = tensor_shape(cur);
        entry.layout = compute_tensor_layout(entry.shape, entry.n_elements);
        entry.tensor_index = gguf_find_tensor(state->gguf.get(), entry.name.c_str());
        if (entry.tensor_index >= 0) {
            entry.offset = gguf_get_tensor_offset(state->gguf.get(), entry.tensor_index);
        } else {
            entry.offset = 0;
        }
        entry.state_index = state->tensors.size();
        state->tensor_index_by_name.emplace(entry.name, entry.state_index);
        state->slice_stats_cache.emplace_back(entry.layout.depth);
        state->tensors.push_back(std::move(entry));
    }

    build_tokenizer_info(*state);
    return state;
}

void print_usage(const char * argv0) {
    fprintf(stderr, "Usage: %s --root <path> [--host 127.0.0.1] [--port 8080]\n", argv0);
}

struct cli_params {
    std::string root;
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
        if (arg == "--root" && i + 1 < argc) {
            params.root = argv[++i];
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

    if (params.root.empty()) {
        fprintf(stderr, "Missing required --root argument\n");
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
        std::error_code ec;
        fs::path root = fs::weakly_canonical(fs::path(params.root), ec);
        if (ec) {
            throw std::runtime_error("failed to resolve root directory");
        }
        if (!fs::exists(root, ec) || ec) {
            throw std::runtime_error("root directory not found");
        }
        if (!fs::is_directory(root, ec) || ec) {
            throw std::runtime_error("root path is not a directory");
        }

        auto app_state = std::make_shared<server_state>();
        app_state->root = std::move(root);

        LOG_INF("Root directory: %s\n", app_state->root.string().c_str());

        httplib::Server server;
        server.set_default_headers({{"Access-Control-Allow-Origin", "*"}});
        setup_routes(server, app_state);

        LOG_INF("Serving llama-gguf-viewer at http://%s:%d\n", params.host.c_str(), params.port);
        server.listen(params.host.c_str(), params.port);
    } catch (const std::exception & ex) {
        LOG_ERR("Fatal error: %s\n", ex.what());
        return 1;
    }

    return 0;
}
