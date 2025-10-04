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

struct tensor_entry {
    std::string              name;
    ggml_tensor *            tensor = nullptr;
    int64_t                  tensor_index = -1;
    size_t                   offset = 0;
    size_t                   n_elements = 0;
    size_t                   n_bytes = 0;
    std::vector<int64_t>     shape;
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
    return node;
}

std::vector<float> tensor_preview_values(const viewer_state & state, const tensor_entry & entry, size_t count, std::string & error) {
    std::vector<float> values;
    if (count == 0) {
        return values;
    }

    const ggml_tensor * tensor = entry.tensor;
    const size_t total = entry.n_elements;
    if (total == 0) {
        return values;
    }

    const size_t take = std::min(count, total);

    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits) {
        error = "Unknown tensor type";
        return {};
    }

    const size_t block_size = traits->blck_size > 0 ? traits->blck_size : 1;
    const size_t blocks = (take + block_size - 1) / block_size;
    const size_t elements_to_convert = blocks * block_size;
    const size_t bytes_to_read = blocks * traits->type_size;

    const size_t tensor_bytes = entry.n_bytes;
    const size_t available_bytes = std::min(bytes_to_read, tensor_bytes);

    std::ifstream file(state.model_path, std::ios::binary);
    if (!file) {
        error = "Failed to open model file";
        return {};
    }

    const size_t absolute_offset = state.data_offset + entry.offset;
    if (absolute_offset + available_bytes > state.file_size) {
        error = "Tensor offset outside file bounds";
        return {};
    }

    file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    std::vector<uint8_t> raw(bytes_to_read, 0);
    file.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(available_bytes));
    if (file.gcount() < static_cast<std::streamsize>(available_bytes)) {
        error = "Failed to read tensor data";
        return {};
    }

    values.resize(elements_to_convert, 0.0f);

    if (tensor->type == GGML_TYPE_F32) {
        const float * src = reinterpret_cast<const float *>(raw.data());
        std::memcpy(values.data(), src, take * sizeof(float));
        values.resize(take);
        return values;
    }

    if (traits->to_float == nullptr) {
        // fallback: interpret raw bytes as signed values scaled to float
        values.resize(take);
        for (size_t i = 0; i < take; ++i) {
            values[i] = static_cast<float>(raw[i]);
        }
        values.resize(take);
        return values;
    }

    traits->to_float(raw.data(), values.data(), static_cast<int64_t>(elements_to_convert));
    values.resize(take);
    return values;
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

    server.Get(R"(/api/tensors/(.+)/preview)", [state](const httplib::Request & req, httplib::Response & res) {
        const std::string name = url_decode(req.matches[1]);
        auto it = state->tensor_index_by_name.find(name);
        if (it == state->tensor_index_by_name.end()) {
            set_json_response(res, make_error("tensor not found"), 404);
            return;
        }
        const tensor_entry & entry = state->tensors[it->second];

        size_t count = 32;
        if (auto value = req.get_param_value("count"); !value.empty()) {
            if (auto parsed = parse_size_t(value)) {
                count = *parsed;
            }
        }

        std::string error;
        std::vector<float> preview = tensor_preview_values(*state, entry, count, error);
        if (!error.empty()) {
            set_json_response(res, make_error(error), 500);
            return;
        }

        json body;
        body["name"] = entry.name;
        body["count"] = preview.size();
        body["total"] = entry.n_elements;
        body["values"] = preview;
        body["type"] = ggml_type_name(entry.tensor->type);
        body["shape"] = entry.shape;
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
