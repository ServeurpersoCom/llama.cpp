#pragma once

#include "common.h"
#include "preset.h"
#include "server-common.h"
#include "server-http.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

/**
 * state diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 *  ▲            │            │               ▲
 *  └───failed───┘            │               │
 *  ▲                         └──sleeping─────┘
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    // TODO: also add downloading state when the logic is added
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_SLEEPING
};

static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") {
        return SERVER_MODEL_STATUS_UNLOADED;
    }
    if (status_str == "loading") {
        return SERVER_MODEL_STATUS_LOADING;
    }
    if (status_str == "loaded") {
        return SERVER_MODEL_STATUS_LOADED;
    }
    if (status_str == "sleeping") {
        return SERVER_MODEL_STATUS_SLEEPING;
    }
    throw std::runtime_error("invalid server model status");
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED: return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:  return "loading";
        case SERVER_MODEL_STATUS_LOADED:   return "loaded";
        case SERVER_MODEL_STATUS_SLEEPING: return "sleeping";
        default:                           return "unknown";
    }
}

struct server_model_meta {
    common_preset preset;
    std::string name;
    std::set<std::string> aliases; // additional names that resolve to this model
    std::set<std::string> tags;    // informational tags, not used for routing
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    json loaded_info; // info to be reflected via /v1/models endpoint
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown
    mtmd_caps multimodal; // multimodal capabilities

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
    void update_caps();
};

struct subprocess_s;

struct server_models {
private:
    struct instance_t {
        std::shared_ptr<subprocess_s> subproc; // shared between main thread and monitoring thread
        std::thread th;
        server_model_meta meta;
        FILE * stdin_file = nullptr;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string> stopping_models;

    // set to true while load_models() is executing a reload; load() will wait until clear
    bool is_reloading = false;

    // conv_id -> model name that currently serves its stream session, lets the resumable stream
    // routes go straight to the owning child instead of polling every one. populated when
    // proxy_request forwards a POST carrying an X-Conversation-Id. best effort: a stale entry just
    // makes the child answer not found and the client recovers. owns its lock, one mutex per struct
    struct conv_model_tracker {
        void remember(const std::string & conv_id, const std::string & model) {
            if (conv_id.empty() || model.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mu);
            map[conv_id] = model;
        }

        std::optional<std::string> lookup(const std::string & conv_id) {
            if (conv_id.empty()) {
                return std::nullopt;
            }
            std::lock_guard<std::mutex> lock(mu);
            auto it = map.find(conv_id);
            if (it == map.end()) {
                return std::nullopt;
            }
            return it->second;
        }

        void forget(const std::string & conv_id) {
            if (conv_id.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mu);
            map.erase(conv_id);
        }

      private:
        std::mutex                                   mu;
        std::unordered_map<std::string, std::string> map;
    };

    common_preset_context ctx_preset;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    common_preset base_preset; // base preset from llama-server CLI args

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

public:
    // conv_id -> model tracker for the resumable stream routes, owns its lock
    conv_model_tracker conv_models;

    server_models(const common_params & params, int argc, char ** argv);

    // (re-)load the list of models from various sources and prepare the metadata mapping
    // - if this is called the first time, simply populate the metadata
    // - if this is called subsequently (e.g. when refreshing from disk):
    //   - if a model is running but updated or removed from the source, it will be unloaded
    //   - if a model is not running, it will be added or updated according to the source
    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance (thread-safe)
    void update_status(const std::string & name, server_model_status status, int exit_code);
    void update_loaded_info(const std::string & name, std::string & raw_info);

    // wait until the model instance is fully loaded (thread-safe)
    // return when the model no longer in "loading" state
    void wait_until_loading_finished(const std::string & name);

    // ensure the model is in ready state (thread-safe)
    // return false if model is ready
    // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
    bool ensure_model_ready(const std::string & name);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used);

    // return true if the current process is a child server instance
    static bool is_child_server();

    // notify the router server that a model instance is ready
    // return the monitoring thread (to be joined by the caller)
    static std::thread setup_child_server(const std::function<void(int)> & shutdown_handler, const json & model_info);

    // notify the router server that the sleeping state has changed
    static void notify_router_sleeping_state(bool sleeping);
};

struct server_models_routes {
    common_params params;
    json ui_settings = json::object();          // Primary: new name
    json webui_settings = json::object();        // Deprecated: use ui_settings (kept for compat)
    server_models models;
    server_models_routes(const common_params & params, int argc, char ** argv)
            : params(params), models(params, argc, argv) {
        // Support both new ui_config_json and deprecated webui_config_json
        const std::string & cfg = !this->params.ui_config_json.empty()
            ? this->params.ui_config_json
            : this->params.webui_config_json;
        if (!cfg.empty()) {
            try {
                json json_settings = json::parse(cfg);
                ui_settings = json_settings;
                webui_settings = json_settings;  // Deprecated: keep in sync
            } catch (const std::exception & e) {
                LOG_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                throw;
            }
        }
        init_routes();
    }

    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_http_context::handler_t get_router_props;
    server_http_context::handler_t proxy_get;
    server_http_context::handler_t proxy_post;
    server_http_context::handler_t get_router_models;
    server_http_context::handler_t post_router_models_load;
    server_http_context::handler_t post_router_models_unload;

    // router side handlers for the resumable streaming routes. each conversation_id may carry
    // an optional ::model suffix to enable direct routing without probing every child. when
    // the suffix is absent the get/delete paths fall back to a loopback probe and a fan out
    // respectively, the list path always fans out and aggregates
    server_http_context::handler_t router_streams_lookup;
    server_http_context::handler_t router_stream_delete;

    // websocket tunnel: the browser connects to the router /ws, the router resolves the owning
    // child once and relays the connection both ways for its whole life. routing is stateful per
    // connection, the conv id to child lookup happens once at open instead of on every call
    server_http_context::ws_handler_t router_stream_ws;
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
    std::function<void()> cleanup = nullptr;
public:
    server_http_proxy(const std::string & method,
                      const std::string & scheme,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::map<std::string, uploaded_file> & files,
                      const std::function<bool()> should_stop,
                      int32_t timeout_read,
                      int32_t timeout_write
                      );
    ~server_http_proxy() {
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};
