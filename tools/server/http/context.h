#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <cpp-httplib/httplib.h>

enum server_http_log_level {
    SERVER_HTTP_LOG_LEVEL_DEBUG = 0,
    SERVER_HTTP_LOG_LEVEL_INFO,
    SERVER_HTTP_LOG_LEVEL_WARN,
    SERVER_HTTP_LOG_LEVEL_ERROR,
};

struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // generator-like API for HTTP response generation
    // this object responds with one of the 2 modes:
    // 1) normal response: `data` contains the full response body
    // 2) streaming response: each call to next(output) generates the next chunk
    //    when next(output) returns false, no more data after the current chunk
    //    note: some chunks can be empty, in which case no data is sent for that chunk
    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;

struct server_http_req {
    std::map<std::string, std::string> params;  // path_params + query_params
    std::map<std::string, std::string> headers; // reserved for future use
    std::string path;                           // reserved for future use
    std::string body;
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
};

class readiness_provider {
public:
    virtual ~readiness_provider() = default;
    virtual bool is_ready() const = 0;
    virtual void set_ready(bool /*ready*/) {}
};

class flag_readiness_provider : public readiness_provider {
public:
    void set_ready(bool ready) override;
    bool is_ready() const override;

private:
    std::atomic<bool> ready{false};
};

struct server_http_middleware_config {
    using middleware_t = std::function<httplib::Server::HandlerResponse(const httplib::Request &, httplib::Response &)>;

    bool enable_cors = true;
    bool enable_auth = true;
    bool enable_readiness = true;
    middleware_t cors;
    middleware_t auth;
    middleware_t readiness;
    std::vector<middleware_t> extra;
    std::vector<std::string> api_keys;
    std::unordered_set<std::string> public_endpoints;
};

struct server_http_config {
    std::function<void(int level, const char * message)> logger;
    std::string path_prefix;
    std::string hostname;
    int port = 0;
    int timeout_read = 30;
    int timeout_write = 30;
    int n_threads_http = 0;
    int parallelism_hint = 0;
    bool webui = false;
    std::string public_path;
    std::string ssl_file_key;
    std::string ssl_file_cert;
    bool enable_logger = true;
    std::string loading_html;  // empty = no loading page
    std::string index_html_gz; // empty = no embedded UI
    std::map<std::string, std::string> default_headers = {{"Server", "llama.cpp"}};
    std::shared_ptr<readiness_provider> readiness;
    server_http_middleware_config middleware;
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread

    std::string path_prefix;
    std::string hostname;
    int port = 0;

    server_http_context();
    ~server_http_context();

    bool init(const server_http_config & config);
    bool start();
    void stop() const;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;

    bool is_ready() const;
    void set_ready(bool ready) const;
    std::shared_ptr<readiness_provider> readiness() const;

    // for debugging
    std::string listening_address;

private:
    server_http_config config;
};
