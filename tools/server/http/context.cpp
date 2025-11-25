#include "http/context.h"

#include "http/streaming.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

using json = nlohmann::ordered_json;

class server_http_context::Impl {
public:
    std::unique_ptr<httplib::Server> srv;
};

namespace {

template <typename... Args>
void log_msg(const std::function<void(int, const char *)> & logger, int level, const char * fmt, Args && ... args) {
    if (!logger) {
        return;
    }

    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    buffer[sizeof(buffer) - 1] = '\0';
    logger(level, buffer);
}

template <typename... Args>
void log_msg(const server_http_config & config, int level, const char * fmt, Args && ... args) {
    log_msg(config.logger, level, fmt, std::forward<Args>(args)...);
}

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split_string(const std::string & value, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, delim)) {
        parts.push_back(part);
    }
    if (!value.empty() && value.back() == delim) {
        parts.emplace_back();
    }
    if (parts.empty()) {
        parts.emplace_back();
    }
    return parts;
}

void log_server_request(const std::function<void(int, const char *)> & logger, const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health") {
        return;
    }

    // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch
    log_msg(logger, SERVER_HTTP_LOG_LEVEL_INFO, "request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
    log_msg(logger, SERVER_HTTP_LOG_LEVEL_DEBUG, "request:  %s\n", req.body.c_str());
    log_msg(logger, SERVER_HTTP_LOG_LEVEL_DEBUG, "response: %s\n", res.body.c_str());
}

std::map<std::string, std::string> get_params(const httplib::Request & req) {
    std::map<std::string, std::string> params;
    for (const auto & [key, value] : req.params) {
        params[key] = value;
    }
    for (const auto & [key, value] : req.path_params) {
        params[key] = value;
    }
    return params;
}

std::map<std::string, std::string> get_headers(const httplib::Request & req) {
    std::map<std::string, std::string> headers;
    for (const auto & [key, value] : req.headers) {
        headers[key] = value;
    }
    return headers;
}

httplib::Server::HandlerResponse run_middlewares(const std::vector<server_http_middleware_config::middleware_t> & mws, const httplib::Request & req, httplib::Response & res) {
    for (const auto & mw : mws) {
        if (!mw) {
            continue;
        }
        auto resp = mw(req, res);
        if (resp == httplib::Server::HandlerResponse::Handled) {
            return resp;
        }
    }
    return httplib::Server::HandlerResponse::Unhandled;
}

} // namespace

server_http_context::server_http_context()
    : pimpl(std::make_unique<server_http_context::Impl>()) {
}

server_http_context::~server_http_context() = default;

void flag_readiness_provider::set_ready(bool ready_) {
    ready.store(ready_);
}

bool flag_readiness_provider::is_ready() const {
    return ready.load();
}

bool server_http_context::init(const server_http_config & cfg) {
    config = cfg;

    if (!config.readiness) {
        config.readiness = std::make_shared<flag_readiness_provider>();
    }

    path_prefix = config.path_prefix;
    port = config.port;
    hostname = config.hostname;

    auto & srv = pimpl->srv;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (!config.ssl_file_key.empty() && !config.ssl_file_cert.empty()) {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "Running with SSL: key = %s, cert = %s\n", config.ssl_file_key.c_str(), config.ssl_file_cert.c_str());
        srv.reset(new httplib::SSLServer(config.ssl_file_cert.c_str(), config.ssl_file_key.c_str()));
    } else {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s", "Running without SSL\n");
        srv.reset(new httplib::Server());
    }
#else
    if (!config.ssl_file_key.empty() && !config.ssl_file_cert.empty()) {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_ERROR, "%s", "Server is built without SSL support\n");
        return false;
    }
    srv.reset(new httplib::Server());
#endif

    if (!config.default_headers.empty()) {
        httplib::Headers headers;
        for (const auto & [key, value] : config.default_headers) {
            headers.emplace(key, value);
        }
        srv->set_default_headers(headers);
    }
    if (config.enable_logger) {
        srv->set_logger([logger = config.logger](const httplib::Request & req, const httplib::Response & res) {
            log_server_request(logger, req, res);
        });
    }
    srv->set_exception_handler([logger = config.logger](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        // this is fail-safe; exceptions should already handled by 'ex_wrapper'
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        res.status = 500;
        res.set_content(message, "text/plain");
        log_msg(logger, SERVER_HTTP_LOG_LEVEL_ERROR, "got exception: %s\n", message.c_str());
    });

    srv->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res.set_content(
                json{
                    {"error", {
                        {"message", "File Not Found"},
                        {"type", "not_found_error"},
                        {"code", 404},
                    }},
                }.dump(),
                "application/json; charset=utf-8");
        }
        // for other error codes, we skip processing here because it's already done by res->error()
    });

    srv->set_read_timeout(config.timeout_read);
    srv->set_write_timeout(config.timeout_write);

    if (config.middleware.api_keys.size() == 1) {
        auto key = config.middleware.api_keys[0];
        std::string substr = key.substr(std::max((int) (key.length() - 4), 0));
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s: api_keys: ****%s\n", __func__, substr.c_str());
    } else if (config.middleware.api_keys.size() > 1) {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s: api_keys: %zu keys loaded\n", __func__, config.middleware.api_keys.size());
    }

    std::vector<server_http_middleware_config::middleware_t> middlewares;
    middlewares.reserve(4 + config.middleware.extra.size());

    if (config.middleware.enable_cors) {
        if (config.middleware.cors) {
            middlewares.push_back(config.middleware.cors);
        } else {
            middlewares.push_back([](const httplib::Request & req, httplib::Response & res) {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                if (req.method == "OPTIONS") {
                    // If this is OPTIONS request, skip validation because browsers don't include Authorization header
                    res.set_header("Access-Control-Allow-Credentials", "true");
                    res.set_header("Access-Control-Allow-Methods", "GET, POST");
                    res.set_header("Access-Control-Allow-Headers", "*");
                    // blank response, no data
                    res.set_content("", "text/html");
                    return httplib::Server::HandlerResponse::Handled;
                }
                return httplib::Server::HandlerResponse::Unhandled;
            });
        }
    }

    if (config.middleware.enable_readiness && config.readiness) {
        if (config.middleware.readiness) {
            middlewares.push_back(config.middleware.readiness);
        } else {
            middlewares.push_back([readiness = config.readiness, loading_page = config.loading_html](const httplib::Request & req, httplib::Response & res) {
                if (readiness->is_ready()) {
                    return httplib::Server::HandlerResponse::Unhandled;
                }

                const auto tmp = split_string(req.path, '.');
                if ((req.path == "/" || (!tmp.empty() && tmp.back() == "html"))) {
                    if (!loading_page.empty()) {
                        res.set_content(loading_page, "text/html; charset=utf-8");
                        res.status = 503;
                    } else {
                        res.status = 503;
                        res.set_content("Loading model", "text/plain");
                    }
                } else if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
                    return httplib::Server::HandlerResponse::Unhandled;
                } else {
                    res.status = 503;
                    res.set_content(
                        json{
                            {"error", {
                                {"message", "Loading model"},
                                {"type", "unavailable_error"},
                                {"code", 503},
                            }},
                        }.dump(),
                        "application/json; charset=utf-8");
                }
                return httplib::Server::HandlerResponse::Handled;
            });
        }
    }

    if (config.middleware.enable_auth) {
        if (config.middleware.auth) {
            middlewares.push_back(config.middleware.auth);
        } else {
            const std::unordered_set<std::string> public_endpoints = config.middleware.public_endpoints.empty()
                ? std::unordered_set<std::string>{
                      "/health",
                      "/v1/health",
                      "/models",
                      "/v1/models",
                      "/api/tags",
                  }
                : config.middleware.public_endpoints;

            middlewares.push_back([api_keys = config.middleware.api_keys, public_endpoints, logger = config.logger](const httplib::Request & req, httplib::Response & res) {
                // If API key is not set, skip validation
                if (api_keys.empty()) {
                    return httplib::Server::HandlerResponse::Unhandled;
                }

                // If path is public or is static file, skip validation
                if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
                    return httplib::Server::HandlerResponse::Unhandled;
                }

                // Check for API key in the header
                auto auth_header = req.get_header_value("Authorization");
                const std::string prefix = "Bearer ";
                if (auth_header.substr(0, prefix.size()) == prefix) {
                    std::string received_api_key = auth_header.substr(prefix.size());
                    if (std::find(api_keys.begin(), api_keys.end(), received_api_key) != api_keys.end()) {
                        return httplib::Server::HandlerResponse::Unhandled;
                    }
                }

                // API key is invalid or not provided
                res.status = 401;
                res.set_content(
                    json{
                        {"error", {
                            {"message", "Invalid API Key"},
                            {"type", "authentication_error"},
                            {"code", 401},
                        }},
                    }.dump(),
                    "application/json; charset=utf-8");
                log_msg(logger, SERVER_HTTP_LOG_LEVEL_WARN, "%s", "Unauthorized: Invalid API Key\n");

                return httplib::Server::HandlerResponse::Handled;
            });
        }
    }

    for (const auto & mw : config.middleware.extra) {
        middlewares.push_back(mw);
    }

    srv->set_pre_routing_handler([middlewares](const httplib::Request & req, httplib::Response & res) {
        return run_middlewares(middlewares, req, res);
    });

    int n_threads_http = config.n_threads_http;
    if (n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        n_threads_http = std::max(config.parallelism_hint + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s: using %d threads for HTTP server\n", __func__, n_threads_http);
    srv->new_task_queue = [n_threads_http] { return new httplib::ThreadPool(n_threads_http); };

    if (config.webui) {
        if (!config.public_path.empty()) {
            // Set the base directory for serving static files
            bool is_found = srv->set_mount_point(config.path_prefix + "/", config.public_path);
            if (!is_found) {
                log_msg(config, SERVER_HTTP_LOG_LEVEL_ERROR, "%s: static assets path not found: %s\n", __func__, config.public_path.c_str());
                return false;
            }
        } else if (!config.index_html_gz.empty()) {
            // using embedded static index.html
            const auto index_html = config.index_html_gz;
            srv->Get(config.path_prefix + "/", [index_html](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    // COEP and COOP headers, required by pyodide (python interpreter)
                    res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                    res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    res.set_content(index_html, "text/html; charset=utf-8");
                }
                return false;
            });
        } else {
            log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s", "Web UI is disabled (no embedded UI configured)\n");
        }
    } else {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s", "Web UI is disabled\n");
    }

    return true;
}

bool server_http_context::start() {
    auto & srv = pimpl->srv;
    bool was_bound = false;
    bool is_sock = false;
    if (ends_with(hostname, ".sock")) {
        is_sock = true;
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s", "setting address family to AF_UNIX\n");
        srv->set_address_family(AF_UNIX);
        was_bound = srv->bind_to_port(hostname, 8080);
    } else {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_INFO, "%s", "binding port with default address family\n");
        if (port == 0) {
            int bound_port = srv->bind_to_any_port(hostname);
            was_bound = (bound_port >= 0);
            if (was_bound) {
                port = bound_port;
            }
        } else {
            was_bound = srv->bind_to_port(hostname, port);
        }
    }

    if (!was_bound) {
        log_msg(config, SERVER_HTTP_LOG_LEVEL_ERROR, "couldn't bind HTTP server socket, hostname: %s, port: %d\n", hostname.c_str(), port);
        return false;
    }

    thread = std::thread([this]() { pimpl->srv->listen_after_bind(); });
    srv->wait_until_ready();

    listening_address = is_sock ? "unix://" + hostname
                                : "http://" + hostname + ":" + std::to_string(port);
    return true;
}

void server_http_context::stop() const {
    if (pimpl->srv) {
        pimpl->srv->stop();
    }
}

void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    pimpl->srv->Get(path_prefix + path, [handler, logger = config.logger](const httplib::Request & req, httplib::Response & res) {
        server_http_res_ptr response = handler(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            req.body,
            req.is_connection_closed,
        });
        http::streaming::apply_chunked_response(response, res, logger);
    });
}

void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    pimpl->srv->Post(path_prefix + path, [handler, logger = config.logger](const httplib::Request & req, httplib::Response & res) {
        server_http_res_ptr response = handler(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            req.body,
            req.is_connection_closed,
        });
        http::streaming::apply_chunked_response(response, res, logger);
    });
}

bool server_http_context::is_ready() const {
    return config.readiness ? config.readiness->is_ready() : true;
}

void server_http_context::set_ready(bool ready) const {
    if (config.readiness) {
        config.readiness->set_ready(ready);
    }
}

std::shared_ptr<readiness_provider> server_http_context::readiness() const {
    return config.readiness;
}
