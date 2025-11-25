#include "http/streaming.h"

#include "http/context.h"

#include <cpp-httplib/httplib.h>
#include <cstdio>
#include <utility>

namespace http::streaming {

namespace {

void set_headers(httplib::Response & res, const std::map<std::string, std::string> & headers) {
    for (const auto & [key, value] : headers) {
        res.set_header(key, value);
    }
}

template <typename... Args>
void log_debug(const std::function<void(int, const char *)> & logger, const char * fmt, Args && ... args) {
    if (!logger) {
        return;
    }
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    buffer[sizeof(buffer) - 1] = '\0';
    logger(SERVER_HTTP_LOG_LEVEL_DEBUG, buffer);
}

} // namespace

void apply_chunked_response(server_http_res_ptr & response, httplib::Response & res, const std::function<void(int, const char *)> & logger) {
    if (!response) {
        res.status = 500;
        res.set_content("Internal Server Error", "text/plain");
        return;
    }

    if (response->is_stream()) {
        res.status = response->status;
        set_headers(res, response->headers);
        std::string content_type = response->content_type;
        std::shared_ptr<server_http_res> r_ptr = std::move(response);
        const auto chunked_content_provider = [response = r_ptr, logger](size_t, httplib::DataSink & sink) -> bool {
            std::string chunk;
            bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                // TODO: maybe handle sink.write unsuccessful? for now, we rely on is_connection_closed()
                sink.write(chunk.data(), chunk.size());
                log_debug(logger, "http: streamed chunk: %s\n", chunk.c_str());
            }
            if (!has_next) {
                sink.done();
                log_debug(logger, "%s", "http: stream ended\n");
            }
            return has_next;
        };
        const auto on_complete = [response = r_ptr](bool) mutable {
            response.reset();
        };
        res.set_chunked_content_provider(content_type, chunked_content_provider, on_complete);
    } else {
        res.status = response->status;
        set_headers(res, response->headers);
        res.set_content(response->data, response->content_type);
    }
}

} // namespace http::streaming
