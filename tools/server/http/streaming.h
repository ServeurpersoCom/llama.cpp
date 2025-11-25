#pragma once

#include <functional>
#include <memory>
#include <string>

struct server_http_res;
using server_http_res_ptr = std::unique_ptr<server_http_res>;

namespace httplib {
class Response;
}

namespace http::streaming {

void apply_chunked_response(server_http_res_ptr & response, httplib::Response & res, const std::function<void(int, const char *)> & logger = nullptr);

} // namespace http::streaming
