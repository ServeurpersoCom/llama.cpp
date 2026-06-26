#pragma once

#include "common.h"
#include "json.h"

#include <string>

// ggml scope: live activation capture streamed during graph execution
//
// a single client holds GET /scope/stream at a time, every llama_decode running while
// the stream is open produces one frame of per node statistics, frames carry a
// monotonic seq so the client detects drops, the pending queue is bounded and sheds
// its oldest frame under backpressure instead of stalling the decode
//
// scope of observation: every llama_context created from the same common_params shares
// the eval callback, frames pair with tokens by decode order on a single active slot
// no callback is installed unless params.endpoint_scope is enabled

void server_scope_install(common_params & params);

// finalizes the frame of the decode that just completed, called from the server loop
void server_scope_on_decode_done();

// opens the exclusive stream and arms the capture, empty filter captures every node
// returns null on success, otherwise { error, error_type } with error_type
// invalid_request for a bad filter regex or unavailable when a stream is already open
common_json server_scope_open(const std::string & filter);

// blocks up to timeout_ms for the next frame, false when none arrived in time
bool server_scope_next_frame(std::string & out_json, int timeout_ms);

// disarms the capture and releases the stream slot, idempotent
void server_scope_close();
