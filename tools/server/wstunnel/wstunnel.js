#!/usr/bin/env node
/**
 * Minimal WebSocket HTTP/SSE relay.
 *
 * Usage:
 *   PORT=3210 TARGET_BASE_URL=https://your-upstream-endpoint node wstunnel.js
 *
 * Environment variables:
 *   PORT              TCP port to bind on (default: 3210)
 *   TARGET_BASE_URL   Optional base URL. Required when the client sends relative paths.
 *   LOG_LEVEL         "silent" to disable logging (default: info)
 *
 * Requires Node.js ≥ 18 for global fetch/Web Streams and the `ws` package (`npm install ws`).
 */

const { WebSocketServer, WebSocket } = require('ws');

const PORT = Number(process.env.PORT || 3210);
const BASE_URL = process.env.TARGET_BASE_URL
  ? process.env.TARGET_BASE_URL.trim()
  : '';
const LOG_LEVEL = (process.env.LOG_LEVEL || 'info').toLowerCase();
const LOG_ENABLED = LOG_LEVEL !== 'silent';

function log(...args) {
  if (LOG_ENABLED) {
    console.log(new Date().toISOString(), '[wstunnel]', ...args);
  }
}

function logError(...args) {
  console.error(new Date().toISOString(), '[wstunnel]', ...args);
}

function resolveTarget(rawUrl) {
  if (!rawUrl || typeof rawUrl !== 'string') {
    throw new Error('Missing target URL in tunnel request');
  }
  const trimmed = rawUrl.trim();
  const hasScheme = /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(trimmed);

  if (hasScheme) {
    if (!/^https?:\/\//i.test(trimmed)) {
      throw new Error('Only http(s) targets are allowed');
    }
    if (!BASE_URL) {
      return trimmed;
    }
    const allowedOrigin = new URL(BASE_URL).origin;
    const requestedOrigin = new URL(trimmed).origin;
    if (allowedOrigin !== requestedOrigin) {
      throw new Error(`Target origin ${requestedOrigin} is not allowed`);
    }
    return trimmed;
  }

  if (!BASE_URL) {
    throw new Error('Relative URLs require TARGET_BASE_URL to be set');
  }
  return new URL(trimmed, BASE_URL).toString();
}

function sanitizeHeaders(headers, mode) {
  if (!headers || typeof headers !== 'object') return {};
  const out = {};
  for (const [key, value] of Object.entries(headers)) {
    if (typeof value !== 'string') continue;
    const lower = key.toLowerCase();
    if (lower === 'connection' || lower === 'content-length' || lower === 'host')
      continue;
    out[key] = value;
  }
  if (
    mode === 'sse' &&
    !Object.keys(out).some((key) => key.toLowerCase() === 'accept')
  ) {
    out.Accept = 'text/event-stream';
  }
  return out;
}

function serializeHeaders(headers) {
  const result = {};
  for (const [key, value] of headers.entries()) {
    result[key] = value;
  }
  return result;
}

function ensureOpen(ws) {
  if (ws.readyState !== WebSocket.OPEN) {
    throw new Error('Client disconnected');
  }
}

function sendFrame(ws, frame) {
  ensureOpen(ws);
  ws.send(JSON.stringify(frame));
}

async function readBody(response) {
  if (!response.body) return null;
  let text = '';
  try {
    text = await response.text();
  } catch (err) {
    throw new Error(
      err?.message ? `Unable to read upstream response: ${err.message}` : 'Unable to read upstream response'
    );
  }
  if (!text.length) {
    return { encoding: 'text', data: '' };
  }
  try {
    return { encoding: 'json', data: JSON.parse(text) };
  } catch {
    return { encoding: 'text', data: text };
  }
}

async function forwardSSE(response, ws) {
  if (!response.body) {
    throw new Error('Upstream response does not contain a body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = createEmptyEvent();

  function createEmptyEvent() {
    return {
      data: [],
      event: undefined,
      id: undefined,
      retry: undefined,
    };
  }

  function emitEvent() {
    if (
      !currentEvent.data.length &&
      currentEvent.event === undefined &&
      currentEvent.id === undefined &&
      currentEvent.retry === undefined
    ) {
      return;
    }

    const payload = currentEvent.data.join('\n');
    sendFrame(ws, {
      type: 'sse',
      data: payload,
      event: currentEvent.event,
      id: currentEvent.id,
      retry: currentEvent.retry,
    });
    currentEvent = createEmptyEvent();
  }

  function processLine(line) {
    if (!line.length) {
      emitEvent();
      return;
    }
    if (line.startsWith(':')) {
      // Comment line – ignore.
      return;
    }

    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? '' : line.slice(separator + 1).trimStart();

    switch (field) {
      case 'data':
        currentEvent.data.push(value);
        break;
      case 'event':
        currentEvent.event = value || undefined;
        break;
      case 'id':
        currentEvent.id = value || undefined;
        break;
      case 'retry': {
        const retry = Number(value);
        if (!Number.isNaN(retry)) {
          currentEvent.retry = retry;
        }
        break;
      }
      default:
        // Ignore unknown fields to keep the tunnel generic.
        break;
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        processLine(line.replace(/\r$/, ''));
      }
    }

    if (buffer.length) {
      processLine(buffer.replace(/\r$/, ''));
    }

    emitEvent();

    if (ws.readyState === WebSocket.OPEN) {
      sendFrame(ws, { type: 'done' });
    }
  } finally {
    reader.releaseLock();
  }
}

const wss = new WebSocketServer({ port: PORT });

wss.on('listening', () => {
  log(`listening on ws://0.0.0.0:${PORT}`);
  if (BASE_URL) {
    log(`allowed upstream base: ${BASE_URL}`);
  } else {
    log('upstream base unrestricted (ensure clients are trusted)');
  }
});

wss.on('connection', (ws, req) => {
  log('client connected from', req.socket.remoteAddress);
  let controller = null;
  let started = false;
  let abortedByClient = false;

  function abortUpstream(reason) {
    if (controller && !controller.signal.aborted) {
      abortedByClient = reason === 'client';
      controller.abort();
    }
  }

  ws.on('close', () => {
    log('client disconnected');
    abortUpstream('client');
  });

  ws.on('error', (err) => {
    logError('socket error:', err.message);
    abortUpstream('client');
  });

  ws.on('message', (raw) => {
    if (!started) {
      started = true;
      handleInitialMessage(raw).catch((err) => {
        if (err?.name === 'AbortError' && abortedByClient) {
          log('upstream aborted by client');
          return;
        }
        if (err && err.message === 'Client disconnected') {
          log('client closed before stream ended');
          return;
        }
        logError('tunnel error:', err?.message || err);
        if (ws.readyState === WebSocket.OPEN) {
          try {
            sendFrame(ws, {
              type: 'error',
              message: err?.message || 'WebSocket tunnel error',
            });
          } catch {
            /* ignore */
          }
          ws.close(1011, 'tunnel error');
        }
      });
      return;
    }

    try {
      const payload = JSON.parse(raw.toString());
      if (payload?.type === 'abort') {
        log('received explicit abort request from client');
        abortUpstream('client');
      }
    } catch {
      /* ignore non-JSON control frames */
    }
  });

  async function handleInitialMessage(raw) {
    let descriptor;
    try {
      descriptor = JSON.parse(raw.toString());
    } catch {
      throw new Error('Invalid JSON payload received by tunnel');
    }

    if (!descriptor || typeof descriptor !== 'object') {
      throw new Error('Malformed tunnel request payload');
    }

    const targetUrl = resolveTarget(descriptor.url);
    const method =
      typeof descriptor.method === 'string'
        ? descriptor.method.toUpperCase()
        : 'GET';
    const mode = descriptor.mode === 'sse' ? 'sse' : 'json';
    const headers = sanitizeHeaders(descriptor.headers, mode);
    const body =
      descriptor.body === undefined
        ? undefined
        : typeof descriptor.body === 'string'
        ? descriptor.body
        : JSON.stringify(descriptor.body);

    controller = new AbortController();
    const signal = controller.signal;

    const fetchInit = { method, headers, signal };
    if (body !== undefined && method !== 'GET' && method !== 'HEAD') {
      fetchInit.body = body;
    }

    log(`proxying ${method} ${targetUrl} (${mode})`);

    let response;
    try {
      response = await fetch(targetUrl, fetchInit);
    } catch (err) {
      if (signal.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      throw new Error(
        err?.message
          ? `Unable to reach upstream: ${err.message}`
          : 'Unable to reach upstream'
      );
    } finally {
      if (!response) {
        controller = null;
      }
    }

    if (!response) {
      throw new Error('Upstream response is empty');
    }

    try {
      sendFrame(ws, {
        type: 'response',
        status: response.status,
        statusText: response.statusText,
        headers: serializeHeaders(response.headers),
      });

      if (mode === 'sse') {
        if (!response.ok) {
          const detail = await readBody(response).catch(() => null);
          if (detail) {
            sendFrame(ws, {
              type: 'data',
              encoding: detail.encoding,
              data: detail.data,
            });
          }
          sendFrame(ws, { type: 'done' });
          return;
        }
        await forwardSSE(response, ws);
        return;
      }

      const payload = await readBody(response).catch((err) => {
        throw err instanceof Error ? err : new Error('Failed to read upstream body');
      });
      if (payload) {
        sendFrame(ws, { type: 'data', encoding: payload.encoding, data: payload.data });
      }
      sendFrame(ws, { type: 'done' });
    } finally {
      controller = null;
    }
  }
});

wss.on('error', (err) => {
  logError('server error:', err.message);
});

process.on('SIGTERM', () => {
  log('shutting down (SIGTERM)');
  wss.close(() => process.exit(0));
});

process.on('SIGINT', () => {
  log('shutting down (SIGINT)');
  wss.close(() => process.exit(0));
});
