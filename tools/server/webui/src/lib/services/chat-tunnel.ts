import { browser } from '$app/environment';
import { SSEWEBSOCKETPROXYURL } from '$lib/constants/chat-tunnel';
import { createSSETunnelResponse } from '$lib/utils/websocket-tunnel';

type NormalizedRequest = {
	url: string;
	method: string;
	headers?: HeadersInit;
	body: string;
	signal?: AbortSignal;
};

const globalState = globalThis as typeof globalThis & {
	__chatTunnelPatched?: boolean;
	__chatTunnelOriginalFetch?: typeof fetch;
};

function resolveTunnelUrl(value: unknown): string | null {
	if (value === null || value === undefined) {
		return null;
	}

	const raw = typeof value === 'string' ? value : value.toString();
	const trimmed = raw.trim();

	return trimmed.length > 0 ? trimmed : null;
}

const RESOLVED_TUNNEL_URL = resolveTunnelUrl(SSEWEBSOCKETPROXYURL);

function normalizeRequest(input: RequestInfo | URL, init?: RequestInit): NormalizedRequest | null {
	let url: string;

	if (typeof input === 'string') {
		url = input;
	} else if (input instanceof URL) {
		url = input.toString();
	} else {
		url = input.url;
	}

	const method = (init?.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
	const headers = init?.headers ?? (input instanceof Request ? input.headers : undefined);
	const signal = init?.signal ?? (input instanceof Request ? input.signal : undefined);

	let body: string | undefined;

	if (init?.body !== undefined) {
		if (typeof init.body === 'string') {
			body = init.body;
		} else {
			return null;
		}
	} else if (!(input instanceof Request)) {
		body = undefined;
	} else {
		// We cannot synchronously read Request bodies; skip interception.
		return null;
	}

	if (body === undefined) {
		return null;
	}

	return { url, method, headers, body, signal };
}

function shouldUseTunnel(request: NormalizedRequest, tunnelUrl: string | null): boolean {
	if (!tunnelUrl) {
		return false;
	}

	if (request.method !== 'POST') {
		return false;
	}

	let payload: unknown;
	try {
		payload = JSON.parse(request.body);
	} catch {
		return false;
	}

	const stream =
		typeof payload === 'object' &&
		payload !== null &&
		(payload as Record<string, unknown>)['stream'];
	if (stream !== true) {
		return false;
	}

	try {
		const resolvedUrl = new URL(request.url, window.location.href);
		if (!resolvedUrl.pathname.endsWith('/v1/chat/completions')) {
			return false;
		}
	} catch {
		return false;
	}

	return true;
}

async function forwardThroughTunnel(
	request: NormalizedRequest,
	tunnelUrl: string
): Promise<Response> {
	const resolvedTarget = new URL(request.url, window.location.href).toString();

	return createSSETunnelResponse({
		socketUrl: tunnelUrl,
		targetUrl: resolvedTarget,
		method: request.method,
		headers: request.headers,
		body: request.body,
		abortSignal: request.signal
	});
}

function patchFetch(): void {
	if (!browser) {
		return;
	}

	if (globalState.__chatTunnelPatched) {
		return;
	}

	const originalFetch = globalThis.fetch.bind(globalThis);
	globalState.__chatTunnelOriginalFetch = originalFetch;

	globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
		const tunnelUrl = RESOLVED_TUNNEL_URL;

		if (!tunnelUrl) {
			return originalFetch(input as RequestInfo, init);
		}

		const normalized = normalizeRequest(input, init);
		if (!normalized || !shouldUseTunnel(normalized, tunnelUrl)) {
			return originalFetch(input as RequestInfo, init);
		}

		try {
			return await forwardThroughTunnel(normalized, tunnelUrl);
		} catch (error) {
			console.warn('WebSocket tunnel failed, falling back to direct fetch:', error);
			return originalFetch(input as RequestInfo, init);
		}
	};

	globalState.__chatTunnelPatched = true;
}

patchFetch();
