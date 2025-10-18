/*
 * Utilities to forward llama.cpp API requests through a WebSocket tunnel.
 *
 * The tunnel protocol is intentionally small: after the websocket connection
 * is established the client sends a JSON payload describing the request to
 * forward (target URL, method, headers, body and the desired mode). The proxy
 * then answers with a "response" frame followed by zero or more payload
 * frames (SSE chunks, JSON fragments, ...). A terminal "done" frame closes the
 * session. If an error occurs the proxy emits an "error" frame instead.
 */

import { browser } from '$app/environment';

export type HeadersRecord = Record<string, string>;

export type TunnelMode = 'json' | 'sse';

export type TunnelRequestPayload = {
	url: string;
	method: string;
	headers: HeadersRecord;
	body?: unknown;
	mode: TunnelMode;
};

type ResponseFrame = {
	type: 'response';
	status: number;
	statusText?: string;
	headers?: HeadersRecord;
};

type DataFrame = {
	type: 'data';
	encoding: 'json' | 'text';
	data: unknown;
};

type SSEFrame = {
	type: 'sse';
	data: unknown;
	event?: unknown;
	id?: unknown;
	retry?: unknown;
};

type DoneFrame = {
	type: 'done';
};

type ErrorFrame = {
	type: 'error';
	message?: string;
};

type IncomingFrame = ResponseFrame | DataFrame | SSEFrame | DoneFrame | ErrorFrame;

type AsyncQueue<T> = {
	push(value: T): void;
	close(): void;
	fail(error: Error): void;
	next(): Promise<IteratorResult<T>>;
};

function createAsyncQueue<T>(): AsyncQueue<T> {
	const values: T[] = [];
	const pending: {
		resolve: (value: IteratorResult<T>) => void;
		reject: (reason?: unknown) => void;
	}[] = [];
	let closed = false;
	let failure: Error | null = null;

	return {
		push(value: T) {
			if (closed || failure) {
				return;
			}

			if (pending.length) {
				pending.shift()!.resolve({ value, done: false });
			} else {
				values.push(value);
			}
		},
		close() {
			if (closed || failure) {
				return;
			}

			closed = true;
			while (pending.length) {
				pending.shift()!.resolve({ value: undefined, done: true });
			}
		},
		fail(error: Error) {
			if (failure) {
				return;
			}

			failure = error;
			while (pending.length) {
				pending.shift()!.reject(error);
			}
		},
		async next() {
			if (values.length) {
				return { value: values.shift()!, done: false };
			}

			if (failure) {
				throw failure;
			}

			if (closed) {
				return { value: undefined, done: true };
			}

			return new Promise<IteratorResult<T>>((resolve, reject) => {
				pending.push({ resolve, reject });
			});
		}
	};
}

function headersInitToRecord(headers: HeadersInit | undefined): HeadersRecord {
	if (!headers) {
		return {};
	}

	if (headers instanceof Headers) {
		return Object.fromEntries(headers.entries());
	}

	if (Array.isArray(headers)) {
		return headers.reduce<HeadersRecord>((acc, [key, value]) => {
			acc[key] = value;
			return acc;
		}, {});
	}

	return { ...(headers as HeadersRecord) };
}

type TunnelConnection = {
	queue: AsyncQueue<IncomingFrame>;
	waitForOpen(): Promise<void>;
	send(payload: TunnelRequestPayload): void;
	cleanup(): void;
};

function openTunnelConnection(socketUrl: string, abortSignal?: AbortSignal): TunnelConnection {
	if (!browser) {
		throw new Error('WebSocket tunnels are only available in the browser environment');
	}

	const ws = new WebSocket(socketUrl);
	const queue = createAsyncQueue<IncomingFrame>();
	let finished = false;
	let failed = false;
	let abortError: Error | null = null;
	let rejectOpen: ((reason?: unknown) => void) | null = null;

	const markFinished = () => {
			if (finished || failed) {
				return;
			}

			finished = true;
			queue.close();
		},
		markFailed = (error: Error) => {
			if (failed) {
				return;
			}

			failed = true;
			queue.fail(error);
		};

	const abortHandler = () => {
		if (abortError) {
			return;
		}

		abortError = new Error('The operation was aborted.');
		abortError.name = 'AbortError';
		markFailed(abortError);

		if (rejectOpen) {
			rejectOpen(abortError);
		}

		if (ws.readyState === WebSocket.OPEN) {
			try {
				ws.send(JSON.stringify({ type: 'abort' }));
			} catch {
				/* noop */
			}
		}

		if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
			try {
				ws.close(4000, 'abort');
			} catch {
				/* noop */
			}
		}
	};

	if (abortSignal?.aborted) {
		abortHandler();
	}

	abortSignal?.addEventListener('abort', abortHandler);

	const handleFrame = (raw: string) => {
		if (finished || failed) {
			return;
		}

		let parsed: IncomingFrame;
		try {
			parsed = JSON.parse(raw) as IncomingFrame;
		} catch {
			markFailed(new Error('Invalid payload received from WebSocket tunnel'));
			return;
		}

		if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) {
			markFailed(new Error('Malformed payload received from WebSocket tunnel'));
			return;
		}

		if (parsed.type === 'error') {
			const message =
				typeof parsed.message === 'string' ? parsed.message : 'WebSocket tunnel error';
			markFailed(new Error(message));
			return;
		}

		queue.push(parsed);

		if (parsed.type === 'done') {
			markFinished();
		}
	};

	const messageHandler = (event: MessageEvent) => {
		if (finished || failed) {
			return;
		}

		if (typeof event.data === 'string') {
			handleFrame(event.data);
			return;
		}

		if (event.data instanceof Blob) {
			event.data
				.text()
				.then(handleFrame)
				.catch(() => {
					markFailed(new Error('Failed to read WebSocket tunnel message payload'));
				});
			return;
		}

		markFailed(new Error('Unsupported message type received from WebSocket tunnel'));
	};

	const errorHandler = () => {
		if (finished || failed) {
			return;
		}

		markFailed(new Error('WebSocket tunnel error'));
	};

	const closeHandler = (event: CloseEvent) => {
		if (failed || finished) {
			return;
		}

		if (event.wasClean && (event.code === 1000 || event.code === 4000)) {
			markFinished();
			return;
		}

		markFailed(new Error(`WebSocket tunnel closed unexpectedly (${event.code})`));
	};

	ws.addEventListener('message', messageHandler);
	ws.addEventListener('error', errorHandler);
	ws.addEventListener('close', closeHandler);

	const openPromise = new Promise<void>((resolve, reject) => {
		const handleOpen = () => {
			ws.removeEventListener('open', handleOpen);
			ws.removeEventListener('error', handleErrorBeforeOpen);
			ws.removeEventListener('close', handleCloseBeforeOpen);
			resolve();
		};

		const handleErrorBeforeOpen = () => {
			ws.removeEventListener('open', handleOpen);
			ws.removeEventListener('error', handleErrorBeforeOpen);
			ws.removeEventListener('close', handleCloseBeforeOpen);
			reject(new Error('Unable to establish WebSocket tunnel'));
		};

		const handleCloseBeforeOpen = () => {
			ws.removeEventListener('open', handleOpen);
			ws.removeEventListener('error', handleErrorBeforeOpen);
			ws.removeEventListener('close', handleCloseBeforeOpen);
			reject(new Error('WebSocket tunnel closed before it was ready'));
		};

		rejectOpen = reject;
		ws.addEventListener('open', handleOpen, { once: true });
		ws.addEventListener('error', handleErrorBeforeOpen, { once: true });
		ws.addEventListener('close', handleCloseBeforeOpen, { once: true });
	});

	const cleanup = () => {
		abortSignal?.removeEventListener('abort', abortHandler);
		ws.removeEventListener('message', messageHandler);
		ws.removeEventListener('error', errorHandler);
		ws.removeEventListener('close', closeHandler);

		if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
			try {
				ws.close(1000, 'done');
			} catch {
				/* noop */
			}
		}
	};

	return {
		queue,
		waitForOpen: () =>
			openPromise.catch((error) => {
				abortSignal?.removeEventListener('abort', abortHandler);
				if (abortError) {
					throw abortError;
				}

				throw error;
			}),
		send(payload: TunnelRequestPayload) {
			try {
				ws.send(JSON.stringify(payload));
			} catch (error) {
				const err =
					error instanceof Error ? error : new Error('Failed to send WebSocket tunnel payload');
				markFailed(err);
				throw err;
			}
		},
		cleanup
	};
}

export type BaseTunnelOptions = {
	socketUrl: string;
	targetUrl: string;
	method: string;
	headers?: HeadersInit;
	body?: unknown;
	abortSignal?: AbortSignal;
};

export type TunnelSSEEvent = {
	data: string;
	event?: string;
	id?: string;
	retry?: number;
};

type TunnelStreamSession = {
	type: 'stream';
	meta: ResponseFrame;
	stream: AsyncGenerator<TunnelSSEEvent>;
};

type TunnelErrorSession = {
	type: 'error';
	meta: ResponseFrame;
	errorBody: string;
};

type TunnelSession = TunnelStreamSession | TunnelErrorSession;

function stringifyFrameData(data: unknown): string {
	if (typeof data === 'string') {
		return data;
	}

	if (data === undefined || data === null) {
		return '';
	}

	try {
		return JSON.stringify(data);
	} catch {
		return String(data);
	}
}

function frameToEvent(frame: SSEFrame | DataFrame): TunnelSSEEvent | null {
	if (frame.type === 'sse') {
		const event: TunnelSSEEvent = {
			data: stringifyFrameData(frame.data)
		};

		if (typeof frame.event === 'string' && frame.event.length) {
			event.event = frame.event;
		}

		if (typeof frame.id === 'string' && frame.id.length) {
			event.id = frame.id;
		}

		if (typeof frame.retry === 'number' && Number.isFinite(frame.retry)) {
			event.retry = frame.retry;
		}

		return event;
	}

	if (frame.type === 'data') {
		return {
			data: stringifyFrameData(frame.data)
		};
	}

	return null;
}

async function readTunnelSession(
	options: BaseTunnelOptions,
	mode: TunnelMode
): Promise<TunnelSession> {
	const connection = openTunnelConnection(options.socketUrl, options.abortSignal);
	let streaming = false;

	try {
		await connection.waitForOpen();
		connection.send({
			url: options.targetUrl,
			method: options.method,
			headers: headersInitToRecord(options.headers),
			body: options.body,
			mode
		});

		const pending: IncomingFrame[] = [];
		let meta: ResponseFrame | null = null;

		while (!meta) {
			const { value, done } = await connection.queue.next();
			if (done) {
				break;
			}

			if (!value) {
				continue;
			}

			if (value.type === 'response') {
				meta = value;
				break;
			}

			pending.push(value);
		}

		if (!meta) {
			throw new Error('WebSocket tunnel did not return response metadata');
		}

		if (mode === 'sse' && meta.status >= 200 && meta.status < 300) {
			streaming = true;
			const iterator = (async function* (): AsyncGenerator<TunnelSSEEvent> {
				try {
					for (const frame of pending) {
						if (frame.type === 'done') {
							return;
						}

						const event = frameToEvent(frame as SSEFrame | DataFrame);
						if (event) {
							yield event;
						}
					}

					while (true) {
						const { value, done } = await connection.queue.next();
						if (done) {
							break;
						}

						if (!value) {
							continue;
						}

						if (value.type === 'done') {
							break;
						}

						const event = frameToEvent(value as SSEFrame | DataFrame);
						if (event) {
							yield event;
						}
					}
				} finally {
					connection.cleanup();
				}
			})();

			return {
				type: 'stream',
				meta,
				stream: iterator
			} satisfies TunnelStreamSession;
		}

		const chunks: string[] = [];

		const appendChunk = (frame: IncomingFrame) => {
			if (frame.type === 'data' || frame.type === 'sse') {
				const payload = stringifyFrameData(frame.data);
				if (payload) {
					chunks.push(payload);
				}
			}
		};

		pending.forEach(appendChunk);

		while (true) {
			const { value, done } = await connection.queue.next();
			if (done) {
				break;
			}

			if (!value) {
				continue;
			}

			if (value.type === 'done') {
				break;
			}

			appendChunk(value);
		}

		return {
			type: 'error',
			meta,
			errorBody: chunks.join('')
		} satisfies TunnelErrorSession;
	} finally {
		// In streaming mode the iterator handles cleanup in its finally block.
		// For non-streaming/error cases ensure we close the connection here.
		if (!streaming) {
			connection.cleanup();
		}
	}
}

export async function createSSETunnelResponse(options: BaseTunnelOptions): Promise<Response> {
	const session = await readTunnelSession(options, 'sse');

	if (session.type === 'error') {
		const body = session.errorBody || session.meta.statusText || 'Upstream error';
		return new Response(body, {
			status: session.meta.status,
			statusText: session.meta.statusText ?? '',
			headers: session.meta.headers
		});
	}

	const encoder = new TextEncoder();

	const formatEvent = (event: TunnelSSEEvent): string => {
		const lines: string[] = [];

		if (event.event) {
			lines.push(`event: ${event.event}`);
		}

		if (event.id) {
			lines.push(`id: ${event.id}`);
		}

		if (event.retry !== undefined) {
			lines.push(`retry: ${event.retry}`);
		}

		const dataLines = event.data.split(/\r?\n/);
		for (const dataLine of dataLines) {
			lines.push(`data: ${dataLine}`);
		}

		lines.push('');

		return lines.join('\n');
	};

	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const pump = async () => {
				try {
					for await (const event of session.stream) {
						controller.enqueue(encoder.encode(formatEvent(event)));
					}

					controller.close();
				} catch (error) {
					controller.error(error);
				}
			};

			pump();
		},
		async cancel() {
			if (typeof session.stream.return === 'function') {
				try {
					await session.stream.return(undefined);
				} catch {
					/* noop */
				}
			}
		}
	});

	return new Response(stream, {
		status: session.meta.status,
		statusText: session.meta.statusText ?? '',
		headers: {
			'Content-Type': 'text/event-stream',
			...(session.meta.headers ?? {})
		}
	});
}

export async function createJsonTunnelResponse(options: BaseTunnelOptions): Promise<Response> {
	const session = await readTunnelSession(options, 'json');

	const body =
		session.type === 'error'
			? session.errorBody
			: await (async () => {
					const chunks: string[] = [];
					for await (const event of session.stream) {
						chunks.push(event.data);
					}
					return chunks.join('');
				})();

	return new Response(body, {
		status: session.meta.status,
		statusText: session.meta.statusText ?? '',
		headers: session.meta.headers
	});
}

export class WebSocketTunnelClient {
	private readonly socketUrl: string;

	constructor(socketUrl: string) {
		if (!socketUrl || typeof socketUrl !== 'string') {
			throw new Error('A WebSocket tunnel URL must be provided');
		}

		this.socketUrl = socketUrl;
	}

	stream(options: Omit<BaseTunnelOptions, 'socketUrl'>): Promise<Response> {
		return createSSETunnelResponse({ ...options, socketUrl: this.socketUrl });
	}

	request(options: Omit<BaseTunnelOptions, 'socketUrl'>): Promise<Response> {
		return createJsonTunnelResponse({ ...options, socketUrl: this.socketUrl });
	}
}
