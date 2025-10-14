/*
 * Utilities to forward API requests through a WebSocket tunnel.
 * The tunnel protocol is intentionally minimal:
 *   - the client sends a single JSON descriptor (url/method/headers/body/mode)
 *   - the proxy answers with a response frame followed by zero or more data
 *     frames (SSE chunks, JSON bodies, error details, ...), finishing with a
 *     final "done" frame.
 */

type HeadersRecord = Record<string, string>;

type TunnelMode = 'json' | 'sse';

type TunnelRequestPayload = {
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
	let closed = false;
	let error: Error | null = null;
	const pending: {
		resolve: (value: IteratorResult<T>) => void;
		reject: (reason?: unknown) => void;
	}[] = [];

	return {
		push(value: T) {
			if (closed || error) return;
			if (pending.length) {
				pending.shift()!.resolve({ value, done: false });
			} else {
				values.push(value);
			}
		},
		close() {
			if (closed || error) return;
			closed = true;
			while (pending.length) {
				pending.shift()!.resolve({ value: undefined, done: true });
			}
		},
		fail(err: Error) {
			if (error) return;
			error = err;
			while (pending.length) {
				pending.shift()!.reject(err);
			}
		},
		async next() {
			if (values.length) {
				return { value: values.shift()!, done: false };
			}
			if (error) {
				throw error;
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

function headersInitToRecord(headers: HeadersInit): HeadersRecord {
	if (headers instanceof Headers) {
		return Object.fromEntries(headers.entries());
	}
	if (Array.isArray(headers)) {
		return headers.reduce((acc, [key, value]) => {
			acc[key] = value;
			return acc;
		}, {} as HeadersRecord);
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
	const ws = new WebSocket(socketUrl);
	const queue = createAsyncQueue<IncomingFrame>();
	let finished = false;
	let failed = false;
	let aborted = false;
	let abortError: Error | null = null;
	let rejectOpen: ((reason?: unknown) => void) | null = null;

	const markFinished = () => {
		if (finished || failed) return;
		finished = true;
		queue.close();
	};

	const markFailed = (error: Error) => {
		if (failed) return;
		failed = true;
		queue.fail(error);
	};

	const abortHandler = () => {
		if (aborted) return;
		aborted = true;
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
				/* ignore */
			}
		}
		if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
			try {
				ws.close(4000, 'abort');
			} catch {
				/* ignore */
			}
		}
	};

	if (abortSignal?.aborted) {
		abortHandler();
	}

	abortSignal?.addEventListener('abort', abortHandler);

	const handleRawFrame = (raw: string) => {
		if (finished || failed) return;
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
		if (finished || failed) return;
		if (typeof event.data === 'string') {
			handleRawFrame(event.data);
			return;
		}
		if (event.data instanceof Blob) {
			event.data
				.text()
				.then(handleRawFrame)
				.catch(() => {
					markFailed(new Error('Failed to read WebSocket tunnel message payload'));
				});
			return;
		}
		markFailed(new Error('Unsupported message type received from WebSocket tunnel'));
	};

	const errorHandler = () => {
		if (finished || failed) return;
		markFailed(new Error('WebSocket tunnel error'));
	};

	const closeHandler = (event: CloseEvent) => {
		if (failed || finished) return;
		if (aborted) {
			markFinished();
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
				/* ignore */
			}
		}
	};

	return {
		queue,
		waitForOpen: () =>
			openPromise.catch((err) => {
				abortSignal?.removeEventListener('abort', abortHandler);
				if (abortError) {
					throw abortError;
				}
				throw err;
			}),
		send(payload: TunnelRequestPayload) {
			try {
				ws.send(JSON.stringify(payload));
			} catch (err) {
				const error =
					err instanceof Error ? err : new Error('Failed to send WebSocket tunnel payload');
				markFailed(error);
				throw error;
			}
		},
		cleanup
	};
}

type TunnelBody =
	| { type: 'json'; data: unknown }
	| { type: 'text'; data: string }
	| { type: 'none' };

export class TunnelResponse {
	status: number;
	statusText: string;
	headers: HeadersRecord;
	ok: boolean;
	private body: TunnelBody;
	private consumed = false;

	constructor(meta: ResponseFrame, body: TunnelBody) {
		this.status = meta.status;
		this.statusText = meta.statusText ?? '';
		this.headers = meta.headers ?? {};
		this.ok = this.status >= 200 && this.status < 300;
		this.body = body;
	}

	private ensureNotConsumed() {
		if (this.consumed) {
			throw new TypeError('Body has already been consumed.');
		}
		this.consumed = true;
	}

	async json<T = unknown>(): Promise<T> {
		this.ensureNotConsumed();
		if (this.body.type === 'json') {
			return this.body.data as T;
		}
		if (this.body.type === 'text') {
			try {
				return JSON.parse(this.body.data) as T;
			} catch (err) {
				throw err instanceof Error
					? err
					: new Error('Unable to parse JSON payload received from tunnel');
			}
		}
		throw new SyntaxError('Unexpected end of JSON input');
	}

	async text(): Promise<string> {
		this.ensureNotConsumed();
		if (this.body.type === 'text') {
			return this.body.data;
		}
		if (this.body.type === 'json') {
			return JSON.stringify(this.body.data);
		}
		return '';
	}
}

export type ResponseLike = Response | TunnelResponse;

type BaseTunnelOptions = {
	socketUrl: string;
	targetUrl: string;
	method: string;
	headers: HeadersInit;
	body?: unknown;
	abortSignal?: AbortSignal;
};

export async function tunnelJsonRequest(options: BaseTunnelOptions): Promise<TunnelResponse> {
	const connection = openTunnelConnection(options.socketUrl, options.abortSignal);

	try {
		await connection.waitForOpen();
		connection.send({
			url: options.targetUrl,
			method: options.method,
			headers: headersInitToRecord(options.headers),
			body: options.body,
			mode: 'json'
		});

		let meta: ResponseFrame | null = null;
		const textChunks: string[] = [];
		let jsonPayload: unknown | undefined;

		while (true) {
			const { value, done } = await connection.queue.next();
			if (done) break;
			if (!value) continue;

			if (value.type === 'response') {
				meta = value;
				continue;
			}

			if (value.type === 'data') {
				if (value.encoding === 'json') {
					if (jsonPayload === undefined) {
						jsonPayload = value.data;
					}
				} else {
					try {
						textChunks.push(
							typeof value.data === 'string' ? value.data : JSON.stringify(value.data)
						);
					} catch {
						textChunks.push(String(value.data));
					}
				}
				continue;
			}

			if (value.type === 'done') {
				break;
			}
		}

		if (!meta) {
			throw new Error('WebSocket tunnel did not return response metadata');
		}

		let payload: TunnelBody = { type: 'none' };
		if (jsonPayload !== undefined) {
			payload = { type: 'json', data: jsonPayload };
		} else if (textChunks.length) {
			payload = { type: 'text', data: textChunks.join('') };
		}

		return new TunnelResponse(meta, payload);
	} finally {
		connection.cleanup();
	}
}

export type TunnelSSEEvent = {
	data: string;
	event?: string;
	id?: string;
	retry?: number;
};

export type TunnelSSEStreamOptions<T> = BaseTunnelOptions & {
	parser?: (event: TunnelSSEEvent) => T | undefined;
};

export async function* tunnelSSEStream<T = TunnelSSEEvent>(
	options: TunnelSSEStreamOptions<T>
): AsyncGenerator<T> {
	const connection = openTunnelConnection(options.socketUrl, options.abortSignal);

	try {
		await connection.waitForOpen();
		connection.send({
			url: options.targetUrl,
			method: options.method,
			headers: headersInitToRecord(options.headers),
			body: options.body,
			mode: 'sse'
		});

		let meta: ResponseFrame | null = null;
		let errorDetail: string | null = null;

		while (true) {
			const { value, done } = await connection.queue.next();
			if (done) break;
			if (!value) continue;

			if (value.type === 'response') {
				meta = value;
				continue;
			}

			if (value.type === 'data') {
				if (typeof value.data === 'string') {
					errorDetail = value.data;
				} else if (value.data !== undefined) {
					try {
						errorDetail = JSON.stringify(value.data);
					} catch {
						errorDetail = String(value.data);
					}
				}
				continue;
			}

			if (value.type === 'sse') {
				if (meta && (meta.status < 200 || meta.status >= 300)) {
					// Ignore SSE payload if we already know the response is an error.
					continue;
				}

				const event: TunnelSSEEvent = {
					data: typeof value.data === 'string' ? value.data : JSON.stringify(value.data ?? '')
				};

				if (typeof value.event === 'string' && value.event.length) {
					event.event = value.event;
				}
				if (typeof value.id === 'string' && value.id.length) {
					event.id = value.id;
				}
				if (typeof value.retry === 'number' && Number.isFinite(value.retry)) {
					event.retry = value.retry;
				}

				if (options.parser) {
					const mapped = options.parser(event);
					if (mapped !== undefined) {
						yield mapped;
					}
				} else {
					yield event as unknown as T;
				}
				continue;
			}

			if (value.type === 'done') {
				break;
			}
		}

		if (!meta) {
			throw new Error('WebSocket tunnel did not return response metadata');
		}

		if (meta.status < 200 || meta.status >= 300) {
			const reason = errorDetail?.trim().length
				? errorDetail
				: meta.statusText || `Upstream error ${meta.status}`;
			throw new Error(reason);
		}
	} finally {
		connection.cleanup();
	}
}

export class WebSocketTunnelClient {
	private readonly socketUrl: string;

	constructor(socketUrl: string) {
		if (!socketUrl || typeof socketUrl !== 'string') {
			throw new Error('A WebSocket tunnel URL must be provided');
		}
		this.socketUrl = socketUrl;
	}

	request(options: Omit<BaseTunnelOptions, 'socketUrl'>) {
		return tunnelJsonRequest({ ...options, socketUrl: this.socketUrl });
	}

	stream<T = TunnelSSEEvent>(
		options: Omit<TunnelSSEStreamOptions<T>, 'socketUrl'>
	): AsyncGenerator<T> {
		return tunnelSSEStream<T>({ ...options, socketUrl: this.socketUrl });
	}
}
