import {
	AUDIO_FORMAT,
	EVENT,
	FRAME_SAMPLES,
	FRAMES_PER_APPEND,
	PCM16_MAX,
	PCM16_SCALE,
	PREBUFFER_MS,
	REALTIME_PATH,
	SAMPLE_RATE,
	WAV_CHANNELS,
	WAV_HEADER_BYTES,
	WAV_MIME,
	WORKLET_MIME,
	WORKLET_NAME
} from './constants';

// One worklet carries the audio both ways. It runs at the rate of the device, so the browser
// echo canceller hears the playback in the same graph as the microphone, and converts to and
// from the stream rate by linear interpolation, averaging two device samples on capture when
// the device runs faster. It ships 20 ms of the microphone with the samples played during the
// same render quanta, and after an underrun it waits for `prebuffer` samples before it plays.
const WORKLET_SOURCE = `
const RATIO = sampleRate / ${SAMPLE_RATE};
const SMOOTH = RATIO > 1;

class Duplex extends AudioWorkletProcessor {
	constructor() {
		super();
		this.mic = new Float32Array(${FRAME_SAMPLES});
		this.ref = new Float32Array(${FRAME_SAMPLES});
		this.filled = 0;
		this.queue = [];
		this.offset = 0;
		this.queued = 0;
		this.prebuffer = 0;
		this.waiting = true;
		this.underruns = 0;
		this.volume = 1;
		this.a = 0;
		this.b = 0;
		this.at = 1;
		this.rawMic = 0;
		this.rawRef = 0;
		this.prevMic = 0;
		this.prevRef = 0;
		this.next = 1;
		this.port.onmessage = (e) => {
			const data = e.data;
			if (data instanceof Float32Array) {
				this.queue.push(data);
				this.queued += data.length;
			} else if (data.prebuffer !== undefined) {
				this.prebuffer = data.prebuffer;
			} else if (data.volume !== undefined) {
				this.volume = data.volume;
			}
		};
	}
	pull() {
		if (this.waiting) {
			if (this.queued === 0 || this.queued < this.prebuffer) return 0;
			this.waiting = false;
		}
		if (this.queue.length === 0) {
			this.waiting = true;
			this.underruns++;
			return 0;
		}
		const chunk = this.queue[0];
		const sample = chunk[this.offset++];
		this.queued--;
		if (this.offset === chunk.length) {
			this.queue.shift();
			this.offset = 0;
		}
		return sample;
	}
	capture(mic, ref) {
		if (SMOOTH) {
			const m = mic;
			const r = ref;
			mic = (mic + this.rawMic) * 0.5;
			ref = (ref + this.rawRef) * 0.5;
			this.rawMic = m;
			this.rawRef = r;
		}
		while (this.next <= 1) {
			this.mic[this.filled] = this.prevMic + (mic - this.prevMic) * this.next;
			this.ref[this.filled] = this.prevRef + (ref - this.prevRef) * this.next;
			if (++this.filled === this.mic.length) {
				this.port.postMessage({ mic: this.mic.slice(), ref: this.ref.slice(), queued: this.queued, underruns: this.underruns });
				this.filled = 0;
			}
			this.next += RATIO;
		}
		this.next -= 1;
		this.prevMic = mic;
		this.prevRef = ref;
	}
	process(inputs, outputs) {
		const output = outputs[0][0];
		const input = inputs[0] && inputs[0][0];
		for (let i = 0; i < output.length; i++) {
			while (this.at >= 1) {
				this.a = this.b;
				this.b = this.pull();
				this.at -= 1;
			}
			output[i] = (this.a + (this.b - this.a) * this.at) * this.volume;
			this.at += 1 / RATIO;
			this.capture(input ? input[i] : 0, output[i]);
		}
		return true;
	}
}
registerProcessor('${WORKLET_NAME}', Duplex);
`;

// Plain true only covers WebRTC remote tracks; lib.dom still types the constraint as a boolean.
const ECHO_CANCELLATION_ALL = 'all' as unknown as ConstrainBoolean;

interface WorkletFrame {
	mic: Float32Array;
	ref: Float32Array;
	queued: number;
	underruns: number;
}

interface ServerEvent {
	type?: string;
	delta?: string;
	error?: { message?: string };
	session?: { model?: string };
}

export interface DuplexStats {
	/** seconds of microphone sent, and of translation received */
	sent: number;
	received: number;
	/** ms of translation waiting in the speaker */
	queued: number;
	underruns: number;
}

export interface DuplexHandlers {
	model: (name: string) => void;
	/** a response opens: the text that follows belongs to it until it closes */
	open: () => void;
	text: (delta: string) => void;
	close: () => void;
	/** one 20 ms frame: the peaks of the microphone and of what played */
	frame: (input: number, output: number, stats: DuplexStats) => void;
	/** the session ended, by stop() or by the server */
	end: (reason: string) => void;
	error: (message: string) => void;
}

function peak(block: Float32Array): number {
	let max = 0;
	for (let i = 0; i < block.length; i++) max = Math.max(max, Math.abs(block[i]));
	return max;
}

function toPcm16(sample: number): number {
	return Math.max(-1, Math.min(1, sample)) * PCM16_MAX;
}

function toBase64(pcm: Float32Array): string {
	const bytes = new Uint8Array(pcm.length * 2);
	const view = new DataView(bytes.buffer);
	for (let i = 0; i < pcm.length; i++) view.setInt16(i * 2, toPcm16(pcm[i]), true);
	let binary = '';
	for (let i = 0; i < bytes.length; i += 0x8000) {
		binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}
	return btoa(binary);
}

function fromBase64(data: string): Float32Array {
	const binary = atob(data);
	const view = new DataView(Uint8Array.from(binary, (c) => c.charCodeAt(0)).buffer);
	const pcm = new Float32Array(binary.length / 2);
	for (let i = 0; i < pcm.length; i++) pcm[i] = view.getInt16(i * 2, true) / PCM16_SCALE;
	return pcm;
}

/** v1/realtime under the page, ws or wss to match http or https: the page is served by llama-server. */
function realtimeUrl(): string {
	const url = new URL(REALTIME_PATH, location.href);
	url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
	return url.toString();
}

/**
 * A full duplex session on llama-server's realtime socket: the microphone streams in, the
 * translation plays as it comes back. The first audio opens a response on the server and
 * cancel() closes it, the next audio opens another one. Everything heard and played is kept,
 * in step, for wav().
 */
export class Duplex {
	private context: AudioContext | null = null;
	private stream: MediaStream | null = null;
	private node: AudioWorkletNode | null = null;
	private ws: WebSocket | null = null;
	private pending: Float32Array[] = [];
	private recordedIn: Float32Array[] = [];
	private recordedOut: Float32Array[] = [];
	private sent = 0;
	private received = 0;

	constructor(private handlers: DuplexHandlers) {}

	get recorded(): boolean {
		return this.recordedIn.length > 0;
	}

	/** Must run inside a user gesture, in a secure context. */
	async start(echoCancellation: boolean, volume: number): Promise<void> {
		this.pending = [];
		this.recordedIn = [];
		this.recordedOut = [];
		this.sent = 0;
		this.received = 0;

		try {
			const context = new AudioContext();
			this.context = context;
			const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: WORKLET_MIME }));
			await context.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url));

			this.stream = await navigator.mediaDevices.getUserMedia({
				audio: { echoCancellation: echoCancellation ? ECHO_CANCELLATION_ALL : false, channelCount: 1 }
			});

			const node = new AudioWorkletNode(context, WORKLET_NAME, {
				numberOfInputs: 1,
				numberOfOutputs: 1,
				outputChannelCount: [1]
			});
			this.node = node;
			node.port.postMessage({ prebuffer: Math.round((PREBUFFER_MS * SAMPLE_RATE) / 1000) });
			node.port.postMessage({ volume });
			node.port.onmessage = (e: MessageEvent<WorkletFrame>) => this.onFrame(e.data);
			context.createMediaStreamSource(this.stream).connect(node);
			node.connect(context.destination);

			await this.connect();
		} catch (e) {
			this.teardown();
			throw e;
		}
	}

	stop(): void {
		this.send({ type: EVENT.CANCEL });
		this.teardown();
		this.handlers.end('stopped');
	}

	/** Closes the response in flight; the next audio opens a fresh one, with its context reset. */
	cancel(): void {
		this.send({ type: EVENT.CANCEL });
	}

	setVolume(volume: number): void {
		this.node?.port.postMessage({ volume });
	}

	/** The session so far as a stereo WAV: the microphone left, the translation right. */
	wav(): Blob {
		const frames = this.recordedIn.length * FRAME_SAMPLES;
		const bytes = frames * WAV_CHANNELS * 2;
		const view = new DataView(new ArrayBuffer(WAV_HEADER_BYTES + bytes));
		const ascii = (offset: number, text: string) => {
			for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
		};
		ascii(0, 'RIFF');
		view.setUint32(4, WAV_HEADER_BYTES - 8 + bytes, true);
		ascii(8, 'WAVE');
		ascii(12, 'fmt ');
		view.setUint32(16, 16, true);
		view.setUint16(20, 1, true);
		view.setUint16(22, WAV_CHANNELS, true);
		view.setUint32(24, SAMPLE_RATE, true);
		view.setUint32(28, SAMPLE_RATE * WAV_CHANNELS * 2, true);
		view.setUint16(32, WAV_CHANNELS * 2, true);
		view.setUint16(34, 16, true);
		ascii(36, 'data');
		view.setUint32(40, bytes, true);

		let offset = WAV_HEADER_BYTES;
		this.recordedIn.forEach((input, f) => {
			const output = this.recordedOut[f];
			for (let i = 0; i < FRAME_SAMPLES; i++) {
				view.setInt16(offset, toPcm16(input[i]), true);
				view.setInt16(offset + 2, toPcm16(output[i]), true);
				offset += 4;
			}
		});
		return new Blob([view.buffer], { type: WAV_MIME });
	}

	private connect(): Promise<void> {
		return new Promise((resolve, reject) => {
			const ws = new WebSocket(realtimeUrl());
			this.ws = ws;
			ws.onopen = () => {
				this.send({
					type: EVENT.SESSION_UPDATE,
					session: { audio: { input: { format: { type: AUDIO_FORMAT, rate: SAMPLE_RATE } } } }
				});
				resolve();
			};
			ws.onerror = () => reject(new Error(`cannot reach ${ws.url}`));
			ws.onclose = (e) => {
				if (ws !== this.ws) return;
				this.teardown();
				this.handlers.end(`connection closed, code ${e.code}${e.reason ? `, ${e.reason}` : ''}`);
			};
			ws.onmessage = (e) => this.onEvent(JSON.parse(e.data as string) as ServerEvent);
		});
	}

	private onFrame({ mic, ref, queued, underruns }: WorkletFrame): void {
		this.recordedIn.push(mic);
		this.recordedOut.push(ref);
		this.pending.push(mic);

		if (this.pending.length === FRAMES_PER_APPEND) {
			const pcm = new Float32Array(FRAME_SAMPLES * FRAMES_PER_APPEND);
			this.pending.forEach((frame, i) => pcm.set(frame, i * FRAME_SAMPLES));
			this.pending = [];
			this.send({ type: EVENT.APPEND, audio: toBase64(pcm) });
			this.sent += pcm.length;
		}

		this.handlers.frame(peak(mic), peak(ref), {
			sent: this.sent / SAMPLE_RATE,
			received: this.received / SAMPLE_RATE,
			queued: (queued * 1000) / SAMPLE_RATE,
			underruns
		});
	}

	private onEvent(event: ServerEvent): void {
		switch (event.type) {
			case EVENT.SESSION_CREATED:
				if (event.session?.model) this.handlers.model(event.session.model);
				break;
			case EVENT.RESPONSE_CREATED:
				this.handlers.open();
				break;
			case EVENT.TEXT_DELTA:
				if (event.delta) this.handlers.text(event.delta);
				break;
			case EVENT.AUDIO_DELTA:
				if (event.delta) {
					const pcm = fromBase64(event.delta);
					this.received += pcm.length;
					this.node?.port.postMessage(pcm, [pcm.buffer]);
				}
				break;
			case EVENT.RESPONSE_DONE:
				this.handlers.close();
				break;
			case EVENT.ERROR:
				this.handlers.error(event.error?.message ?? 'server error');
				break;
		}
	}

	private send(event: object): void {
		if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(event));
	}

	private teardown(): void {
		const ws = this.ws;
		this.ws = null;
		ws?.close();
		this.stream?.getTracks().forEach((track) => track.stop());
		this.stream = null;
		this.context?.close();
		this.context = null;
		this.node = null;
	}
}
