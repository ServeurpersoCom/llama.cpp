import { getJsonHeaders } from '$lib/utils/api-headers';

// Qwen3-Omni text to speech, low latency streaming player. The server streams pcm16 mono
// frames as the talker generates them, in windows of a few seconds. On a slow machine the
// windows arrive less often than they play, so a small jitter buffer absorbs the irregular
// arrival and the scheduler chains every block on a monotonic play head that never rewinds.

const TTS_SAMPLE_RATE = 24000;

// seconds of audio to accumulate before starting playback, absorbs arrival jitter so a late
// window does not create a gap. larger is safer on slow machines, smaller starts sooner.
const TTS_PREBUFFER_SECONDS = 0.3;

const TTS_ENDPOINT = './v1/audio/speech';

export interface TtsOptions {
	voice?: string;
	// model name to speak with, required in router mode so the proxy targets the talker
	model?: string;
	signal?: AbortSignal;
	onFirstSound?: (latencySeconds: number) => void;
}

// drives one playback. call stop() to cancel network and silence the output.
export class TtsStreamPlayer {
	private ctx: AudioContext | null = null;
	private controller = new AbortController();
	private sources: AudioBufferSourceNode[] = [];

	// next start time on the audio clock, only ever moves forward
	private playHead = 0;
	private started = false;

	// pcm16 little endian, a stray odd byte is carried to the next chunk
	private carry = new Uint8Array(0);

	async speak(text: string, options: TtsOptions = {}): Promise<void> {
		const signal = options.signal
			? anySignal(options.signal, this.controller.signal)
			: this.controller.signal;

		this.ctx = new AudioContext({ sampleRate: TTS_SAMPLE_RATE });

		// browsers start the context suspended until a user gesture, resume explicitly. the
		// speak call sits behind a button click so this resolves to running.
		if (this.ctx.state === 'suspended') {
			await this.ctx.resume();
		}

		const t0 = performance.now();

		const body: Record<string, unknown> = {
			input: text,
			voice: options.voice ?? 'ethan',
			response_format: 'pcm'
		};
		if (options.model) body.model = options.model;

		const response = await fetch(TTS_ENDPOINT, {
			method: 'POST',
			headers: getJsonHeaders(),
			body: JSON.stringify(body),
			signal
		});
		if (!response.ok || !response.body) {
			throw new Error(`tts request failed: ${response.status}`);
		}

		const reader = response.body.getReader();

		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			if (!value) continue;

			// prepend any half sample carried from the previous chunk
			let bytes: Uint8Array = value;
			if (this.carry.length) {
				const merged = new Uint8Array(this.carry.length + value.length);
				merged.set(this.carry, 0);
				merged.set(value, this.carry.length);
				bytes = merged;
				this.carry = new Uint8Array(0);
			}
			const usable = bytes.length - (bytes.length % 2);
			if (usable < bytes.length) this.carry = bytes.slice(usable);
			if (usable === 0) continue;

			const samples = new Int16Array(bytes.buffer, bytes.byteOffset, usable / 2);
			this.schedule(samples);

			if (!this.started) {
				this.started = true;
				options.onFirstSound?.((performance.now() - t0) / 1000);
			}
		}
	}

	stop(): void {
		this.controller.abort();
		for (const s of this.sources) {
			try {
				s.stop();
			} catch {
				// already stopped, ignore
			}
		}
		this.sources = [];
		if (this.ctx) {
			void this.ctx.close();
			this.ctx = null;
		}
	}

	// schedule one pcm block on the play head. the first block starts after a small
	// prebuffer, later blocks chain on the running head. if a block arrives after the head
	// has already played out, the head catches up to now so the next block plays immediately
	// rather than scheduling in the past.
	private schedule(samples: Int16Array): void {
		if (!this.ctx || samples.length === 0) return;

		const buffer = this.ctx.createBuffer(1, samples.length, TTS_SAMPLE_RATE);
		const channel = buffer.getChannelData(0);
		for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;

		const now = this.ctx.currentTime;
		if (!this.started) {
			this.playHead = now + TTS_PREBUFFER_SECONDS;
		} else if (this.playHead < now) {
			// the talker fell behind playback, resume from now, a short silence is unavoidable
			this.playHead = now;
		}

		const source = this.ctx.createBufferSource();
		source.buffer = buffer;
		source.connect(this.ctx.destination);
		source.start(this.playHead);
		this.playHead += buffer.duration;

		this.sources.push(source);
		source.onended = () => {
			const i = this.sources.indexOf(source);
			if (i >= 0) this.sources.splice(i, 1);
		};
	}
}

// merge several abort signals into one, aborts when any of them aborts
function anySignal(...signals: AbortSignal[]): AbortSignal {
	const controller = new AbortController();
	for (const s of signals) {
		if (s.aborted) {
			controller.abort();
			break;
		}
		s.addEventListener('abort', () => controller.abort(), { once: true });
	}
	return controller.signal;
}
