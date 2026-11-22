<script lang="ts">
	import { onMount } from 'svelte';
	import { LEVELS_LENGTH, LEVELS_RANGE_DB } from './lib/constants';
	import { Duplex, type DuplexStats } from './lib/duplex';

	let running = $state(false);
	let starting = $state(false);
	let responding = $state(false);
	let recorded = $state(false);
	let echoCancellation = $state(true);
	let volume = $state(1);
	let model = $state('');
	let segments = $state<string[]>([]);
	let stats = $state<DuplexStats | null>(null);
	let status = $state('');
	let failed = $state(false);

	let transcript: HTMLDivElement;
	let inputCanvas: HTMLCanvasElement;
	let outputCanvas: HTMLCanvasElement;

	const inputLevels = new Float32Array(LEVELS_LENGTH);
	const outputLevels = new Float32Array(LEVELS_LENGTH);
	let levelAt = 0;

	function report(message: string, error = false) {
		status = message;
		failed = error;
	}

	function append(text: string) {
		if (segments.length === 0) segments.push('');
		segments[segments.length - 1] += text;
		queueMicrotask(() => transcript?.scrollTo({ top: transcript.scrollHeight }));
	}

	const duplex = new Duplex({
		model: (name) => (model = name),
		open: () => {
			responding = true;
			segments.push('');
		},
		text: append,
		close: () => (responding = false),
		frame: (input, output, now) => {
			inputLevels[levelAt] = input;
			outputLevels[levelAt] = output;
			levelAt = (levelAt + 1) % LEVELS_LENGTH;
			stats = now;
		},
		end: (reason) => {
			running = false;
			responding = false;
			recorded = duplex.recorded;
			report(reason, reason !== 'stopped');
		},
		error: (message) => report(message, true)
	});

	async function toggle() {
		if (running) {
			duplex.stop();
			return;
		}
		if (!isSecureContext) {
			report('the microphone needs https, or localhost', true);
			return;
		}
		starting = true;
		try {
			await duplex.start(echoCancellation, volume);
			running = true;
			recorded = false;
			report('listening');
		} catch (e) {
			report(`start failed: ${e instanceof Error ? e.message : String(e)}`, true);
		} finally {
			starting = false;
		}
	}

	function download() {
		const a = document.createElement('a');
		a.href = URL.createObjectURL(duplex.wav());
		a.download = `duplex-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.wav`;
		a.click();
		URL.revokeObjectURL(a.href);
	}

	function draw(canvas: HTMLCanvasElement, levels: Float32Array, color: string) {
		const ratio = devicePixelRatio;
		const width = canvas.clientWidth * ratio;
		const height = canvas.clientHeight * ratio;
		if (canvas.width !== width || canvas.height !== height) {
			canvas.width = width;
			canvas.height = height;
		}
		const g = canvas.getContext('2d')!;
		g.clearRect(0, 0, width, height);
		g.fillStyle = color;
		const bar = width / LEVELS_LENGTH;
		for (let i = 0; i < LEVELS_LENGTH; i++) {
			const level = levels[(levelAt + i) % LEVELS_LENGTH];
			const db = level > 0 ? (20 * Math.log10(level)) / LEVELS_RANGE_DB + 1 : 0;
			const h = Math.max(ratio, Math.max(0, db) * height);
			g.fillRect(i * bar, (height - h) / 2, Math.max(ratio, bar - ratio), h);
		}
	}

	onMount(() => {
		let frame = requestAnimationFrame(function loop() {
			const style = getComputedStyle(document.documentElement);
			draw(inputCanvas, inputLevels, style.getPropertyValue('--fg-info'));
			draw(outputCanvas, outputLevels, style.getPropertyValue('--fg-warning'));
			frame = requestAnimationFrame(loop);
		});
		return () => cancelAnimationFrame(frame);
	});
</script>

<main>
	<header>
		<h1>Llama Duplex</h1>
		<span class="muted">{model || 'live speech translation'}</span>
	</header>

	<div class="controls">
		<button class="primary" class:danger={running} disabled={starting} onclick={toggle}>
			{running ? 'Stop' : 'Start'}
		</button>
		<button
			disabled={!running}
			onclick={() => duplex.cancel()}
			title="Close the stream; the next audio opens a fresh one"
		>
			New segment
		</button>
		<button
			disabled={!recorded}
			onclick={download}
			title="Stereo WAV: the microphone left, the translation right, in step"
		>
			Download WAV
		</button>
		<label title="Off with headphones: the browser leaves the microphone as it is">
			<input type="checkbox" bind:checked={echoCancellation} disabled={running} />
			Echo cancellation
		</label>
		<label>
			Volume
			<input
				type="range"
				min="0"
				max="1.5"
				step="0.05"
				value={volume}
				oninput={(e) => duplex.setVolume((volume = Number(e.currentTarget.value)))}
			/>
		</label>
		<span class="status" class:live={running} class:failed>{status}</span>
	</div>

	<div class="meters">
		<section>
			<h2>Microphone</h2>
			<canvas bind:this={inputCanvas}></canvas>
			<p class="muted mono">{stats ? `sent ${stats.sent.toFixed(1)} s` : ''}</p>
		</section>
		<section>
			<h2>Translation</h2>
			<canvas bind:this={outputCanvas}></canvas>
			<p class="muted mono">
				{stats
					? `received ${stats.received.toFixed(1)} s, queued ${stats.queued.toFixed(0)} ms, underruns ${stats.underruns}`
					: ''}
			</p>
		</section>
	</div>

	<div class="transcript" bind:this={transcript}>
		{#each segments as segment, i (i)}
			<p class:current={responding && i === segments.length - 1}>{segment}</p>
		{:else}
			<p class="muted">Speak French: the English translation shows here as it is spoken.</p>
		{/each}
	</div>

	<footer class="muted">
		A full duplex model on llama-server's <code>/v1/realtime</code>: every 80 ms of speech in gives
		80 ms of speech out, a few seconds behind, the delay the model takes to hear enough.
	</footer>
</main>

<style>
	main {
		max-width: 1080px;
		margin: 0 auto;
		padding: 24px 20px 40px;
	}

	header {
		display: flex;
		align-items: baseline;
		gap: 12px;
		flex-wrap: wrap;
	}

	h1 {
		margin: 0;
		font-size: 22px;
		font-weight: 600;
	}

	h2 {
		margin: 0 0 8px;
		font-size: 13px;
		font-weight: 500;
		color: var(--fg-muted);
	}

	.muted {
		color: var(--fg-muted);
		font-size: 13px;
	}

	.mono {
		font-family: 'Llama Mono', ui-monospace, monospace;
		font-size: 12px;
		min-height: 1.5em;
		margin: 6px 0 0;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
		margin: 18px 0;
	}

	button {
		font: inherit;
		padding: 7px 14px;
		border-radius: 8px;
		border: 1px solid var(--line-control);
		background: var(--surface-raised);
		color: var(--fg-default);
		cursor: pointer;
	}

	button:disabled {
		opacity: 0.45;
		cursor: default;
	}

	button.primary {
		min-width: 96px;
		border-color: transparent;
		background: var(--action-primary);
		color: var(--fg-on-action);
	}

	button.primary.danger {
		background: var(--action-danger);
		color: var(--fg-on-danger);
	}

	label {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--fg-muted);
		font-size: 13px;
	}

	.status {
		margin-left: auto;
		color: var(--fg-muted);
		font-size: 13px;
	}

	.status.live {
		color: var(--fg-success);
	}

	.status.failed {
		color: var(--fg-danger);
	}

	.meters {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}

	section,
	.transcript {
		background: var(--surface-raised);
		border: 1px solid var(--line-default);
		border-radius: 12px;
		padding: 14px 16px;
	}

	canvas {
		display: block;
		width: 100%;
		height: 80px;
		border-radius: 8px;
		background: var(--surface-muted);
	}

	.transcript {
		margin-top: 16px;
		min-height: 200px;
		max-height: 50vh;
		overflow-y: auto;
		font-size: 20px;
	}

	.transcript p {
		margin: 0 0 10px;
	}

	.transcript p.current::after {
		content: '\258d';
		color: var(--fg-warning);
	}

	footer {
		margin-top: 16px;
	}

	@media (max-width: 720px) {
		.meters {
			grid-template-columns: 1fr;
		}

		.status {
			margin-left: 0;
		}
	}
</style>
