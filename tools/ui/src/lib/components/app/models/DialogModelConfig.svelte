<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Slider } from '$lib/components/ui/slider';
	import { toast } from 'svelte-sonner';
	import { ModelsService } from '$lib/services/models.service';
	import { modelsStore, routerModels } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums';

	interface Props {
		open: boolean;
		modelId: string | null;
		onOpenChange: (open: boolean) => void;
	}

	let { open = $bindable(), modelId, onOpenChange }: Props = $props();

	// Below this the model is barely usable (no room for a real prompt), so
	// it's a fixed floor rather than something read from anywhere per-model.
	const MIN_CTX_SIZE = 2048;
	const FALLBACK_MAX_CTX_SIZE = 131072; // used only if the GGUF header read fails

	let currentRouterModels = $derived(routerModels());
	let model = $derived(currentRouterModels.find((m) => m.id === modelId) ?? null);
	let isRunning = $derived(
		model?.status.value === ServerModelStatus.LOADED ||
			model?.status.value === ServerModelStatus.LOADING ||
			model?.status.value === ServerModelStatus.SLEEPING
	);

	// The preset comes back as raw INI text (see server-models.cpp get_router_models) -
	// pull ctx-size out of it with a plain line match rather than a full INI parser.
	let currentCtxSize = $derived.by(() => {
		const preset = model?.status.preset;
		if (!preset) return null;
		const match = preset.match(/^ctx-size\s*=\s*(\d+)/m);
		return match ? Number(match[1]) : null;
	});

	let maxCtxSize = $state(FALLBACK_MAX_CTX_SIZE);
	let loadingMaxCtx = $state(false);
	let ctxSize = $state(MIN_CTX_SIZE);
	let saving = $state(false);

	$effect(() => {
		if (!open || !modelId) return;
		ctxSize = currentCtxSize ?? Math.min(FALLBACK_MAX_CTX_SIZE, maxCtxSize);
		loadingMaxCtx = true;
		ModelsService.getMaxContext(modelId)
			.then(({ max_context }) => {
				maxCtxSize = Math.max(MIN_CTX_SIZE, max_context);
				// clamp in case the saved value is stale (e.g. weights swapped for a
				// smaller quant since the override was last written)
				ctxSize = Math.min(ctxSize, maxCtxSize);
			})
			.catch(() => {
				// Model not yet resolvable (e.g. weights missing) - keep the fallback
				// bound rather than blocking the dialog on a failed lookup.
			})
			.finally(() => {
				loadingMaxCtx = false;
			});
	});

	async function handleSave() {
		if (!modelId) return;
		if (!Number.isInteger(ctxSize) || ctxSize < MIN_CTX_SIZE) {
			toast.error(`Context size must be at least ${MIN_CTX_SIZE}`);
			return;
		}
		saving = true;
		try {
			await ModelsService.setConfig(modelId, { ctxSize });
			toast.success(`Saved - takes effect next time ${modelId} loads`);
			await modelsStore.fetchRouterModels();
			onOpenChange(false);
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to save config');
		} finally {
			saving = false;
		}
	}
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="max-w-sm">
		<Dialog.Header>
			<Dialog.Title>Model config</Dialog.Title>
			<Dialog.Description>{modelId}</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 py-2">
			{#if isRunning}
				<div class="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-600 dark:text-amber-400">
					Unload the model first - context size is a startup setting and can't change while it's running.
				</div>
			{/if}
			<div class="space-y-3">
				<div class="flex items-center justify-between">
					<Label for="ctx-size-slider">Context size (tokens)</Label>
					<Input
						type="number"
						min={MIN_CTX_SIZE}
						max={maxCtxSize}
						bind:value={ctxSize}
						disabled={isRunning}
						class="h-7 w-28 text-right"
					/>
				</div>
				<Slider
					id="ctx-size-slider"
					bind:value={ctxSize}
					min={MIN_CTX_SIZE}
					max={maxCtxSize}
					step={256}
					disabled={isRunning}
				/>
				<div class="flex justify-between text-xs text-muted-foreground">
					<span>{MIN_CTX_SIZE.toLocaleString()}</span>
					<span>
						{loadingMaxCtx ? 'reading model max…' : `max ${maxCtxSize.toLocaleString()}`}
					</span>
				</div>
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="outline" onclick={() => onOpenChange(false)}>Cancel</Button>
			<Button onclick={handleSave} disabled={isRunning || saving}>
				{saving ? 'Saving…' : 'Save'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
