<script lang="ts">
	import {
		CircleAlert,
		Heart,
		HeartOff,
		Info,
		Loader2,
		Power,
		PowerOff,
		RotateCw,
		Settings
	} from '@lucide/svelte';
	import { ActionIcon, ModelId, DialogModelConfig } from '$lib/components/app';
	import ModelLoadHighlight from './ModelLoadHighlight.svelte';
	import type { ModelOption } from '$lib/types/models';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, routerModels } from '$lib/stores/models.svelte';
	import { modelLoadFraction, modelLoadProgressText } from '$lib/utils';

	let configDialogOpen = $state(false);

	interface Props {
		option: ModelOption;
		isSelected: boolean;
		isHighlighted: boolean;
		isFav: boolean;
		hideOrgName?: boolean;
		onSelect: (modelId: string) => void;
		onMouseEnter: () => void;
		onKeyDown: (e: KeyboardEvent) => void;
		onInfoClick?: (modelName: string) => void;
	}

	let {
		option,
		isSelected,
		isHighlighted,
		isFav,
		hideOrgName = false,
		onSelect,
		onMouseEnter,
		onKeyDown,
		onInfoClick
	}: Props = $props();

	let currentRouterModels = $derived(routerModels());
	let matchingRouterModel = $derived(currentRouterModels.find((m) => m.id === option.model));
	let serverStatus = $derived.by(() => {
		return (matchingRouterModel?.status?.value as ServerModelStatus) ?? null;
	});
	let isOperationInProgress = $derived(modelsStore.isModelOperationInProgress(option.model));
	// The backend only ever reports status.value as unloaded/loading/loaded/sleeping - a
	// failed load attempt stays "unloaded" but adds status.failed=true + status.exit_code
	// (see server-models.cpp meta.is_failed()). That's the signal for whether this preset
	// will actually run on our cluster config, so the row tint below keys off it too.
	let hasFailed = $derived(
		serverStatus === ServerModelStatus.FAILED || Boolean(matchingRouterModel?.status?.failed)
	);
	let isFailed = $derived(hasFailed);
	let isSleeping = $derived(serverStatus === ServerModelStatus.SLEEPING);
	let isLoaded = $derived(
		(serverStatus === ServerModelStatus.LOADED || isSleeping) && !isOperationInProgress
	);
	let isLoading = $derived(serverStatus === ServerModelStatus.LOADING || isOperationInProgress);

	let loadProgress = $derived(isLoading ? modelsStore.getLoadProgress(option.model) : null);
	let loadPercent = $derived(Math.round(modelLoadFraction(loadProgress) * 100));
	let loadTitle = $derived(modelLoadProgressText(loadProgress));
</script>

<div
	class={[
		'group relative flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
		'cursor-pointer hover:bg-muted focus:bg-muted',
		(isSelected || isHighlighted) && 'bg-accent text-accent-foreground',
		!(isSelected || isHighlighted) && 'hover:bg-accent hover:text-accent-foreground',
		// Translucent cluster-fit tint: green = this preset is expected to load fine on
		// our config, red = it already failed to load (incompatible/OOM/etc). Kept subtle
		// (low opacity) so it doesn't fight the selection/hover highlight or text contrast.
		// Resting-state only (no hover: variant) - it'd compete on specificity with the
		// hover-bg classes above at equal Tailwind-utility specificity.
		!(isSelected || isHighlighted) && (hasFailed ? 'bg-red-500/10' : 'bg-green-500/5'),
		isLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
	]}
	role="option"
	aria-selected={isSelected || isHighlighted}
	title={loadTitle}
	tabindex="0"
	onclick={() => onSelect(option.id)}
	onmouseenter={onMouseEnter}
	onkeydown={onKeyDown}
>
	<ModelId
		modelId={option.model}
		{hideOrgName}
		aliases={option.aliases}
		tags={option.tags}
		class="flex-1"
	/>

	<div class="flex shrink-0 items-center gap-1">
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class="pointer-events-none flex items-center justify-center gap-0.75 pl-2 opacity-0 group-hover:pointer-events-auto group-hover:opacity-100 [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
			onclick={(e) => e.stopPropagation()}
		>
			{#if isFav}
				<ActionIcon
					iconSize="h-2.5 w-2.5"
					icon={HeartOff}
					tooltip="Remove from favorites"
					class="h-3 w-3 hover:text-foreground"
					onclick={() => modelsStore.toggleFavorite(option.model)}
				/>
			{:else}
				<ActionIcon
					iconSize="h-2.5 w-2.5"
					icon={Heart}
					tooltip="Add to favorites"
					class="h-3 w-3 hover:text-foreground"
					onclick={() => modelsStore.toggleFavorite(option.model)}
				/>
			{/if}

			<!-- info button: only shown when model is loaded and callback is provided -->
			{#if isLoaded && onInfoClick}
				<ActionIcon
					iconSize="h-2.5 w-2.5"
					icon={Info}
					tooltip="Model information"
					class="h-3 w-3 hover:text-foreground"
					onclick={() => onInfoClick(option.model)}
				/>
			{/if}

			<ActionIcon
				iconSize="h-2.5 w-2.5"
				icon={Settings}
				tooltip="Model config (context size)"
				class="h-3 w-3 hover:text-foreground"
				onclick={(e) => {
					e?.stopPropagation();
					configDialogOpen = true;
				}}
			/>
		</div>

		{#if isLoading}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-5">
				<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
			</div>
		{:else if isFailed}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<CircleAlert
					class="h-3.5 w-3.5 text-red-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
				/>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={RotateCw}
						tooltip="Retry loading model"
						class="h-3 w-3 text-red-500 hover:text-foreground"
						onclick={() => modelsStore.loadModel(option.model)}
						stopPropagationOnClick
					/>
				</div>
			</div>
		{:else if isSleeping}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<span
					class="h-2 w-2 rounded-full bg-orange-400 group-hover:hidden [@media(pointer:coarse)]:hidden"
				></span>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={PowerOff}
						tooltip="Unload model"
						class="h-3 w-3 text-red-500 hover:text-red-600 [@media(pointer:coarse)]:text-amber-500 [@media(pointer:coarse)]:hover:text-amber-600"
						onclick={(e) => {
							e?.stopPropagation();
							modelsStore.unloadModel(option.model);
						}}
					/>
				</div>
			</div>
		{:else if isLoaded}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<span
					class="h-2 w-2 rounded-full bg-green-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
				></span>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={PowerOff}
						tooltip="Unload model"
						class="h-3 w-3 text-red-500 hover:text-red-600 [@media(pointer:coarse)]:text-green-500 [@media(pointer:coarse)]:hover:text-green-600"
						onclick={() => modelsStore.unloadModel(option.model)}
						stopPropagationOnClick
					/>
				</div>
			</div>
		{:else}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<span
					class="h-2 w-2 rounded-full bg-muted-foreground/50 group-hover:hidden [@media(pointer:coarse)]:hidden"
				></span>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						iconSize="h-2.5 w-2.5"
						icon={Power}
						tooltip="Load model"
						class="h-3 w-3 [@media(pointer:coarse)]:text-muted-foreground"
						onclick={() => modelsStore.loadModel(option.model)}
						stopPropagationOnClick
					/>
				</div>
			</div>
		{/if}
	</div>

	{#if isLoading}
		<ModelLoadHighlight percent={loadPercent} />
	{/if}
</div>

<DialogModelConfig
	bind:open={configDialogOpen}
	modelId={option.model}
	onOpenChange={(v) => (configDialogOpen = v)}
/>
