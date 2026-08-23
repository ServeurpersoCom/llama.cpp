<script lang="ts">
	import { ChevronLeft, Download, HardDriveDownload, Search, X } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { Input } from '$lib/components/ui/input';
	import { HF_SEARCH_DEBOUNCE_MS, ROUTES } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelSummary, HfModelVariant } from '$lib/types';
	import { formatFileSize } from '$lib/utils';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let query = $state('');
	let results = $state<HfModelSummary[]>([]);
	let selectedRepo = $state<string | null>(null);
	let variants = $state<HfModelVariant[]>([]);
	let searching = $state(false);
	let loadingVariants = $state(false);
	let addingTag = $state<string | null>(null);
	let error = $state<string | null>(null);

	let previousRouteId = $state<string | null>(null);

	let searchTimer: ReturnType<typeof setTimeout> | null = null;
	let searchAbort: AbortController | null = null;
	let variantsAbort: AbortController | null = null;

	$effect(() => {
		const currentId = page.route.id;

		return () => {
			previousRouteId = currentId;
		};
	});

	function handleClose() {
		const prevIsModels = previousRouteId === '/models';

		if (browser && window.history.length > 1 && !prevIsModels) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}

	async function runSearch(value: string) {
		searchAbort?.abort();
		searchAbort = new AbortController();
		searching = true;
		error = null;

		try {
			results = await HuggingFaceService.searchModels(value, searchAbort.signal);
		} catch (e) {
			if ((e as Error).name === 'AbortError') return;

			error = (e as Error).message;
			results = [];
		} finally {
			searching = false;
		}
	}

	function handleQueryInput(value: string) {
		query = value;
		selectedRepo = null;
		variants = [];

		if (searchTimer) clearTimeout(searchTimer);

		searchTimer = setTimeout(() => void runSearch(value), HF_SEARCH_DEBOUNCE_MS);
	}

	async function selectRepo(repoId: string) {
		selectedRepo = repoId;
		variantsAbort?.abort();
		variantsAbort = new AbortController();
		loadingVariants = true;
		error = null;

		try {
			variants = await HuggingFaceService.listVariants(repoId, variantsAbort.signal);
		} catch (e) {
			if ((e as Error).name === 'AbortError') return;

			error = (e as Error).message;
			variants = [];
		} finally {
			loadingVariants = false;
		}
	}

	async function addVariant(variant: HfModelVariant) {
		error = null;
		addingTag = variant.tag;

		try {
			await ModelsService.add(variant.tag);
			await modelsStore.fetch(true);
			goto(ROUTES.START);
		} catch (e) {
			error = (e as Error).message;
		} finally {
			addingTag = null;
		}
	}
</script>

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div
		class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 px-4 p-0 md:justify-between md:p-4 md:px-8"
	>
		<div class="flex items-center gap-2">
			<HardDriveDownload class="h-5 w-5 md:h-6 md:w-6" />

			<h1 class="text-lg font-semibold md:text-2xl">Add a model</h1>
		</div>
	</div>

	<div class="{className} flex flex-col gap-4">
		<div class="relative">
			<Search
				class="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted-foreground"
			/>

			<Input
				class="pl-9"
				placeholder="Search Hugging Face for GGUF models"
				value={query}
				oninput={(e) => handleQueryInput(e.currentTarget.value)}
			/>
		</div>

		{#if error}
			<p class="text-sm text-destructive">{error}</p>
		{/if}

		{#if selectedRepo}
			<div class="flex items-center gap-2">
				<Button variant="ghost" size="sm" onclick={() => (selectedRepo = null)}>
					<ChevronLeft />

					Back to results
				</Button>

				<span class="truncate font-mono text-sm">{selectedRepo}</span>
			</div>

			{#if loadingVariants}
				<p class="text-sm text-muted-foreground">Reading repository...</p>
			{:else if variants.length === 0}
				<p class="text-sm text-muted-foreground">No GGUF model found in this repository.</p>
			{:else}
				<div
					class="grid gap-3"
					style="grid-template-columns: repeat(auto-fill, minmax(min(28rem, calc(100dvw - 2rem)), 1fr));"
				>
					{#each variants as variant (variant.tag)}
						<div class="flex items-center justify-between gap-4 rounded-lg border p-4">
							<div class="min-w-0">
								<p class="truncate font-medium">{variant.label}</p>

								<p class="text-sm text-muted-foreground">
									{formatFileSize(variant.size)}

									{#if variant.parts > 1}
										&middot; {variant.parts} parts
									{/if}
								</p>
							</div>

							<Button
								size="sm"
								disabled={addingTag !== null}
								onclick={() => void addVariant(variant)}
							>
								<Download />

								{addingTag === variant.tag ? 'Starting...' : 'Download'}
							</Button>
						</div>
					{/each}
				</div>
			{/if}
		{:else if searching}
			<p class="text-sm text-muted-foreground">Searching...</p>
		{:else if results.length === 0}
			<div class="flex flex-1 items-center justify-center py-16">
				<Empty.Root class="max-w-md">
					<Empty.Header>
						<Empty.Media variant="icon">
							<Search />
						</Empty.Media>

						<Empty.Title>Search the Hugging Face Hub</Empty.Title>

						<Empty.Description>
							Find a repository holding GGUF weights, then pick the quantization to download.
						</Empty.Description>
					</Empty.Header>
				</Empty.Root>
			</div>
		{:else}
			<div
				class="grid gap-3"
				style="grid-template-columns: repeat(auto-fill, minmax(min(28rem, calc(100dvw - 2rem)), 1fr));"
			>
				{#each results as result (result.id)}
					<button
						type="button"
						class="flex items-center justify-between gap-4 rounded-lg border p-4 text-left transition hover:bg-muted"
						onclick={() => void selectRepo(result.id)}
					>
						<span class="min-w-0 truncate font-medium">{result.id}</span>

						<span class="shrink-0 text-sm text-muted-foreground">
							{result.downloads.toLocaleString()} downloads
						</span>
					</button>
				{/each}
			</div>
		{/if}
	</div>
</div>
