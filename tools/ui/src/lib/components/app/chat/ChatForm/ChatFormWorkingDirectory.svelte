<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { Folder, FolderOpen, GitBranch, X } from '@lucide/svelte';
	import { FilesystemService } from '$lib/services';
	import { abbreviateWorkingDir, ApiError } from '$lib/utils';
	import { debounce } from '$lib/utils/debounce';
	import {
		browseRoots,
		browseRootsError,
		defaultBrowseRootPath,
		ensureBrowseRoots,
		isBrowseEndpointDisabled,
		markBrowseEndpointDisabled
	} from '$lib/stores/browse-roots.svelte';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import HighlightedMatch from '$lib/components/app/forms/HighlightedMatch.svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import { ChatFormPickerList, ChatFormPickerListItem } from '$lib/components/app/chat';
	import type { ApiFilesystemSearchEntry } from '$lib/types';

	interface Props {
		class?: string;
		disabled?: boolean;
		directory?: string | null;
		onChange?: (directory: string | null) => void;
	}

	let {
		class: className = '',
		disabled = false,
		directory = $bindable(null),
		onChange
	}: Props = $props();

	// File System Access API is opt-in: when available (Chrome / Edge / Opera) the popover
	// exposes a "Browse" button that opens the native folder picker. When unavailable the
	// popover still works via the text input - no alerts, no upload semantics.
	const pickerSupported =
		typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

	// Popover open state. The popover element handles outside-click and Escape;
	// we just react to open and seed the search field with the active path.
	let isOpen = $state(false);
	let inputValue = $state('');
	let searchInputRef: HTMLInputElement | null = $state(null);

	// Search / autocomplete state
	let queryResults = $state<ApiFilesystemSearchEntry[]>([]);
	let isSearching = $state(false);
	let searchError = $state<string | null>(null);
	let hoveredIndex = $state(-1);
	// Bumped only by ArrowUp/ArrowDown handlers; the list's
	// ChatFormPickerList uses this to scroll the picked row into view
	// without scrolling on mouse hover.
	let scrollTrigger = $state(0);
	// Browse roots live in the shared browse-roots store; both this component
	// and the file-mention picker (Phase B) read from it. defaultRootPath is
	// a derived view used to scope server-side search calls.

	// Invisible anchor for popover positioning - sits at the top edge of the
	// chat form so the popover floats above the box (matching the MCP picker
	// pattern). The visible chip is the click target (Popover.Trigger) but
	// positions via this anchor instead of the chip's own bounding box.
	let popoverAnchor = $state<HTMLDivElement | null>(null);

	let defaultRootPath = $derived(defaultBrowseRootPath());

	// Label on the trigger button: abbreviated active path, or the ghost
	// prompt. The default browse root is intentionally NOT previewed on
	// the chip - the user picks explicitly via the popover.
	let displayLabel = $derived.by(() => {
		if (!directory) return 'Select working directory';
		return abbreviateWorkingDir(directory, browseRoots());
	});

	// Full path surface for the chip - lets the user hover the abbreviated
	// label to recall exactly which directory is set.
	let displayLabelTitle = $derived(directory ?? '');

	// Git metadata for the picked directory. Probed by the server walking up
	// from `directory` looking for `.git/`. Updated whenever the active
	// `directory` changes; stale responses from earlier paths are dropped.
	let gitInfo = $state<{ is_repo: boolean; branch: string } | null>(null);
	let gitController: AbortController | null = null;
	let gitSeq = 0;

	$effect(() => {
		const path = directory;

		// Cancel any in-flight pose for a previous directory before kicking
		// off the next one.
		gitController?.abort();
		gitSeq++;

		if (!path) {
			gitInfo = null;
			return;
		}

		const controller = new AbortController();
		gitController = controller;
		const mySeq = gitSeq;

		FilesystemService.getGitInfo({ path }, controller.signal)
			.then((response) => {
				if (mySeq !== gitSeq) return;
				gitInfo = response.is_repo ? { is_repo: true, branch: response.branch } : null;
			})
			.catch((err: unknown) => {
				if (mySeq !== gitSeq) return;
				// 501 from servers without --tools / --agent is a normal
				// operational state; just hide the branch badge silently.
				if (err instanceof ApiError && err.status === 501) {
					gitInfo = null;
					return;
				}
				if (controller.signal.aborted) return;
				gitInfo = null;
			});
	});

	// AbortController + sequence counter to discard stale responses when the user
	// keeps typing; a newer call aborts the previous one. The sequence counter
	// also covers the gap between abort and the catch handler.
	let searchController: AbortController | null = null;
	let searchSeq = 0;

	// Auto-focus the search input when the popover opens.
	// HTML `autofocus` is unreliable on dynamically shown elements, so we
	// use a microtask (0ms setTimeout) after the effect flushes.
	$effect(() => {
		if (!isOpen) return;
		setTimeout(() => searchInputRef?.focus(), 0);
	});

	const runSearch = debounce((query: string) => {
		void doSearch(query);
	}, 180);

	// Local 501 handler removed; the search endpoint is gated by the same
	// server flag as the roots endpoint, so a 501 on /search implies the
	// whole browsing feature is off - markBrowseEndpointDisabled() tells
	// the shared store so other consumers see the same state.

	// Load browse roots eagerly on mount so the trigger can advertise the
	// default browse scope before the user opens the picker. ensureBrowseRoots()
	// is idempotent and promise-cached, so the call from handleOpenChange
	// stays a no-op once resolved.
	$effect(() => {
		if (typeof window === 'undefined') return;
		void ensureBrowseRoots();
	});

	function cancelSearch() {
		searchController?.abort();
		searchSeq++;
		isSearching = false;
	}

	async function doSearch(query: string) {
		// The inputValue $effect handles empty queries synchronously
		// before scheduling this fetch; doSearch is never called with
		// empty input, so an empty-input guard here would be dead
		// code. Trim is taken once at the top and reused.

		const trimmed = query.trim();
		cancelSearch();
		const controller = new AbortController();
		searchController = controller;
		const mySeq = ++searchSeq;

		isSearching = true;
		try {
			const response = await FilesystemService.search(
				{
					query: trimmed,
					type: 'directory',
					path: defaultRootPath ?? '',
					limit: 20,
					max_depth: 6,
					show_hidden: true
				},
				controller.signal
			);
			if (mySeq !== searchSeq) return;
			queryResults = response.results;
			// Auto-highlight the first row once results land so a fresh
			// query is immediately commit-able via Enter. Reset to -1
			// when the query returned nothing so the empty-state owns
			// the visual cue.
			hoveredIndex = queryResults.length > 0 ? 0 : -1;
			searchError = null;
		} catch (err) {
			if (mySeq !== searchSeq) return;
			queryResults = [];
			hoveredIndex = -1;
			if (controller.signal.aborted) return;
			if (err instanceof ApiError && err.status === 501) {
				markBrowseEndpointDisabled();
				searchError = null;
			} else {
				searchError = err instanceof Error ? err.message : String(err);
			}
		} finally {
			if (mySeq === searchSeq) isSearching = false;
		}
	}

	function commit(entry: ApiFilesystemSearchEntry) {
		directory = entry.path;
		onChange?.(entry.path);
		isOpen = false;
	}

	function setDirectory(value: string) {
		const trimmed = value.trim();
		if (!trimmed) return;
		directory = trimmed;
		onChange?.(trimmed);
	}

	// Resolve a folder name picked via the browser-native picker (which exposes
	// only the leaf name) to a server-side absolute path. Falls back to the
	// leaf name when the server cannot locate a matching directory.
	async function resolveNativeName(name: string): Promise<string> {
		if (isBrowseEndpointDisabled() || !defaultRootPath) return name;
		try {
			const res = await FilesystemService.search(
				{
					query: name,
					type: 'directory',
					path: defaultRootPath,
					limit: 1,
					max_depth: 4
				},
				new AbortController().signal
			);
			const match = res.results[0];
			return match && match.name === name ? match.path : name;
		} catch {
			return name;
		}
	}

	async function browseNative() {
		if (disabled || !window.showDirectoryPicker) return;
		try {
			const handle = await window.showDirectoryPicker();
			const path = await resolveNativeName(handle.name);
			setDirectory(path);
			isOpen = false;
		} catch (err) {
			// user cancelled - silently ignore; other errors are logged
			if (err instanceof DOMException && err.name === 'AbortError') return;
			console.error('[ChatFormWorkingDirectory] showDirectoryPicker failed:', err);
		}
	}

	function handleSubmit() {
		const value = inputValue.trim();
		if (!value) {
			isOpen = false;
			return;
		}
		setDirectory(value);
		isOpen = false;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			event.preventDefault();
			// Commit the highlighted search result when there is one
			// (the user may have arrow-keyed to it or it is the auto-
			// selected first row after results landed). Fall back to
			// the raw input only when the query returned no matches,
			// so the user can still type a known absolute path.
			if (hoveredIndex >= 0 && queryResults[hoveredIndex]) {
				commit(queryResults[hoveredIndex]);
			} else if (queryResults.length === 0) {
				handleSubmit();
			}
		} else if (event.key === 'ArrowDown') {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = (hoveredIndex + 1) % queryResults.length;
				scrollTrigger++;
			}
		} else if (event.key === 'ArrowUp') {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = hoveredIndex <= 0 ? queryResults.length - 1 : hoveredIndex - 1;
				scrollTrigger++;
			}
		}
	}

	// Drive the search from `inputValue` itself — the picker binds its
	// header input to `inputValue`, so user typing in the search head
	// (and the seeded value on open) both pick up the same debounced
	// fetch without needing a side-channel onInput callback.
	//
	// Synchronously reflects the correct state during the 180ms
	// debounce window so the picker never flashes the "API responded
	// with no matches" empty-state for a query that hasn't actually
	// been sent yet. Active query → skeleton immediately; empty
	// query → clears state and renders nothing; API-returned-empty →
	// renders the "No matching folders" message (recorded only by
	// doSearch).
	$effect(() => {
		const q = inputValue;
		if (!isOpen) {
			cancelSearch();
			return;
		}
		if (isBrowseEndpointDisabled()) {
			cancelSearch();
			queryResults = [];
			isSearching = false;
			searchError = null;
			hoveredIndex = -1;
			return;
		}
		const trimmed = q.trim();
		if (!trimmed) {
			cancelSearch();
			queryResults = [];
			isSearching = false;
			searchError = null;
			hoveredIndex = -1;
			return;
		}
		// Active query — cancel any in-flight stale request and show
		// the skeleton synchronously so the picker doesn't sit on
		// previously-returned empty data while debouncing the new
		// request.
		cancelSearch();
		isSearching = true;
		runSearch(q);
	});

	function clearDirectory(event?: MouseEvent) {
		// Stop the click from bubbling into the popover trigger and re-opening
		// the picker on top of the now-cleared state.
		event?.stopPropagation();
		event?.preventDefault();
		directory = null;
		onChange?.(null);
		isOpen = false;
	}

	// Chip is always visible - the X just clears the picked directory and
	// reveals the empty "Select working directory" placeholder again. No-op
	// when there's already nothing to clear.
	function handleDismiss(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();
		if (directory) {
			clearDirectory(event);
		}
	}

	function handleOpenChange(open: boolean) {
		isOpen = open;
		if (open) {
			// Seed the search field with the current path so the user can refine it
			// (or hit Enter to confirm / clear via the X icon).
			inputValue = directory ?? '';
			hoveredIndex = -1;
			queryResults = [];
			searchError = null;
			// The inputValue $effect picks up the seeded search automatically,
			// but it would only run after browse roots resolve; warm the cache
			// here so the first request is not gated on the initial fetch.
			void ensureBrowseRoots();
		} else {
			cancelSearch();
		}
	}

	// Imperative API: opens the picker without requiring the chip's own
	// trigger to be clicked. Used by ChatForm so picking the "Working
	// Directory" item from the Add dropdown reveals the chip and instantly
	// drops the user into the picker.
	export function openPicker() {
		isOpen = true;
	}

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Mirrors the gate used in ActionIcon.
	let innerWidth = $state(0);
	const showTooltip = $derived(innerWidth > 768);

	// Branch label resolved down to a string so the chip's two branches
	// (with / without Tooltip) don't have to re-narrow `gitInfo` inside
	// a snippet body - svelte-check loses the outer narrowing once the
	// markup crosses a Tooltip.Trigger boundary.
	const gitBranchLabel = $derived(gitInfo && gitInfo.is_repo ? gitInfo.branch : '');
</script>

<div class={['flex min-w-0 items-center gap-1 pt-2.5 px-2', className]}>
	<div
		bind:this={popoverAnchor}
		class="pointer-events-none absolute top-0 right-0 left-0 h-px"
		aria-hidden="true"
	></div>
	<Popover.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
		<Popover.Trigger {disabled} class="w-full flex justify-start">
			<span
				class="text-muted-foreground inline-flex items-center gap-1 text-xs group"
				class:text-foreground={directory}
			>
				<div class="flex min-w-0 items-center gap-1 cursor-pointer">
					<Folder class="w-3.5 h-3.5" />

					{#if showTooltip && displayLabelTitle}
						<Tooltip.Root>
							<Tooltip.Trigger>
								{#snippet child({ props })}
									<span {...props} class="max-w-64 truncate">{displayLabel}</span>
								{/snippet}
							</Tooltip.Trigger>
							<Tooltip.Content>
								<p>{displayLabelTitle}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{:else}
						<span class="max-w-64 truncate">{displayLabel}</span>
					{/if}

					{#if gitBranchLabel}
						{#if showTooltip}
							<Tooltip.Root>
								<Tooltip.Trigger>
									{#snippet child({ props })}
										<span
											{...props}
											class="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted/70 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
										>
											<GitBranch class="h-2.5 w-2.5" />
											<span>{gitBranchLabel}</span>
										</span>
									{/snippet}
								</Tooltip.Trigger>
								<Tooltip.Content>
									<p>Git branch on disk</p>
								</Tooltip.Content>
							</Tooltip.Root>
						{:else}
							<span
								class="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted/70 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
							>
								<GitBranch class="h-2.5 w-2.5" />
								<span>{gitBranchLabel}</span>
							</span>
						{/if}
					{/if}
				</div>

				{#if directory}
					<ActionIcon
						icon={X}
						tooltip="Reset working directory"
						ariaLabel="Reset working directory"
						{disabled}
						onclick={handleDismiss}
						iconSize="h-3 w-3"
						stopPropagationOnClick
						class="!h-4 !w-4 shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
					/>
				{/if}
			</span>
		</Popover.Trigger>

		<Popover.Content
			side="top"
			align="start"
			sideOffset={12}
			class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl"
			onkeydown={handleKeydown}
			onOpenAutoFocus={(event) => event.preventDefault()}
			onCloseAutoFocus={(event) => event.preventDefault()}
			customAnchor={popoverAnchor}
		>
			{#if isBrowseEndpointDisabled()}
				<div class="px-3 py-4 text-sm text-muted-foreground">
					Filesystem browsing is disabled. Start the server with
					<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--tools</code>
					or
					<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--agent</code>
					to enable it.
				</div>
			{:else}
				<ChatFormPickerList
					items={queryResults}
					isLoading={isSearching}
					selectedIndex={hoveredIndex}
					showSearchInput={true}
					bind:searchQuery={inputValue}
					bind:inputRef={searchInputRef}
					autofocus={true}
					onSearchClose={() => (isOpen = false)}
					searchPlaceholder="Choose working directory"
					emptyMessage={searchError
						? `Search failed - ${searchError}`
						: inputValue.trim() ? 'No matching folders' : undefined}
					itemKey={(entry) => entry.path}
					{scrollTrigger}
				>
					<!--
						Skeleton rows mirror the real row layout (folder
						icon + a single truncating monospace path bar).
						Widths vary per index so the placeholder reads as
						a list of directories, not six identical boxes;
						that swap-in is the only thing standing between the
						search request and any visible layout shift.
					-->
					{#snippet skeleton()}
						<div aria-busy="true" aria-live="polite" class="flex flex-col">
							{#each { length: 8 } as _, rowIndex (rowIndex)}
								{@const widths = ['w-3/5', 'w-4/5', 'w-2/5', 'w-3/4', 'w-1/2', 'w-5/6', 'w-3/4', 'w-4/5']}
								{@const widthClass = widths[rowIndex % widths.length]}
								<div class="flex items-start gap-2 rounded-lg px-3 py-2">
									<div class="mt-1 size-4 shrink-0 rounded-md bg-muted/60"></div>
									<div
										class={[
											'h-4 mt-1 animate-pulse rounded-sm bg-muted/60',
											widthClass
										]}
									></div>
								</div>
							{/each}
						</div>
					{/snippet}

					{#snippet item(entry, index, isSelected)}
						<ChatFormPickerListItem
						    class="gap-2!"
							dataIndex={index}
							{isSelected}
							onclick={() => commit(entry)}
							onmouseenter={() => (hoveredIndex = index)}
						>
							<Folder class="size-4 shrink-0 text-muted-foreground pt-1" />

							<span class="min-w-0 flex-1 truncate font-mono text-left text-sm">
								<HighlightedMatch text={entry.path} query={inputValue.trim()} />
							</span>
						</ChatFormPickerListItem>
					{/snippet}
				</ChatFormPickerList>

				<div class="px-2 pb-2">
					{#if pickerSupported}
						<button
							type="button"
							class="-mt-1 flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none hover:bg-accent hover:text-accent-foreground"
							onclick={browseNative}
						>
							<FolderOpen class="size-4 shrink-0 text-muted-foreground" />
							<span>Browse</span>
						</button>
					{/if}

					{#if defaultRootPath || browseRootsError()}
						<div class="-mx-2 my-1 h-px bg-border/20" aria-hidden="true"></div>

						{#if defaultRootPath}
							<span class="px-2 py-1.5 font-mono text-[10px]">
								Searching in:

								{#if showTooltip}
									<Tooltip.Root>
										<Tooltip.Trigger>
											{#snippet child({ props })}
												<span {...props} class="truncate text-muted-foreground/70">
													{abbreviateWorkingDir(defaultRootPath, browseRoots())}
												</span>
											{/snippet}
										</Tooltip.Trigger>
										<Tooltip.Content>
											<p>{defaultRootPath}</p>
										</Tooltip.Content>
									</Tooltip.Root>
								{:else}
									<span class="truncate text-muted-foreground/70">
										{abbreviateWorkingDir(defaultRootPath, browseRoots())}
									</span>
								{/if}
							</span>
						{:else if browseRootsError()}
							<div class="px-2 py-1.5 text-xs text-destructive">
								Cannot load browse roots - {browseRootsError()}
							</div>
						{/if}
					{/if}
				</div>
			{/if}
		</Popover.Content>
	</Popover.Root>
</div>

<svelte:window bind:innerWidth />
