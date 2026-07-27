<script lang="ts" generics="T">
	import { untrack, type Snippet } from 'svelte';
	import { SearchInput } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import { CHAT_FORM_POPOVER_MAX_HEIGHT } from '$lib/constants';

	interface Props {
		items: T[];
		isLoading: boolean;
		selectedIndex: number;
		searchQuery: string;
		showSearchInput: boolean;
		searchPlaceholder?: string;
		/**
		 * Shown only when the API has responded with empty data. The
		 * default is `undefined` so that pickers which need to
		 * distinguish "haven't searched yet" from "search returned
		 * nothing" can pass `undefined` and silence the message.
		 * Svelte treats `undefined` as missing for prop defaults, so
		 * any picker that does NOT want an empty-state message must
		 * pass it explicitly (or omit it) and not rely on the
		 * defaulting behaviour falling through.
		 */
		emptyMessage?: string;
		autofocus?: boolean;
		inputRef?: HTMLInputElement | null;
		onSearchClose?: () => void;
		itemKey: (item: T, index: number) => string;
		item: Snippet<[T, number, boolean]>;
		skeleton?: Snippet;
		skeletonCount?: number;
		footer?: Snippet;
		/**
		 * Monotonically-increasing counter the picker bumps when the
		 * user navigates the list via keyboard. When the counter
		 * changes the list scrolls the selected row into view; mouse
		 * hover bumps the index without bumping the counter, so hover
		 * never moves the scroll position. `undefined` disables auto
		 * scroll entirely. Mount and reactively-driven selection
		 * changes (typing in the search input) leave the counter
		 * untouched, so the list stays put while the result set
		 * replaces itself.
		 */
		scrollTrigger?: number;
	}

	let {
		items,
		isLoading,
		selectedIndex,
		searchQuery = $bindable(),
		showSearchInput,
		searchPlaceholder = 'Search...',
		emptyMessage,
		autofocus = false,
		inputRef = $bindable(null),
		onSearchClose,
		itemKey,
		item,
		skeleton,
		skeletonCount = 6,
		footer,
		scrollTrigger
	}: Props = $props();

	let listContainer = $state<HTMLDivElement | null>(null);

	// Track the previous scrollTrigger so the very first mount-render
	// pass doesn't fire a scroll — the picker starts with the counter
	// at 0 and no user keyboard nav has happened yet.
	let lastScrollTrigger: number | null = null;

	/**
	 * Snap the keyboard-selected row into view without disturbing the
	 * scroll position on mouse hover.
	 *
	 * Only `scrollTrigger` is a reactive dependency here. Reads of
	 * `selectedIndex`/`items.length` happen inside `untrack` so the
	 * effect does not re-run when those mutate elsewhere (e.g. on
	 * `onmouseenter` rewriting `hoveredIndex`). Without untracking,
	 * every cursor twitch would re-fire `scrollIntoView` even though
	 * the `scrollTrigger === undefined` gate passes (the parent sets
	 * the counter to `0`, not `undefined`, on first render); near-edge
	 * rows in long result sets would visibly drift on each hover.
	 *
	 * Mouse hover, search-driven result replacement, and the initial
	 * mount therefore never scroll. Keyboard `ArrowUp`/`ArrowDown` is
	 * the only path that bumps the trigger and scrolls.
	 */
	$effect(() => {
		if (scrollTrigger === undefined || scrollTrigger === lastScrollTrigger) return;
		lastScrollTrigger = scrollTrigger;
		untrack(() => {
			if (!listContainer) return;
			if (selectedIndex < 0 || selectedIndex >= items.length) return;
			const selectedElement = listContainer.querySelector(
				`[data-picker-index="${selectedIndex}"]`
			) as HTMLElement | null;
			selectedElement?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
		});
	});
</script>

<ScrollArea>
	{#if showSearchInput}
		<div class="absolute top-0 right-0 left-0 z-10 p-2 pb-0">
			<SearchInput
				{autofocus}
				placeholder={searchPlaceholder}
				bind:value={searchQuery}
				bind:ref={inputRef}
				onClose={onSearchClose}
			/>
		</div>
	{/if}

	<div
		bind:this={listContainer}
		class={[`${CHAT_FORM_POPOVER_MAX_HEIGHT} p-2`, showSearchInput && (isLoading || items.length > 0) ? 'pt-13' : showSearchInput ? 'pt-10' : '']}
	>
		{#if isLoading}
			{#if skeleton}
				{@render skeleton()}
			{:else}
				<div aria-busy="true" aria-live="polite" class="flex flex-col">
					{#each { length: skeletonCount } as _, rowIndex (rowIndex)}
						<div class="flex items-start gap-3 rounded-lg px-3 py-2">
							<div class="mt-0.5 size-4 shrink-0 animate-pulse rounded-md bg-muted/60"></div>
							<div class="flex min-w-0 flex-1 flex-col">
								<div class="h-5 w-2/5 animate-pulse rounded-sm bg-muted/60"></div>
								<div class="h-4 w-1/3 animate-pulse rounded-sm bg-muted/40"></div>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		{:else if items && items.length === 0}
			{#if emptyMessage}
				<div class="py-6 text-center text-sm text-muted-foreground">{emptyMessage}</div>
			{/if}
		{:else}
			{#each items as itemData, index (itemKey(itemData, index))}
				{@render item(itemData, index, index === selectedIndex)}
			{/each}
		{/if}
	</div>

	{#if footer}
		{@render footer()}
	{/if}
</ScrollArea>
