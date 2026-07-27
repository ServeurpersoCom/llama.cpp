<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import {
		buildFragment,
		rangeToTextOffset,
		serializeContent,
		tokenizeContent,
		textOffsetToRange
	} from '$lib/utils';
	import type { ContentToken } from '$lib/utils';

	interface Props {
		class?: string;
		disabled?: boolean;
		onInput?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		placeholder?: string;
		value?: string;
	}

	let {
		class: className = '',
		disabled = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		value = $bindable('')
	}: Props = $props();

	let rootElement: HTMLDivElement | undefined = $state();
	let lastEmittedValue = '';
	let isComposing = $state(false);

	/**
	 * Track whether the editable area is visually empty. Browsers
	 * disagree on what an empty contenteditable root contains (some
	 * insert a placeholder `<br>` so the caret has a home, some
	 * leave it as a pure empty node) - we drive the placeholder via
	 * `data-empty="true|false"` so both shapes render correctly.
	 */
	function syncEmptyState() {
		if (!rootElement) return;
		const text = rootElement.textContent ?? '';
		const onlyBr =
			rootElement.childNodes.length === 1 && rootElement.firstChild?.nodeName === 'BR';
		const isEmpty = text.trim().length === 0 && onlyBr;
		rootElement.dataset.empty = isEmpty ? 'true' : 'false';
	}

	/**
	 * Render `tokens` into the contenteditable root. Captures the
	 * caret position before clearing so the cursor lands at the
	 * same logical character after the rebuild.
	 *
	 * Used both for the initial mount and for any external value
	 * change (system prompt insertion, paste handlers replacing
	 * the buffer, two-way sync from the @-mention picker, ...).
	 *
	 * @param tokens - Token stream produced by `tokenizeContent`.
	 */
	function renderTokens(tokens: ContentToken[]) {
		if (!rootElement) return;

		const caret = rangeToTextOffset(rootElement, safeRange());

		while (rootElement.firstChild) {
			rootElement.removeChild(rootElement.firstChild);
		}

		rootElement.appendChild(buildFragment(tokens));

		restoreCaret(caret);
		resizeHeight();
		syncEmptyState();
	}

	/**
	 * Pull a `Range` from the live selection. Returns `null` when
	 * nothing is selected, when selection collapsed to a node
	 * outside the editable root, or when the document lost focus -
	 * callers fall back to the buffer end.
	 */
	function safeRange(): Range | null {
		if (!rootElement) return null;

		const selection = window.getSelection();
		if (!selection || selection.rangeCount === 0) return null;

		const range = selection.getRangeAt(0);

		if (!rootElement.contains(range.startContainer) || !rootElement.contains(range.endContainer)) {
			return null;
		}

		return range;
	}

	function restoreCaret(offset: number) {
		if (!rootElement) return;

		const target = textOffsetToRange(rootElement, offset);
		const selection = window.getSelection();
		if (!selection) return;

		selection.removeAllRanges();
		selection.addRange(target);
	}

	/**
	 * Auto-grow the editable area to fit its content, capped via the
	 * shared `--max-message-height` CSS variable. Mirrors the
	 * dimensions of the legacy `<textarea>` so the surrounding flex
	 * layout (padding, scroll behaviour) keeps its current shape.
	 */
	function resizeHeight() {
		if (!rootElement) return;
		// content-height drives the auto-grow; max-height is enforced by CSS
		rootElement.style.height = 'auto';
		rootElement.style.height = `${rootElement.scrollHeight}px`;
	}

	/**
	 * Re-emit the current markdown source value to the parent.
	 * Bails out when the DOM diff is below the threshold reported by
	 * the browser (browser coalesces rapid IME keystrokes) - the
	 * native event already fired, so let the parent react when
	 * `compositionend` finishes.
	 */
	function handleInput() {
		if (isComposing || !rootElement) return;

		syncEmptyState();
		const serialized = serializeContent(rootElement);
		if (serialized === lastEmittedValue) return;

		lastEmittedValue = serialized;
		value = serialized;
		onInput?.();
	}

	function handleCompositionStart() {
		isComposing = true;
	}

	function handleCompositionEnd() {
		isComposing = false;
		if (!rootElement) return;
		const serialized = serializeContent(rootElement);
		if (serialized === lastEmittedValue) return;
		lastEmittedValue = serialized;
		value = serialized;
		onInput?.();
		syncEmptyState();
	}

	/**
	 * Native caret-keys that have to be forwarded by us: Enter and
	 * Escape are consumed by `handleKeydown` to trigger submit /
	 * picker-dismiss in `ChatForm`. We do not consume them here -
	 * just expose the event to the parent.
	 *
	 * Tab is intercepted locally so focus order is predictable
	 * without escaping the form area.
	 */
	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Tab') {
			event.preventDefault();
			return;
		}
		onKeydown?.(event);
	}

	/**
	 * Plain-text paste (the parent's `handlePaste` already short-
	 * circuits file/clipboard-quote cases) - feed the inserted
	 * string through the tokenizer so newly pasted `[name](file://...)`
	 * segments immediately render as inline badges.
	 *
	 * Passing through event.preventDefault() + manual insertText
	 * guarantees the browser does not produce stray HTML elements
	 * mid-paste (keeping Chromium's sanitize-on-paste path from
	 * introducing `<div>` wrappers that would otherwise appear at
	 * every line break).
	 */
	function handlePasteEvent(event: ClipboardEvent) {
		const pasted = event.clipboardData?.getData('text/plain');
		if (pasted && pasted.length > 0) {
			event.preventDefault();
			document.execCommand('insertText', false, pasted);
		}
	}

	/**
	 * Triggers above (parent paste handler: long text -> file,
	 * quoted clipboard prompt, MCP file attach, ...) bubble here
	 * before we run the local sanitize path. The parent handler
	 * calls preventDefault itself when it intends to do something.
	 * If the parent did not consume the event, fall back to handlePasteEvent.
	 */
	function handlePaste(event: ClipboardEvent) {
		onPaste?.(event);
		if (!event.defaultPrevented) {
			handlePasteEvent(event);
		}
	}

	onMount(() => {
		// Initial mount: render whatever value the parent handed us.
		// untrack so the effect doesn't re-fire on every keystroke
		// (we manage the DOM manually from input events).
		renderTokens(tokenizeContent(untrack(() => value)));
		lastEmittedValue = untrack(() => value ?? '');
		resizeHeight();
		syncEmptyState();
		if (!isMobile.current) {
			rootElement?.focus({ preventScroll: true });
		}
	});

	// External `value` updates (system-prompt insertion, two-way sync from
	// the mention picker, imperative clears, ...). We compare against
	// `lastEmittedValue`: when equal, the dispatch came from our own input,
	// so leave the DOM alone (the browser already owns the right shape).
	$effect(() => {
		const incoming = value ?? '';
		if (incoming === lastEmittedValue) return;

		renderTokens(tokenizeContent(incoming));
		lastEmittedValue = incoming;
	});

	export function getElement() {
		return rootElement;
	}

	/**
	 * Plain-text position of the current caret. Falls back to the
	 * buffer end when no selection lives inside the root (e.g. the
	 * component just lost focus to a picker or system dialog).
	 */
	export function getCaretOffset(): number {
		if (!rootElement) return 0;
		return rangeToTextOffset(rootElement, safeRange());
	}

	/**
	 * Place the caret at a plain-text position inside the buffer.
	 * Used by the mention picker round-trip and the post-splice
	 * caret restoration in `ChatForm`.
	 *
	 * `selection.addRange` requires the editable to have focus on
	 * some browsers, so we focus first when called from outside the
	 * component (the parent does it too as belt-and-braces).
	 *
	 * @param offset - The position to land on, expressed as the
	 * length of the serialized source string up to that caret.
	 */
	export function setCaretOffset(offset: number) {
		if (rootElement && rootElement !== document.activeElement) {
			rootElement.focus({ preventScroll: true });
		}
		restoreCaret(offset);
	}

	export function focus() {
		if (isMobile.current) return;
		rootElement?.focus({ preventScroll: true });
	}

	export function resetHeight() {
		if (rootElement) {
			rootElement.style.height = '';
			resizeHeight();
		}
	}
</script>

<div class="flex-1 {className}">
	<div
		bind:this={rootElement}
		contenteditable={!disabled}
		role="textbox"
		aria-multiline="true"
		aria-disabled={disabled}
		aria-placeholder={placeholder}
		data-placeholder={placeholder}
		tabindex={disabled ? -1 : 0}
		class={[
			'chat-form-contenteditable text-md min-h-12 w-full whitespace-pre-wrap break-words border-0 bg-transparent p-0 leading-6 outline-none focus-visible:ring-0 focus-visible:ring-offset-0',
			disabled && 'cursor-not-allowed'
		]}
		style="max-height: var(--max-message-height);"
		oncompositionstart={handleCompositionStart}
		oncompositionend={handleCompositionEnd}
		oninput={handleInput}
		onkeydown={handleKeydown}
		onpaste={handlePaste}
	></div>
</div>

<style>
	.chat-form-contenteditable:global([data-empty='true'])::before {
		content: attr(data-placeholder);
		color: var(--muted-foreground);
		pointer-events: none;
	}
</style>
