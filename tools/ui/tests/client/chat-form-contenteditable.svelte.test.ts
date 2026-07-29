// Guards the clipboard contract of the chat-form contenteditable:
// copy/cut expose the markdown SOURCE of the selection (each mention
// badge contributes its full `[name](file://...)` link) and pasting
// such markdown re-renders the badges. The browser dispatches the
// same copy/cut/paste events for keyboard shortcuts, the Edit menu
// and the mouse context menu, so covering the event handlers covers
// all three entry points.

import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { userEvent } from 'vitest/browser';
import { tick } from 'svelte';
import { rangeToTextOffset, serializeContent, textOffsetToRange } from '$lib/utils';
import ChatFormContenteditable from '$lib/components/app/chat/ChatForm/ChatFormContenteditable.svelte';

const SOURCE = 'hello [docs](file:///a/b) world';
const BADGE_SELECTOR = '[data-mention-badge="true"]';

function editableIn(container: HTMLElement): HTMLElement {
	const el = container.querySelector('[role="textbox"]');
	if (!(el instanceof HTMLElement)) throw new Error('contenteditable not rendered');
	return el;
}

function setSelection(root: HTMLElement, place: (range: Range, root: HTMLElement) => void) {
	const range = document.createRange();
	place(range, root);
	const selection = window.getSelection();
	if (!selection) throw new Error('no selection');
	selection.removeAllRanges();
	selection.addRange(range);
}

function clipboardEvent(type: 'copy' | 'cut' | 'paste', text = '') {
	const data = new DataTransfer();
	if (text) data.setData('text/plain', text);
	const event = new ClipboardEvent(type, { clipboardData: data, bubbles: true, cancelable: true });
	return { event, data };
}

describe('ChatFormContenteditable clipboard', () => {
	it('copy exposes the markdown source of the selection', async () => {
		const { container } = render(ChatFormContenteditable, { value: SOURCE });
		await tick();

		const root = editableIn(container);
		setSelection(root, (range) => range.selectNodeContents(root));

		const { event, data } = clipboardEvent('copy');
		root.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(true);
		expect(data.getData('text/plain')).toBe(SOURCE);
	});

	it('cut exposes the markdown source and removes the slice', async () => {
		const { container } = render(ChatFormContenteditable, { value: SOURCE });
		await tick();

		const root = editableIn(container);
		setSelection(root, (range) => {
			const badge = root.querySelector(BADGE_SELECTOR);
			if (!badge) throw new Error('badge not rendered');
			range.setStartBefore(badge);
			range.setEndAfter(badge);
		});

		const { event, data } = clipboardEvent('cut');
		root.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(true);
		expect(data.getData('text/plain')).toBe('[docs](file:///a/b)');
		expect(root.querySelector(BADGE_SELECTOR)).toBeNull();
		expect(root.textContent).toBe('hello  world');
	});

	it('paste of markdown mention links re-renders badges', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'hello ' });
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});

		const { event } = clipboardEvent('paste', '[docs](file:///a/b) world');
		root.dispatchEvent(event);
		await tick();

		expect(event.defaultPrevented).toBe(true);
		const badge = root.querySelector(BADGE_SELECTOR);
		expect(badge).not.toBeNull();
		expect(badge!.getAttribute('data-mention-name')).toBe('docs');
		expect(root.textContent).toContain('world');
	});

	it('paste without mention links keeps the DOM untouched', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'hello ' });
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});
		const firstChild = root.firstChild;

		const { event } = clipboardEvent('paste', 'plain text');
		root.dispatchEvent(event);
		await tick();

		expect(event.defaultPrevented).toBe(true);
		expect(root.querySelector(BADGE_SELECTOR)).toBeNull();
		// no rebuild: the live text node is the same instance
		expect(root.firstChild).toBe(firstChild);
	});
});

describe('ChatFormContenteditable code spans', () => {
	it('renders inline code from the initial value', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'run `npm test` now' });
		await tick();

		const root = editableIn(container);
		const code = root.querySelector('code[data-code-token="inline"]');
		expect(code).not.toBeNull();
		expect(code!.textContent).toBe('`npm test`');
	});

	it('renders a fenced code block with a language', async () => {
		const source = 'before\n```js\nconst a = 1;\n```\nafter';
		const { container } = render(ChatFormContenteditable, { value: source });
		await tick();

		const root = editableIn(container);
		const code = root.querySelector('code[data-code-token="block"]');
		expect(code).not.toBeNull();
		expect(code!.textContent).toBe('```js\nconst a = 1;\n```');
	});

	it('copy exposes the markdown source of a selection spanning code', async () => {
		const source = 'run `npm test` now';
		const { container } = render(ChatFormContenteditable, { value: source });
		await tick();

		const root = editableIn(container);
		setSelection(root, (range) => range.selectNodeContents(root));

		const { event, data } = clipboardEvent('copy');
		root.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(true);
		expect(data.getData('text/plain')).toBe(source);
	});

	it('paste of a code span renders the styled element', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'run ' });
		await tick();

		const root = editableIn(container);
		root.focus();
		setSelection(root, (range) => {
			range.selectNodeContents(root);
			range.collapse(false);
		});

		const { event } = clipboardEvent('paste', '`npm test` now');
		root.dispatchEvent(event);
		await tick();

		expect(event.defaultPrevented).toBe(true);
		const code = root.querySelector('code[data-code-token="inline"]');
		expect(code).not.toBeNull();
		expect(code!.textContent).toBe('`npm test`');
		expect(root.textContent).toContain('now');
	});

	it('highlights a fenced block content and stays byte-exact', async () => {
		const source = '```js\nconst a = 1;\n```';
		const { container } = render(ChatFormContenteditable, { value: source });
		await tick();

		const root = editableIn(container);
		const code = root.querySelector('code[data-code-token="block"]');
		expect(code).not.toBeNull();
		expect(code!.querySelector('.hljs-keyword')).not.toBeNull();
		expect(code!.textContent).toBe(source);
	});

	it('does not highlight inline code', async () => {
		const { container } = render(ChatFormContenteditable, { value: 'run `const` now' });
		await tick();

		const root = editableIn(container);
		expect(root.querySelector('[class*="hljs-"]')).toBeNull();
	});
});

describe('ChatFormContenteditable code block escape hatches', () => {
	const BLOCK_SOURCE = '```js\nconst a = 1;\n```';
	const BLOCK_SELECTOR = 'code[data-code-token="block"]';

	function blockIn(root: HTMLElement): HTMLElement {
		const el = root.querySelector(BLOCK_SELECTOR);
		if (!(el instanceof HTMLElement)) throw new Error('code block not rendered');
		return el;
	}

	// Caret at the very start/end of the block's text (across highlight spans)
	function placeCaretInBlock(root: HTMLElement, where: 'start' | 'end') {
		const code = blockIn(root);
		const walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
		let target: Node | null = null;
		for (let n = walker.nextNode(); n; n = walker.nextNode()) {
			target = where === 'start' ? (target ?? n) : n;
		}
		if (!target) throw new Error('no text inside code block');
		setSelection(root, (range) => {
			range.setStart(target!, where === 'start' ? 0 : (target!.textContent ?? '').length);
			range.collapse(true);
		});
	}

	function caretContainer(): Node {
		const selection = window.getSelection();
		if (!selection || selection.rangeCount === 0) throw new Error('no selection');
		return selection.getRangeAt(0).startContainer;
	}

	it('pads a trailing code block with a br hatch that stays invisible to copy', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		// no permanent empty line above a leading block
		expect(root.firstChild).toBe(blockIn(root));
		expect(root.lastChild?.nodeName).toBe('BR');

		setSelection(root, (range) => range.selectNodeContents(root));
		const { event, data } = clipboardEvent('copy');
		root.dispatchEvent(event);

		expect(data.getData('text/plain')).toBe(BLOCK_SOURCE);
	});

	it('escapes a trailing code block with ArrowDown and types after it', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{ArrowDown}');
		expect(blockIn(root).contains(caretContainer())).toBe(false);

		await userEvent.keyboard('x');
		await tick();

		expect(blockIn(root).textContent).toBe(BLOCK_SOURCE);
		expect(root.textContent).toBe(BLOCK_SOURCE + 'x');
		// the stale trailing hatch is removed once real text follows the block
		expect(root.lastChild?.nodeName).not.toBe('BR');
	});

	it('escapes a leading code block with ArrowUp and types before it', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'start');

		await userEvent.keyboard('{ArrowUp}');
		expect(blockIn(root).contains(caretContainer())).toBe(false);
		// the transient hatch line exists while the caret sits on it
		expect(root.firstChild?.nodeName).toBe('BR');

		await userEvent.keyboard('y');
		await tick();

		expect(blockIn(root).textContent).toBe(BLOCK_SOURCE);
		expect(root.textContent).toBe('y' + BLOCK_SOURCE);
		// the typed text consumed the hatch
		expect(root.firstChild?.nodeName).not.toBe('BR');
	});

	it('escapes a leading code block with ArrowLeft from its first character', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'start');

		await userEvent.keyboard('{ArrowLeft}');
		expect(blockIn(root).contains(caretContainer())).toBe(false);
		expect(root.firstChild?.nodeName).toBe('BR');
	});

	it('removes the transient leading hatch when the caret moves back into the block', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'start');

		await userEvent.keyboard('{ArrowUp}');
		expect(root.firstChild?.nodeName).toBe('BR');

		await userEvent.keyboard('{ArrowDown}');
		await tick();

		expect(blockIn(root).contains(caretContainer())).toBe(true);
		expect(root.firstChild).toBe(blockIn(root));
	});

	it('extends the selection out of the block with Shift+ArrowDown', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();
		placeCaretInBlock(root, 'end');

		await userEvent.keyboard('{Shift>}{ArrowDown}{/Shift}');

		const selection = window.getSelection();
		expect(selection).not.toBeNull();
		expect(selection!.isCollapsed).toBe(false);
		expect(blockIn(root).contains(selection!.getRangeAt(0).endContainer)).toBe(false);
	});

	it('re-highlights while typing inside a block and keeps the caret', async () => {
		const { container } = render(ChatFormContenteditable, { value: BLOCK_SOURCE });
		await tick();

		const root = editableIn(container);
		root.focus();

		// caret at the start of the block content (after the opening fence)
		setSelection(root, (range) => {
			const target = textOffsetToRange(root, 6);
			range.setStart(target.startContainer, target.startOffset);
			range.collapse(true);
		});

		await userEvent.keyboard('x');
		await tick();

		const code = blockIn(root);
		expect(serializeContent(root)).toBe('```js\nxconst a = 1;\n```');
		expect(code.textContent).toBe('```js\nxconst a = 1;\n```');
		expect(code.querySelector('.hljs-number')).not.toBeNull();

		const selection = window.getSelection();
		expect(code.contains(selection!.getRangeAt(0).startContainer)).toBe(true);
		expect(rangeToTextOffset(root, selection!.getRangeAt(0))).toBe(7);
	});
});
