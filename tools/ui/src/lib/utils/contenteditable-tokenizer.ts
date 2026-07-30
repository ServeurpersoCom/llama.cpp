/**
 * Tokenizer for the chat-form contenteditable input.
 *
 * The chat input renders some user-typed segments as plain text,
 * others (`[name](file://...)` markdown links produced by the @
 * picker) as inline badge chips, and complete `code` spans (inline
 * and fenced blocks) with the markdown code look. This module owns
 * the two-way mapping between the underlying markdown source string
 * and a flat token stream the DOM is built from.
 *
 * Hard rules that the invariants below rely on:
 *
 * 1. A badge span's own inner DOM (`<svg>`, label `<span>`, inner
 *    text) must NEVER be observed as input source. The badge's
 *    inner text length is NOT the badge's source length; a chip
 *    labelled "chat" can represent `[chat](file://long/path)` in
 *    the source. Iterating a badge's subtree would leak the label
 *    into the source; only `root.childNodes` is walked and each
 *    badge is one opaque contribution.
 *
 * 2. Anything inside a badge subtree is invisible to source/diff
 *    math. The badge root contributes `[name](file://path)` and
 *    nothing else; its descendants (svg, label span, inner text)
 *    are implementation detail that must not leak into source.
 *
 * 3. `textOffsetToRange` cannot place the caret inside a badge
 *    (the badge is `contenteditable=false`). We collapse to the
 *    nearest editable edge (`setStartBefore` / `setStartAfter`)
 *    so the user-visible caret lands cleanly.
 *
 * 4. Code spans (`<code data-code-token>`) are EDITABLE, unlike
 *    badges: they carry the full source segment (backtick fences
 *    included) as their text, so their textContent serializes
 *    verbatim and source offsets map 1:1 to text offsets.
 *
 * 5. The newline separating a fenced block from adjacent content is
 *    a SOURCE-level concept, never stored in the DOM: the block is
 *    display:block, so a leading `\n` in the following text node
 *    would render as a phantom empty line. Serialization synthesizes
 *    exactly one `\n` at every block boundary (see
 *    `hasBlockBoundary`), and `buildFragment` strips it from text
 *    tokens. A text node's own leading/trailing `\n` next to a block
 *    is an ADDITIONAL blank line.
 */

import {
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	getMentionBadgeIconPaths
} from './mention-badge';

export type ContentToken =
	| { kind: 'text'; text: string }
	| { kind: 'badge'; name: string; path: string }
	| { kind: 'inlineCode'; text: string }
	| { kind: 'codeBlock'; text: string };

/**
 * Recognize completed `[name](file://path)` insertions across the buffer.
 *
 * - `file://` is required so a normal web link like `[foo](https://...)`
 *   is left untouched in the stream.
 * - The path allows `)` only when it is not followed by whitespace or
 *   `[` - this admits macOS paths like `Screenshot (1).png` and
 *   folders named `Foo (Stuff)/bar` while still cutting the match
 *   at the closing `)` of an adjacent badge (`[a](file:///p)[b]...`)
 *   and at the link's actual end. This is the same shape
 *   `handleMentionSelect` emits from the picker.
 * - The match consumes the markdown link only; any trailing whitespace
 *   typed or pasted after stays in a separate text token so the
 *   round trip is byte-exact.
 */
const MENTION_BADGE_RE = /\[([^\]\n]+?)\]\(file:\/\/((?:[^)\n]|\)(?![\s[]))+)\)/g;

/**
 * Compute the byte-length contribution of one badge in source form.
 * Centralized so `serializeContent`, `rangeToTextOffset` and
 * `textOffsetToRange` agree on what counts; otherwise math of
 * `caret offset -> markdown offset` silently breaks.
 */
function badgeSourceLength(name: string, path: string): number {
	if (!name || !path) return 0;
	return `[${name}](file://${path})`.length;
}

/**
 * Recognize complete code spans. Fenced blocks (triple backticks,
 * optional language, possibly multiline) take priority over inline
 * spans (single backticks, single line, non-empty). Only CLOSED
 * spans match: an unclosed fence stays plain text until the closing
 * backticks land. The match includes the fences so the token's
 * source length equals its rendered text length.
 */
const CODE_SPAN_RE = /(```[\s\S]*?```)|(`[^`\n]+`)/g;

/**
 * Cheap gate check for `ChatForm`: does the buffer contain a
 * complete code span (inline or fenced)? Used to promote the plain
 * textarea to the contenteditable renderer.
 */
export function containsCodeSpan(value: string): boolean {
	CODE_SPAN_RE.lastIndex = 0;
	return CODE_SPAN_RE.test(value);
}

const CODE_FENCE_RE = /```/g;

/**
 * Is `offset` inside a fenced code block region? Toggle-based: an
 * odd number of ``` fences before the offset means the position
 * sits in block content. Unlike `containsCodeSpan` this also
 * counts the still-OPEN fence while the user is typing a block
 * (no closing ``` yet), so Enter can add a line instead of
 * submitting the message.
 */
export function isOffsetInCodeBlock(source: string, offset: number): boolean {
	let inside = false;
	CODE_FENCE_RE.lastIndex = 0;

	let match: RegExpExecArray | null;
	while ((match = CODE_FENCE_RE.exec(source)) !== null) {
		if (match.index + match[0].length > offset) break;
		inside = !inside;
	}

	return inside;
}

/**
 * Tokenize a markdown source value into the segments the
 * contenteditable will render. Code spans are carved out first
 * (their content is literal - a `file://` link inside backticks
 * must NOT render as a badge), then plain text and badges
 * interleave in the remaining gaps. Any whitespace after a badge
 * stays in a plain text token so the round trip is byte-exact.
 */
export function tokenizeContent(input: string): ContentToken[] {
	const tokens: ContentToken[] = [];
	let cursor = 0;
	CODE_SPAN_RE.lastIndex = 0;

	let match: RegExpExecArray | null;
	while ((match = CODE_SPAN_RE.exec(input)) !== null) {
		const start = match.index;

		if (start > cursor) {
			pushTextAndBadgeTokens(input.slice(cursor, start), tokens);
		}

		tokens.push(
			match[1] !== undefined
				? { kind: 'codeBlock', text: match[1] }
				: { kind: 'inlineCode', text: match[2] }
		);
		cursor = start + match[0].length;
	}

	if (cursor < input.length) {
		pushTextAndBadgeTokens(input.slice(cursor), tokens);
	}

	return tokens;
}

/**
 * Tokenize a code-free segment into text and badge tokens.
 */
function pushTextAndBadgeTokens(input: string, tokens: ContentToken[]) {
	let cursor = 0;
	MENTION_BADGE_RE.lastIndex = 0;

	let match: RegExpExecArray | null;
	while ((match = MENTION_BADGE_RE.exec(input)) !== null) {
		const [whole, name, path] = match;
		const start = match.index;

		if (start > cursor) {
			tokens.push({ kind: 'text', text: input.slice(cursor, start) });
		}

		tokens.push({ kind: 'badge', name, path });
		cursor = start + whole.length;
	}

	if (cursor < input.length) {
		tokens.push({ kind: 'text', text: input.slice(cursor) });
	}
}

/**
 * Serialize a contenteditable subtree back to markdown source form.
 *
 * Iterates `root.childNodes` directly so a badge is one opaque
 * contribution: its descendants are NEVER walked. Code spans
 * contribute their textContent verbatim (fences included). Text
 * nodes (direct children) contribute their textContent verbatim.
 * Any non-text, non-badge, non-code element is skipped (defensive -
 * the contenteditable root should not contain anything else by
 * construction, but browsers can inject wrappers in some edit
 * scenarios and we don't want those to leak into the source).
 *
 * One separator `\n` is synthesized at every fenced-block boundary
 * (rule 5): the DOM never stores it, so the source keeps fences on
 * their own lines without the form rendering a phantom empty line.
 */
export function serializeContent(root: HTMLElement): string {
	let out = '';
	let prev: Node | null = null;

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType === Node.TEXT_NODE) {
			const text = child.textContent ?? '';
			if (text.length === 0) continue;

			if (hasBlockBoundary(prev, child)) out += '\n';
			out += text;
			prev = child;
			continue;
		}

		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;

		// Code spans serialize verbatim: their textContent IS the source
		// segment, backtick fences included.
		if (el.dataset.codeToken !== undefined) {
			if (hasBlockBoundary(prev, el)) out += '\n';
			out += el.textContent ?? '';
			prev = el;
			continue;
		}

		if (el.dataset.mentionBadge !== 'true') continue;

		const name = el.dataset.mentionName ?? '';
		const path = el.dataset.mentionPath ?? '';
		if (name && path) {
			if (hasBlockBoundary(prev, el)) out += '\n';
			out += `[${name}](file://${path})`;
			prev = el;
		}
	}

	return out;
}

/**
 * Compare the live DOM's non-text structure against a token stream.
 * Only element contributions are compared (badges by name/path, code
 * spans by kind and source segment): text nodes are owned by the
 * browser between rebuilds, so their split/merge state is irrelevant.
 * A mismatch means token boundaries shifted (a code span was just
 * completed or broken) and the DOM needs a rebuild to restyle.
 */
export function domMatchesTokens(root: HTMLElement, tokens: ContentToken[]): boolean {
	const expected = tokens.filter((token) => token.kind !== 'text');
	let index = 0;

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;
		const isBadge = el.dataset.mentionBadge === 'true';
		const isCode = el.dataset.codeToken !== undefined;
		if (!isBadge && !isCode) continue;

		const token = expected[index++];
		if (!token) return false;

		if (isBadge) {
			if (token.kind !== 'badge') return false;
			if (token.name !== (el.dataset.mentionName ?? '')) return false;
			if (token.path !== (el.dataset.mentionPath ?? '')) return false;
			continue;
		}

		const codeKind = el.dataset.codeToken === 'block' ? 'codeBlock' : 'inlineCode';
		if (token.kind !== codeKind) return false;
		if (
			(token.kind === 'inlineCode' || token.kind === 'codeBlock') &&
			token.text !== (el.textContent ?? '')
		) {
			return false;
		}
	}

	return index === expected.length;
}

/**
 * Compute the plain-text character offset of a `Range` anchored
 * inside the contenteditable root. Used to capture caret position
 * before any DOM rebuild so we can restore it after.
 *
 * If `range` is null (selection lost during teardown) the position
 * falls back to buffer length. The body walks `tmp.childNodes`
 * only, so badges contribute their full source length, not their
 * visible label width. `cloneContents()` truncates the trailing
 * text node properly via the browser's range semantics, so its
 * `textContent` is the buffer length up to and including the caret.
 */
export function rangeToTextOffset(root: HTMLElement, range: Range | null): number {
	if (!range) return serializeContent(root).length;

	const pre = range.cloneRange();
	pre.selectNodeContents(root);
	pre.setEnd(range.endContainer, range.endOffset);

	const tmp = document.createElement('div');
	tmp.appendChild(pre.cloneContents());

	let total = 0;
	let prev: Node | null = null;

	for (const child of Array.from(tmp.childNodes)) {
		if (child.nodeType === Node.TEXT_NODE) {
			const text = child.textContent ?? '';
			if (text.length === 0) continue;

			if (hasBlockBoundary(prev, child)) total += 1;
			total += text.length;
			prev = child;
			continue;
		}

		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;

		if (el.dataset.codeToken !== undefined) {
			if (hasBlockBoundary(prev, el)) total += 1;
			total += (el.textContent ?? '').length;
			prev = el;
			continue;
		}

		if (el.dataset.mentionBadge !== 'true') continue;

		if (hasBlockBoundary(prev, el)) total += 1;
		total += badgeSourceLength(el.dataset.mentionName ?? '', el.dataset.mentionPath ?? '');
		prev = el;
	}

	return total;
}

/**
 * Materialize a single token stream into a freshly-built DOM subtree
 * suitable for inserting in place of the live contenteditable body.
 * The returned fragment contains plain text nodes for text tokens
 * and `<span data-mention-badge="true">` elements for badges. The
 * badge's class string + inline folder SVG mirror
 * `MentionBadge.svelte` exactly; Tailwind scans both and gets the
 * same style applied.
 */
export function buildFragment(tokens: ContentToken[]): DocumentFragment {
	const fragment = document.createDocumentFragment();

	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];

		if (token.kind === 'text') {
			let text = token.text;

			// The separator \n at a fenced-block boundary is synthesized
			// at serialization time (rule 5); keeping it in the DOM would
			// render a phantom empty line next to the block.
			if (tokens[index - 1]?.kind === 'codeBlock' && text.startsWith('\n')) {
				text = text.slice(1);
			}
			if (tokens[index + 1]?.kind === 'codeBlock' && text.endsWith('\n')) {
				text = text.slice(0, -1);
			}
			if (text.length === 0) continue;

			fragment.appendChild(document.createTextNode(text));
			continue;
		}

		if (token.kind === 'inlineCode' || token.kind === 'codeBlock') {
			const code = document.createElement('code');
			code.dataset.codeToken = token.kind === 'codeBlock' ? 'block' : 'inline';
			code.textContent = token.text;
			fragment.appendChild(code);
			continue;
		}

		// A leading badge gets an empty text node prepended: without a
		// real text position at the buffer start, the spot before the
		// badge is unreachable via keyboard (ArrowLeft/Home). The empty
		// node serializes to nothing, so the round trip stays byte-exact.
		if (!fragment.lastChild) {
			fragment.appendChild(document.createTextNode(''));
		}

		const badge = document.createElement('span');
		badge.dataset.mentionBadge = 'true';
		badge.dataset.mentionName = token.name;
		badge.dataset.mentionPath = token.path;
		badge.title = token.path;
		badge.className = MENTION_BADGE_CLASSNAME;
		badge.contentEditable = 'false';

		// Icon - matches the lucide component picked by MentionBadge.svelte
		// so the DOM-built badge is visually identical.
		const SVG_NS = 'http://www.w3.org/2000/svg';
		const svg = document.createElementNS(SVG_NS, 'svg');
		svg.setAttribute('viewBox', '0 0 24 24');
		svg.setAttribute('fill', 'none');
		svg.setAttribute('stroke', 'currentColor');
		svg.setAttribute('stroke-width', '2');
		svg.setAttribute('stroke-linecap', 'round');
		svg.setAttribute('stroke-linejoin', 'round');
		svg.setAttribute('aria-hidden', 'true');
		for (const cls of MENTION_BADGE_ICON_CLASSNAME.split(/\s+/).filter(Boolean)) {
			svg.classList.add(cls);
		}

		for (const d of getMentionBadgeIconPaths(token.path)) {
			const path = document.createElementNS(SVG_NS, 'path');
			path.setAttribute('d', d);
			svg.appendChild(path);
		}

		const label = document.createElement('span');
		label.classList.add('shrink-0', 'truncate');
		label.textContent = token.name;

		badge.appendChild(svg);
		badge.appendChild(label);
		fragment.appendChild(badge);
	}

	return fragment;
}

function isCodeBlockElement(node: Node | null): node is HTMLElement {
	return node instanceof HTMLElement && node.dataset.codeToken === 'block';
}

function isContentElement(node: Node): node is HTMLElement {
	return (
		node instanceof HTMLElement &&
		(node.dataset.codeToken !== undefined || node.dataset.mentionBadge === 'true')
	);
}

/**
 * Does the source carry a separator `\n` between two adjacent
 * root-level content nodes? Exactly one side of the boundary must be
 * a fenced block; the other side is a text node or any content
 * element (code span, badge). Callers skip empty text nodes, so an
 * element following a block through an empty text node still gets
 * its separator.
 */
function hasBlockBoundary(prev: Node | null, next: Node): boolean {
	if (isCodeBlockElement(prev)) {
		return next.nodeType === Node.TEXT_NODE || isContentElement(next);
	}

	if (!isCodeBlockElement(next) || prev === null) return false;

	return prev.nodeType === Node.TEXT_NODE || isContentElement(prev);
}

// A sibling provides a reachable caret line when it is an element
// (badge, another block, an existing hatch) or a non-empty text node.
function hasLineBeside(node: Node | null): boolean {
	if (!node) return false;
	if (node.nodeType === Node.ELEMENT_NODE) return true;
	return (node.textContent ?? '') !== '';
}

/**
 * A code block at the END of the buffer needs an editable line after
 * it: without one the caret cannot leave the block with
 * ArrowDown/ArrowRight. A trailing `<br>` provides that line while
 * staying transparent to serialization (skipped as a non-text,
 * non-token element), and is removed again once real content takes
 * its place.
 *
 * No hatch is added BEFORE a leading block: the empty line above it
 * is transient and managed by the component (created when the caret
 * arrows onto it, removed when the caret leaves). A transient
 * leading hatch found here is kept; the browser's lone placeholder
 * `<br>` in an empty root is left untouched.
 */
export function syncCodeBlockHatches(root: HTMLElement) {
	for (const child of Array.from(root.childNodes)) {
		if (child.nodeName !== 'BR') continue;

		const isPlaceholder = root.childNodes.length === 1;
		const isLeadingHatch = !child.previousSibling && isCodeBlockElement(child.nextSibling);
		const isTrailingHatch = !child.nextSibling && isCodeBlockElement(child.previousSibling);

		if (!isPlaceholder && !isLeadingHatch && !isTrailingHatch) {
			child.remove();
		}
	}

	for (const child of Array.from(root.childNodes)) {
		if (!isCodeBlockElement(child)) continue;

		if (!hasLineBeside(child.nextSibling)) {
			child.after(document.createElement('br'));
		}
	}
}

/**
 * Strip the separator and artificial newlines from an all-newline text
 * node directly after a fenced block. Chromium's line break at the
 * buffer end inserts an extra artificial `\n` so the new line has
 * height, and the first `\n` after a block doubles as the fence's
 * separator line (synthesized at serialization time, rule 5). Removing
 * both makes Shift+Enter after a block land the caret on the line
 * directly below the block, like a plain textarea would.
 *
 * Only all-newline text nodes are touched: a node with real content
 * carries intentional blank lines and is left alone. Returns true when
 * the DOM changed.
 */
export function stripBlockBoundaryLineBreaks(root: HTMLElement): boolean {
	let changed = false;

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType !== Node.TEXT_NODE) continue;
		if (!isCodeBlockElement(child.previousSibling)) continue;

		let text = child.textContent ?? '';
		if (!/^\n{2,}$/.test(text)) continue;

		text = text.slice(1);

		const atBufferEnd = !child.nextSibling || child.nextSibling.nodeName === 'BR';
		if (atBufferEnd) {
			text = text.slice(0, -1);
		}

		child.textContent = text;
		changed = true;
	}

	return changed;
}

const WORD_CHAR_RE = /[\p{L}\p{N}_]/u;

/**
 * Word-jump target (Option+Arrow / Ctrl+Arrow) in source offsets, or
 * null when the jump crosses no badge and native word movement should
 * handle it. Badge spans are masked to word characters and act as
 * hard word-run boundaries, so a badge counts as exactly one word:
 * native word iteration treats the non-editable badge element
 * inconsistently and overshoots it by a full word in either direction.
 */
export function badgeAwareWordJump(
	source: string,
	offset: number,
	direction: 'forward' | 'backward'
): number | null {
	let masked = '';
	const badgeSpans: Array<[number, number]> = [];

	for (const token of tokenizeContent(source)) {
		const len =
			token.kind === 'badge' ? badgeSourceLength(token.name, token.path) : token.text.length;
		if (token.kind === 'badge') badgeSpans.push([masked.length, masked.length + len]);
		masked += token.kind === 'badge' ? 'a'.repeat(len) : token.text;
	}

	if (badgeSpans.length === 0) return null;

	const isWord = (index: number) => WORD_CHAR_RE.test(masked[index]);
	const spanStartingAt = (index: number) => badgeSpans.find(([start]) => start === index);
	const spanEndingAt = (index: number) => badgeSpans.find(([, end]) => end === index);
	const n = masked.length;
	let i = offset;

	if (direction === 'forward') {
		// Skip non-word run when starting outside a word, then skip the
		// word run itself. Entering a badge completes the word phase at
		// the badge's end edge.
		if (!(i < n && isWord(i))) {
			while (i < n && !isWord(i)) i++;
		}
		while (i < n && isWord(i)) {
			const span = spanStartingAt(i);
			if (span) {
				i = span[1];
				break;
			}
			i++;
		}
	} else {
		if (!(i > 0 && isWord(i - 1))) {
			while (i > 0 && !isWord(i - 1)) i--;
		}
		while (i > 0 && isWord(i - 1)) {
			const span = spanEndingAt(i);
			if (span) {
				i = span[0];
				break;
			}
			i--;
		}
	}

	if (i === offset) return null;

	const lo = Math.min(offset, i);
	const hi = Math.max(offset, i);
	return badgeSpans.some(([start, end]) => start < hi && end > lo) ? i : null;
}

/**
 * Returns 0 when `caret` sits exactly at a leading badge's end edge,
 * null otherwise. Plain ArrowLeft at that spot has no native previous
 * position (the buffer starts with a non-editable element), so the
 * host snaps the caret to the buffer start manually. Covers post-edit
 * states where the leading pad from `buildFragment` is gone (e.g. the
 * user deleted the text before a mid-text badge).
 */
export function leadingBadgeEdgeOffset(source: string, caret: number): number | null {
	const [first] = tokenizeContent(source);
	if (!first || first.kind !== 'badge') return null;
	return caret === badgeSourceLength(first.name, first.path) ? 0 : null;
}

/**
 * Translate a plain-text character offset into a `Range` placed at
 * that position in the DOM. Returns a degenerate range (collapsed
 * to a single point). Out-of-range `offset` clamps to buffer end.
 *
 * Inside a badge we cannot land caret, so the offset resolves to
 * one of the two badge edges: zero offset lands BEFORE the badge,
 * any positive source offset lands AFTER. This matches the
 * visible-edit behavior the user expects from a non-editable
 * inline element.
 */
export function textOffsetToRange(root: HTMLElement, offset: number): Range {
	const range = document.createRange();
	let remaining = offset;
	let prev: Node | null = null;

	for (const child of Array.from(root.childNodes)) {
		if (child.nodeType === Node.TEXT_NODE) {
			const text = child.textContent ?? '';

			// The synthetic separator position (rule 5) maps to the
			// near edge of the content that follows the block.
			if (text.length > 0 && hasBlockBoundary(prev, child)) {
				if (remaining === 0) {
					range.setStart(child, 0);
					range.setEnd(child, 0);
					return range;
				}
				remaining -= 1;
			}

			if (remaining <= text.length) {
				range.setStart(child, remaining);
				range.setEnd(child, remaining);
				return range;
			}
			remaining -= text.length;
			if (text.length > 0) prev = child;
			continue;
		}

		if (child.nodeType !== Node.ELEMENT_NODE) continue;

		const el = child as HTMLElement;

		if (el.nodeName === 'BR') {
			// Escape-hatch line around an edge code block: contributes no
			// source length. Offset 0 lands before a leading hatch so
			// text typed there takes its place.
			if (remaining === 0) {
				range.setStartBefore(el);
				range.setEndBefore(el);
				return range;
			}
			continue;
		}

		if (el.dataset.codeToken !== undefined) {
			if (hasBlockBoundary(prev, el)) {
				if (remaining === 0) {
					range.setStartBefore(el);
					range.setEndBefore(el);
					return range;
				}
				remaining -= 1;
			}

			const codeLen = (el.textContent ?? '').length;
			if (remaining > codeLen) {
				remaining -= codeLen;
				prev = el;
				continue;
			}

			// Boundary offsets land OUTSIDE the element so typing at a
			// code span's edge extends the surrounding text, not the
			// span. Interior offsets land in the element's text.
			if (remaining === 0) {
				range.setStartBefore(el);
				range.setEndBefore(el);
				return range;
			}
			if (remaining === codeLen) {
				range.setStartAfter(el);
				range.setEndAfter(el);
				return range;
			}

			// Interior offsets land in the element's text. Descendants
			// are walked with a TreeWalker because syntax highlighting
			// nests the text inside token spans.
			const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
			let rest = remaining;
			for (let node = walker.nextNode(); node; node = walker.nextNode()) {
				const len = (node.textContent ?? '').length;
				if (rest <= len) {
					range.setStart(node, rest);
					range.setEnd(node, rest);
					return range;
				}
				rest -= len;
			}

			range.setStartAfter(el);
			range.setEndAfter(el);
			return range;
		}

		if (el.dataset.mentionBadge !== 'true') continue;

		if (hasBlockBoundary(prev, el)) {
			if (remaining === 0) {
				range.setStartBefore(el);
				range.setEndBefore(el);
				return range;
			}
			remaining -= 1;
		}

		const badgeLen = badgeSourceLength(el.dataset.mentionName ?? '', el.dataset.mentionPath ?? '');
		if (remaining <= badgeLen) {
			if (remaining === 0) {
				range.setStartBefore(el);
				range.setEndBefore(el);
			} else {
				range.setStartAfter(el);
				range.setEndAfter(el);
			}
			return range;
		}
		remaining -= badgeLen;
		prev = el;
	}

	// Out-of-range offsets clamp to the buffer end - before a
	// trailing escape hatch, not after it.
	const last = root.lastChild;
	if (last && last.nodeName === 'BR') {
		range.setStartBefore(last);
		range.setEndBefore(last);
		return range;
	}

	range.selectNodeContents(root);
	range.collapse(false);
	return range;
}
