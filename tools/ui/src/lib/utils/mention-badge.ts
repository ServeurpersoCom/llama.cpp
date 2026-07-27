/**
 * Visual contract shared between `MentionBadge.svelte` and the two
 * DOM-only paths (the contenteditable tokenizer + the rehype plugin
 * that renders `file://` anchors inside `MarkdownContent`). Svelte
 * cannot be mounted at the per-keystroke tokenizer hot path, nor
 * from within a hast tree, so both paths emit the badge with the
 * exact same class string literal as the component. Tailwind's
 * content scanner picks the literal up in all three sources, which
 * is what keeps the styles in sync without runtime mounting.
 */

const FILE_MENTION_LINK = /\[([^\]\n]+?)\]\(file:\/\/[^\s\)\n]+\)/;

export function containsFileMentionLink(value: string): boolean {
	return FILE_MENTION_LINK.test(value);
}

export const MENTION_BADGE_CLASSNAME =
	'inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border/50 bg-foreground/15 px-1.5 py-0.5 text-xs font-mono text-foreground hover:bg-foreground/25 dark:bg-foreground/10 dark:text-secondary-foreground';

export const MENTION_BADGE_ICON_CLASSNAME = 'h-3 w-3 shrink-0';

/**
 * Folder-path coordinate string for the badge's inline SVG. Matches
 * `lucide-svelte`'s `<Folder />` glyph so the Svelte-rendered and
 * DOM-built paths produce visually identical icons. Used by both
 * `contenteditable-tokenizer.ts` (which calls `createElementNS`) and
 * `MarkdownContent/plugins/rehype/file-badge.ts` (which builds a
 * hast `<svg>` node).
 */
export const MENTION_BADGE_FOLDER_D =
	'M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z';
