<script lang="ts">
	import {
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormFileInputInvisible,
		ChatFormMcpResourcesList,
		ChatFormPickers,
		ChatFormTextarea,
		ChatFormWorkingDirectory,
		DialogMcpResourcesBrowser
	} from '$lib/components/app';
	import {
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		INPUT_CLASSES,
		SETTING_CONFIG_DEFAULT,
		INITIAL_FILE_SIZE,
		PROMPT_CONTENT_SEPARATOR,
		PROMPT_TRIGGER_PREFIX,
		MENTION_TRIGGER_PREFIX
	} from '$lib/constants';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MimeTypeText,
		SpecialFileType
	} from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
	import { modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { BuiltInTool } from '$lib/enums';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpHasResourceAttachments } from '$lib/stores/mcp-resources.svelte';
	import {
		conversationsStore,
		activeMessages,
		activeConversation,
		pendingWorkingDirectory
	} from '$lib/stores/conversations.svelte';
	import type {
		ApiFilesystemSearchEntry,
		GetPromptResult,
		MCPPromptInfo,
		MCPResourceInfo,
		PromptMessage
	} from '$lib/types';
	import {
		findMentionToken,
		isIMEComposing,
		lastPathSegment,
		parseClipboardContent,
		takeMentionDismissSnapshot,
		uuid,
		type MentionDismissSnapshot
	} from '$lib/utils';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount } from 'svelte';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;
		showMcpPromptButton?: boolean;
		showAddButton?: boolean;
		showModelSelector?: boolean;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSubmit?: () => void;
		onSystemPromptClick?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		onUploadedFileRemove?: (fileId: string) => void;
		onUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		onValueChange?: (value: string) => void;
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		isLoading = false,
		placeholder = 'Type a message...',
		showMcpPromptButton = false,
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = $bindable([]),
		value = $bindable(''),
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onUploadedFilesChange,
		onValueChange
	}: Props = $props();

	// Component References
	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let textareaRef: ChatFormTextarea | undefined = $state(undefined);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Picker State
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');
	let isMentionPickerOpen = $state(false);
	let mentionQuery = $state('');

	/**
	 * Snapshot of the most recent `@`-mention token the user dismissed
	 * (via Escape, outside-click, or simply by deleting it). When the
	 * picker is closed AND the same token is still intact in the buffer,
	 * we do NOT auto-reopen - the user has explicitly told us this
	 * `@<query>` should be treated as literal text. The snapshot
	 * becomes stale the moment any character inside the token changes,
	 * at which point the picker is allowed to reopen on the next input.
	 */
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;

	// Invisible anchor for the mention picker: sits at the top edge of the
	// chat form so the popover floats above the box (matches the working-
	// directory picker's `customAnchor` pattern). One anchor per popover we
	// want to anchor above the form.
	let mentionAnchor: HTMLDivElement | null = $state(null);

	// Working Directory State
	// Sourced from the active conversation so the picked cwd is restored when
	// the user reopens the same chat. On the empty new-chat screen there's no
	// active conversation yet; falls back to the pending cwd that the user just
	// picked (and which createConversation() will persist on first message).
	let workingDirectory = $derived(
		activeConversation()?.workingDirectory ?? pendingWorkingDirectory() ?? null
	);

	async function handleWorkingDirectoryChange(value: string | null) {
		await conversationsStore.setWorkingDirectory(value);
		// If the conversation has already started, drop a synthetic
		// `set_working_directory` tool call into chat history so the model
		// sees the change on its next turn. Pending mode (no active conv
		// yet) is handled in `chatStore.sendMessage` at first-send time.
		if (conversationsStore.activeConversation) {
			const trimmed = value?.trim();
			await chatStore.recordUserToolCall(
				BuiltInTool.SET_WORKING_DIRECTORY,
				{ path: trimmed ?? '' },
				{
					content: trimmed ? `Working directory set to: ${trimmed}` : 'Working directory cleared',
					isError: false
				}
			);
		}
	}

	// Resource Dialog State
	let isResourceDialogOpen = $state(false);
	let preSelectedResourceUri = $state<string | undefined>(undefined);

	let currentConfig = $derived(config());

	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);
		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	let isRouter = $derived(isRouterMode());
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);
	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	export function focus() {
		textareaRef?.focus();
	}

	export function resetTextareaHeight() {
		textareaRef?.resetHeight();
	}

	export function openModelSelector() {
		chatFormActionsRef?.openModelSelector();
	}

	export function checkModelSelected(): boolean {
		if (!hasModelSelected) {
			chatFormActionsRef?.openModelSelector();
			return false;
		}
		return true;
	}

	function handleFileSelect(files: File[]) {
		onFilesAdd?.(files);
	}

	function handleFileUpload() {
		fileInputRef?.click();
	}

	function handleFileRemove(fileId: string) {
		if (fileId.startsWith('attachment-')) {
			const index = parseInt(fileId.replace('attachment-', ''), 10);
			if (!isNaN(index) && index >= 0 && index < attachments.length) {
				onAttachmentRemove?.(index);
			}
		} else {
			onUploadedFileRemove?.(fileId);
		}
	}

	function handleInput() {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
		const hasServers = mcpStore.hasEnabledServers(perChatOverrides);
		const cursor = textareaRef?.getElement()?.selectionStart ?? value.length;

		if (value.startsWith(PROMPT_TRIGGER_PREFIX) && hasServers) {
			isPromptPickerOpen = true;
			promptSearchQuery = value.slice(1);
			isMentionPickerOpen = false;
			mentionQuery = '';
		} else {
			const token = findMentionToken(value, cursor);

			if (token) {
				// Picker's been dismissed for THIS exact token - honor the
				// "literal until delete + retype" rule: don't reopen until the
				// token changes (typed-then-Esc'd a slot, then kept typing
				// inside the same `@<q>`).
				const isDismissedSticky =
					mentionDismissedSnapshot !== null &&
					mentionDismissedSnapshot.start === token.start &&
					mentionDismissedSnapshot.query === token.query;

				if (!isDismissedSticky) {
					mentionDismissedSnapshot = null;
					isMentionPickerOpen = true;
					mentionQuery = token.query;
					isPromptPickerOpen = false;
					promptSearchQuery = '';
					return;
				}
			}

			isPromptPickerOpen = false;
			promptSearchQuery = '';
			isMentionPickerOpen = false;
			mentionQuery = '';

			// Token gone or no longer intact - the snapshot is stale. Reset so
			// the next fresh `@` opens immediately even at the same offset.
			if (mentionDismissedSnapshot !== null && !token) {
				mentionDismissedSnapshot = null;
			}
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (pickersRef?.handleKeydown(event)) {
			return;
		}

		if (event.key === KeyboardKey.ESCAPE && isPromptPickerOpen) {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			const sendOnEnter = currentConfig.sendOnEnter !== false;

			if (sendOnEnter || isModifier) {
				event.preventDefault();

				if (!canSubmit || disabled || hasLoadingAttachments) return;

				onSubmit?.();
			}
		}
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFilesAdd?.(files);
			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (text.startsWith(CLIPBOARD_CONTENT_QUOTE_PREFIX)) {
			const parsed = parseClipboardContent(text);

			if (parsed.textAttachments.length > 0 || parsed.mcpPromptAttachments.length > 0) {
				event.preventDefault();
				value = parsed.message;
				onValueChange?.(parsed.message);

				// Handle text attachments as files
				if (parsed.textAttachments.length > 0) {
					const attachmentFiles = parsed.textAttachments.map(
						(att) =>
							new File([att.content], att.name, {
								type: MimeTypeText.PLAIN
							})
					);
					onFilesAdd?.(attachmentFiles);
				}

				// Handle MCP prompt attachments as ChatUploadedFile with mcpPrompt data
				if (parsed.mcpPromptAttachments.length > 0) {
					const mcpPromptFiles: ChatUploadedFile[] = parsed.mcpPromptAttachments.map((att) => ({
						id: uuid(),
						name: att.name,
						size: att.content.length,
						type: SpecialFileType.MCP_PROMPT,
						file: new File([att.content], `${att.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						isLoading: false,
						textContent: att.content,
						mcpPrompt: {
							serverName: att.serverName,
							promptName: att.promptName,
							arguments: att.arguments
						}
					}));

					uploadedFiles = [...uploadedFiles, ...mcpPromptFiles];
					onUploadedFilesChange?.(uploadedFiles);
				}

				setTimeout(() => {
					textareaRef?.focus();
				}, 10);

				return;
			}
		}

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			onFilesAdd?.([textFile]);
		}
	}

	function handlePromptLoadStart(
		placeholderId: string,
		promptInfo: MCPPromptInfo,
		args?: Record<string, string>
	) {
		// Only clear the value if the prompt was triggered by typing '/'
		if (value.startsWith(PROMPT_TRIGGER_PREFIX)) {
			value = '';
			onValueChange?.('');
		}
		isPromptPickerOpen = false;
		promptSearchQuery = '';

		const promptName = promptInfo.title || promptInfo.name;
		const placeholder: ChatUploadedFile = {
			id: placeholderId,
			name: promptName,
			size: INITIAL_FILE_SIZE,
			type: SpecialFileType.MCP_PROMPT,
			file: new File([], 'loading'),
			isLoading: true,
			mcpPrompt: {
				serverName: promptInfo.serverName,
				promptName: promptInfo.name,
				arguments: args ? { ...args } : undefined
			}
		};

		uploadedFiles = [...uploadedFiles, placeholder];
		onUploadedFilesChange?.(uploadedFiles);
		textareaRef?.focus();
	}

	function handlePromptLoadComplete(placeholderId: string, result: GetPromptResult) {
		const promptText = result.messages
			?.map((msg: PromptMessage) => {
				if (typeof msg.content === 'string') {
					return msg.content;
				}

				if (msg.content.type === ContentPartType.TEXT) {
					return msg.content.text;
				}

				return '';
			})
			.filter(Boolean)
			.join(PROMPT_CONTENT_SEPARATOR);

		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId
				? {
						...f,
						isLoading: false,
						textContent: promptText,
						size: promptText.length,
						file: new File([promptText], `${f.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						})
					}
				: f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	function handlePromptLoadError(placeholderId: string, error: string) {
		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId ? { ...f, isLoading: false, loadError: error } : f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	function handlePromptPickerClose() {
		isPromptPickerOpen = false;
		promptSearchQuery = '';
		textareaRef?.focus();
	}

	/**
	 * Mention picker dismissed (Esc, outside-click, or selection-complete).
	 * Capture a `(start, query)` snapshot of the live token so subsequent
	 * input events that produce the SAME token won't reopen the picker -
	 * the user has explicitly told us that `@<query>` should be literal
	 * until they delete or retype a fresh `@`.
	 */
	function handleMentionPickerClose() {
		if (isMentionPickerOpen) {
			const cursor = textareaRef?.getElement()?.selectionStart ?? value.length;
			mentionDismissedSnapshot = takeMentionDismissSnapshot(value, cursor);
		}
		isMentionPickerOpen = false;
		mentionQuery = '';
	}

	/**
	 * Selection from the mention picker: splice `[name](file://<abs>)`
	 * + trailing space in place of the `@<query>` token. Cursor lands
	 * right after the trailing space so the user can keep typing
	 * naturally. Uses the live cursor position (not the stale snapshot)
	 * because the token might have been edited since we last saw it.
	 *
	 * URI shape follows RFC 8089: `file:` + `//` + absolute path. The
	 * search entry's `path` is already rooted (begins with `/`), so the
	 * prefix is `file://` not `file:///` - that yields the canonical
	 * three-slash form `file:///Users/foo/bar` without an extra `/`.
	 *
	 * Directories get a trailing `/` so the link resolves to a folder
	 * rather than being interpreted as a file with no extension.
	 */
	function handleMentionSelect(entry: ApiFilesystemSearchEntry) {
		const cursor = textareaRef?.getElement()?.selectionStart ?? value.length;
		const token = findMentionToken(value, cursor);
		if (!token) return;

		const basename = lastPathSegment(entry.path) || entry.name;
		const pathWithSeparator = entry.type === 'directory' ? `${entry.path}/` : entry.path;
		const insertion = `[${basename}](file://${pathWithSeparator}) `;
		const newValue = value.slice(0, token.start) + insertion + value.slice(token.end);
		const cursorOffset = token.start + insertion.length;

		value = newValue;
		onValueChange?.(newValue);

		// Place the caret right after the trailing space we just inserted.
		queueMicrotask(() => {
			textareaRef?.getElement()?.setSelectionRange(cursorOffset, cursorOffset);
		});
	}

	async function handleMicClick() {
		if (!audioRecorder || !recordingSupported) {
			console.warn('Audio recording not supported');
			return;
		}

		if (isRecording) {
			isRecording = false;
			try {
				const audioBlob = await audioRecorder.stopRecording();
				const wavBlob = await convertToWav(audioBlob);
				const audioFile = createAudioFile(wavBlob);

				onFilesAdd?.([audioFile]);
			} catch (error) {
				console.error('Failed to stop recording:', error);
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (error) {
				console.error('Failed to start recording:', error);
			}
		}
	}
</script>

<ChatFormFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	class="relative {className}"
	onsubmit={(event) => {
		event.preventDefault();

		if (!canSubmit || disabled || hasLoadingAttachments) return;

		onSubmit?.();
	}}
>
	<ChatFormPickers
		bind:this={pickersRef}
		{isPromptPickerOpen}
		{promptSearchQuery}
		{isMentionPickerOpen}
		{mentionQuery}
		{mentionAnchor}
		onPromptPickerClose={handlePromptPickerClose}
		onMentionPickerClose={handleMentionPickerClose}
		onMentionSelect={handleMentionSelect}
		onPromptLoadStart={handlePromptLoadStart}
		onPromptLoadComplete={handlePromptLoadComplete}
		onPromptLoadError={handlePromptLoadError}
	/>

	<div
		bind:this={mentionAnchor}
		class="pointer-events-none absolute top-0 right-0 left-0 h-px"
		aria-hidden="true"
	></div>

	<div
		class="{INPUT_CLASSES} overflow-hidden rounded-4xl md:rounded-3xl backdrop-blur-md {disabled
			? 'cursor-not-allowed opacity-60'
			: ''}"
		data-slot="input-area"
	>
		<ChatAttachmentsList
			{attachments}
			bind:uploadedFiles
			onFileRemove={handleFileRemove}
			limitToSingleRow
			class="py-5"
			style="scroll-padding: 1rem;"
			activeModelId={activeModelId ?? undefined}
		/>

		<div
			class="flex-column relative min-h-12 items-center rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3!"
			onpaste={handlePaste}
		>
			<ChatFormTextarea
				class="px-5 py-1.5 md:pt-0"
				bind:this={textareaRef}
				bind:value
				onKeydown={handleKeydown}
				onInput={() => {
					handleInput();
					onValueChange?.(value);
				}}
				{disabled}
				{placeholder}
			/>

			{#if mcpHasResourceAttachments()}
				<ChatFormMcpResourcesList
					class="mb-3"
					onResourceClick={(uri) => {
						preSelectedResourceUri = uri;
						isResourceDialogOpen = true;
					}}
				/>
			{/if}

			<ChatFormActions
				class="px-3"
				bind:this={chatFormActionsRef}
				canSend={canSubmit}
				{disabled}
				{isLoading}
				isReasoning={chatStore.isReasoning}
				{isRecording}
				{showAddButton}
				{showModelSelector}
				{uploadedFiles}
				onFileUpload={handleFileUpload}
				onMicClick={handleMicClick}
				{onStop}
				onSystemPromptClick={() => onSystemPromptClick?.({ message: value, files: uploadedFiles })}
				onMcpPromptClick={showMcpPromptButton ? () => (isPromptPickerOpen = true) : undefined}
				onMcpResourcesClick={() => (isResourceDialogOpen = true)}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	<ChatFormWorkingDirectory
		directory={workingDirectory}
		onChange={handleWorkingDirectoryChange}
		{disabled}
	/>
</form>

<DialogMcpResourcesBrowser
	bind:open={isResourceDialogOpen}
	preSelectedUri={preSelectedResourceUri}
	onAttach={(resource: MCPResourceInfo) => {
		mcpStore.attachResource(resource.uri);
	}}
	onOpenChange={(newOpen: boolean) => {
		if (!newOpen) {
			preSelectedResourceUri = undefined;
		}
	}}
/>
