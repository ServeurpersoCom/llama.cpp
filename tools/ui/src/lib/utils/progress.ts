/**
 * Model progress helpers for the /models/sse surfaces (selector trigger and
 * row, chat message). A model transfers its files then loads them, and the
 * two phases report different shapes; only the unified pair is public, so a
 * surface never has to know which phase is running.
 */

import { formatFileSize } from './formatters';
import { MODEL_LOAD_STAGE_LABELS, MODEL_LOAD_TAIL_SHARE } from '$lib/constants';
import { ModelProgressKind } from '$lib/enums';

/**
 * Human label for a model load stage.
 */
function modelLoadStageLabel(stage: ApiModelLoadStage): string {
	return MODEL_LOAD_STAGE_LABELS[stage];
}

/**
 * Overall load fraction (0.0 -> 1.0) across the declared stage plan.
 * text_model fills [0, 1 - tail], each later phase owns one tail slice.
 */
function modelLoadFraction(progress: ModelLoadProgress | null): number {
	if (!progress) return 0;

	// The server may emit a progress event before the stage plan is known, so
	// `stages` can be absent. Fall back to the raw value in that case.
	const { current, stages = [], value } = progress;
	const tailCount = Math.max(stages.length - 1, 0);
	const textCeiling = 1 - tailCount * MODEL_LOAD_TAIL_SHARE;
	const idx = stages.indexOf(current);

	if (idx <= 0) {
		return value * textCeiling;
	}

	return textCeiling + (idx - 1 + value) * MODEL_LOAD_TAIL_SHARE;
}

/**
 * Single line describing load progress: active stage label and overall percent.
 * Returns null when there is no progress to show.
 */
function modelLoadProgressText(progress: ModelLoadProgress | null): string | null {
	if (!progress) return null;

	const label = modelLoadStageLabel(progress.current);

	if (!label) return null;

	return `${label} ${Math.round(modelLoadFraction(progress) * 100)}%`;
}

/**
 * Overall download fraction (0.0 -> 1.0). The server aggregates across every
 * file it transfers, so this reads its value and guards the unknown size case.
 */
function modelDownloadFraction(progress: ModelDownloadProgress | null): number {
	return progress?.value ?? 0;
}

/**
 * Single line describing download progress: transferred amount, and the total
 * with a percent once every file size is known.
 * Returns null when there is no transfer to show.
 */
function modelDownloadProgressText(progress: ModelDownloadProgress | null): string | null {
	if (!progress) return null;

	const transferred = formatFileSize(progress.downloaded);

	if (progress.value === null) {
		return `Downloading ${transferred}`;
	}

	const percent = Math.round(progress.value * 100);

	return `Downloading ${transferred} of ${formatFileSize(progress.total)} (${percent}%)`;
}

/**
 * Overall fraction (0.0 -> 1.0) of whichever phase the model is in.
 */
export function modelProgressFraction(progress: ModelProgress | null): number {
	if (!progress) return 0;

	return progress.kind === ModelProgressKind.DOWNLOAD
		? modelDownloadFraction(progress.progress)
		: modelLoadFraction(progress.progress);
}

/**
 * Single line describing whichever phase the model is in.
 * Returns null when there is nothing to show.
 */
export function modelProgressText(progress: ModelProgress | null): string | null {
	if (!progress) return null;

	return progress.kind === ModelProgressKind.DOWNLOAD
		? modelDownloadProgressText(progress.progress)
		: modelLoadProgressText(progress.progress);
}
