/**
 * Formats file size in bytes to human readable format
 * @param bytes - File size in bytes
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number): string {
	if (bytes === 0) return '0 Bytes';

	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));

	return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Gets a display label for a file type
 * @param fileType - The file type/mime type
 * @returns Formatted file type label
 */
export function getFileTypeLabel(fileType: string): string {
	return fileType.split('/').pop()?.toUpperCase() || 'FILE';
}
