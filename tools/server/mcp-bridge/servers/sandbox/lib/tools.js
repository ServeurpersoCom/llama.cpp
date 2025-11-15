/**
 * Sandbox Tools Module
 * Provides isolated code execution in a Podman container
 * Isomorphic to Anthropic's Claude computer-use tools
 *
 * This module implements four core tools for secure code execution:
 *
 * - bash_tool: Execute arbitrary bash commands in isolated container
 *   - Automatic output truncation for large results
 *   - Execution timing and exit code reporting
 *   - Smart truncation preserving line boundaries
 *
 * - view: Read files or list directories with intelligent handling
 *   - Directory listing with size information (2 levels deep)
 *   - Binary file detection and rejection
 *   - Line-numbered file display with optional range selection
 *   - Filters hidden files and node_modules
 *
 * - create_file: Create new files with automatic directory creation
 *   - Base64 encoding to handle special characters safely
 *   - Automatic parent directory creation
 *   - Atomic file writing
 *
 * - str_replace: Replace unique strings in files with validation
 *   - Enforces string uniqueness (prevents ambiguous replacements)
 *   - Safe base64 encoding for content preservation
 *   - Detailed error reporting for not found or multiple occurrences
 *
 * All tools execute commands via Podman in an isolated container environment,
 * providing security through containerization while maintaining full filesystem
 * access within the container boundaries.
 *
 * Configuration is managed through the podman-executor module, allowing
 * customization of container name, user, timeouts, and output limits.
 */

const fs = require('fs');
const path = require('path');
const config = require('../config.json');
const { podmanExec, escapeShell, escapeRegex } = require('./podman-executor');

function textResult(text, isError = false) {
	return {
		type: 'text',
		text: text == null ? '' : String(text),
		isError: Boolean(isError)
	};
}

function imageResult(data, mimeType, isError = false) {
	return {
		type: 'image',
		data: data == null ? '' : String(data),
		mimeType: mimeType || 'application/octet-stream',
		isError: Boolean(isError)
	};
}

// Load tools definitions from tools.json once at startup
const toolsDefinitionPath = path.join(__dirname, '..', 'tools.json');
const rawDefinitions = JSON.parse(fs.readFileSync(toolsDefinitionPath, 'utf8'));
const TOOLS_DEFINITIONS = rawDefinitions.map((tool) => ({
	name: tool.function.name,
	description: tool.function.description,
	inputSchema: tool.function.parameters
}));

/**
 * Tool: bash_tool
 * Execute bash command in container with output truncation
 * @param {object} args - Tool arguments
 * @param {string} args.command - Bash command to execute
 * @returns {Promise<string>} Tool result
 */
async function tool_bash(args = {}) {
	const cmd = args.command;

	if (!cmd) {
		return textResult('❌ Command argument required', true);
	}

	if (!args.description) {
		return textResult('❌ Description argument required', true);
	}

	const t0 = Date.now();
	const result = await podmanExec(cmd);
	const elapsed = Date.now() - t0;

	let output = result.stdout.replace(/\n$/, '');

	if (output.length > config.podman.bashOutputLimitBytes) {
		let tail = output.slice(-config.podman.bashOutputLimitBytes);

		const firstNewline = tail.indexOf('\n');
		if (firstNewline !== -1 && firstNewline < 512) {
			tail = tail.slice(firstNewline + 1);
		}

		const truncatedBytes = output.length - tail.length;
		output = tail + `\n⚠️ Long output with ${truncatedBytes} bytes hidden from context`;
	}

	const statusEmoji = result.exitCode === 0 ? '✅' : '❌';
	const newLine = output ? '\n' : '';
	const justification = args.description ? `🎯 ${args.description}\n` : '';
	return textResult(
		`${output}${newLine}${justification}#️⃣ ${cmd}\n${statusEmoji} Exit code ${result.exitCode} (${elapsed} ms)`,
		result.exitCode !== 0
	);
}

/**
 * Tool: view
 * Display file content or list directory
 * @param {object} args - Tool arguments
 * @param {string} args.path - File or directory path
 * @param {Array<number>} [args.view_range] - Optional line range [start, end]
 * @returns {Promise<string>} Tool result
 */
async function tool_view(args = {}) {
	const filepath = args.path?.trim();
	const range = args.view_range;

	if (!filepath) {
		return textResult('❌ Path argument required', true);
	}

	if (!args.description) {
		return textResult('❌ Description argument required', true);
	}

	const testScript = `if [ -d ${escapeShell(filepath)} ]; then echo DIR; else echo FILE; fi`;
	const testResult = await podmanExec(testScript);

	const isDir = testResult.stdout.trim() === 'DIR';

	if (isDir) {
		const script = `find ${escapeShell(filepath)} -maxdepth 2 ! -path '*/.*' ! -path '*/node_modules/*' -exec du -sh {} \\; 2>/dev/null | sort -k2`;
		const result = await podmanExec(script);

		const justification = args.description ? `🎯 ${args.description}\n` : '';
		return textResult(result.stdout + `${justification}👁️ Directory listing of ${filepath}`);
	}

	const mimeScript = `file --mime-encoding ${escapeShell(filepath)} 2>/dev/null`;
	const mimeResult = await podmanExec(mimeScript);
	const mimeLine = mimeResult.stdout.trim();
	const mimeTypeScript = `file --mime-type ${escapeShell(filepath)} 2>/dev/null`;
	const mimeTypeResult = await podmanExec(mimeTypeScript);
	const mimeType = mimeTypeResult.stdout.split(':').pop()?.trim().toLowerCase();

	const supportedImageTypes = new Set([
		'image/png',
		'image/jpeg',
		'image/jpg',
		'image/gif',
		'image/webp'
	]);

	if (mimeLine.toLowerCase().includes('binary')) {
		if (mimeType && supportedImageTypes.has(mimeType)) {
			const base64Script = `base64 -w 0 ${escapeShell(filepath)} 2>/dev/null`;
			const base64Result = await podmanExec(base64Script);

			if (base64Result.exitCode !== 0) {
				return textResult(
					base64Result.stdout + `❌ Error encoding image ${filepath}`,
					true
				);
			}

			const base64Data = base64Result.stdout.replace(/\s+/g, '');
			return imageResult(base64Data, mimeType);
		}

		return textResult(`❌ Binary file detected (${mimeLine})`, true);
	}

	const catScript = `cat ${escapeShell(filepath)}`;
	const catResult = await podmanExec(catScript);

	if (catResult.exitCode !== 0) {
		return textResult(catResult.stdout + `❌ Error reading file ${filepath}`, true);
	}

	const lines = catResult.stdout.replace(/\n$/, '').split('\n');
	const totalLines = lines.length;

	let startLine = 1;
	let endLine = totalLines;

	if (range && Array.isArray(range) && range.length === 2) {
		startLine = Math.max(1, parseInt(range[0], 10) || 1);
		endLine =
			range[1] === -1
				? totalLines
				: Math.min(totalLines, parseInt(range[1], 10) || totalLines);
	}

	const selectedLines = lines.slice(startLine - 1, endLine);

	const numberedLines = selectedLines
		.map((line, idx) => {
			const lineNum = startLine + idx;
			return `${lineNum}\t${line}`;
		})
		.join('\n');

	const justification = args.description ? `🎯 ${args.description}\n` : '';
	if (range && Array.isArray(range) && range.length === 2) {
		return textResult(
			`${numberedLines}\n${justification}👁️ File content of ${filepath} (lines ${startLine}-${endLine} out of ${totalLines})`
		);
	}

	return textResult(
		`${numberedLines}\n${justification}👁️ File content of ${filepath} (${totalLines} lines)`
	);
}

/**
 * Tool: create_file
 * Create new file with automatic directory creation and base64 safety
 * @param {object} args - Tool arguments
 * @param {string} args.path - File path
 * @param {string} args.file_text - File contents
 * @returns {Promise<string>} Tool result
 */
async function tool_create_file(args = {}) {
	const filepath = args.path?.trim();
	const content = args.file_text;

	if (!filepath || content === undefined) {
		return textResult('❌ Path and content arguments required', true);
	}

	if (!args.description) {
		return textResult('❌ Description argument required', true);
	}

	const b64 = Buffer.from(content).toString('base64');
	const dirname = filepath.split('/').slice(0, -1).join('/') || '/';

	const script = `mkdir -p ${escapeShell(dirname)} && echo '${b64}' | base64 -d > ${escapeShell(filepath)}`;
	const result = await podmanExec(script);

	if (result.exitCode !== 0) {
		return textResult(result.stdout + `❌ Error creating file ${filepath}`, true);
	}

	const size = content.length;
	const justification = args.description ? `🎯 ${args.description}\n` : '';
	return textResult(`${justification}✨ File ${filepath} created (${size} bytes)`);
}

/**
 * Tool: str_replace
 * Replace unique string in file with validation and base64-safe write
 * @param {object} args - Tool arguments
 * @param {string} args.path - File path
 * @param {string} args.old_str - String to replace
 * @param {string} [args.new_str] - Replacement string (default: empty)
 * @returns {Promise<string>} Tool result
 */
async function tool_str_replace(args = {}) {
	const filepath = args.path?.trim();
	const oldStr = args.old_str;
	const newStr = args.new_str ?? '';

	if (!filepath || oldStr === undefined) {
		return textResult('❌ Path and old_str arguments required', true);
	}

	if (!args.description) {
		return textResult('❌ Description argument required', true);
	}

	// Read file contents via podman
	const readScript = `cat ${escapeShell(filepath)}`;
	const readResult = await podmanExec(readScript);

	if (readResult.exitCode !== 0) {
		return textResult(readResult.stdout + `❌ Error reading file ${filepath}`, true);
	}

	const content = readResult.stdout;
	const occurrences = (content.match(new RegExp(escapeRegex(oldStr), 'g')) || []).length;

	if (occurrences === 0) {
		return textResult(`❌ String "${oldStr}" not found in ${filepath}`, true);
	}

	if (occurrences > 1) {
		return textResult(
			`❌ String "${oldStr}" found ${occurrences} times in ${filepath} (must be unique)`,
			true
		);
	}

	const newContent = content.replace(oldStr, newStr);
	const b64 = Buffer.from(newContent).toString('base64');
	const writeScript = `echo '${b64}' | base64 -d > ${escapeShell(filepath)}`;
	const writeResult = await podmanExec(writeScript);

	if (writeResult.exitCode !== 0) {
		return textResult(writeResult.stdout + `❌ Error writing file ${filepath}`, true);
	}

	const oldLen = oldStr.length;
	const newLen = newStr.length;
	const justification = args.description ? `🎯 ${args.description}\n` : '';
	return textResult(
		`${justification}🔄 Replacement done in ${filepath} (${oldLen} -> ${newLen} bytes)`
	);
}

const TOOLS_MAPPING = {
	bash_tool: tool_bash,
	view: tool_view,
	create_file: tool_create_file,
	str_replace: tool_str_replace
};

module.exports = {
	TOOLS_DEFINITIONS,
	TOOLS_MAPPING
};
