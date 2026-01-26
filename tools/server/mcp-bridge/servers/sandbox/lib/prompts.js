/**
 * Prompts module for sandbox MCP server
 * Loads prompt definitions from config.json
 */

function createPromptsModule(config) {
	const promptsConfig = config.prompts || [];

	// Build definitions for prompts/list (without messages)
	const PROMPTS_DEFINITIONS = promptsConfig.map((prompt) => ({
		name: prompt.name,
		title: prompt.title,
		description: prompt.description,
		arguments: prompt.arguments || []
	}));

	// Build mapping for prompts/get
	const PROMPTS_MAPPING = {};
	for (const prompt of promptsConfig) {
		PROMPTS_MAPPING[prompt.name] = {
			description: prompt.description,
			messages: prompt.messages || []
		};
	}

	return {
		PROMPTS_DEFINITIONS,
		PROMPTS_MAPPING
	};
}

module.exports = { createPromptsModule };
