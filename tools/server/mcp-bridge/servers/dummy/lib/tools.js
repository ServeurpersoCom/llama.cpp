const TOOLS_DEFINITIONS = [
	{
		name: 'echo',
		description: 'Echo back the input text',
		inputSchema: {
			type: 'object',
			properties: {
				text: {
					type: 'string',
					description: 'Text to echo'
				}
			},
			required: ['text']
		}
	},
	{
		name: 'add',
		description: 'Add two numbers',
		inputSchema: {
			type: 'object',
			properties: {
				a: {
					type: 'number',
					description: 'First number'
				},
				b: {
					type: 'number',
					description: 'Second number'
				}
			},
			required: ['a', 'b']
		}
	}
];

const TOOLS_MAPPING = {
	echo: async ({ text = '' } = {}) => `ECHO: ${text}`,
	add: async ({ a = 0, b = 0 } = {}) => `${a} + ${b} = ${a + b}`
};

module.exports = {
	TOOLS_DEFINITIONS,
	TOOLS_MAPPING
};
