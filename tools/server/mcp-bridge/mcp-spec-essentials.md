# MCP Specification 2025-06-18 - Essential Reference

> **Developer synthesis for implementation** - Essential pages extracted and compiled from the official specification at https://modelcontextprotocol.io/specification/2025-06-18
>
> This document focuses on core protocol requirements for building MCP clients and servers, omitting optional features like OAuth authorization, prompts, sampling, and advanced utilities.

---

# Overview

Protocol Revision: 2025-06-18

Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools.

## Key Architecture

MCP uses JSON-RPC 2.0 messages to establish communication between:
- **Hosts**: LLM applications that initiate connections
- **Clients**: Connectors within the host application
- **Servers**: Services that provide context and capabilities

## Base Protocol

- JSON-RPC message format
- Stateful connections
- Server and client capability negotiation

## Core Features

Servers offer these features to clients:
- **Resources**: Context and data for user or AI model
- **Prompts**: Templated messages and workflows
- **Tools**: Functions for AI model to execute
- **Sampling**: Server-initiated agentic behaviors
- **Roots**: Server-initiated inquiries into uri/filesystem boundaries
- **Elicitation**: Server-initiated requests for user information

## Security Principles

1. **User Consent**: Users must explicitly consent to all data access and operations
2. **Data Privacy**: Hosts must obtain consent before exposing user data
3. **Tool Safety**: Tools represent arbitrary code execution, require explicit consent
4. **LLM Sampling Controls**: Users must approve sampling requests

---

# Architecture

The Model Context Protocol follows a **client-host-server architecture** where each host can run multiple client instances.

## Core Components

### Hosts
- LLM applications (Claude Desktop, IDEs, AI tools)
- Maintain user interactions and communication
- Can run multiple client instances
- In practice, hosts often embed one or several protocol clients, each maintaining an independent server session

### Clients
- Protocol clients within host application
- 1:1 relationship with servers
- Maintain independent server connections
- Isolate server interactions

### Servers
- Lightweight programs exposing specific capabilities
- Provide context, tools, prompts to clients
- Stateful sessions lasting duration of connection
- Can serve multiple clients simultaneously

## Design Principles

### Servers as Lightweight Services
- Host multiple servers simultaneously
- Servers provide focused, specific capabilities
- Composable and modular architecture

### Capability Negotiation
- Explicit feature declaration during initialization
- Clients and servers advertise supported capabilities
- Features enabled based on mutual support

Examples:
- Implemented server features must be advertised in capabilities
- Resource subscriptions require server to declare support
- Tool invocation requires server to declare tool capabilities

---

# Lifecycle

Protocol Revision: 2025-06-18

MCP defines a rigorous lifecycle for client-server connections ensuring proper capability negotiation and state management.

## Lifecycle Phases

### 1. Initialization
- Capability negotiation
- Protocol version agreement
- Server information exchange

### 2. Operation
- Normal protocol communication
- Request/response exchanges
- Notifications

### 3. Shutdown
- Clean connection termination

## Initialization Phase

The client MUST initiate the connection by sending an `initialize` request as the **first message**.

### Initialize Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "roots": {
        "listChanged": true
      },
      "sampling": {}
    },
    "clientInfo": {
      "name": "ExampleClient",
      "version": "1.0.0"
    }
  }
}
```

### Initialize Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "logging": {},
      "prompts": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      },
      "tools": {
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "ExampleServer",
      "version": "1.0.0"
    }
  }
}
```

### Version Negotiation

- Client proposes protocol version in `initialize`
- Server responds with version it will use
- Server MUST NOT use higher version than client proposed
- If server cannot support client's version, error returned
- When using HTTP transport, the `protocolVersion` field MUST correspond to the `MCP-Protocol-Version` HTTP header

### Capability Negotiation

Both parties declare capabilities:
- **Client capabilities**: Features client supports
- **Server capabilities**: Features server provides

Common capabilities:
- `roots`: Filesystem roots support
- `sampling`: LLM sampling support (allows servers to request model-driven decision making or multi-step agentic behaviors)
- `prompts`: Prompt templates
- `resources`: Resource access
- `tools`: Tool execution
- `logging`: Logging support

### Initialized Notification

After successful `initialize`, client MUST send `initialized` notification:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

This signals client is ready for normal operations.

## Operation Phase

After initialization, normal protocol operations begin:
- Requests/responses
- Notifications
- Tool calls
- Resource access

## Shutdown Phase

Clean termination of protocol connection.

### stdio Transport
Client initiates shutdown by:
1. Closing input stream to server process
2. Waiting for server exit
3. Sending SIGTERM if needed
4. Sending SIGKILL if SIGTERM fails

Server MAY initiate by closing output stream and exiting.

### HTTP Transports
Shutdown indicated by closing HTTP connections.

## Timeouts

Implementations SHOULD establish timeouts for all requests to prevent hung connections.

When timeout occurs:
1. Sender SHOULD issue cancellation notification
2. Sender SHOULD treat request as failed

## Error Handling

When initialize fails:
- Connection SHOULD be terminated
- No further messages sent
- Client MAY retry with different parameters

---

# Transports

Protocol Revision: 2025-06-18

MCP uses JSON-RPC to encode messages. JSON-RPC messages MUST be UTF-8 encoded.

## Transport Mechanisms

Two standard transports:
1. **stdio** - Standard input/output (RECOMMENDED for local)
2. **Streamable HTTP** - HTTP POST/GET with SSE

Clients SHOULD support stdio whenever possible.

## stdio Transport

### How It Works

- Client launches MCP server as subprocess
- Server reads JSON-RPC from stdin
- Server writes JSON-RPC to stdout
- Messages delimited by newlines
- Messages MUST NOT contain embedded newlines

### Logging

- Server MAY write UTF-8 to stderr for logging
- Clients MAY capture, forward, or ignore stderr
- Server MUST NOT write non-MCP to stdout
- Client MUST NOT write non-MCP to stdin

### Message Flow

```
Client
  ↓ Launch subprocess
Server Process
  ↓ stdin  ← Client writes JSON-RPC
  ↓ stdout → Client reads JSON-RPC
  ↓ stderr → Optional logs
```

### Example

Client sends (via stdin):
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
```

Server responds (via stdout):
```json
{"jsonrpc":"2.0","id":1,"result":{...}}
```

Server logs (via stderr):
```
[Server] Initialized successfully
```

## Streamable HTTP Transport

### Overview

Server operates independently, handles multiple clients.
Uses HTTP POST/GET with optional Server-Sent Events (SSE).

### MCP Endpoint

Server MUST provide single HTTP endpoint supporting POST and GET.
Example: `https://example.com/mcp`

### Security Warning

When implementing HTTP transport:
- MUST validate Origin header (prevent DNS rebinding)
- SHOULD bind to localhost only when local
- SHOULD implement authentication

### Sending Messages (Client → Server)

Every JSON-RPC message = new HTTP POST to MCP endpoint.

**Requirements:**
- POST to MCP endpoint
- Include `Accept: application/json, text/event-stream`
- Body = single JSON-RPC request/notification/response

**Response Types:**

For notifications/responses:
- 202 Accepted (success)
- 4xx error (failure)

For requests:
- `Content-Type: text/event-stream` (SSE stream)
- `Content-Type: application/json` (single response)

### Listening for Messages (Server → Client)

Client MAY issue HTTP GET to MCP endpoint for SSE stream.

**Requirements:**
- GET to MCP endpoint
- Include `Accept: text/event-stream`

**Response:**
- `Content-Type: text/event-stream` (stream opened)
- HTTP 405 (no SSE support)

### Session Management

Server MAY assign session ID during initialization via `Mcp-Session-Id` header.

**Session ID Requirements:**
- Globally unique
- Cryptographically secure
- Only visible ASCII (0x21 to 0x7E)

**Client Requirements:**
- Include `Mcp-Session-Id` in all subsequent requests
- Start new session on HTTP 404

**Termination:**
- Client sends HTTP DELETE with session ID
- Server MAY respond 405 if termination not allowed

### Protocol Version Header

When using HTTP, client MUST include:
```
MCP-Protocol-Version: 2025-06-18
```

On all requests after initialization.

If missing, server SHOULD assume `2025-03-26`.
If invalid/unsupported, server MUST respond 400.

## Custom Transports

Implementers MAY create custom transports.

**Requirements:**
- Preserve JSON-RPC message format
- Maintain lifecycle requirements
- Document connection patterns

---

# Tools

Protocol Revision: 2025-06-18

MCP allows servers to expose tools that language models can invoke.

## Overview

Tools enable models to interact with external systems:
- Query databases
- Call APIs
- Perform computations

Each tool has:
- Unique name identifier
- Metadata with schema
- Input parameters schema
- Optional output schema

## Design Philosophy

**Model-Controlled**: LLM discovers and invokes tools automatically based on context.

**Human-in-the-Loop**: For trust & safety, there SHOULD always be human approval for tool invocations.

## Capabilities

Server declares tool support:

```json
{
  "capabilities": {
    "tools": {
      "listChanged": true
    }
  }
}
```

`listChanged`: Server will emit `notifications/tools/list_changed` when tools change.

## Protocol Messages

### Listing Tools

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list"
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "inputSchema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            }
          },
          "required": ["location"]
        }
      }
    ]
  }
}
```

### Calling Tools

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "San Francisco"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather in San Francisco: 18°C, partly cloudy"
      }
    ]
  }
}
```

### List Changed Notification

When tools list changes:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

Client SHOULD call `tools/list` to get updated list.

## Input Schema

Tools MUST define input schema using JSON Schema.

Schemas SHOULD conform to JSON Schema Draft 7 or later.

**Example:**
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "limit": {
      "type": "number",
      "description": "Max results",
      "minimum": 1,
      "maximum": 100
    }
  },
  "required": ["query"]
}
```

## Output Schema

Tools MAY define output schema for structured results.

**Example:**
```json
{
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": {
        "type": "number"
      },
      "conditions": {
        "type": "string"
      }
    }
  }
}
```

When provided:
- Servers MUST return results conforming to schema
- Clients SHOULD validate results against schema

## Result Types

Tool results can contain multiple content types:

### Text Content
```json
{
  "type": "text",
  "text": "Result text"
}
```

### Image Content
```json
{
  "type": "image",
  "data": "base64-encoded-data",
  "mimeType": "image/png"
}
```

### Resource Content
```json
{
  "type": "resource",
  "resource": {
    "uri": "file:///path/to/file",
    "mimeType": "text/plain",
    "text": "File contents"
  }
}
```

### Structured Content (NEW in 2025-06-18)

For backwards compatibility, tools returning structured content SHOULD also return serialized JSON in TextContent.

## Error Handling

Tool execution errors returned as JSON-RPC errors:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32000,
    "message": "Tool execution failed",
    "data": {
      "details": "Connection timeout"
    }
  }
}
```

## Security Considerations

**Applications SHOULD:**
- Provide UI showing which tools are exposed
- Require explicit user approval before invocation
- Show tool results to users
- Allow users to deny tool calls

**Servers SHOULD:**
- Validate all inputs
- Implement appropriate access controls
- Rate limit tool calls
- Log tool invocations for audit

---

# Resources

Protocol Revision: 2025-06-18

Resources provide context and data to language models through MCP.

## Overview

Resources are data that servers expose to clients:
- Files
- Database records
- API responses
- Live system data

Each resource has:
- Unique URI identifier
- MIME type
- Metadata
- Optional text or binary content

## Capabilities

Server declares resource support:

```json
{
  "capabilities": {
    "resources": {
      "subscribe": true,
      "listChanged": true
    }
  }
}
```

- `subscribe`: Clients can subscribe to resource updates
- `listChanged`: Server will notify when list changes

## Protocol Messages

### Listing Resources

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list"
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resources": [
      {
        "uri": "file:///project/README.md",
        "name": "Project README",
        "description": "Main project documentation",
        "mimeType": "text/markdown"
      }
    ]
  }
}
```

### Reading Resources

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "resources/read",
  "params": {
    "uri": "file:///project/README.md"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "contents": [
      {
        "uri": "file:///project/README.md",
        "mimeType": "text/markdown",
        "text": "# Project\n\nDocumentation here..."
      }
    ]
  }
}
```

### Resource Templates

Resources can use URI templates:

```json
{
  "uri": "file:///logs/{date}.log",
  "name": "Daily logs",
  "description": "Logs for specific date (YYYY-MM-DD)"
}
```

Client reads by filling template:
```
file:///logs/2025-06-18.log
```

### Subscribing to Resources

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "resources/subscribe",
  "params": {
    "uri": "file:///project/config.json"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {}
}
```

### Update Notifications

When subscribed resource changes:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///project/config.json"
  }
}
```

Client SHOULD call `resources/read` to get new content.

### List Changed Notification

When available resources change:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/list_changed"
}
```

## Resource Annotations

Resources can include annotations:

```json
{
  "uri": "file:///data.csv",
  "mimeType": "text/csv",
  "annotations": {
    "audience": ["user", "assistant"],
    "priority": 0.8
  }
}
```

- `audience`: Who should see this (user/assistant)
- `priority`: Relative importance (0.0-1.0)

---

# Schema Reference

## Core Types

### JSON-RPC Message

All MCP messages follow JSON-RPC 2.0:

```typescript
{
  jsonrpc: "2.0",
  id?: string | number,
  method?: string,
  params?: object,
  result?: any,
  error?: {
    code: number,
    message: string,
    data?: any
  }
}
```

### Result Types

```typescript
type Result = {
  content: Content[],
  isError?: boolean
}

type Content =
  | TextContent
  | ImageContent
  | ResourceContent

type TextContent = {
  type: "text",
  text: string
}

type ImageContent = {
  type: "image",
  data: string,        // base64
  mimeType: string
}

type ResourceContent = {
  type: "resource",
  resource: {
    uri: string,
    mimeType?: string,
    text?: string,
    blob?: string      // base64
  }
}
```

### Tool Definition

```typescript
type Tool = {
  name: string,
  description: string,
  inputSchema: JSONSchema,
  outputSchema?: JSONSchema
}
```

### Resource Definition

```typescript
type Resource = {
  uri: string,
  name: string,
  description?: string,
  mimeType?: string,
  annotations?: {
    audience?: ("user" | "assistant")[],
    priority?: number
  }
}
```

### Capabilities

```typescript
type ServerCapabilities = {
  prompts?: {
    listChanged?: boolean
  },
  resources?: {
    subscribe?: boolean,
    listChanged?: boolean
  },
  tools?: {
    listChanged?: boolean
  },
  logging?: {},
  experimental?: Record<string, object>
}

type ClientCapabilities = {
  roots?: {
    listChanged?: boolean
  },
  sampling?: {},
  experimental?: Record<string, object>
}
```

---

# Implementation Checklist

## Minimum Viable MCP Client

- [ ] stdio transport support
- [ ] initialize/initialized handshake
- [ ] Protocol version negotiation
- [ ] tools/list request
- [ ] tools/call request
- [ ] Error handling
- [ ] Process lifecycle (spawn/kill)

## Production Ready

- [ ] HTTP transport (optional)
- [ ] Resource support (optional)
- [ ] Subscriptions (optional)
- [ ] List changed notifications
- [ ] Timeout handling
- [ ] Reconnection logic
- [ ] Logging/debugging
- [ ] Capability negotiation test
- [ ] Version mismatch fallback (graceful error)

## Security

- [ ] User approval for tool calls
- [ ] Input validation
- [ ] Rate limiting
- [ ] Audit logging

---

# Quick Reference

## Common Methods

| Method | Direction | Purpose |
|--------|-----------|---------|
| `initialize` | Client → Server | Start connection |
| `initialized` | Client → Server | Confirm ready |
| `tools/list` | Client → Server | Get available tools |
| `tools/call` | Client → Server | Execute tool |
| `resources/list` | Client → Server | Get resources |
| `resources/read` | Client → Server | Read resource |
| `notifications/tools/list_changed` | Server → Client | Tools updated |
| `notifications/resources/updated` | Server → Client | Resource changed |
| `notifications/resources/list_changed` | Server → Client | Resources updated |

## Error Codes

| Code | Meaning |
|------|---------|
| -32700 | Parse error |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |
| -32000 to -32099 | Server error (reserved for implementation-defined errors) |

---

End of Essential MCP Specification Reference
