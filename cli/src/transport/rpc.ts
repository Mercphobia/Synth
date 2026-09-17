// JSON-RPC 2.0 types for the Synth bridge (spec 6.16).
// The TS CLI and Python runtime exchange NDJSON (one JSON object per line)
// over stdio. Each message is either a Request, Response, or Notification.

export interface RpcRequest {
  jsonrpc: "2.0";
  id: number | string;
  method: string;
  params?: Record<string, unknown> | unknown[];
}

export interface RpcResponse {
  jsonrpc: "2.0";
  id: number | string;
  result?: unknown;
  error?: RpcError;
}

export interface RpcError {
  code: number;
  message: string;
  data?: unknown;
}

export interface RpcNotification {
  jsonrpc: "2.0";
  method: string;
  params?: Record<string, unknown> | unknown[];
}

// Standard error codes (JSON-RPC 2.0 spec).
export const PARSE_ERROR = -32700;
export const INVALID_REQUEST = -32600;
export const METHOD_NOT_FOUND = -32601;
export const INVALID_PARAMS = -32602;
export const INTERNAL_ERROR = -32603;

// Synth-specific error codes (mapped to Python exit codes).
export const SYNTH_CONFIG_ERROR = -32001;
export const SYNTH_LLM_ERROR = -32004;
export const SYNTH_TOOL_ERROR = -32005;
export const SYNTH_CANCELLED = -32009;

export type RpcMessage = RpcRequest | RpcResponse | RpcNotification;

// Type guard: is this a response (has result or error)?
export function isResponse(msg: RpcMessage): msg is RpcResponse {
  return "id" in msg && ("result" in msg || "error" in msg);
}

// Type guard: is this a request (has method + id)?
export function isRequest(msg: RpcMessage): msg is RpcRequest {
  return "method" in msg && "id" in msg && !("result" in msg) && !("error" in msg);
}

// Type guard: is this a notification (has method, no id)?
export function isNotification(msg: RpcMessage): msg is RpcNotification {
  return "method" in msg && !("id" in msg);
}
