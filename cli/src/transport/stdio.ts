// Stdio transport: spawns the Python runtime as a child process and
// exchanges NDJSON lines over stdin/stdout. Lines are newline-delimited
// JSON-RPC messages (spec 6.16 JSON-RPC bridge).

import { spawn, ChildProcess } from "child_process";
import { existsSync } from "fs";
import { EventEmitter } from "events";
import { createInterface } from "readline";
import { RpcMessage, RpcResponse } from "./rpc.js";

export interface TransportOptions {
  // Path to the Python runtime entry point (default: synth __main__).
  runtimeBin?: string;
  // Extra args passed to the runtime (e.g. ["--no-stream"]).
  runtimeArgs?: string[];
  // Working directory for the runtime process.
  cwd?: string;
  // Env vars passed to the runtime.
  env?: Record<string, string>;
}

export class StdioTransport extends EventEmitter {
  private proc: ChildProcess | null = null;
  private nextId = 1;
  private pending = new Map<number | string, {
    resolve: (r: unknown) => void;
    reject: (e: Error) => void;
  }>();

  constructor(private opts: TransportOptions = {}) {
    super();
  }

  // Spawn the Python runtime and start reading its stdout line-by-line.
  start(): void {
    if (this.proc) return;
    // Prefer the project's .venv Python when available (litellm lives there).
    const candidates = [".venv/bin/python3", ".venv/bin/python"];
    let bin = this.opts.runtimeBin ?? "python3";
    for (const c of candidates) {
      if (existsSync(c)) { bin = c; break; }
    }
    const args = ["-m", "synth", ...(this.opts.runtimeArgs ?? [])];
    this.proc = spawn(bin, args, {
      cwd: this.opts.cwd,
      env: { ...process.env, ...this.opts.env },
      stdio: ["pipe", "pipe", "inherit"],
    });

    const rl = createInterface({ input: this.proc.stdout! });
    rl.on("line", (line: string) => this.handleLine(line));
    this.proc.on("exit", (code) => {
      this.emit("exit", code);
      // Reject all pending promises so callers don't hang.
      for (const [, p] of this.pending) p.reject(new Error("runtime exited"));
      this.pending.clear();
    });
  }

  // Send a JSON-RPC request and return a promise for the response.
  call(method: string, params?: Record<string, unknown> | unknown[]): Promise<unknown> {
    if (!this.proc || !this.proc.stdin) throw new Error("transport not started");
    const id = this.nextId++;
    const msg = JSON.stringify({ jsonrpc: "2.0", id, method, params });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc!.stdin!.write(msg + "\n");
    });
  }

  // Send a notification (no response expected).
  notify(method: string, params?: Record<string, unknown> | unknown[]): void {
    if (!this.proc || !this.proc.stdin) throw new Error("transport not started");
    const msg = JSON.stringify({ jsonrpc: "2.0", method, params });
    this.proc.stdin.write(msg + "\n");
  }

  private handleLine(line: string): void {
    if (!line.trim()) return;
    let msg: RpcMessage;
    try {
      msg = JSON.parse(line);
    } catch {
      // Non-JSON stdout: emit as raw so the TUI can show runtime logs.
      this.emit("log", line);
      return;
    }

    // Response to a pending request?
    if ("id" in msg && ("result" in msg || "error" in msg)) {
      const resp = msg as RpcResponse;
      const p = this.pending.get(resp.id);
      if (p) {
        this.pending.delete(resp.id);
        if (resp.error) p.reject(new Error(resp.error.message));
        else p.resolve(resp.result);
      }
      return;
    }

    // Notification (streaming tokens, tool-call events, etc.)
    if ("method" in msg && !("id" in msg)) {
      this.emit("notification", msg);
      return;
    }

    // Anything else: emit as generic.
    this.emit("message", msg);
  }

  stop(): void {
    if (this.proc) {
      this.proc.stdin?.end();
      this.proc.kill("SIGTERM");
      this.proc = null;
    }
  }
}
