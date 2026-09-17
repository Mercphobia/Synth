// Ink TUI for Synth chat (spec 6.16: Ink TUI).
// Renders streaming tokens, tool-call events, and the conversation history
// in a scrollable terminal UI.

import React, { useState, useEffect, useCallback } from "react";
import { render, Box, Text, useInput, useApp } from "ink";
import TextInput from "ink-text-input";
import { StdioTransport } from "../transport/stdio.js";

interface ChatMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
}

interface AppProps {
  transport: StdioTransport;
}

function App({ transport }: AppProps) {
  const { exit } = useApp();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    transport.start();

    transport.on("notification", (msg: { method: string; params?: Record<string, unknown> }) => {
      if (msg.method === "stream" && msg.params?.token) {
        setStreaming((prev) => prev + String(msg.params!.token));
      } else if (msg.method === "tool_call") {
        const p = msg.params ?? {};
        setMessages((prev) => [
          ...prev,
          { role: "tool" as const, content: String(p.result ?? ""), toolName: String(p.name ?? "") },
        ]);
      }
    });

    return () => transport.stop();
  }, [transport]);

  const submit = useCallback(async () => {
    const text = input.trim();
    if (!text) return;
    if (text === "/exit" || text === "/quit") {
      exit();
      return;
    }
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setBusy(true);
    setStreaming("");

    try {
      const result = await transport.call("run", { prompt: text });
      const responseText = (result as { text?: string })?.text ?? "";
      if (responseText) {
        setMessages((prev) => [...prev, { role: "assistant", content: responseText }]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${(err as Error).message}` },
      ]);
    } finally {
      setBusy(false);
      setStreaming("");
    }
  }, [input, transport, exit]);

  useInput((input, key) => {
    if (key.return) void submit();
  });

  return (
    <Box flexDirection="column" height="100%">
      {/* Messages */}
      <Box flexDirection="column" flexGrow={1} overflowY="hidden">
        {messages.map((msg, i) => (
          <Box key={i} flexDirection="column">
            {msg.role === "user" ? (
              <Text color="cyan">❯ {msg.content}</Text>
            ) : msg.role === "tool" ? (
              <Text color="gray" dimColor>
                ▸ {msg.toolName}: {msg.content.slice(0, 200)}
              </Text>
            ) : (
              <Text>{msg.content}</Text>
            )}
          </Box>
        ))}
        {streaming && <Text color="green">{streaming}</Text>}
        {busy && !streaming && <Text color="yellow">∿ thinking...</Text>}
      </Box>

      {/* Input */}
      <Box marginTop={1}>
        <Text color="cyan">❯ </Text>
        <TextInput value={input} onChange={setInput} placeholder="Type a task (/exit to quit)" />
      </Box>
    </Box>
  );
}

export function startTui(transport: StdioTransport): void {
  render(React.createElement(App, { transport }));
}
