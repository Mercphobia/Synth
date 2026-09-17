// Synth TypeScript CLI entry point (spec 6.16).
// Usage:
//   synth-tui chat          — Ink TUI chat (default)
//   synth-tui run "task"    — one-shot via JSON-RPC to Python runtime
//   synth-tui status        — check runtime availability

import yargs from "yargs";
import type { Argv } from "yargs";
import { hideBin } from "yargs/helpers";
import { StdioTransport } from "./transport/stdio.js";
import { startTui } from "./tui/App.js";

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .scriptName("synth-tui")
    .command("chat", "Start interactive TUI chat", () => {}, () => {
      const transport = new StdioTransport({
        runtimeArgs: ["serve"],
        env: { PYTHONPATH: "src", ...process.env } as Record<string, string>,
      });
      startTui(transport);
    })
    .command("run <prompt>", "Run a one-shot task", (y: Argv) =>
      y.positional("prompt", { type: "string", demandOption: true }),
      async (args: { prompt: string }) => {
        const transport = new StdioTransport({
          runtimeArgs: ["serve"],
          env: { PYTHONPATH: "src", ...process.env } as Record<string, string>,
        });
        transport.start();
        try {
          const result = await transport.call("run", { prompt: args.prompt });
          console.log(JSON.stringify(result, null, 2));
        } catch (err) {
          console.error("Error:", (err as Error).message);
          process.exit(1);
        } finally {
          transport.stop();
        }
      }
    )
    .command("status", "Check if the Python runtime is reachable", () => {}, async () => {
      const transport = new StdioTransport({
        runtimeArgs: ["serve"],
        env: { PYTHONPATH: "src", ...process.env } as Record<string, string>,
      });
      transport.start();
      try {
        const result = await transport.call("ping", {});
        console.log("Runtime:", result);
      } catch {
        console.log("Runtime: not reachable (check python3 + synth package)");
        process.exit(1);
      } finally {
        transport.stop();
      }
    })
    .demandCommand(1)
    .strict()
    .help()
    .parse();
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
