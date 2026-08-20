/* The real renderer: a long-lived Python process running diffusers.

   Loading a 4B model takes tens of seconds, so the process is started once and
   kept warm for as long as the worker runs. Communication is line-delimited
   JSON over stdin and stdout — no port, no socket, nothing to firewall. */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";

const here = dirname(fileURLToPath(import.meta.url));
const SCRIPT = join(here, "..", "..", "python", "render.py");

function findPython() {
  const candidates = [
    process.env.PEERPIXEL_PYTHON,
    join(here, "..", "..", ".venv", "bin", "python"),
    join(here, "..", "..", ".venv", "Scripts", "python.exe"),
    "python3",
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (candidate === "python3" || existsSync(candidate)) return candidate;
  }
  return "python3";
}

export function createTorchRenderer() {
  let child = null;
  let accelerator = "detecting…";
  const waiting = new Map();
  let onProgress = () => {};

  function start() {
    if (child) return;
    child = spawn(findPython(), [SCRIPT], { stdio: ["pipe", "pipe", "pipe"] });
    child.stderr.on("data", (chunk) => {
      const text = String(chunk).trim();
      /* diffusers and torch are chatty on stderr; only surface real trouble. */
      if (/error|traceback|not enough memory|out of memory/i.test(text)) console.error(text);
    });
    createInterface({ input: child.stdout }).on("line", (line) => {
      let message;
      try { message = JSON.parse(line); } catch { return; }
      if (message.accelerator) accelerator = message.accelerator;
      if (message.type === "progress") { onProgress(message.step, message.total); return; }
      const pending = waiting.get(message.id ?? "_");
      if (!pending) return;
      waiting.delete(message.id ?? "_");
      if (message.type === "error") pending.reject(new Error(message.message));
      else pending.resolve(message);
    });
    child.on("exit", (code) => {
      child = null;
      for (const pending of waiting.values()) pending.reject(new Error(`renderer exited (${code})`));
      waiting.clear();
    });
  }

  function ask(request, timeoutMs) {
    start();
    return new Promise((resolve, reject) => {
      waiting.set(request.id, { resolve, reject });
      const timer = setTimeout(() => {
        waiting.delete(request.id);
        reject(new Error("renderer timed out"));
      }, timeoutMs);
      const settle = (fn) => (value) => { clearTimeout(timer); fn(value); };
      waiting.set(request.id, { resolve: settle(resolve), reject: settle(reject) });
      child.stdin.write(JSON.stringify(request) + "\n");
    });
  }

  return {
    name: "torch",
    get accelerator() { return accelerator; },

    async warm() {
      start();
      /* Loading is slow and only happens once; ten minutes is generous rather
         than optimistic, because a cold page cache on a spinning disk is real. */
      await ask({ cmd: "render", id: "warm", prompt: "a grey square", seed: 0, steps: 1 }, 600_000);
    },

    async render(job, progress = () => {}) {
      onProgress = progress;
      const reply = await ask({
        cmd: "render",
        id: job.id,
        prompt: job.prompt,
        seed: job.seed,
        steps: job.steps,
        guidance: job.guidance ?? 4,
        init: job.init || null,
        strength: job.strength ?? 0.55,
      }, 15 * 60_000);
      onProgress = () => {};
      return Buffer.from(reply.jpeg, "base64");
    },

    stop() {
      try { child?.stdin.write(JSON.stringify({ cmd: "quit" }) + "\n"); } catch { /* already gone */ }
      child?.kill();
    },
  };
}
