/* The worker loop.

   Connects to the dispatcher over a WebSocket and stays there. The connection
   is the queue: jobs are pushed, not polled, so an idle machine costs nothing
   at either end. Everything here works headless — no window, no desktop
   environment, no display. */

import { API, readConfig } from "./config.mjs";
import { submitResult } from "./api.mjs";

const RECONNECT_MIN = 2_000;
const RECONNECT_MAX = 60_000;

const stamp = () => new Date().toISOString().slice(11, 19);
const log = (...parts) => console.log(`[${stamp()}]`, ...parts);

export async function runWorker(renderer, { once = false, quiet = false } = {}) {
  const config = readConfig();
  if (!config.token) throw new Error("this machine is not paired yet — run `peerpixel pair <CODE>`");

  await renderer.warm();
  let backoff = RECONNECT_MIN;
  let stopping = false;
  let rendered = 0;

  process.on("SIGINT", () => { log("stopping"); process.exit(0); });
  process.on("SIGTERM", () => { stopping = true; });

  while (!stopping) {
    const closed = await new Promise((resolve) => {
      const url = `${API.replace(/^http/, "ws")}/api/device/connect`;
      const socket = new WebSocket(url, { headers: { authorization: `Bearer ${config.token}` } });
      let heartbeat;

      socket.addEventListener("open", () => {
        backoff = RECONNECT_MIN;
        log(`connected to ${API} as ${renderer.name} (${renderer.accelerator})`);
      });

      socket.addEventListener("message", async (event) => {
        let message;
        try { message = JSON.parse(event.data); } catch { return; }

        if (message.type === "welcome") {
          clearInterval(heartbeat);
          heartbeat = setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "heartbeat" }));
          }, message.heartbeatMs || 30_000);
          log("waiting for work");
          return;
        }

        if (message.type !== "job") return;
        const job = message.job;
        log(`job ${job.id} · ${job.steps} steps · "${String(job.prompt).slice(0, 60)}"`);
        const started = Date.now();
        try {
          const jpeg = await renderer.render(job, (step, total) => {
            if (!quiet && step % Math.max(1, Math.round(total / 4)) === 0) {
              process.stdout.write(`\r  step ${step}/${total}   `);
            }
          });
          if (!quiet) process.stdout.write("\r");
          const result = await submitResult(job.id, jpeg);
          rendered++;
          log(`done in ${((Date.now() - started) / 1000).toFixed(1)}s · +${result.earnedCredits} credits`);
          socket.send(JSON.stringify({ type: "finished", jobId: job.id }));
          if (once) { stopping = true; socket.close(); }
        } catch (error) {
          log(`failed: ${error.message}`);
          socket.send(JSON.stringify({ type: "failed", jobId: job.id }));
        }
      });

      socket.addEventListener("close", (event) => { clearInterval(heartbeat); resolve(event); });
      socket.addEventListener("error", () => { /* a close event always follows */ });
    });

    if (stopping) break;
    /* 1008 means the dispatcher will never accept this device, so retrying
       forever would just be noise in somebody's logs. */
    if (closed?.code === 1008) { log(`refused: ${closed.reason || "not accepted"}`); break; }
    log(`disconnected (${closed?.code ?? "?"}) — retrying in ${Math.round(backoff / 1000)}s`);
    await new Promise((r) => setTimeout(r, backoff));
    backoff = Math.min(RECONNECT_MAX, Math.round(backoff * 1.8));
  }
  return rendered;
}
