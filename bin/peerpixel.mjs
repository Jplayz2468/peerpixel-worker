#!/usr/bin/env node
/* PeerPixel worker.

   Headless by design: pair once, then `peerpixel run` and leave it. No window,
   no desktop environment. A server in a cupboard is a perfectly good peer. */

import { platform, arch, hostname, cpus } from "node:os";
import { pair, submitBench, pool, ApiError } from "../src/api.mjs";
import { readConfig, writeConfig, configPath, API } from "../src/config.mjs";
import { runWorker } from "../src/run.mjs";
import { createStubRenderer } from "../src/renderers/stub.mjs";

/* The native renderer lands here. Until it does, --stub is the only option and
   the CLI says so rather than pretending. */
async function pickRenderer(argv) {
  if (argv.includes("--stub")) return createStubRenderer();
  try {
    const { createOrtRenderer } = await import("../src/renderers/ort.mjs");
    return createOrtRenderer();
  } catch {
    console.error("No native renderer is built yet. Re-run with --stub to exercise the loop.");
    process.exit(2);
  }
}

const machine = () => ({
  name: hostname(),
  platform: `${platform()}-${arch()}`,
  cores: cpus().length,
});

const commands = {
  async pair(argv) {
    const code = argv[0];
    if (!code) {
      console.error("usage: peerpixel pair <CODE>   (get one from peerpixel.cc)");
      process.exit(1);
    }
    const renderer = await pickRenderer(argv);
    const info = machine();
    const result = await pair(code.toUpperCase(), { ...info, accelerator: renderer.accelerator });
    writeConfig({ deviceId: result.deviceId, token: result.token, api: API });
    console.log(`Paired as ${info.name}.`);
    console.log(`Saved to ${configPath()}`);
    console.log(`Next: peerpixel bench${argv.includes("--stub") ? " --stub" : ""}`);
  },

  async bench(argv) {
    const renderer = await pickRenderer(argv);
    console.log(`Benchmarking ${renderer.name} (${renderer.accelerator}) - 4 steps...`);
    await renderer.warm();
    const started = Date.now();
    await renderer.render({ id: "bench", prompt: "a lighthouse made of blown glass", seed: 1, steps: 4, guidance: 4 });
    const ms = Date.now() - started;
    const result = await submitBench(ms, renderer.accelerator);
    console.log(`${(ms / 1000).toFixed(1)}s for 4 steps (limit ${(result.limitMs / 1000).toFixed(0)}s)`);
    console.log(result.approved
      ? "Approved. Run `peerpixel run` to start earning."
      : "Not approved: this machine is too slow to keep people waiting.");
    if (!result.approved) process.exit(1);
  },

  async run(argv) {
    const renderer = await pickRenderer(argv);
    await runWorker(renderer, { once: argv.includes("--once"), quiet: argv.includes("--quiet") });
  },

  async status() {
    const config = readConfig();
    const online = await pool();
    console.log(`api      ${API}`);
    console.log(`device   ${config.deviceId || "not paired"}`);
    console.log(`pool     ${online.workersOnline} online, ${online.workersIdle} idle, ${online.queued} queued, ${online.running} running`);
  },
};

const [command, ...argv] = process.argv.slice(2);
const run = commands[command];
if (!run) {
  console.log([
    "peerpixel - render for people who cannot",
    "",
    "  peerpixel pair <CODE>   link this machine to your account",
    "  peerpixel bench         prove it is fast enough",
    "  peerpixel run           start rendering (Ctrl-C to stop)",
    "  peerpixel status        pool and device state",
    "",
    "  --stub                  run without the model, to test the loop",
    "  --once                  render one job and exit",
    "  --quiet                 no per-step output",
  ].join("\n"));
  process.exit(command ? 1 : 0);
}
run(argv).catch((error) => {
  if (error instanceof ApiError) console.error(`${error.code}${error.body?.reason ? ` (${error.body.reason})` : ""}`);
  else console.error(error.message);
  process.exit(1);
});
