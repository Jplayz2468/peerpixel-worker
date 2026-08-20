# PeerPixel Worker

The desktop half of [PeerPixel](https://peerpixel.cc). It downloads the model
once, runs it natively on your own GPU, and — when you switch it on — renders
images for people who do not have a card fast enough to run one themselves.

You earn 90% of the credits every image you render is worth. They show up on
your dashboard on the website.

> **Status: the loop works, the renderer does not.** Pairing, the benchmark
> gate, the WebSocket connection, job delivery, upload and payment are all
> live against peerpixel.cc. What is missing is the part that actually runs
> the model — `--stub` stands in for it today.

## Using it

```bash
node bin/peerpixel.mjs pair <CODE> --stub   # code comes from peerpixel.cc
node bin/peerpixel.mjs bench --stub         # must finish 4 steps under 30s
node bin/peerpixel.mjs run --stub           # renders until you stop it
```

No window, no desktop environment, no display. A box in a cupboard is a
perfectly good peer. Identity lives in `~/.peerpixel/config.json`, written
0600; the device token authorises this machine to render and nothing else.

## Adding the real renderer

Drop in `src/renderers/ort.mjs` exporting `createOrtRenderer()`. A renderer is:

```js
{ name, accelerator, async warm(), async render(job, onProgress) -> Buffer }
```

Return JPEG bytes. The CLI picks it up automatically and `--stub` stops being
needed. It should use native ONNX Runtime — DirectML on Windows, CoreML on
macOS — rather than WebNN, which is the browser's constraint and the reason
the browser build needs 8 GB and a flag.

## What it will do

- Download the FLUX.2 Klein weights from Hugging Face, resumable, once
- Run them through native ONNX Runtime, using whichever accelerator this machine
  actually has rather than the browser's lowest common denominator
- One switch: sharing on, sharing off. Nothing runs in the background when it
  is off, and nothing runs at all until you say so
- Show the job it is working on right now, live, with who asked for it
- Update itself, because a stale worker returning bad images is worse than no
  worker at all

## Why not just the browser

The browser build needs about 7.9 GB of memory and WebNN, which rules out most
Windows machines with an 8 GB card. Running natively means picking the right
execution provider per platform, quantising further, and offloading layers —
none of which the browser will let us do.

## Licence

MIT.
