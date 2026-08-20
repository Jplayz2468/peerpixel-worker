# PeerPixel Worker

The desktop half of [PeerPixel](https://peerpixel.cc). It downloads the model
once, runs it natively on your own GPU, and — when you switch it on — renders
images for people who do not have a card fast enough to run one themselves.

You earn 90% of the credits every image you render is worth. They show up on
your dashboard on the website.

> **Status: empty.** Nothing here works yet. The web app is live and generates
> locally in the browser; this repository is where the native worker will live.

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
