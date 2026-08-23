# PeerPixel Worker

Renders images for people whose computers cannot. The pixels you earn show up
on your dashboard at [peerpixel.cc](https://peerpixel.cc).

There are three kinds of job. A **draft** is small and quick: somebody is
working out whether a composition is the one they wanted, and they are asking
several times. It never becomes a file -- the JPEG goes back down this socket
and is relayed straight to their browser. A **master** is the full picture at
512x512, rendered from nothing but a prompt and a seed. An **upscale** runs
AuraSR-v2 only after somebody chooses the paid 4x download; those bytes go
straight to that browser and are never saved by the server.

Each enhanced draft runs Qwen3-1.7B independently with its own variation seed.
The selected draft's enhanced prompt is then reused verbatim for its master.
Seven prompt-only styles use distinct Qwen medium directives; style selection
never loads or stacks adapter weights. Every
render is classified locally before delivery, and prompt, moderation, render,
and upscale operations all return evidence for random trusted-user rechecks.

The two are the same picture because they start from the same noise. A seed
names one 1024px noise tensor, and a preview renders that tensor averaged down
to its own smaller shape -- which is still an exact sample of the distribution
the model expects, unlike anything you get by scaling noise the other way. So a
final needs nothing from the browser that asked for it, and closing a tab can
no longer destroy a render somebody paid for. `seeded_latents` in
`peerpixel/render.py` has the numbers.

Whatever a job costs the person who asked for it is what you are paid for
rendering it. The current prices and sizes live on the site rather than here, so
this page cannot go stale on you.

## Install

Download the zip, unzip it, and double-click one file.

| | |
| --- | --- |
| Windows | `PeerPixel.cmd` |
| macOS | `PeerPixel.command` |
| Linux | `PeerPixel.sh`, or `PeerPixel.desktop` |

A terminal opens and walks you through it: a standalone Python, the rendering
libraries, the model, a pairing code, and a speed check, in the only order they
can happen in. Everything lands inside this folder or in the usual per-user
caches. Nothing is installed system-wide and nothing needs administrator rights.

The first run downloads the 4B base model from its pinned Hugging Face revision.
Smaller prompt, safety, and upscale models come from PeerPixel's private, signed Cloudflare R2
registry only when first needed. Hashes and manifest signatures are checked
before anything is loaded, and downloads resume after interruption.

Apple-silicon Macs render through the published MLX 4-bit FLUX.2 Klein Base
package. Their real benchmark is used for scheduling: current Macs are much
slower than recent NVIDIA cards and may receive very few image jobs, but remain
useful for verification and transient upscaling work.

> On macOS, anything downloaded from the internet is quarantined until you have
> opened it once deliberately: right-click `PeerPixel.command` and choose
> **Open**, then **Open** again. After that it is an ordinary double-click.

## Using it

Double-clicking the launcher again -- or typing `peerpixel` -- sets up anything
still missing and then renders until you stop it. That is the whole of normal
use. Everything else is a named command for people who want one.

```bash
peerpixel                # set up if needed, then render until stopped
peerpixel setup          # the guided first run, on its own
peerpixel pair CODE      # link this machine to your account
peerpixel download       # fetch the model ahead of time
peerpixel bench          # time this machine against the admission limit
peerpixel status         # this machine, and the state of the pool
peerpixel settings       # list or change the knobs
peerpixel doctor         # what this machine is, and a test render
peerpixel update         # fetch and install a newer worker
```

From a clone, put `uv run` in front of any of them.

**`peerpixel doctor` is the one to run when something looks wrong.** It prints
what this machine is, what precision it is rendering in, and then renders a
lighthouse and tells you where it put it. A lighthouse should look like a
lighthouse.

## Settings

```bash
peerpixel settings                  # list them, with what each one does
peerpixel settings dtype            # explain one
peerpixel settings free on          # change one
```

| | |
| --- | --- |
| `free` | also render for people with no pixels |
| `dtype` | arithmetic precision: `auto`, `bfloat16`, `float16`, `float32` |
| `keep-last` | write the last picture rendered to disk, so you can look at it |
| `unload-after` | minutes idle before the model leaves memory; `0` never |
| `colour` | colour and animation in this terminal |
| `api` | which server to talk to |

## Progress bars

**Everything that takes time has a bar on it, and every bar says when it ends.**
This is a rule rather than a nicety, and `peerpixel/progress.py` is where it is
written down and tested. It rules out the two bars everybody has met: the one
that sits at 0% for ten seconds because it only knows how to count bytes and no
byte has arrived, and the one that reaches 99% and stops because the last piece
of work was never in the plan.

Three ideas do all of it.

**Phases declared up front.** A task lists its phases before it starts, so the
last percent is a phase like any other and finishing it is visible. Nothing may
discover halfway through that there is more work.

**Time as a floor, never a ceiling.** Every phase carries an estimate of how
long it takes, learned from the last time this machine did it. Measured
progress -- bytes, steps, files -- always wins when there is any. When there is
not, the clock keeps the bar moving at the speed the estimate implies. And the
bar is painted by a ticker twenty times a second rather than by the work, so a
ninety-second model load that cannot report anything at all still moves.

**Past the estimate, slow down; never stop and never lie.** An overrunning
phase decelerates toward its own ceiling and never arrives. Only the work
actually finishing shows 100%.

## Free work

Somebody signed in can choose the free queue instead of their own balance: one
candidate at a time, up to 12 a day. Those jobs pay nothing and always queue
behind paid ones. A free master additionally needs the pool to have a machine
left over afterwards, so unpaid work can never put a paying person in a queue.

They only go to machines whose owner said yes. The switch belongs to your
account rather than to this machine, so setting it needs you signed in: either
use the Contribute page on the site, or export `PEERPIXEL_SESSION` with your
`pp` cookie first. `peerpixel settings` says plainly when a switch only ever got
as far as this machine.

## When a render comes out wrong

A machine whose arithmetic this model does not get on with does not fail. It
quietly fills the latents with NaN, and the decoder turns that into flat grey
with a few black specks, which looks exactly like a broken model to anybody
watching. That is nearly always precision rather than hardware.

So the worker looks. Every render is checked before it is delivered, and one
that came back as nonsense is never sent and never paid for: the worker drops
to the next precision down, remembers, and renders it again. `peerpixel
settings dtype` overrules that if you know better.

## Requirements and platforms

An NVIDIA card with 8 GB or more, or an Apple silicon Mac with 16 GB or more.
CPU works but is far too slow to pass the benchmark. No Python needed: the
launcher installs a standalone one.

The worker uses platform-neutral cache paths and Python APIs. Windows uses
`PeerPixel.cmd`; Linux uses `PeerPixel.sh` or the desktop launcher; macOS uses
`PeerPixel.command`. CUDA is selected on Windows/Linux when available, MPS on
Apple silicon, and CPU is the portable fallback. Model activation explicitly
installs PEFT, and AuraSR selects CUDA, MPS, or CPU at runtime.

## The code

Plain Python, standard library except where it renders. There is no window, no
localhost server and no build step; the terminal is the whole interface.

| | |
| --- | --- |
| `peerpixel/progress.py` | the rules every bar obeys -- pure, and tested |
| `peerpixel/console.py` | drawing on a terminal, and degrading when there is none |
| `peerpixel/cli.py` | the commands, and the guide somebody meets first |
| `peerpixel/settings.py` | the knobs, declared once |
| `peerpixel/plans.py` | what the long jobs are, and the shape of each one's bar |
| `peerpixel/render.py` | runs the model - start here if an image looks wrong |
| `peerpixel/worker.py` | holds the connection, takes jobs |
| `peerpixel/relay.py` | the binary frame that carries preview and reference bytes |
| `peerpixel/api.py` | talks to peerpixel.cc |
| `peerpixel/download.py` | fetches the weights and shows how far along it is |
| `peerpixel/weights.py` | are they already here? asked without the Hub |
| `peerpixel/updater.py` | finding a newer PeerPixel, and becoming it |
| `peerpixel/runtime.py` | where this install keeps its Python |
| `peerpixel/config.py` | the device token, in `~/.peerpixel` |
| `launch/` | the two bootstrap scripts, and the only thing that runs before Python |

Edit any of them and restart. There is nothing to rebuild.

```bash
python -m unittest discover -s tests -t .
```

## Licence and model notices

The distribution includes the standard Apache 2.0 text in `APACHE-2.0.txt`.
`THIRD_PARTY_NOTICES.txt` lists every bundled or on-demand model and travels in
the built wheel alongside the licence; neither notices nor model identifiers
are embedded into downloaded images.
