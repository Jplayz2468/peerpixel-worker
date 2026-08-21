# PeerPixel Worker

Renders images for people whose computers cannot. The pixels you earn show up
on your dashboard at [peerpixel.cc](https://peerpixel.cc).

There are two kinds of job. A **draft** is small and quick: somebody is working
out whether a composition is the one they wanted, and they are asking several
times. It never becomes a file -- the JPEG goes back down this socket and is
relayed straight to their browser. A **master** is the full picture,
conditioned on the draft they chose, which arrives over the same socket.

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

A window opens and does the rest: a standalone Python, the rendering libraries,
the model, and a speed check, in the only order they can happen in. Everything
lands inside this folder or in the usual per-user caches. Nothing is installed
system-wide and nothing needs administrator rights.

The first run downloads about 18 GB and takes a while. You can close the window
and come back; every download resumes from where it stopped.

The one thing the window needs from you is a pairing code, which you get from
peerpixel.cc. Paste it in and press Pair.

> On macOS, an app downloaded from the internet is quarantined until you have
> opened it once deliberately: right-click `PeerPixel.command` and choose
> **Open**, then **Open** again. After that it is an ordinary double-click.

## Update

Press **Install it** when the window says a newer version is out, or run
`./update.sh`. Either way the update is downloaded in full before anything is
replaced, your `.venv` and any local edits are left alone, and it never happens
while a render is in flight.

## The window

It is an application, not a browser tab: the interface is HTML, but it is
hosted in a native webview owned by this process, in a window with its own
title bar and its own dock or taskbar entry. On a machine with no webview
available -- usually a Linux box without WebKitGTK -- it falls back to a
chromeless browser window with its own profile, which behaves the same way.

**Everything that takes time has a bar on it, and every bar has an estimate.**
That is a rule rather than a nicety, and `peerpixel/progress.py` is where it is
written down and tested. It rules out the two bars everybody has met: the one
that sits at 0% for ten seconds because it only knows how to count bytes and no
byte has arrived, and the one that reaches 99% and stops because the last piece
of work was never in the plan.

Three ideas do all of it. A task declares its phases up front, so the last
percent is a phase like any other. Every phase carries an estimate learned from
the last time this machine did it, and the clock keeps the bar moving at that
speed whenever there is no measurement -- so a bar is never still. And past its
estimate a phase slows down and approaches its own end without arriving; only
the work actually finishing shows 100%.

## Headless

A box in a cupboard has no window and does not want one.

```bash
uv run peerpixel run          # renders until you stop it
uv run peerpixel run --free   # the same, and take unpaid work too
```

`run` shows a live panel -- what is connected, what it is rendering, how far
through, what you have earned this session. Piped to a file or a systemd
journal it prints plain timestamped lines instead, because escape codes in a
journal are noise nobody can read back.

```bash
uv run peerpixel pair ABC123    # get a code from peerpixel.cc
uv run peerpixel download       # fetch the model ahead of time
uv run peerpixel bench          # warm once, then measure 4 steady-state steps
uv run peerpixel status         # the state of the pool and of this machine
uv run peerpixel update         # fetch and install a newer worker
uv run peerpixel app            # the window, if you want it after all
```

`bench` and `run` fetch the model themselves if it is not there yet, so
`download` is only for getting it over with first.

## Free work

Somebody signed in can choose the free queue instead of their own balance: one
candidate at a time, up to 12 a day. Those jobs pay nothing and always queue
behind paid ones. A free master additionally needs the pool to have a machine
left over afterwards, so unpaid work can never put a paying person in a queue.

They only go to machines whose owner said yes -- the switch in the window, or:

```bash
uv run peerpixel free on
uv run peerpixel free off       # the default
```

The switch belongs to your account rather than to the machine, so the worker
needs you signed in to set it: either use the Contribute page on the site, or
export `PEERPIXEL_SESSION` with your `pp` cookie first. The window tells you if
a switch only ever got as far as this machine.

## Requirements

An NVIDIA card with 8 GB or more, or an Apple silicon Mac with 16 GB or more.
CPU works but is far too slow to pass the benchmark. No Python needed: the
launcher installs a standalone one.

## The code

Two halves. The app is standard library only and starts in a second on a
machine where nothing is installed yet -- which is the whole reason the
dependency install can have a progress bar, because the thing drawing it does
not depend on what it is installing. Everything heavy runs in a child process
and reports back over a pipe.

| | |
| --- | --- |
| `peerpixel/progress.py` | the rules every bar obeys -- pure, and tested |
| `peerpixel/app.py` | the local server, the state, and what the buttons do |
| `peerpixel/window.py` | getting a real window: webview, then app mode, then a tab |
| `peerpixel/ui/` | the interface. Three files, no build step |
| `peerpixel/tasks.py` | what the long jobs are, and the shape of each one's bar |
| `peerpixel/events.py` | how a child process says what it is doing |
| `peerpixel/runtime.py` | where this install keeps its Python |
| `peerpixel/updater.py` | finding a newer PeerPixel, and becoming it |
| `peerpixel/render.py` | runs the model - start here if an image looks wrong |
| `peerpixel/worker.py` | holds the connection, takes jobs |
| `peerpixel/relay.py` | the binary frame that carries draft and reference bytes |
| `peerpixel/ui.py` | the terminal panel, for when there is no window |
| `peerpixel/api.py` | talks to peerpixel.cc |
| `peerpixel/download.py` | fetches the weights and shows how far along it is |
| `peerpixel/weights.py` | are they already here? asked without the Hub |
| `peerpixel/config.py` | the device token, in `~/.peerpixel` |
| `launch/` | the two bootstrap scripts, and the only console you will see |

Edit any of them and restart. There is nothing to rebuild.

```bash
python -m unittest discover -s tests -t .
```

## Licence

MIT
