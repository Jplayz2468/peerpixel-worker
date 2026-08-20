# PeerPixel Worker

Renders images for people whose computers cannot. You earn 90% of the pixels
each image is worth, and they show up on your dashboard at
[peerpixel.cc](https://peerpixel.cc).

Runs on Windows, macOS and Linux. Setup opens a small dashboard in your browser,
served only from this machine. After that it can run headless with no window.

## Install

```bash
git clone https://github.com/Jplayz2468/peerpixel-worker
cd peerpixel-worker
./setup.sh          # Windows: .\setup.ps1
```

That puts a standalone Python and every library into `.venv` inside this
folder. Nothing is installed system-wide.

## Commands

```bash
uv run peerpixel pair ABC123    # get a code from peerpixel.cc
uv run peerpixel dashboard     # local pairing, benchmark and worker controls
uv run peerpixel download       # fetch the model (~15 GB), optional
uv run peerpixel bench          # warm once, then measure 4 steady-state steps
uv run peerpixel run            # renders until you stop it
```

`bench` and `run` fetch the model themselves if it is not there yet, so
`download` is only for getting it over with first. Either way you get a
progress line, and stopping part way costs you nothing: it resumes.

`run` shows a live panel - what is connected, what it is rendering, how far
through, what you have earned this session. Piped to a file or a systemd
journal it prints plain timestamped lines instead.

Two more: `peerpixel status` prints the state of the pool, and
`peerpixel free on|off` is below.

## Free work

People without an account can ask for one 4-step draft at a time, up to 12 a
day. Those jobs pay nothing and always queue behind paid ones.

They only go to machines whose owner said yes:

```bash
uv run peerpixel free on        # also: uv run peerpixel run --free
uv run peerpixel free off       # the default
```

The switch belongs to your account rather than to the machine, so the worker
needs you signed in to set it: either use the Contribute page on the site, or
export `PEERPIXEL_SESSION` with your `pp` cookie first. `peerpixel status`
tells you if it only ever got as far as this machine.

## Requirements

An NVIDIA card with 8 GB or more, or an Apple silicon Mac with 16 GB or more.
CPU works but is far too slow to pass the benchmark. Python 3.11 or newer,
which `setup.sh` installs for you.

## The code

Eight short files, all plain Python:

| | |
| --- | --- |
| `peerpixel/render.py` | runs the model - start here if an image looks wrong |
| `peerpixel/worker.py` | holds the connection, takes jobs |
| `peerpixel/ui.py` | the live panel, and the plain lines when nobody is watching |
| `peerpixel/api.py` | talks to peerpixel.cc |
| `peerpixel/download.py` | fetches the weights and shows how far along it is |
| `peerpixel/config.py` | the device token, in `~/.peerpixel` |
| `peerpixel/update.py` | says when a newer release exists, installs nothing |
| `peerpixel/__main__.py` | the commands above |

Edit any of them and restart. There is nothing to rebuild.

## Licence

MIT
