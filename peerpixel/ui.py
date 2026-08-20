"""Everything that writes to the terminal.

The same facts two ways. When a person is watching, a small panel that rewrites
itself in place. When one is not - systemd, docker, nohup - plain timestamped
lines, because escape codes in a journal are noise nobody can read back.
sys.stdout.isatty() decides, and nothing here emits an escape without it.

Standard library only. A worker that cannot start because a display library is
missing would be a ridiculous way to lose a machine.
"""
from __future__ import annotations

import shutil
import sys
import time

REDRAW = 0.25   # seconds between panel redraws
PLAIN = 60.0    # seconds between status lines when there is no terminal
SETTLE = 3.0    # seconds before a measured rate is worth an ETA
TAIL = 3        # events kept under the panel
WIDTH = 76      # the panel stops here however wide the window is
LABEL = 11      # width of the label column

HIDE, SHOW = "\x1b[?25l", "\x1b[?25h"
UP, WIPE = "\x1b[{}A", "\x1b[2K"
DIM, BOLD, OFF = "\x1b[2m", "\x1b[1m", "\x1b[0m"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def width() -> int:
    # Re-measured every draw, so resizing the window is not a special case.
    return max(40, min(WIDTH, shutil.get_terminal_size((80, 24)).columns - 1))


def bar(done: int, total: int, size: int) -> str:
    filled = min(size, int(size * done / total)) if total > 0 else 0
    return "[" + "#" * filled + " " * (size - filled) + "]"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _row(label: str, value: str, cap: int) -> str:
    text = f"{label:<{LABEL}}{value}"[:cap]
    return DIM + text[:LABEL] + OFF + text[LABEL:]


def _split(left: str, right: str, cap: int) -> str:
    gap = cap - len(left) - len(right)
    return left + " " * gap + right if gap >= 1 else f"{left} {right}"[:cap]


class Progress:
    """One line that rewrites itself, or an occasional plain line in a log."""

    def __init__(self, total: int, done: int):
        self.total = total
        self.initial = done
        self.started = time.monotonic()
        self.tty = sys.stdout.isatty()
        self.last = float("-inf")  # so the first line is drawn straight away
        self.width = 0

    def update(self, done: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < (REDRAW if self.tty else PLAIN):
            return
        self.last = now

        # Rate over the whole session rather than the last few seconds: an ETA
        # measured in tens of minutes should not swing with every hiccup.
        elapsed = now - self.started
        rate = (done - self.initial) / elapsed if elapsed > 0 else 0.0

        if self.total > 0:
            done = min(done, self.total)
            parts = [f"{100 * done / self.total:5.1f}%", f"{human(done)} / {human(self.total)}"]
        else:
            parts = [human(done)]
        parts.append(f"{human(rate)}/s")
        # Under a few seconds the rate is noise and the ETA it gives is absurd.
        if rate > 0 and elapsed > SETTLE and self.total > done:
            parts.append(f"ETA {clock((self.total - done) / rate)}")

        line = "   ".join(parts)
        if self.tty:
            print("\r" + line.ljust(self.width), end="", flush=True)
            self.width = max(self.width, len(line))
        else:
            print(line, flush=True)

    def close(self, done: int | None = None) -> None:
        """Leave the cursor on its own line. Draws a last update if given one."""
        if done is not None:
            self.update(done, force=True)
        if self.tty:
            print()


class Status:
    """What the worker knows about itself. The display only ever reads it."""

    def __init__(self, api: str, machine: str, device: str, accelerator: str,
                 free: bool = False, free_confirmed: bool = False):
        self.api = api
        self.machine = machine
        self.device = device
        self.accelerator = accelerator
        self.free = free
        self.free_confirmed = free_confirmed

        self.state = "connecting"
        self.prompt = ""
        self.steps = 0
        self.step = 0
        self.job_started = 0.0

        self.images = 0
        self.pixels = 0.0
        self.started = time.monotonic()
        self.idle_since = time.monotonic()

    def begin(self, prompt: str, steps: int) -> None:
        self.state = "rendering"
        self.prompt = prompt
        self.steps = steps
        self.step = 0
        self.job_started = time.monotonic()

    def finish(self, pixels: float = 0.0) -> None:
        self.images += 1
        self.pixels += pixels
        self.idle()

    def idle(self) -> None:
        self.state = "online"
        self.prompt = ""
        self.step = self.steps = 0
        self.idle_since = time.monotonic()

    def uptime(self) -> float:
        return time.monotonic() - self.started

    def free_label(self) -> str:
        if not self.free:
            return "off"
        # The switch that decides what arrives lives on the account, so a local
        # yes that the server never acknowledged is only an intention.
        return "on" if self.free_confirmed else "on (unconfirmed)"


class Display:
    """A live panel on a terminal, plain periodic lines anywhere else."""

    def __init__(self, status: Status):
        self.status = status
        self.tty = sys.stdout.isatty()
        self.tail: list[str] = []
        self.painted = 0
        self.opened = False
        self.last = float("-inf")

    def event(self, text: str) -> None:
        """Something discrete happened: connected, finished, failed."""
        self.tail.append(f"{time.strftime('%H:%M:%S')}  {text}")
        del self.tail[:-TAIL]
        if self.tty:
            self.refresh(force=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    def refresh(self, force: bool = False) -> None:
        """Redraw if enough time has passed. Safe to call on every step."""
        now = time.monotonic()
        if not force and now - self.last < (REDRAW if self.tty else PLAIN):
            return
        self.last = now
        if self.tty:
            self._paint(self._panel())
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {self._line()}", flush=True)

    def close(self) -> None:
        """Give the cursor back. Anything else would leave the shell blind."""
        if self.tty and self.opened:
            sys.stdout.write(SHOW)
            sys.stdout.flush()
            self.opened = False

    def _line(self) -> str:
        s = self.status
        parts = [s.state]
        if s.state == "rendering" and s.steps:
            parts.append(f"{s.step}/{s.steps}")
        parts += [plural(s.images, "image"), f"{s.pixels:g} pixels", f"up {clock(s.uptime())}"]
        parts.append(f"free {s.free_label()}")
        return "  ".join(parts)

    def _panel(self) -> list[str]:
        s = self.status
        cap = width()
        host = s.api.split("://")[-1]
        rule = DIM + "-" * cap + OFF

        # While a job is running the elapsed time is on the meter line, and an
        # idle clock frozen at the moment work arrived would only mislead.
        idle = "" if s.state == "rendering" else f"   idle {clock(time.monotonic() - s.idle_since)}"

        if s.state == "rendering":
            elapsed = time.monotonic() - s.job_started
            size = max(10, min(28, cap - LABEL - 22))
            job = s.prompt
            meter = f"{bar(s.step, s.steps, size)}  {s.step}/{s.steps}  {elapsed:.1f}s"
        else:
            job = "waiting for work" if s.state == "online" else "-"
            meter = ""

        lines = [
            BOLD + _split(host, f"{s.state}   up {clock(s.uptime())}", cap)[:cap] + OFF,
            rule,
            _row("machine", f"{s.machine}   device {s.device}", cap),
            _row("render", s.accelerator, cap),
            _row("free work", s.free_label(), cap),
            _row("job", job, cap),
            _row("", meter, cap),
            _row("session", f'{plural(s.images, "image")}   {s.pixels:g} pixels{idle}', cap),
            rule,
        ]
        lines += [DIM + line[:9] + OFF + line[9:cap] for line in reversed(self.tail)]

        # A panel taller than the window would scroll, and then cursor-up lands
        # somewhere else and the screen fills with copies of itself.
        room = shutil.get_terminal_size((80, 24)).lines - 1
        return lines[:room]

    def _paint(self, lines: list[str]) -> None:
        out = []
        if not self.opened:
            out.append(HIDE)
            self.opened = True
        elif self.painted:
            out.append(UP.format(self.painted))
        out += [WIPE + line + "\n" for line in lines]
        self.painted = len(lines)
        sys.stdout.write("".join(out))
        sys.stdout.flush()
