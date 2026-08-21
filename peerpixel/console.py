"""Drawing on a terminal, nicely, with nothing but the standard library.

A worker that would not start because a display library was missing would be a
ridiculous way to lose a machine, so there is no display library. What there is
instead is about two hundred lines that know how to draw a bar, and one idea
that matters more than any of the drawing:

**The bar repaints itself on a thread.**

Everything slow in this program is slow inside a single blocking call. Loading a
4B model is one `from_pretrained` that returns ninety seconds later. `uv sync`
is one subprocess that says nothing for four minutes. A diffusion step is one
matrix multiply. None of them can call back to say they are still alive, and a
bar drawn only when the work reports would sit still through all three -- which
is the exact failure this whole design exists to avoid.

So the bar is painted by a ticker, twenty times a second, from the tracker's own
clock. The work moves it when it has something to say, and the clock moves it
when the work does not. See `progress.py` for the rules the numbers obey.

Everything degrades. No terminal, no escape codes: plain timestamped lines, so a
systemd journal or a piped log stays readable. No UTF-8, ASCII blocks. NO_COLOR
or a dumb terminal, no colour.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
import threading
import time

FPS = 20.0        # repaints a second on a terminal
PLAIN = 30.0      # seconds between lines when nobody is watching
MAX_WIDTH = 88

# -- what this terminal can do ------------------------------------------------

def _windows_ansi() -> None:
    """Windows consoles understand escape codes, once asked."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL
    except Exception:  # noqa: BLE001 - an old console simply gets no colour
        pass


_windows_ansi()

def _tty() -> bool:
    """Is anybody watching this in a terminal?

    PEERPIXEL_TTY overrules the guess in both directions. Piping through a
    pager, recording a demo and testing the drawing all want the live version
    on something that is not a terminal, and none of them should have to fake
    one.
    """
    told = os.environ.get("PEERPIXEL_TTY")
    if told in ("0", "1"):
        return told == "1"
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


TTY = _tty()
COLOUR = TTY and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


UNICODE = _unicode()

#: Eight partial blocks, so a bar advances a fifth of a character at a time
#: rather than jumping a whole one. At the widths used here that is the
#: difference between motion and a row of steps.
BLOCKS = " ▏▎▍▌▋▊▉█" if UNICODE else " ####    "
TRACK = "░" if UNICODE else "."
TICK = "✓" if UNICODE else "+"
DOT = "•" if UNICODE else "*"
ARROW = "→" if UNICODE else "->"


def _c(code: str) -> str:
    return code if COLOUR else ""


AMBER = _c("\033[38;5;179m")
CREAM = _c("\033[38;5;223m")
GREEN = _c("\033[38;5;114m")
RED = _c("\033[38;5;167m")
DIM = _c("\033[2m")
BOLD = _c("\033[1m")
OFF = _c("\033[0m")
HIDE, SHOW = _c("\033[?25l"), _c("\033[?25h")
UP, WIPE = "\033[{}A", "\033[2K"


def width() -> int:
    return max(40, min(MAX_WIDTH, shutil.get_terminal_size((80, 24)).columns - 2))


# -- words --------------------------------------------------------------------

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def how_long(seconds: float | None) -> str:
    """The ETA, in words somebody can act on rather than a bare number."""
    if seconds is None:
        return "working out how long"
    if seconds < 5:
        return "a few seconds left"
    if seconds < 90:
        return f"{round(seconds)}s left"
    return f"about {clock(seconds)} left"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# -- lines --------------------------------------------------------------------

def say(text: str = "") -> None:
    print(text, flush=True)


def title(text: str) -> None:
    say()
    say(f"  {BOLD}{AMBER}{text}{OFF}")


def note(text: str) -> None:
    """Dim prose, wrapped to the window.

    Wrapped per paragraph rather than as one blob, so the deliberate line
    breaks in the longer explanations survive.
    """
    for paragraph in text.split("\n"):
        stripped = paragraph.strip()
        if not stripped:
            say()
            continue
        for line in textwrap.wrap(stripped, width=width() - 2) or [""]:
            say(f"  {DIM}{line}{OFF}")


def step_line(done: bool, text: str, detail: str = "") -> None:
    mark = f"{GREEN}{TICK}{OFF}" if done else f"{DIM}{DOT}{OFF}"
    tail = f"  {DIM}{detail}{OFF}" if detail else ""
    say(f"  {mark} {text}{tail}")


def block(text: str) -> None:
    """Dim text exactly as written. For anything whose shape carries meaning.

    `note` reflows to the window, which is right for a paragraph and wrong for
    a numbered list or a column of commands: those stop being a list.
    """
    for line in text.split("\n"):
        say(f"  {DIM}{line}{OFF}" if line.strip() else "")


def problem(text: str) -> None:
    say()
    say(f"  {RED}{text}{OFF}")


def rule() -> None:
    say(f"  {DIM}{'─' * (width() - 2) if UNICODE else '-' * (width() - 2)}{OFF}")


def ask(question: str, default: str = "") -> str:
    """One question. Never dies on a closed stdin: a pipe just gets the default."""
    suffix = f" {DIM}[{default}]{OFF}" if default else ""
    try:
        answer = input(f"  {CREAM}{question}{OFF}{suffix} ").strip()
    except (EOFError, KeyboardInterrupt):
        say()
        return default
    return answer or default


def confirm(question: str, default: bool = True) -> bool:
    answer = ask(f"{question}", "yes" if default else "no").lower()
    return answer[:1] in ("y", "1", "t") if answer else default


# -- the bar ------------------------------------------------------------------

def bar(fraction: float, size: int) -> str:
    """A bar with eighth-of-a-character resolution."""
    fraction = max(0.0, min(1.0, fraction))
    if not UNICODE:
        filled = int(size * fraction)
        return "#" * filled + TRACK * (size - filled)
    eighths = int(round(size * fraction * 8))
    full, remainder = divmod(eighths, 8)
    full = min(full, size)
    out = "█" * full
    if remainder and full < size:
        out += BLOCKS[remainder]
    return out + TRACK * (size - len(out))


class Live:
    """One tracker, painted until it stops.

    Used as a context manager around the slow thing. The ticker runs whatever
    the slow thing is doing, including nothing at all, which is the point.
    """

    def __init__(self, tracker, *, heading: str = "", footer=None):
        self.tracker = tracker
        self.heading = heading
        self.footer = footer
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.lines = 0
        self.opened = False
        self.last_plain = 0.0

    def __enter__(self) -> "Live":
        if self.heading:
            title(self.heading)
        self.thread = threading.Thread(target=self._tick, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        self._paint(final=True)
        if TTY and self.opened:
            sys.stdout.write(SHOW)
            sys.stdout.flush()
        return False

    def _tick(self) -> None:
        while not self.stop.is_set():
            self._paint()
            self.stop.wait(1.0 / FPS if TTY else 1.0)

    def _paint(self, final: bool = False) -> None:
        snapshot = self.tracker.snapshot()
        if not TTY:
            now = time.monotonic()
            if not final and now - self.last_plain < PLAIN:
                return
            self.last_plain = now
            tail = "done" if snapshot.finished else how_long(snapshot.eta_seconds)
            status = f"  {self.footer()}" if self.footer else ""
            say(f"[{time.strftime('%H:%M:%S')}] {snapshot.percent:.0f}%  "
                f"{snapshot.label}  {snapshot.detail}  {tail}{status}".rstrip())
            return

        cap = width()
        size = max(12, cap - 10)
        colour = RED if snapshot.failed else (GREEN if snapshot.finished else AMBER)
        left = snapshot.label
        right = f"{snapshot.percent:5.1f}%"
        gap = max(1, cap - len(left) - len(right))
        under = snapshot.detail or ""
        if snapshot.failed:
            eta = ""
        elif snapshot.finished:
            eta = "done"          # not "a few seconds left", which it is not
        else:
            eta = how_long(snapshot.eta_seconds)
        if under and eta:
            under = f"{under}  {DIM}{ARROW}{OFF}  {eta}"
        else:
            under = under or eta

        rows = [
            f"  {CREAM}{left}{OFF}{' ' * gap}{DIM}{right}{OFF}",
            f"  {colour}{bar(snapshot.fraction, size)}{OFF}",
            f"  {DIM}{under}{OFF}"[:cap + len(DIM) + len(OFF)],
        ]
        if self.footer:
            rows.append(f"  {DIM}{self.footer()}{OFF}"[:cap + len(DIM) + len(OFF)])
        out = []
        if not self.opened:
            out.append(HIDE)
            self.opened = True
        elif self.lines:
            out.append(UP.format(self.lines))
        out += [WIPE + row + "\n" for row in rows]
        self.lines = len(rows)
        try:
            sys.stdout.write("".join(out))
            sys.stdout.flush()
        except (OSError, ValueError):
            pass


SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if UNICODE else "|/-\\"


class Line:
    """One line, repainted in place from whatever it is asked.

    For the stretches that are not a task with an end -- a worker connected and
    waiting -- where a progress bar would be a lie but a frozen screen looks
    like a crash. `paint` is called on the ticker and returns the line.
    """

    def __init__(self, paint):
        self.paint = paint
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.shown = False
        self.frame = 0

    def __enter__(self) -> "Line":
        self.thread = threading.Thread(target=self._tick, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.clear()
        return False

    def clear(self) -> None:
        """Take the line back, so whatever prints next starts on clean ground."""
        if TTY and self.shown:
            sys.stdout.write("\r" + WIPE + SHOW)
            sys.stdout.flush()
            self.shown = False

    def _tick(self) -> None:
        last_plain = 0.0
        while not self.stop.is_set():
            text = self.paint()
            if TTY:
                self.frame = (self.frame + 1) % len(SPINNER)
                spin = f"{AMBER}{SPINNER[self.frame]}{OFF}"
                row = f"\r{WIPE}  {spin} {DIM}{text}{OFF}"
                if not self.shown:
                    row = HIDE + row
                    self.shown = True
                try:
                    sys.stdout.write(row[:width() + 40])
                    sys.stdout.flush()
                except (OSError, ValueError):
                    pass
                self.stop.wait(0.1)
            else:
                now = time.monotonic()
                if now - last_plain >= PLAIN:
                    last_plain = now
                    say(f"[{time.strftime('%H:%M:%S')}] {text}")
                self.stop.wait(1.0)
