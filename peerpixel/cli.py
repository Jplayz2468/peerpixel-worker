"""The commands, and the guide somebody meets the first time.

PeerPixel is a terminal program that people who do not live in terminals are
expected to run, which pulls in two directions. So there are two doors. Typing
`peerpixel` with nothing after it walks you through setting the machine up and
then starts it -- that is what the launcher in this folder does, and it is the
only thing most people ever use. Everything else is a named subcommand for
people who want one.

Nothing here blocks without a bar on it. Every wait in this program is either
under a second or drawn; see `console.py` for why the drawing happens on a
thread, and `progress.py` for the rules the numbers obey.
"""
from __future__ import annotations

import os
import sys

from . import api, config, console, plans, runtime, settings, updater, weights
from .console import (DIM, OFF, ask, block, confirm, note, problem, rule, say,
                      step_line, title)

BANNER = r"""
  ██████╗ ███████╗███████╗██████╗ ██████╗ ██╗██╗  ██╗███████╗██╗
  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██║╚██╗██╔╝██╔════╝██║
  ██████╔╝█████╗  █████╗  ██████╔╝██████╔╝██║ ╚███╔╝ █████╗  ██║
  ██╔═══╝ ██╔══╝  ██╔══╝  ██╔══██╗██╔═══╝ ██║ ██╔██╗ ██╔══╝  ██║
  ██║     ███████╗███████╗██║  ██║██║     ██║██╔╝ ██╗███████╗███████╗
  ╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
"""

PLAIN_BANNER = "  P E E R P I X E L"


def banner() -> None:
    say(console.AMBER + (BANNER if console.UNICODE and console.width() >= 74
                         else "\n" + PLAIN_BANNER + "\n") + console.OFF)
    note("Rendering for people whose computers cannot.")


# -- what is and is not done yet ---------------------------------------------

def state() -> dict:
    saved = config.read()
    return {
        "libraries": runtime.dependencies_ready(),
        "paired": bool(saved.get("token")),
        "model": weights.cached(),
        "approved": bool(saved.get("approved")),
    }


def ready(where: dict | None = None) -> bool:
    return all((where or state()).values())


def run_plan(name: str, work, *, estimates: dict | None = None):
    """Run something long under its bar, and learn how long it took.

    `work` is handed the tracker and moves it. Everything it does not say is
    covered by the clock, which is what keeps the bar alive through a model
    load that cannot report anything at all.
    """
    made = plans.tracker(name, estimates)
    try:
        with console.Live(made, heading=plans.PLANS[name].title):
            result = work(made)
            made.finish()
    except BaseException as error:
        made.fail(str(error) or type(error).__name__)
        raise
    plans.remember(made)
    return result


# -- the steps ----------------------------------------------------------------

def need_libraries() -> None:
    """Refuse to go on without the rendering stack, in a sentence.

    Reached only when something has gone wrong that the setup could not fix by
    itself, so it says what to do rather than dying on `import torch` twelve
    frames deep.
    """
    runtime.use_venv()
    if runtime.torch_here():
        return
    raise RuntimeError(
        "the rendering libraries are not installed in this folder. "
        "Run `peerpixel setup` here, or delete .venv and run it again.")


def install_libraries() -> None:
    run_plan("install", lambda made: plans.install_dependencies(made))
    step_line(True, "Libraries installed.")
    # Everything after this needs them, and this process cannot see them: it is
    # running on the bare interpreter that did the installing.
    runtime.use_venv()


def fetch_model() -> None:
    from . import download

    def work(made):
        download.ensure(on_phase=made.begin, on_progress=made.report)

    run_plan("model", work)


def benchmark() -> dict:
    from .benchmark import generation_warning, likely_generation_work, run_benchmark
    from .render import Renderer

    def work(made):
        made.begin("load")
        renderer = Renderer()
        renderer.warm()
        made.note(renderer.accelerator)
        phases = iter(("warm", "measure"))
        made.begin(next(phases))

        def stepped(done, total):
            made.report(done, total, detail=f"step {done} of {total}")

        ms, result = run_benchmark(renderer, on_step=stepped,
                                   between=lambda: made.begin(next(phases, "measure")))
        made.begin("submit")
        config.write(benchMs=ms, approved=bool(result.get("approved")),
                     accelerator=renderer.accelerator)
        return ms, result

    ms, result = run_plan("bench", work)
    say()
    limit = result.get("limitMs", 0)
    step_line(bool(result.get("approved")),
              f"{ms / 1000:.1f}s for four steps"
              + (f"  (the limit is {limit / 1000:.0f}s)" if limit else ""))
    if not result.get("approved"):
        problem("This machine is not fast enough to keep people waiting, so the "
                "network will not send it work.")
        note("Nothing is wrong with it. It is just slower than a render can afford.")
    elif not likely_generation_work(ms):
        note(generation_warning(ms, renderer.accelerator))
    return result


def pair_machine(code: str) -> dict:
    from .render import describe_accelerator

    import platform
    import socket

    name = describe_accelerator()
    import hashlib
    token = code.strip()
    if len(token) < 16:
        raise RuntimeError("ask a PeerPixel moderator for a permanent worker key")
    device_id = "dev_" + hashlib.sha256(
        f"{socket.gethostname()}:{platform.node()}".encode()).hexdigest()[:16]
    config.write(deviceId=device_id, token=token, accelerator=name)
    return {"deviceId": device_id}


# -- the guide ----------------------------------------------------------------

WELCOME = """PeerPixel pays you in pixels for rendering other people's pictures, and \
spends those pixels when you want one of your own. There is no data centre \
behind it: it is machines like this one, taking turns.

Setting up happens once and takes a while, mostly downloading. You can stop it \
at any point and run this again; nothing starts over."""


def onboard(*, interactive: bool = True) -> bool:
    """The first run. Returns True when the machine is ready to render."""
    where = state()
    if ready(where):
        return True

    banner()
    say()
    note(WELCOME)
    say()
    rule()
    say()
    step_line(where["libraries"], "Rendering libraries",
              "" if where["libraries"] else "a few gigabytes, kept in this folder")
    step_line(where["paired"], "Paired with your account",
              "" if where["paired"] else "needs a code from peerpixel.cc")
    step_line(where["model"], "The model",
              "" if where["model"] else "about 15 GB, and it resumes if interrupted")
    step_line(where["approved"], "Speed check", "one timed render")
    say()
    if interactive and not confirm("Go ahead?"):
        note("Nothing done. Run `peerpixel setup` when you are ready.")
        return False

    if not where["libraries"]:
        install_libraries()
    if not weights.cached():
        fetch_model()
    if not where["paired"]:
        if not pair_interactively(interactive):
            return False
    if not config.read().get("approved"):
        if not benchmark().get("approved"):
            return False
    if interactive:
        offer_free_work()

    say()
    title("Ready.")
    note("This machine can render now. `peerpixel` starts it; Ctrl-C stops it.")
    return True


PAIRING = """A machine has to belong to an account, so the pixels it earns have
somewhere to go.

  1. Open  https://peerpixel.cc/app  and sign in
  2. Ask a PeerPixel moderator for a permanent worker key
  3. Type the code here"""


def pair_interactively(interactive: bool) -> bool:
    title("Pair this machine")
    block(PAIRING)
    say()
    if not interactive:
        problem("Not bound. Ask a PeerPixel moderator for a permanent worker key.")
        return False
    while True:
        code = ask("Pairing code:")
        if not code:
            note("Skipped. Run `peerpixel pair KEY` after a moderator gives you one.")
            return False
        try:
            result = pair_machine(code)
        except api.ApiError as error:
            problem(f"That code was not accepted ({error.code}).")
            note("Codes expire after a few minutes. Get a fresh one and try again.")
            continue
        step_line(True, f"Paired as {result['deviceId']}")
        return True


def offer_free_work() -> None:
    if config.read().get("allowFree"):
        return
    say()
    title("Free work")
    note("Some people have no pixels yet. You can render for them as well as for "
         "people who pay. Those jobs earn nothing and always wait behind paid ones, "
         "so they only ever use a card that would be idle anyway.")
    say()
    if confirm("Take free work too?", default=False):
        say(f"  {settings.put('free', 'on')}")


# -- commands -----------------------------------------------------------------

#: Set on the way into a restart, so a release whose tag runs ahead of its own
#: version cannot make this update, restart, still look old, and update again.
UPDATED = "PEERPIXEL_UPDATED"


def self_update() -> None:
    """Install a newer PeerPixel before starting, if there is one.

    Only on the way in. A worker that has already picked up a job is holding a
    picture somebody paid for, and no amount of being out of date is worth
    dropping that -- so this is the one moment it can happen, and it is over
    before the first job is claimed.

    Silent on every failure. Being offline, behind a proxy or rate limited by
    GitHub is not a reason to refuse to render.
    """
    mode = settings.update_mode()
    if mode == "off":
        return
    with console.Line(lambda: "checking for a newer PeerPixel"):
        latest = updater.latest(timeout=5.0)
    server_update(latest, mode=mode)


def server_update(required: str, *, mode: str | None = None) -> bool:
    """Install the minimum version named by the coordinator, while idle."""
    mode = settings.update_mode() if mode is None else mode
    here = updater.installed()
    if not required or not updater.newer(required, here):
        return False
    if os.environ.get(UPDATED) == required:
        note(f"{required} says it is newer than {here}, and installing it did not "
             f"change that. Staying on {here}.")
        return False
    if mode == "off":
        return False
    if mode == "notify":
        note(f"{required} is required; you have {here}. Install it with: peerpixel update")
        return False

    result = run_plan("update", updater.apply)
    if not result.get("updated"):
        return False
    step_line(True, f"Updated to {result['version']}.")
    note("Restarting into it.")
    os.environ[UPDATED] = str(result["version"])
    runtime.restart()
    return True


def cmd_start(argv: list[str]) -> None:
    """Set up if needed, then render until stopped."""
    if "--no-update" not in argv:
        self_update()
    if not onboard(interactive=sys.stdin.isatty() and "--yes" not in argv):
        raise SystemExit(1)
    need_libraries()
    from .render import Renderer
    from .worker import run as run_worker

    run_worker(Renderer(), once="--once" in argv)


def cmd_setup(argv: list[str]) -> None:
    if not onboard(interactive="--yes" not in argv):
        raise SystemExit(1)


def cmd_pair(argv: list[str]) -> None:
    if not argv:
        if not pair_interactively(sys.stdin.isatty()):
            raise SystemExit(1)
        return
    result = pair_machine(argv[0])
    step_line(True, f"Paired as {result['deviceId']}")
    note(f"Saved to {config.FILE}")


def cmd_download(_argv: list[str]) -> None:
    if weights.cached():
        step_line(True, "The model is already here.")
        return
    need_libraries()
    fetch_model()


def cmd_bench(_argv: list[str]) -> None:
    need_libraries()
    if not weights.cached():
        fetch_model()
    if not benchmark().get("approved"):
        raise SystemExit(1)


def cmd_settings(argv: list[str]) -> None:
    if not argv:
        show_settings()
        return
    name = argv[0]
    if len(argv) == 1:
        setting = settings.BY_NAME.get(name)
        if setting is None:
            raise SystemExit(f"there is no setting called {name!r}")
        value = dict((s.name, v) for s, v, _ in settings.current())[name]
        title(f"{setting.name}  {DIM}{value}{OFF}")
        note(setting.detail)
        if setting.values:
            note(f"Accepts: {', '.join(setting.values)}")
        return
    say(f"  {settings.put(name, ' '.join(argv[1:]))}")


def show_settings() -> None:
    title("Settings")
    note("Change one with:  peerpixel settings <name> <value>")
    say()
    for setting, value, warning in settings.current():
        line = f"  {setting.name:<14}{console.CREAM}{value:<26}{OFF}{DIM}{setting.summary}{OFF}"
        say(line)
        if warning:
            say(f"  {' ' * 14}{console.RED}{warning}{OFF}")
    say()
    note(f"Stored in {config.FILE}")


def cmd_status(_argv: list[str]) -> None:
    from .benchmark import generation_warning

    saved = config.read()
    where = state()
    title("This machine")
    step_line(where["libraries"], "Rendering libraries")
    step_line(where["model"], "The model")
    step_line(where["paired"], "Paired",
              saved.get("deviceId", "") if where["paired"] else "ask a moderator for a worker key")
    step_line(where["approved"], "Speed check",
              f"{saved.get('benchMs', 0) / 1000:.1f}s" if saved.get("benchMs") else "")
    warning = generation_warning(saved.get("benchMs", 0), saved.get("accelerator", "")) \
        if saved.get("benchMs") else ""
    if warning:
        note(warning)

    title("The pool")
    try:
        pool = api.pool()
    except api.ApiError as error:
        problem(f"Could not reach {config.API}: {error}")
        return
    say(f"  {pool.get('workersOnline', 0)} online, {pool.get('workersIdle', 0)} idle, "
        f"{pool.get('workersFree', 0)} taking free work")
    say(f"  {pool.get('queued', 0)} paid queued, {pool.get('queuedFree', 0)} free queued, "
        f"{pool.get('running', 0)} rendering")


def cmd_update(_argv: list[str]) -> None:
    def work(made):
        return updater.apply(made)

    result = run_plan("update", work)
    say()
    if result.get("updated"):
        step_line(True, f"Updated to {result['version']}.")
        note("Start PeerPixel again to use it.")
    else:
        step_line(True, f"Already up to date ({result['version']}).")


def cmd_doctor(_argv: list[str]) -> None:
    """Everything somebody would have to ask for about this install."""
    import platform

    title("PeerPixel")
    say(f"  version      {updater.installed()}")
    say(f"  server       {config.API}")
    say(f"  config       {config.FILE}")
    say(f"  folder       {runtime.ROOT}")

    title("This machine")
    say(f"  system       {platform.platform()}")
    say(f"  python       {runtime.python_version()}")
    say(f"  uv           {runtime.uv() or 'not found'}")
    saved = config.read()
    say(f"  device       {saved.get('deviceId', 'not paired')}")
    say(f"  precision    {saved.get('dtype') or 'auto'}"
        + ("   (chosen automatically after a bad render)" if saved.get("dtypeDemoted") else ""))

    if not runtime.dependencies_ready():
        problem("The rendering libraries are not installed, so there is nothing "
                "to test. Run: peerpixel setup")
        return

    from .render import describe_accelerator

    say(f"  renders on   {describe_accelerator()}")
    if not weights.cached():
        problem("The model is not downloaded.")
        return

def cmd_help(_argv: list[str]) -> None:
    banner()
    say()


def cmd_licenses(_argv: list[str]) -> None:
    """Display the model notices shipped beside the worker."""
    notices = runtime.ROOT / "THIRD_PARTY_NOTICES.txt"
    license_file = runtime.ROOT / "APACHE-2.0.txt"
    title("Third-party model notices")
    block(notices.read_text(encoding="utf-8"))
    note(f"Full Apache 2.0 terms: {license_file}")
    note("Run `peerpixel` on its own and it does the right thing: it sets the "
         "machine up if it needs it, then renders until you stop it.")
    say()
    title("Commands")
    for name, summary in HELP:
        say(f"  {console.CREAM}{name:<22}{OFF}{DIM}{summary}{OFF}")
    say()
    block("Settings:  peerpixel settings            to list them\n"
          "           peerpixel settings free on    to change one")
    say()


HELP = (
    ("peerpixel", "update, set up if needed, then render until stopped"),
    ("peerpixel setup", "the guided first run, on its own"),
    ("peerpixel pair KEY", "save the permanent key issued by a moderator"),
    ("peerpixel download", "fetch the model ahead of time"),
    ("peerpixel bench", "time this machine against the admission limit"),
    ("peerpixel status", "this machine, and the state of the pool"),
    ("peerpixel settings", "list or change the knobs"),
    ("peerpixel doctor", "what this machine and install can use"),
    ("peerpixel update", "fetch and install a newer worker"),
    ("peerpixel licenses", "model licenses and third-party notices"),
)

COMMANDS = {
    "start": cmd_start,
    "run": cmd_start,
    "setup": cmd_setup,
    "pair": cmd_pair,
    "download": cmd_download,
    "bench": cmd_bench,
    "settings": cmd_settings,
    "config": cmd_settings,
    "status": cmd_status,
    "update": cmd_update,
    "licenses": cmd_licenses,
    "doctor": cmd_doctor,
    "help": cmd_help,
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if config.read().get("colour") == "off":
        os.environ["NO_COLOR"] = "1"
    # Started on the bare interpreter the launcher provides; move to the one
    # that can see torch as soon as there is one. See runtime.use_venv.
    runtime.use_venv()
    name = argv[0] if argv and not argv[0].startswith("-") else "start"
    handler = COMMANDS.get(name)
    if handler is None:
        problem(f"There is no command called {name!r}.")
        cmd_help([])
        raise SystemExit(1)
    rest = argv[1:] if argv and argv[0] == name else argv
    try:
        handler(rest)
    except KeyboardInterrupt:
        say()
        note("Stopped.")
        raise SystemExit(130) from None
    except api.ApiError as error:
        problem(str(error))
        raise SystemExit(1) from None
    except (RuntimeError, ValueError, OSError) as error:
        problem(str(error))
        raise SystemExit(1) from None
