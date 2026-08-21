"""The knobs, and what each one is for.

Declared rather than scattered, so `peerpixel settings` can list them, explain
them and check them without a second copy of the same list going stale. Each
one says what it accepts and what it means; the help you get in the terminal is
generated from exactly the same rows the setter validates against.

Two of these reach past this machine. `free` belongs to the account, not the
install, so setting it also asks peerpixel.cc -- and says so plainly when the
answer never arrived. The rest are local and take effect on the next render.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from . import api, config


@dataclass(frozen=True)
class Setting:
    name: str
    summary: str
    values: tuple[str, ...]        # empty means free text
    default: str
    detail: str = ""

    def parse(self, raw: str):
        text = raw.strip().lower()
        if self.values and text not in self.values:
            raise ValueError(
                f"{self.name} takes {' or '.join(self.values)}, not {raw!r}")
        if self.values == ("on", "off"):
            return text == "on"
        return raw.strip()

    def blank(self):
        """The value when nothing has been stored.

        Typed, not the display string. `default="off"` read back as a stored
        value is a non-empty string, and every switch in the program would read
        as on -- which is exactly what it did.
        """
        if self.values == ("on", "off"):
            return self.default == "on"
        return self.default

    def show(self, stored) -> str:
        if self.values == ("on", "off"):
            return "on" if stored else "off"
        return str(self.default if stored in (None, "") else stored)


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "free", "Also render for people with no pixels", ("on", "off"), "off",
        "Free jobs pay nothing and always queue behind paid ones. Nobody's card "
        "renders for strangers without this. It belongs to your account rather "
        "than to this machine, so setting it needs peerpixel.cc to agree."),
    Setting(
        "dtype", "Arithmetic precision", ("auto", "bfloat16", "float16", "float32"),
        "auto",
        "auto is what the checkpoint was trained in, and is right on nearly "
        "every machine. Nearly: where it is not, renders come back as flat grey "
        "with black specks, and the worker drops down this list on its own and "
        "remembers. Set it by hand only to overrule that."),
    Setting(
        "keep-last", "Write the last picture rendered to disk", ("on", "off"), "on",
        "One file, replaced each time, so you can see what this machine is "
        "actually producing. `peerpixel doctor` tells you where it is."),
    Setting(
        "unload-after", "Minutes idle before the model leaves memory", (), "120",
        "A loaded model holds several gigabytes. Dropping it frees them and "
        "costs the next job a reload. 0 keeps it forever."),
    Setting(
        "colour", "Colour and animation in this terminal", ("auto", "off"), "auto",
        "auto follows the terminal, NO_COLOR and whether output is a pipe."),
    Setting(
        "api", "Which server to talk to", (), "https://peerpixel.cc",
        "Only useful if you are running your own."),
)

BY_NAME = {setting.name: setting for setting in SETTINGS}

#: What each setting is called in the config file, where the names are older
#: than this table and are not worth a migration.
STORED = {"free": "allowFree", "dtype": "dtype", "keep-last": "keepLast",
          "unload-after": "unloadAfterMinutes", "colour": "colour", "api": "api"}


def current() -> list[tuple[Setting, str, str]]:
    """Every setting, its value, and a note when the value is not the whole story."""
    saved = config.read()
    out = []
    for setting in SETTINGS:
        value = setting.show(saved.get(STORED[setting.name], setting.blank()))
        note = ""
        if setting.name == "free" and value == "on" and not saved.get("allowFreeSyncedAt"):
            note = "saved here, never accepted by your account"
        if setting.name == "dtype" and value != "auto" and saved.get("dtypeDemoted"):
            note = "chosen automatically after a render came back as nan"
        out.append((setting, value, note))
    return out


def put(name: str, raw: str) -> str:
    """Set one. Returns a sentence for the person who set it."""
    setting = BY_NAME.get(name)
    if setting is None:
        raise ValueError(f"there is no setting called {name!r}")
    value = setting.parse(raw)
    config.write(**{STORED[name]: value})

    if name == "free":
        return _sync_free(bool(value))
    if name == "dtype":
        config.write(dtypeDemoted=False)
        return ("Precision is chosen automatically again."
                if value == "auto" else f"Rendering in {value} from now on.")
    if name == "unload-after":
        try:
            minutes = int(value)
        except ValueError:
            raise ValueError("unload-after takes a number of minutes") from None
        config.write(unloadAfterMinutes=minutes)
        return ("The model stays loaded." if minutes <= 0
                else f"The model leaves memory after {minutes} idle minutes.")
    return f"{name} is {setting.show(value)}."


def _sync_free(wanted: bool) -> str:
    """The one switch this machine cannot decide on its own.

    Remembered locally either way, so a machine that was offline when it was
    thrown still knows what was asked for. The confirmation is remembered
    separately, because one carried over from an older setting would be a lie.
    """
    config.write(allowFreeSyncedAt=0)
    device = config.read().get("deviceId")
    if not device:
        return "Saved. It will be sent to your account when this machine is paired."
    try:
        api.set_free(device, wanted)
    except api.ApiError as error:
        if error.status in (401, 403):
            return ("Saved on this machine, but your account has not agreed to it. "
                    "The free switch belongs to the account: turn it on from the "
                    "Contribute page at peerpixel.cc, or export PEERPIXEL_SESSION "
                    "with your pp cookie and set it again.")
        raise
    config.write(allowFreeSyncedAt=int(time.time()))
    return ("Now taking free work as well as paid." if wanted else "Paid work only.")


def unload_seconds() -> float:
    try:
        minutes = float(config.read().get("unloadAfterMinutes", 120))
    except (TypeError, ValueError):
        minutes = 120.0
    return max(0.0, minutes) * 60.0


def keep_last() -> bool:
    return bool(config.read().get("keepLast", True))
