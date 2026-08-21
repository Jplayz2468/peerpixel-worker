"""A window. An actual one, with a title bar and a dock icon.

PeerPixel is a thing you unzip and double-click, so what should appear is an
application, not a tab in whatever browser happened to be open with somebody's
email next to it. The interface is still HTML -- there is no build step here and
there is not going to be one -- but it is hosted in a native webview owned by
this process.

Three ways to get a window, tried in order, because no single one of them is
available everywhere:

1. **pywebview.** The real answer: WKWebView on macOS, WebView2 on Windows,
   WebKitGTK on Linux. A native window drawn by the operating system, with the
   operating system's own web engine inside it. Installed alongside the
   launcher's interpreter, which is why it is there before anything else is.

2. **A Chromium-family browser in app mode.** `--app=` gives a chromeless
   window with its own taskbar entry and no address bar, tabs or bookmarks. It
   is a browser, but it does not look or behave like one. This is what saves a
   Linux box with no WebKitGTK.

3. **An ordinary browser tab**, and a line in the console saying so. Last
   resort, and never silent -- somebody who expected an app should be told they
   got a tab, not left wondering why it looks like that.

The server is running in a thread either way; this file only decides what is
looking at it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

TITLE = "PeerPixel"
SIZE = (1140, 840)
MINIMUM = (860, 640)

#: Chromium-family browsers understand --app=. Order is deliberate: whichever
#: is most likely to be the one already installed and updated on that platform.
CHROMIUM = {
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
    "win32": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
}
CHROMIUM_NAMES = ("google-chrome", "google-chrome-stable", "chromium",
                  "chromium-browser", "brave-browser", "microsoft-edge")


def _chromium() -> str | None:
    for path in CHROMIUM.get(sys.platform, []):
        if Path(path).exists():
            return path
    for name in CHROMIUM_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def native(url: str, on_close) -> bool:
    """A window drawn by the operating system. False if there is no webview.

    Blocks until the window is closed, and has to be called from the main
    thread: every one of these toolkits insists on it, and macOS crashes rather
    than complains.
    """
    try:
        import webview  # type: ignore
    except Exception:  # noqa: BLE001 - not installed, or no GUI backend here
        return False
    try:
        handle = webview.create_window(
            TITLE, url,
            width=SIZE[0], height=SIZE[1],
            min_size=MINIMUM,
            background_color="#0a0908",
        )
        handle.events.closed += on_close
        globals()["_handle"] = handle
        webview.start()
        return True
    except Exception as error:  # noqa: BLE001 - fall through to a browser
        print(f"native window unavailable ({error}); falling back", flush=True)
        return False


def close() -> None:
    """Shut the native window, if that is what we ended up with."""
    handle = globals().get("_handle")
    if handle is not None:
        try:
            handle.destroy()
        except Exception:  # noqa: BLE001 - already gone is the desired state
            pass


def app_mode(url: str, profile: Path, on_close) -> bool:
    """A chromeless browser window. Looks like an app; is not a tab.

    Its own profile directory, so it never joins an existing browser session,
    never inherits extensions, and closing it does not close somebody's work.
    """
    browser = _chromium()
    if not browser:
        return False
    profile.mkdir(parents=True, exist_ok=True)
    argv = [
        browser,
        f"--app={url}",
        f"--user-data-dir={profile}",
        f"--window-size={SIZE[0]},{SIZE[1]}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=Translate,AutofillServerCommunication",
    ]
    try:
        process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False

    def watch():
        process.wait()
        on_close()

    threading.Thread(target=watch, daemon=True).start()
    globals()["_process"] = process
    return True


def open_window(url: str, profile: Path, on_close) -> str:
    """Get a window in front of somebody. Returns which kind they got.

    `native` blocks for as long as the window is open, so it is last in the
    body even though it is first in preference: everything after it only runs
    when there was no native window to be had.
    """
    if native(url, on_close):
        return "native"
    if app_mode(url, profile, on_close):
        return "app"
    print("Opening PeerPixel in your browser: no app window is available on this "
          "machine.", flush=True)
    webbrowser.open(url)
    return "browser"


def stop() -> None:
    close()
    process = globals().get("_process")
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
