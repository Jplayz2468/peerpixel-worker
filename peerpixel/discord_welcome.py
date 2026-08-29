"""Owner-only Discord Gateway listener for welcoming new server members."""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

from . import config

GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
DISCORD_API = "https://discord.com/api/v10"
GATEWAY_INTENTS = (1 << 0) | (1 << 1)  # GUILDS + privileged GUILD_MEMBERS
_started = False
_start_lock = threading.Lock()


def welcome_message(guild_id: str, channel_id: str) -> str:
    channel_url = f"https://discord.com/channels/{guild_id}/{channel_id}"
    return (
        "Welcome to **PeerPixel**! 🎨\n\n"
        f"To make your first image, open **#imagine in the PeerPixel server**: {channel_url}\n"
        "Then type `/imagine`, choose the `prompt` field, and describe anything you want to see. "
        "You'll receive four images with buttons underneath to upscale or vary your favorites.\n\n"
        "Don't send `/imagine` in this DM—the command must be run inside the PeerPixel server."
    )


class WelcomeState:
    """Persist who was considered so reconnects cannot spam DMs."""

    def __init__(self, guild_id: str, channel_id: str, state_file: Path,
                 send_dm: Callable[[str, str], None]):
        self.guild_id = str(guild_id)
        self.channel_id = str(channel_id)
        self.state_file = Path(state_file)
        self.send_dm = send_dm
        self.bootstrapped = self.state_file.exists()
        self.seen = self._read()

    def _read(self) -> set[str]:
        try:
            return {str(value) for value in json.loads(self.state_file.read_text())}
        except (OSError, json.JSONDecodeError, TypeError):
            return set()

    def _write(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(sorted(self.seen)))
        temporary.replace(self.state_file)

    def _consider(self, user: dict, *, notify: bool) -> None:
        user_id = str(user.get("id") or "")
        if not user_id or user.get("bot") is True or user_id in self.seen:
            return
        self.seen.add(user_id)
        self._write()
        if not notify:
            return
        try:
            self.send_dm(user_id, welcome_message(self.guild_id, self.channel_id))
        except Exception:
            pass  # closed DMs are normal and must not affect rendering

    def handle_dispatch(self, event: str, data: dict) -> None:
        if event == "GUILD_CREATE" and str(data.get("id")) == self.guild_id:
            notify = self.bootstrapped
            for member in data.get("members") or []:
                self._consider(member.get("user") or {}, notify=notify)
            if not self.bootstrapped:
                self.bootstrapped = True
                self._write()
            return
        if event == "GUILD_MEMBER_ADD" and str(data.get("guild_id")) == self.guild_id:
            self._consider(data.get("user") or {}, notify=True)


def _discord_json(token: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{DISCORD_API}{path}", data=json.dumps(payload).encode(), method="POST",
        headers={
            "authorization": f"Bot {token}",
            "content-type": "application/json",
            "user-agent": "PeerPixel welcome listener (+https://peerpixel.cc)",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode() or "{}")


def send_direct_message(token: str, user_id: str, message: str) -> None:
    channel = _discord_json(token, "/users/@me/channels", {"recipient_id": user_id})
    _discord_json(token, f"/channels/{channel['id']}/messages", {
        "content": message, "allowed_mentions": {"parse": []},
    })


def run_gateway(token: str, guild_id: str, channel_id: str,
                *, state_file: Path | None = None) -> None:
    """Maintain the minimal Gateway session forever on a daemon thread."""
    from websockets.sync.client import connect

    state = WelcomeState(
        guild_id, channel_id,
        state_file or (config.HOME / "welcomed-members.json"),
        lambda user_id, message: send_direct_message(token, user_id, message),
    )
    backoff = 2
    while True:
        try:
            with connect(GATEWAY, ping_interval=None, close_timeout=5) as socket:
                sequence = None
                heartbeat_at = None
                heartbeat_seconds = 40.0
                while True:
                    timeout = max(0.1, (heartbeat_at or time.monotonic() + 60) - time.monotonic())
                    try:
                        payload = json.loads(socket.recv(timeout=timeout))
                    except TimeoutError:
                        socket.send(json.dumps({"op": 1, "d": sequence}))
                        heartbeat_at = time.monotonic() + heartbeat_seconds
                        continue
                    if payload.get("s") is not None:
                        sequence = payload["s"]
                    opcode = payload.get("op")
                    if opcode == 10:
                        heartbeat_seconds = float(payload["d"]["heartbeat_interval"]) / 1000
                        heartbeat_at = time.monotonic() + random.random() * heartbeat_seconds
                        socket.send(json.dumps({"op": 2, "d": {
                            "token": token, "intents": GATEWAY_INTENTS,
                            "properties": {"os": "peerpixel-worker",
                                           "browser": "peerpixel-welcome",
                                           "device": "peerpixel-welcome"},
                            "large_threshold": 250,
                        }}))
                    elif opcode == 1:
                        socket.send(json.dumps({"op": 1, "d": sequence}))
                        heartbeat_at = time.monotonic() + heartbeat_seconds
                    elif opcode == 0:
                        state.handle_dispatch(str(payload.get("t") or ""), payload.get("d") or {})
                    elif opcode in (7, 9):
                        break
                backoff = 2
        except Exception:
            time.sleep(backoff)
            backoff = min(60, int(backoff * 1.8) or 2)


def start_gateway_listener() -> bool:
    """Start only on the owner's explicitly configured worker."""
    token = os.environ.get("PEERPIXEL_DISCORD_BOT_TOKEN", "").strip()
    guild_id = os.environ.get("PEERPIXEL_DISCORD_GUILD_ID", "").strip()
    channel_id = os.environ.get("PEERPIXEL_DISCORD_IMAGINE_CHANNEL_ID", "").strip()
    if not all((token, guild_id, channel_id)):
        return False
    global _started
    with _start_lock:
        if _started:
            return True
        threading.Thread(target=run_gateway, args=(token, guild_id, channel_id),
                         name="peerpixel-discord-welcome", daemon=True).start()
        _started = True
    return True
