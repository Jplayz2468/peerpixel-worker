"""Thin wrapper over the PeerPixel HTTP API. Standard library only."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class ApiError(Exception):
    def __init__(self, status: int, code: str, body: dict | None = None):
        super().__init__(f"{code} ({status})")
        self.status = status
        self.code = code
        self.body = body or {}


#: Cloudflare blocks urllib's default user agent outright, so identify properly.
USER_AGENT = "peerpixel-worker/0.8.9 (+https://github.com/Jplayz2468/peerpixel-worker)"


def _call(path: str, *, method="GET", payload=None, raw=None, auth=True, cookie=False, timeout=120):
    headers = {"user-agent": USER_AGENT, "accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["content-type"] = "application/json"
    elif raw is not None:
        body = raw
        headers["content-type"] = "image/jpeg"
    if auth:
        token = config.read().get("token")
        if token:
            headers["authorization"] = f"Bearer {token}"
    if cookie:
        session = config.session()
        if session:
            headers["cookie"] = f"pp={session}"

    request = urllib.request.Request(f"{config.API}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode() or "{}"
            return json.loads(text)
    except urllib.error.HTTPError as error:
        text = error.read().decode() or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"error": text[:200]}
        raise ApiError(error.code, parsed.get("error", "http_error"), parsed) from None


def pair(code: str, info: dict) -> dict:
    return _call("/api/pair/claim", method="POST", payload={"code": code, **info}, auth=False)


def submit_bench(ms: int, accelerator: str) -> dict:
    return _call("/api/device/bench", method="POST", payload={"ms": ms, "accelerator": accelerator})


def submit_result(job_id: str, frame: bytes) -> dict:
    return _call(f"/api/device/job/{job_id}/result", method="POST", raw=frame, timeout=300)


def verify_asset(check_id: str, which: str) -> bytes:
    """One of the two pictures a check compares. Raw bytes, not JSON."""
    import urllib.request

    token = config.read().get("token", "")
    request = urllib.request.Request(
        f"{config.API}/api/device/verify/{check_id}/{which}",
        headers={"user-agent": USER_AGENT, "authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, "verify_asset_failed") from None


def submit_verification(check_id: str, measurements: dict) -> dict:
    return _call(f"/api/device/verify/{check_id}/result", method="POST",
                 payload=measurements, timeout=120)


def upscale_source(job_id: str) -> bytes:
    token = config.read().get("token", "")
    request = urllib.request.Request(
        f"{config.API}/api/device/upscale/{job_id}/source",
        headers={"user-agent": USER_AGENT, "authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, "upscale_source_failed") from None


def auxiliary_input(check_id: str) -> bytes:
    token = config.read().get("token", "")
    request = urllib.request.Request(
        f"{config.API}/api/device/auxiliary/{check_id}/input",
        headers={"user-agent": USER_AGENT, "authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, "auxiliary_input_failed") from None


def submit_auxiliary(check_id: str, output_digest: str) -> dict:
    return _call(f"/api/device/auxiliary/{check_id}/result", method="POST",
                 payload={"outputDigest": output_digest}, timeout=300)


def set_free(device_id: str, allow: bool) -> dict:
    """Opt this machine in or out of unpaid work.

    Account-level, so it wants the website session; a device token gets a 401
    here no matter how valid it is. The caller explains that.
    """
    return _call("/api/device/free", method="POST", cookie=True,
                 payload={"deviceId": device_id, "allowFree": bool(allow)})


def pool() -> dict:
    return _call("/api/pool", auth=False)
