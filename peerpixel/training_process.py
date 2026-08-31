"""Run optional LoRA training in a disposable spawned process."""
from __future__ import annotations

import ctypes
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Callable

from .trainer import TrainingError, TrainingLease, TrainingReport


def _die_with_parent() -> None:
    """Never outlive the worker that started us, holding the card.

    Training keeps gigabytes of VRAM. If the parent is killed outright the
    `finally` that stops this child never runs, and the leftover is invisible:
    it holds the GPU while nothing alive admits to owning it. Linux can promise
    the kernel ends us instead. Elsewhere this is a no-op and the start-time
    sweep is the safety net.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001 - a missing promise is not a reason to skip training
        pass


def _training_child(connection, lease: TrainingLease, train_backend: Callable | None) -> None:
    _die_with_parent()
    try:
        from .trainer import run_training

        def progress(step, total):
            connection.send({"type": "progress", "step": step, "total": total})

        report = run_training(
            lease, train_backend=train_backend, progress=progress)
        connection.send({"type": "result", "report": report})
    except BaseException as error:
        detail = " ".join(str(error).split())[:160]
        connection.send({
            "type": "error",
            "reason": f"{type(error).__name__}:{detail}" if detail else type(error).__name__,
        })
    finally:
        connection.close()


def _stop(process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
    _stop_descendants(process.pid)


def _stop_descendants(pid: int, *, psutil_module=None) -> list[int]:
    """End anything training spawned for itself.

    Ending the child releases only what the child held. A dataloader or backend
    worker it started is a separate process on the same card, and killing its
    parent leaves it running and holding memory.
    """
    if psutil_module is None:
        try:
            import psutil as psutil_module
        except Exception:  # noqa: BLE001 - best effort, the sweep still covers this
            return []
    try:
        survivors = psutil_module.Process(pid).children(recursive=True)
    except Exception:  # noqa: BLE001 - the parent is gone, so its children are reparented
        survivors = [child for child in _orphans(psutil_module, pid)]
    ended = []
    for child in survivors:
        try:
            child.kill()
            ended.append(child.pid)
        except Exception:  # noqa: BLE001 - already gone is the outcome we wanted
            pass
    return ended


def _orphans(psutil_module, pid: int):
    """Processes this run started that outlived the lookup of their parent."""
    try:
        for process in psutil_module.process_iter(["name", "ppid"]):
            if process.info.get("ppid") == pid:
                yield process
    except Exception:  # noqa: BLE001 - diagnostics are allowed to be partial
        return


def run_training_isolated(
    lease: TrainingLease,
    *,
    timeout: float,
    progress: Callable | None = None,
    train_backend: Callable | None = None,
) -> TrainingReport:
    """Return a report from a fresh process or fail within a fixed deadline."""
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_training_child,
        args=(sending, lease, train_backend),
        name="peerpixel-training",
    )
    process.start()
    sending.close()
    deadline = time.monotonic() + max(0.01, float(timeout))
    try:
        while time.monotonic() < deadline:
            if receiving.poll(min(0.25, max(0.0, deadline - time.monotonic()))):
                message = receiving.recv()
                if message.get("type") == "progress":
                    if progress is not None:
                        progress(message["step"], message["total"])
                    continue
                if message.get("type") == "result":
                    process.join(timeout=5)
                    report = message.get("report")
                    if not isinstance(report, TrainingReport):
                        raise TrainingError("training_result_invalid")
                    return report
                if message.get("type") == "error":
                    raise TrainingError(str(message.get("reason") or "training_child_failed"))
            if not process.is_alive():
                break
        if process.is_alive():
            raise TrainingError("training_timeout")
        raise TrainingError(f"training_child_exit:{process.exitcode}")
    except (EOFError, OSError) as error:
        raise TrainingError(f"training_child_connection:{type(error).__name__}") from error
    finally:
        receiving.close()
        _stop(process)

