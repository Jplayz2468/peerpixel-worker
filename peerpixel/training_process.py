"""Run optional LoRA training in a disposable spawned process."""
from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable

from .trainer import TrainingError, TrainingLease, TrainingReport


def _training_child(connection, lease: TrainingLease, train_backend: Callable | None) -> None:
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

