"""Only our own leftovers, and only when they are worth taking back."""
from __future__ import annotations

import os

import pytest

from peerpixel import gpu_reclaim

GIGABYTE = 1024 ** 3


class FakeProcess:
    def __init__(self, pid, name, cmdline, username, parents=(), children=()):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline
        self._username = username
        self._parents = list(parents)
        self._children = list(children)
        self.terminated = False
        self.killed = False

    def name(self):
        return self._name

    def cmdline(self):
        return self._cmdline

    def username(self):
        return self._username

    def parents(self):
        return self._parents

    def children(self, recursive=False):
        return self._children

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakePsutil:
    def __init__(self, processes, *, stubborn=()):
        self._processes = {process.pid: process for process in processes}
        self._stubborn = set(stubborn)

    def Process(self, pid):  # noqa: N802 - mirrors psutil's own name
        if pid not in self._processes:
            raise LookupError(f"no process {pid}")
        return self._processes[pid]

    def wait_procs(self, processes, timeout=None):
        alive = [process for process in processes if process.pid in self._stubborn]
        gone = [process for process in processes if process.pid not in self._stubborn]
        return gone, alive


def owner():
    return "renderer"


def worker(pid=100):
    return FakeProcess(pid, "python", ["python", "-m", "peerpixel"], owner())


def test_a_leftover_trainer_is_ended_and_its_memory_reported():
    leftover = FakeProcess(777, "peerpixel-training",
                           ["python", "-c", "from peerpixel.training_process import _training_child"],
                           owner())
    psutil_module = FakePsutil([worker(), leftover])

    verdicts = gpu_reclaim.reclaim([("python", 777, 15 * GIGABYTE)],
                                   psutil_module=psutil_module, pid=100)

    assert leftover.terminated is True
    assert verdicts == [{"pid": 777, "name": "python", "bytes": 15 * GIGABYTE,
                         "reclaim": True, "reason": "PeerPixel leftover"}]
    assert gpu_reclaim.describe(verdicts) == [
        "Reclaimed 15.0 GB from a leftover PeerPixel process (PID 777).",
    ]


def test_a_process_that_ignores_terminate_is_killed():
    leftover = FakeProcess(777, "peerpixel-training", ["peerpixel", "train"], owner())
    psutil_module = FakePsutil([worker(), leftover], stubborn={777})

    gpu_reclaim.reclaim([("python", 777, 9 * GIGABYTE)], psutil_module=psutil_module, pid=100)

    assert (leftover.terminated, leftover.killed) == (True, True)


@pytest.mark.parametrize("name,cmdline,username,reason", [
    ("blender", ["blender", "scene.blend"], owner(), "not a PeerPixel process"),
    ("python", ["python", "train_my_own_model.py"], owner(), "not a PeerPixel process"),
    ("python", ["python", "-m", "peerpixel"], "someone-else", "owned by someone-else"),
])
def test_somebody_elses_work_is_never_ended(name, cmdline, username, reason):
    other = FakeProcess(777, name, cmdline, username)
    psutil_module = FakePsutil([worker(), other])

    verdicts = gpu_reclaim.reclaim([(name, 777, 14 * GIGABYTE)],
                                   psutil_module=psutil_module, pid=100)

    assert (other.terminated, other.killed) == (False, False)
    assert verdicts[0]["reclaim"] is False
    assert verdicts[0]["reason"] == reason


def test_an_unreadable_process_is_left_alone():
    psutil_module = FakePsutil([worker()])

    verdicts = gpu_reclaim.reclaim([("python", 777, 14 * GIGABYTE)],
                                   psutil_module=psutil_module, pid=100)

    assert verdicts[0]["reclaim"] is False
    assert verdicts[0]["reason"] == "not readable"


def test_this_run_and_its_own_family_are_never_ended():
    parent = FakeProcess(1, "systemd", ["systemd"], owner())
    child = FakeProcess(101, "peerpixel-training", ["peerpixel", "train"], owner())
    running = FakeProcess(100, "python", ["python", "-m", "peerpixel"], owner(),
                          parents=[parent], children=[child])
    psutil_module = FakePsutil([running, parent, child])

    verdicts = gpu_reclaim.reclaim(
        [("python", 100, 8 * GIGABYTE), ("python", 101, 7 * GIGABYTE)],
        psutil_module=psutil_module, pid=100)

    assert [record["reason"] for record in verdicts] == ["part of this run", "part of this run"]
    assert (child.terminated, child.killed) == (False, False)


def test_small_holdings_are_beneath_notice():
    leftover = FakeProcess(777, "peerpixel-training", ["peerpixel", "train"], owner())
    psutil_module = FakePsutil([worker(), leftover])

    verdicts = gpu_reclaim.reclaim([("python", 777, 40 * 1024 ** 2)],
                                   psutil_module=psutil_module, pid=100)

    assert verdicts == []
    assert leftover.terminated is False


def test_protected_pids_survive_an_unreadable_process_tree():
    class Hostile:
        def Process(self, pid):  # noqa: N802 - mirrors psutil's own name
            raise PermissionError("no")

    assert gpu_reclaim.protected_pids(Hostile(), 4242) == {4242}
    assert os.getpid() in gpu_reclaim.protected_pids(Hostile())
