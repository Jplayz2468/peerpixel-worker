"""What the worker says on the socket, and what it waits for.

No network and no model: a fake link plays back messages the dispatcher would
send. What is being pinned down is the protocol the server relies on, because a
mismatch here means somebody is charged for a picture that never arrives.
"""
import json
import unittest

from peerpixel import relay, worker


class FakeLink:
    """A websocket that hands back a scripted sequence of messages.

    An entry of None is a quiet tick, delivered as the TimeoutError the real
    `recv(timeout=...)` raises, so the waiting loops are exercised as they
    actually run rather than as a straight line.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.sent = []

    def recv(self, timeout=None):
        if not self.script:
            raise TimeoutError
        message = self.script.pop(0)
        if message is None:
            raise TimeoutError
        return message

    def send(self, data):
        self.sent.append(data)


def ticking(values):
    """A monotonic clock that advances a tenth of a second per reading."""
    state = {"t": 0.0}

    def clock():
        state["t"] += values
        return state["t"]

    return clock


#: What each protocol version means, in the only terms the server and this
#: worker have to agree on. Both ends pin sizes and steps, and both refuse a
#: payload that disagrees -- so a size change without a version bump means the
#: dispatcher confidently hands every job to a machine certain to fail it.
SIZES_BY_VERSION = {
    3: {"draft": (128, 16), "master": (512, 50)},
    4: {"draft": (256, 6), "master": (1024, 50)},
    # Same sizes as 4; what changed is that a final is rendered from its seed
    # alone. A version-4 worker would wait forty-five seconds for conditioning
    # bytes that are never sent and then fail the job, so it still has to be
    # kept away from today's work.
    5: {"draft": (256, 6), "master": (1024, 50)},
    # Styled LoRA recipes, optional Qwen prompt enhancement, a pinned auxiliary
    # model manifest, and mandatory pre-delivery moderation evidence.
    6: {"draft": (256, 6), "master": (1024, 50)},
    # Faster pixels and measured phase reporting. Finals retain all fifty
    # guided steps; only their spatial contract changes.
    7: {"draft": (128, 6), "master": (512, 50)},
}


class ProtocolVersionTests(unittest.TestCase):
    def test_the_advertised_version_matches_the_sizes_this_install_pins(self):
        """The bump and the sizes have to move together, or neither is safe.

        This is the test that fails when somebody edits OPERATIONS and forgets
        the version. Getting that wrong is not a degraded network -- it is one
        where every job is dispatched to a worker that refuses it.
        """
        from peerpixel.render import OPERATIONS

        expected = SIZES_BY_VERSION.get(worker.PROTOCOL_VERSION)
        self.assertIsNotNone(
            expected,
            f"protocol {worker.PROTOCOL_VERSION} is not described in SIZES_BY_VERSION; "
            "add what it means, and change it on the server too")
        for name, (size, steps) in expected.items():
            self.assertEqual((OPERATIONS[name]["width"], OPERATIONS[name]["height"]),
                             (size, size), f"{name} size")
            self.assertEqual(OPERATIONS[name]["steps"], steps, f"{name} steps")

    def test_the_current_version_is_the_highest_one_described(self):
        self.assertEqual(worker.PROTOCOL_VERSION, max(SIZES_BY_VERSION))

    def test_the_model_is_kept_loaded_for_two_idle_hours(self):
        self.assertFalse(worker.should_unload_model(100, 100 + 7199, loaded=True))
        self.assertTrue(worker.should_unload_model(100, 100 + 7200, loaded=True))

    def test_idle_status_keeps_session_and_hardware_visible_together(self):
        session = type("Session", (), {"line": lambda self, state: f"{state} · 2 images"})()
        hardware = type("Hardware", (), {"line": lambda self: "CPU 12% · RAM 4/8 GB"})()

        self.assertEqual(
            worker.status_line(session, "online, waiting for work", hardware),
            "CPU 12% · RAM 4/8 GB · online, waiting for work · 2 images",
        )


class DraftSettlementTests(unittest.TestCase):
    def test_an_accepted_draft_reports_what_it_earned(self):
        link = FakeLink([
            None,
            json.dumps({"type": "result_accepted", "jobId": "other", "earnedCredits": 99}),
            json.dumps({"type": "result_accepted", "jobId": "d1", "earnedCredits": 0.1}),
        ])
        self.assertEqual(worker.await_settlement(link, "d1", clock=ticking(0.1)), 0.1)

    def test_a_rejected_draft_raises_so_the_job_is_reported_failed(self):
        link = FakeLink([
            json.dumps({"type": "result_rejected", "jobId": "d1", "reason": "browser_gone"}),
        ])
        with self.assertRaises(RuntimeError) as caught:
            worker.await_settlement(link, "d1", clock=ticking(0.1))
        self.assertIn("browser_gone", str(caught.exception))

    def test_silence_settles_at_zero_rather_than_hanging(self):
        # The ledger is the truth; a missing acknowledgement only costs the
        # local earnings display until the next refresh.
        link = FakeLink([None, None])
        self.assertEqual(worker.await_settlement(link, "d1", timeout=1, clock=ticking(0.4)), 0)


class ResultSizeTests(unittest.TestCase):
    def test_the_worker_knows_the_same_ceiling_the_dispatcher_enforces(self):
        self.assertEqual(relay.MAX_RESULT_BYTES, 256 * 1024)


if __name__ == "__main__":
    unittest.main()
