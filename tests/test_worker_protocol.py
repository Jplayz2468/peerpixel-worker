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


class ProtocolVersionTests(unittest.TestCase):
    def test_this_install_advertises_version_two(self):
        self.assertEqual(worker.PROTOCOL_VERSION, 2)

    def test_the_model_is_kept_loaded_for_two_idle_hours(self):
        self.assertFalse(worker.should_unload_model(100, 100 + 7199, loaded=True))
        self.assertTrue(worker.should_unload_model(100, 100 + 7200, loaded=True))


class ConditioningTests(unittest.TestCase):
    def test_a_master_waits_for_the_draft_that_was_chosen(self):
        reference = bytes([0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9])
        link = FakeLink([
            None,                                                  # a quiet tick
            json.dumps({"type": "ack"}),                            # unrelated chatter
            relay.encode({"type": "conditioning", "jobId": "m1"}, reference),
        ])
        self.assertEqual(
            worker.await_reference(link, "m1", timeout=5, clock=ticking(0.1)),
            reference,
        )

    def test_conditioning_for_a_different_job_is_ignored(self):
        link = FakeLink([
            relay.encode({"type": "conditioning", "jobId": "somebody-else"}, b"\xff\xd8\xff\xd9"),
        ])
        self.assertIsNone(worker.await_reference(link, "m1", timeout=1, clock=ticking(0.3)))

    def test_a_browser_that_never_answers_ends_the_wait(self):
        link = FakeLink([None, None, None])
        self.assertIsNone(worker.await_reference(link, "m1", timeout=1, clock=ticking(0.4)))


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
