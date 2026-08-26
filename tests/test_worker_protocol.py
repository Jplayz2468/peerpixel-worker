"""What the worker says on the socket, and what it waits for.

No network and no model: a fake link plays back messages the dispatcher would
send. What is being pinned down is the protocol the server relies on, because a
mismatch here means somebody is charged for a picture that never arrives.
"""
import json
import unittest
from contextlib import nullcontext
from unittest import mock

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
    # Styled recipes, optional Qwen prompt enhancement, a pinned model-set
    # contract, and mandatory pre-delivery moderation evidence.
    6: {"draft": (256, 6), "master": (1024, 50)},
    # Faster pixels and measured phase reporting. Finals retain all fifty
    # guided steps; only their spatial contract changes.
    7: {"draft": (128, 6), "master": (512, 50)},
    # Auto-style requests are resolved by Qwen and returned as a concrete
    # style/recipe pair in the generation evidence.
    8: {"draft": (128, 6), "master": (512, 50)},
    # Temporary quality experiment: drafts retain their tiny canvas but use
    # the same full denoising schedule as finals.
    9: {"draft": (128, 50), "master": (512, 50)},
    # Full-size finals return while temporary drafts keep fifty steps.
    10: {"draft": (128, 50), "master": (1024, 50)},
    # Public generation is one direct final. Fraud checks are explicitly
    # internal probes with their own fixed wire contract.
    11: {"master": (1024, 50), "probe": (128, 50)},
    # Private-job consent and content-free official worker presentation.
    12: {"master": (1024, 50), "probe": (128, 50)},
    # Five trusted near-one-megapixel aspect ratios. The default remains square;
    # the complete accepted set is pinned by the render operation tests.
    13: {"master": (1024, 50), "probe": (128, 50)},
    # Source-conditioned variation and masked inpainting on capable CUDA workers.
    14: {"master": (1024, 50), "probe": (128, 50)},
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

    def test_protocol_fourteen_accepts_server_directed_updates_while_idle(self):
        requested = []
        handled = worker.handle_idle_control(
            {"type": "ack", "requiredWorkerVersion": "0.14.1"},
            installed="0.14.0",
            update=requested.append,
        )
        self.assertTrue(handled)
        self.assertEqual(requested, ["0.14.1"])

    def test_an_equal_or_malformed_server_version_is_not_an_update(self):
        requested = []
        for message in (
            {"type": "ack", "requiredWorkerVersion": "0.14.0"},
            {"type": "ack", "requiredWorkerVersion": ""},
            {"type": "job", "requiredWorkerVersion": "9.0.0"},
        ):
            self.assertFalse(worker.handle_idle_control(
                message, installed="0.14.0", update=requested.append,
            ))
        self.assertEqual(requested, [])

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


class SocketResultTests(unittest.TestCase):
    def test_an_edit_fetches_private_source_and_mask_only_after_assignment(self):
        link = FakeLink()
        renderer = mock.Mock()
        renderer.pipe = object()
        renderer._precision_mode = "native"
        renderer._memory_mode = "resident"
        renderer.generate_job.return_value = (b"jpeg", {"attestations": []})
        session = mock.Mock(images=0, pixels=0.0)
        bar = mock.Mock()
        reporter = mock.Mock()
        job = {
            "id": "edit-1", "prompt": "fix the hand", "seed": 7,
            "operation": "master", "steps": 50, "width": 1024, "height": 1024,
            "editMode": "inpaint", "editStrength": .65,
            "sourceImageId": "source", "hasMask": True,
        }

        with mock.patch.object(worker.plans, "tracker", return_value=bar), \
             mock.patch.object(worker.plans, "remember"), \
             mock.patch.object(worker.console, "Live", return_value=nullcontext()), \
             mock.patch("peerpixel.job_phases.PhaseReporter", return_value=reporter), \
             mock.patch.object(worker.api, "edit_asset", side_effect=[b"source", b"mask"]) as fetch, \
             mock.patch.object(worker.api, "submit_result", return_value={"earnedCredits": 1}):
            worker._do_job(link, job, renderer, session, [link, ""], [0.0], [""])

        self.assertEqual(fetch.call_args_list, [
            mock.call("edit-1", "source"), mock.call("edit-1", "mask"),
        ])
        rendered = renderer.generate_job.call_args.args[0]
        self.assertEqual(rendered["_editSource"], b"source")
        self.assertEqual(rendered["_editMask"], b"mask")

    def test_an_accepted_socket_result_reports_what_it_earned(self):
        link = FakeLink([
            None,
            json.dumps({"type": "result_accepted", "jobId": "other", "earnedCredits": 99}),
            json.dumps({"type": "result_accepted", "jobId": "p1", "earnedCredits": 0.2}),
        ])
        self.assertEqual(worker.await_settlement(link, "p1", clock=ticking(0.1)), 0.2)

    def test_a_rejected_socket_result_raises_so_the_job_is_reported_failed(self):
        link = FakeLink([
            json.dumps({"type": "result_rejected", "jobId": "p1", "reason": "byte_length"}),
        ])
        with self.assertRaises(RuntimeError) as caught:
            worker.await_settlement(link, "p1", clock=ticking(0.1))
        self.assertIn("byte_length", str(caught.exception))

    def test_silence_settles_at_zero_rather_than_hanging(self):
        # The ledger is the truth; a missing acknowledgement only costs the
        # local earnings display until the next refresh.
        link = FakeLink([None, None])
        self.assertEqual(worker.await_settlement(link, "p1", timeout=1, clock=ticking(0.4)), 0)

    def test_an_internal_probe_uses_probe_result_framing(self):
        link = FakeLink([
            json.dumps({"type": "result_accepted", "jobId": "probe-1", "earnedCredits": 0.2}),
        ])
        renderer = mock.Mock()
        renderer.pipe = object()
        renderer._precision_mode = "native"
        renderer._memory_mode = "resident"
        renderer.generate_job.return_value = (b"jpeg", {"attestations": []})
        session = mock.Mock(images=0, pixels=0.0)
        bar = mock.Mock()
        reporter = mock.Mock()
        job = {
            "id": "probe-1", "prompt": "a quiet harbour", "seed": 7,
            "operation": "probe", "steps": 50, "width": 128, "height": 128,
            "transient": True,
        }

        with mock.patch.object(worker.plans, "tracker", return_value=bar), \
             mock.patch.object(worker.plans, "remember"), \
             mock.patch.object(worker.console, "Live", return_value=nullcontext()), \
             mock.patch("peerpixel.job_phases.PhaseReporter", return_value=reporter):
            earned = worker._do_job(
                link, job, renderer, session, [link, ""], [0.0], [""],
            )

        frame = next(message for message in link.sent if isinstance(message, bytes))
        header, payload = relay.decode(frame)
        self.assertEqual(header["type"], "probe_result")
        self.assertEqual(header["jobId"], "probe-1")
        self.assertNotIn("draftId", header)
        self.assertEqual(payload, b"jpeg")
        self.assertEqual(earned, 0.2)


class CompletedResultNotificationTests(unittest.TestCase):
    def test_a_closed_socket_after_http_acceptance_does_not_fail_the_render(self):
        class ClosedLink:
            def send(self, _data):
                raise RuntimeError("no close frame received or sent")

        self.assertFalse(worker.notify_finished(ClosedLink(), "master1"))


class ResultSizeTests(unittest.TestCase):
    def test_the_worker_knows_the_same_ceiling_the_dispatcher_enforces(self):
        self.assertEqual(relay.MAX_RESULT_BYTES, 256 * 1024)


if __name__ == "__main__":
    unittest.main()
