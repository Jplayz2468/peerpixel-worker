import unittest
from unittest import mock

from peerpixel.liveness import PhaseLease, cleanup_after_task, redact_reason


class PhaseLeaseTests(unittest.TestCase):
    def test_pulses_do_not_extend_the_fixed_deadline(self):
        now = [10.0]
        events = []
        lease = PhaseLease("decoding", 90, events.append, clock=lambda: now[0])
        self.assertEqual(lease.deadline, 100.0)
        now[0] = 50.0
        lease.pulse()
        now[0] = 99.0
        lease.pulse()
        self.assertEqual(lease.deadline, 100.0)
        self.assertEqual([event.phase for event in events], ["decoding", "decoding"])

    def test_timeout_reasons_are_stable_and_do_not_leak_job_content(self):
        secret = 'prompt "private castle" token=abc123'
        self.assertEqual(redact_reason("decoding", TimeoutError(secret)),
                         "phase_timeout:decoding")
        self.assertNotIn("private castle", redact_reason("decoding", TimeoutError(secret)))


class CleanupTests(unittest.TestCase):
    def test_cleanup_collects_python_and_releases_the_active_accelerator_cache(self):
        renderer = mock.Mock(_device="cuda")
        torch = mock.Mock()
        torch.cuda.is_available.return_value = True
        with mock.patch("peerpixel.liveness.gc.collect") as collect:
            cleanup_after_task(renderer, torch_module=torch,
                               psutil_module=mock.Mock(Process=lambda: mock.Mock(
                                   memory_info=lambda: mock.Mock(rss=10),
                                   memory_full_info=lambda: mock.Mock(swap=0))))
        collect.assert_called_once_with()
        torch.cuda.empty_cache.assert_called_once_with()

    def test_cleanup_failures_never_replace_a_completed_task(self):
        renderer = mock.Mock(_device="cuda")
        torch = mock.Mock()
        torch.cuda.empty_cache.side_effect = RuntimeError("driver")
        cleanup_after_task(renderer, torch_module=torch,
                           psutil_module=mock.Mock(Process=lambda: mock.Mock(
                               memory_info=lambda: mock.Mock(rss=10),
                               memory_full_info=lambda: mock.Mock(swap=0))))


if __name__ == "__main__":
    unittest.main()
