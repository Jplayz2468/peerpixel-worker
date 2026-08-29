import unittest

from peerpixel.liveness import MemorySnapshot
from peerpixel.supervisor import (
    RUNTIME_CHILD, UPDATE_EXIT, RuntimePolicy, RuntimeState, restart_delay,
)


class RuntimeStateTests(unittest.TestCase):
    def test_repeated_phase_pulses_never_extend_the_original_deadline(self):
        state = RuntimeState(started_at=0)
        state.accept({"type": "phase", "phase": "decoding", "timeout": 90}, now=10)
        self.assertEqual(state.deadline, 100)
        state.accept({"type": "phase", "phase": "decoding", "timeout": 90}, now=80)
        self.assertEqual(state.deadline, 100)
        self.assertTrue(state.expired(100))

    def test_a_new_phase_gets_its_own_fixed_deadline(self):
        state = RuntimeState(started_at=0)
        state.accept({"type": "phase", "phase": "decoding", "timeout": 90}, now=10)
        state.accept({"type": "phase", "phase": "safety_check", "timeout": 30}, now=30)
        self.assertEqual((state.phase, state.deadline), ("safety_check", 60))

    def test_health_recycling_happens_only_while_idle(self):
        policy = RuntimePolicy(max_renders=100, max_age_seconds=86400, max_swap_bytes=10)
        unhealthy = MemorySnapshot(rss_bytes=95, swap_bytes=11, accelerator_bytes=0)
        busy = RuntimeState(started_at=0, idle=False, completed=100, memory=unhealthy)
        idle = RuntimeState(started_at=0, idle=True, completed=100, memory=unhealthy)
        self.assertIsNone(policy.recycle_reason(busy, now=90000, physical_memory=100))
        self.assertEqual(policy.recycle_reason(idle, now=90000, physical_memory=100),
                         "runtime_swap")

    def test_every_recycling_threshold_has_a_stable_reason(self):
        policy = RuntimePolicy(max_renders=100, max_age_seconds=86400,
                               accelerator_idle_watermark=20, max_swap_bytes=10)
        cases = [
            (RuntimeState(0, idle=True, memory=MemorySnapshot(1, 11, 0)), 1, "runtime_swap"),
            (RuntimeState(0, idle=True, memory=MemorySnapshot(91, 0, 0)), 100, "runtime_rss"),
            (RuntimeState(0, idle=True, memory=MemorySnapshot(1, 0, 21)), 100, "runtime_accelerator"),
            (RuntimeState(0, idle=True, completed=100), 100, "runtime_task_limit"),
            (RuntimeState(0, idle=True), 86400, "runtime_age"),
        ]
        for state, value, reason in cases:
            with self.subTest(reason=reason):
                now = value if reason == "runtime_age" else 1
                physical = value if reason == "runtime_rss" else 100
                self.assertEqual(policy.recycle_reason(state, now=now,
                                                       physical_memory=physical), reason)

    def test_small_background_swap_does_not_recycle_a_healthy_runtime(self):
        policy = RuntimePolicy(max_swap_bytes=512 * 1024 * 1024)
        state = RuntimeState(0, idle=True,
                             memory=MemorySnapshot(400_000_000, 50 * 1024 * 1024, 0))
        self.assertIsNone(policy.recycle_reason(
            state, now=1, physical_memory=32 * 1024 * 1024 * 1024))


class BackoffTests(unittest.TestCase):
    def test_restart_backoff_is_bounded_and_resets_after_ready(self):
        self.assertEqual([restart_delay(index) for index in range(9)],
                         [2, 4, 7, 12, 21, 37, 60, 60, 60])

    def test_update_exit_is_distinct_from_failure(self):
        self.assertEqual(UPDATE_EXIT, 75)
        self.assertEqual(RUNTIME_CHILD, "PEERPIXEL_RUNTIME_CHILD")


if __name__ == "__main__":
    unittest.main()
