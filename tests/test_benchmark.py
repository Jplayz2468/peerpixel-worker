import unittest

from peerpixel.benchmark import BENCH_STEPS, JOB, run_benchmark
from peerpixel.render import NETWORK_OPERATIONS, OPERATIONS


class FakeRenderer:
    accelerator = "test gpu"

    def __init__(self):
        self.warmed = 0
        self.jobs = []

    def warm(self):
        self.warmed += 1

    def render(self, job, on_step=None):
        self.jobs.append(job)
        steps = OPERATIONS[job["operation"]]["steps"]
        if on_step:
            for step in range(1, steps + 1):
                on_step(step, steps)
        return b"jpeg"


class BenchmarkTests(unittest.TestCase):
    def test_only_second_render_is_timed_and_submitted(self):
        renderer = FakeRenderer()
        submitted = []
        ticks = iter((100.0, 112.345))

        result = run_benchmark(
            renderer,
            submit=lambda ms, accelerator: submitted.append((ms, accelerator)) or {"approved": True},
            clock=lambda: next(ticks),
        )

        self.assertEqual(renderer.warmed, 1)
        self.assertEqual(len(renderer.jobs), 2)
        self.assertEqual(renderer.jobs[1]["operation"], "bench")
        self.assertNotIn("width", renderer.jobs[1])
        self.assertEqual(submitted, [(12345, "test gpu")])
        self.assertEqual(result[0], 12345)

    def test_the_benchmark_runs_the_number_of_steps_it_claims_to(self):
        """It did not, and nothing said so.

        Step count is pinned by the operation and ignored from the payload, so
        that a job cannot talk a machine into rendering fewer steps than it was
        paid for. The benchmark asked for four steps as a payload field, which
        that rule quietly discarded: it ran fifty, took twelve times as long as
        intended, and was then timed against a limit written for four.
        """
        from peerpixel.render import operation_of

        spec = operation_of(JOB)
        self.assertEqual(spec["steps"], BENCH_STEPS)
        # Still master resolution: that is what catches a card which cannot
        # hold a real render, and it is the whole reason for the size.
        self.assertEqual(spec["width"], OPERATIONS["master"]["width"])

    def test_the_network_can_never_send_a_benchmark(self):
        # Four steps of work submitted for a fifty-step price.
        self.assertNotIn("bench", NETWORK_OPERATIONS)

    def test_the_bar_is_told_when_the_warm_up_ends(self):
        """Otherwise it fills, resets and fills again, which reads as a failure."""
        renderer = FakeRenderer()
        ticks = iter((100.0, 101.0))
        seen = []

        run_benchmark(
            renderer,
            submit=lambda ms, accelerator: {"approved": True},
            clock=lambda: next(ticks),
            on_step=lambda done, total: seen.append(("step", done, total)),
            between=lambda: seen.append(("between",)),
        )

        self.assertEqual(seen.count(("between",)), 1)
        self.assertEqual(seen.index(("between",)), BENCH_STEPS,
                         "after the first render, not part way through it")
        self.assertEqual(len(seen), BENCH_STEPS * 2 + 1)


if __name__ == "__main__":
    unittest.main()
