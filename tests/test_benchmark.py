import unittest

from peerpixel.benchmark import run_benchmark


class FakeRenderer:
    accelerator = "test gpu"

    def __init__(self):
        self.warmed = 0
        self.jobs = []

    def warm(self):
        self.warmed += 1

    def render(self, job):
        self.jobs.append(job)
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
        self.assertEqual(renderer.jobs[0]["steps"], 4)
        self.assertEqual(renderer.jobs[1]["steps"], 4)
        # Master resolution so a card that cannot hold a real render fails here
        # rather than on somebody's paid job, but nowhere near the master step
        # count: a fifty-step admission test would take minutes and be judged
        # against a limit written for a four-step one.
        self.assertEqual(renderer.jobs[1]["operation"], "master")
        self.assertNotIn("width", renderer.jobs[1])
        self.assertEqual(submitted, [(12345, "test gpu")])
        self.assertEqual(result[0], 12345)


if __name__ == "__main__":
    unittest.main()
