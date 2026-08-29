import os
import tempfile
import time
import unittest
from pathlib import Path

from peerpixel.trainer import TrainingError, TrainingLease
from peerpixel.training_process import run_training_isolated
from tests.test_trainer import active_adapter, lease_payload, snapshot


def pid_backend(lease, training, evaluation, partial):
    (partial / "adapter_model.safetensors").write_bytes(b"candidate")
    return {"childPid": os.getpid(), "loss": 0.1}


def stalled_backend(lease, training, evaluation, partial):
    time.sleep(30)


class IsolatedTrainingTests(unittest.TestCase):
    def bound_lease(self, root: Path) -> TrainingLease:
        body = snapshot()
        return TrainingLease.from_payload(
            lease_payload(body), expected_device_id="owner-device", now=1_000,
        ).bind_local(
            body, active_adapter=active_adapter(root), staging_root=root / "staging",
        )

    def test_training_runs_in_a_disposable_spawned_process(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_training_isolated(
                self.bound_lease(Path(folder)), timeout=10, train_backend=pid_backend,
            )
            self.assertEqual(report.status, "staged")
            self.assertNotEqual(report.metrics["childPid"], os.getpid())
            self.assertTrue(report.artifact_path.exists())

    def test_a_stalled_training_process_is_terminated_at_the_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            started = time.monotonic()
            with self.assertRaisesRegex(TrainingError, "training_timeout"):
                run_training_isolated(
                    self.bound_lease(Path(folder)), timeout=.1,
                    train_backend=stalled_backend,
                )
            self.assertLess(time.monotonic() - started, 5)


if __name__ == "__main__":
    unittest.main()
