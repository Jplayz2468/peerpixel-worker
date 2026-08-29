import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from peerpixel.lora_manifest import load_manifest, write_manifest
from peerpixel.trainer import (
    TrainerCapability, TrainingHttpClient, TrainingLease, TrainingReport,
    package_candidate, run_training,
)


def active_adapter(root: Path) -> Path:
    adapter = root / "active" / "bootstrap-0001"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"active adapter")
    write_manifest(adapter, {
        "schemaVersion": 1,
        "version": "bootstrap-0001",
        "kind": "bootstrap",
        "baseModel": "Qwen/Qwen3-1.7B",
        "parentVersion": None,
        "dataset": {},
        "training": {},
        "evaluation": {"passed": True},
        "createdAt": "2026-08-27T00:00:00Z",
    })
    return adapter


def snapshot() -> bytes:
    return json.dumps({
      "version": "preference-snapshot-v1",
      "metadata": {"anchors": {"version": "prompt-anchors-v1"}},
      "train": [
        {"prompt": "fox", "chosen": "A red fox in a cedar grove.",
         "rejected": "A fox.", "weight": 1.0, "eventId": "event-u"},
      ],
      "evaluation": [
        {"prompt": "harbor", "chosen": "A quiet harbor at blue hour.",
         "rejected": "A harbor.", "weight": 0.5, "eventId": "event-eval"},
      ],
      "replay": [
        {"prompt": "owl", "chosen": "A barn owl above a moonlit field.",
         "rejected": "An owl.", "weight": 0.5, "eventId": "event-replay"},
      ],
    }, sort_keys=True, separators=(",", ":")).encode()


def lease_payload(body: bytes) -> dict:
    return {
        "runId": "run-0001",
        "leaseToken": "signed-short-lived-token",
        "deviceId": "owner-device",
        "expiresAt": 2_000.0,
        "snapshotDigest": hashlib.sha256(body).hexdigest(),
        "parentVersion": "bootstrap-0001",
        "candidateVersion": "preference-0001",
        "training": {
            "beta": 0.1,
            "maxSteps": 4,
            "learningRate": 1e-6,
            "batchSize": 1,
            "gradientAccumulationSteps": 2,
            "seed": 2468,
        },
    }


class FakeTrainingClient:
    def __init__(self, payload=None, body=b""):
        self.payload = payload
        self.body = body
        self.polls = 0
        self.uploads = []
        self.reports = []

    def lease_training(self, device_id):
        self.polls += 1
        return self.payload

    def download_snapshot(self, lease):
        return self.body

    def upload_candidate(self, lease, artifact_path, report):
        self.uploads.append((lease.run_id, Path(artifact_path), report.status))

    def report_training(self, lease, report):
        self.reports.append((lease.run_id, report.status, report.reason))


class TrainerLifecycleTests(unittest.TestCase):
    def test_retry_replaces_only_an_incomplete_staging_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = snapshot()
            lease = TrainingLease.from_payload(
                lease_payload(body), expected_device_id="owner-device", now=1_000.0,
            ).bind_local(body, active_adapter=active_adapter(root), staging_root=root / "staged")
            partial = root / "staged" / "preference-0001.partial"
            partial.mkdir(parents=True)
            (partial / "stale").write_text("failed attempt")

            def backend(_lease, _rows, _evaluation, output_dir):
                (Path(output_dir) / "adapter_model.safetensors").write_bytes(b"candidate")
                return {"trainLoss": 0.2, "steps": 1}

            report = run_training(lease, train_backend=backend)

            self.assertEqual(report.status, "staged")
            self.assertFalse(partial.exists())
            self.assertTrue((root / "staged" / "preference-0001").is_dir())

    def test_capability_is_disabled_by_default_and_never_polls_when_busy(self):
        client = FakeTrainingClient()
        capability = TrainerCapability(client, device_id="owner-device")
        self.assertEqual(capability.capabilities, ())
        self.assertFalse(capability.poll_when_idle())
        self.assertEqual(client.polls, 0)

        opted_in_but_busy = TrainerCapability(
            client, device_id="owner-device", enabled=True, is_idle=lambda: False,
        )
        self.assertEqual(opted_in_but_busy.capabilities, ("train",))
        self.assertFalse(opted_in_but_busy.poll_when_idle())
        self.assertEqual(client.polls, 0)

    def test_owner_opt_in_verifies_snapshot_and_stages_without_mutating_active_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = active_adapter(root)
            before = {path.name: path.read_bytes() for path in active.iterdir()}
            body = snapshot()
            client = FakeTrainingClient(lease_payload(body), body)
            calls = []

            def train_backend(lease, rows, evaluation, output_dir):
                calls.append((lease.parent_version, rows, evaluation, Path(output_dir)))
                (Path(output_dir) / "adapter_model.safetensors").write_bytes(b"candidate")
                return {"trainLoss": 0.25, "steps": 4}

            capability = TrainerCapability(
                client, device_id="owner-device", enabled=True,
                active_adapter=active, staging_root=root / "staged",
                train_backend=train_backend, now=lambda: 1_000.0,
            )
            self.assertTrue(capability.poll_when_idle())
            self.assertFalse(capability.training)

            candidate = root / "staged" / "preference-0001"
            self.assertTrue(candidate.is_dir())
            self.assertFalse(candidate.with_name("preference-0001.partial").exists())
            manifest = load_manifest(candidate / "manifest.json")
            self.assertEqual(manifest["kind"], "preference")
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["parentVersion"], "bootstrap-0001")
            self.assertEqual(manifest["snapshotDigest"], hashlib.sha256(body).hexdigest())
            self.assertEqual(manifest["dataset"]["snapshotDigest"],
                             hashlib.sha256(body).hexdigest())
            self.assertEqual([row["eventId"] for row in calls[0][1]],
                             ["event-u", "event-replay"])
            self.assertEqual([row["eventId"] for row in calls[0][2]],
                             ["event-eval"])
            gates = manifest["evaluation"]["gates"]
            self.assertEqual(set(gates), {"responseStructure", "weightedPreferenceAccuracy",
                "promptFidelity", "diversity", "phraseConcentration", "anchorDrift"})
            self.assertTrue(all(value is True for value in gates.values()))
            self.assertTrue(manifest["evaluation"]["passed"])
            self.assertEqual(manifest["evaluation"]["mode"], "smoke")
            self.assertEqual(
                {path.name: path.read_bytes() for path in active.iterdir()}, before,
            )
            self.assertEqual(client.uploads,
                             [("run-0001", candidate.resolve(), "staged")])
            self.assertEqual(client.reports,
                             [("run-0001", "staged", None)])

    def test_snapshot_tampering_fails_before_training_or_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = snapshot()
            client = FakeTrainingClient(lease_payload(body), body + b"tampered")
            calls = []
            capability = TrainerCapability(
                client, device_id="owner-device", enabled=True,
                active_adapter=active_adapter(root), staging_root=root / "staged",
                train_backend=lambda *args: calls.append(args), now=lambda: 1_000.0,
            )

            self.assertTrue(capability.poll_when_idle())
            self.assertFalse(capability.training)
            self.assertEqual(calls, [])
            self.assertEqual(client.uploads, [])
            self.assertEqual(client.reports,
                             [("run-0001", "failed", "snapshot_digest_mismatch")])
            self.assertFalse((root / "staged" / "preference-0001").exists())

    def test_interruption_is_reported_and_training_state_is_always_released(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = snapshot()
            client = FakeTrainingClient(lease_payload(body), body)

            def interrupt(_lease, _rows, _evaluation, _output_dir):
                raise KeyboardInterrupt

            capability = TrainerCapability(
                client, device_id="owner-device", enabled=True,
                active_adapter=active_adapter(root), staging_root=root / "staged",
                train_backend=interrupt, now=lambda: 1_000.0,
            )
            with self.assertRaises(KeyboardInterrupt):
                capability.poll_when_idle()
            self.assertFalse(capability.training)
            self.assertEqual(client.uploads, [])
            self.assertEqual(client.reports,
                             [("run-0001", "failed", "interrupted")])
            self.assertFalse((root / "staged" / "preference-0001").exists())

    def test_expired_or_wrong_device_leases_are_rejected(self):
        body = snapshot()
        payload = lease_payload(body)
        for patch, error in (
            ({"expiresAt": 999.0}, "expired"),
            ({"deviceId": "somebody-else"}, "device"),
        ):
            candidate = {**payload, **patch}
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, error):
                TrainingLease.from_payload(
                    candidate, expected_device_id="owner-device", now=1_000.0,
                )


class TrainingHttpClientTests(unittest.TestCase):
    def test_candidate_archive_and_payload_are_deterministic_and_bound_to_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / "preference-0001"
            candidate.mkdir()
            manifest = {
                "complete": True, "kind": "preference", "version": "preference-0001",
                "baseModel": "Qwen/Qwen3-1.7B", "parentVersion": "bootstrap-0001",
                "snapshotDigest": "a" * 64,
                "evaluation": {"gates": {name: False for name in (
                    "responseStructure", "weightedPreferenceAccuracy", "promptFidelity",
                    "diversity", "phraseConcentration", "anchorDrift")}},
            }
            (candidate / "manifest.json").write_text(json.dumps(manifest))
            (candidate / "adapter_model.safetensors").write_bytes(b"adapter")
            self.assertEqual(package_candidate(candidate), package_candidate(candidate))

            lease = TrainingLease.from_payload({
                **lease_payload(b"snapshot"), "snapshotDigest": "a" * 64,
            }, expected_device_id="owner-device", now=1_000.0)
            report = TrainingReport("run-0001", "staged", artifact_path=candidate)
            client = TrainingHttpClient()
            reply = mock.MagicMock()
            reply.__enter__.return_value.read.return_value = b'{"accepted":true,"promoted":false}'
            with mock.patch("peerpixel.trainer.urllib.request.urlopen", return_value=reply) as call:
                response = client.upload_candidate(lease, candidate, report)
            self.assertFalse(response["promoted"])
            request = call.call_args.args[0]
            self.assertEqual(request.data, package_candidate(candidate))
            self.assertEqual(request.headers["Content-type"], "application/zip")
            self.assertEqual(request.headers["X-peerpixel-training-lease"], "signed-short-lived-token")
            self.assertEqual(request.headers["X-peerpixel-artifact-digest"],
                             hashlib.sha256(request.data).hexdigest())

    def test_lease_and_report_use_the_coordinator_routes(self):
        client = TrainingHttpClient()
        body = snapshot()
        payload = lease_payload(body)
        lease = TrainingLease.from_payload(
            payload, expected_device_id="owner-device", now=1_000.0)
        with mock.patch("peerpixel.trainer.api._call", side_effect=[payload, {"acknowledged": True}]) as call:
            self.assertEqual(client.lease_training("owner-device"), payload)
            client.report_training(lease, TrainingReport.failed("run-0001", "interrupted"))
        self.assertEqual(call.call_args_list[0].args, ("/api/worker/training/lease",))
        self.assertEqual(call.call_args_list[0].kwargs["payload"], {"deviceId": "owner-device"})
        self.assertEqual(call.call_args_list[1].args, ("/api/worker/training/report/run-0001",))
        self.assertEqual(call.call_args_list[1].kwargs["payload"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
