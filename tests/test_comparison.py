import hashlib
import io
import json
import unittest
import urllib.error
import zipfile
from unittest import mock

from PIL import Image

from peerpixel import comparison


def adapter(version):
    out = io.BytesIO()
    manifest = {"version": version, "kind": "bootstrap", "evaluation": {"passed": True}}
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("adapter_model.safetensors", b"adapter")
    return out.getvalue()


class ComparisonTests(unittest.TestCase):
    @mock.patch("time.sleep")
    @mock.patch("peerpixel.comparison.api._call")
    def test_progress_retries_a_transient_tls_timeout(self, call, sleep):
        call.side_effect = [
            urllib.error.URLError(TimeoutError("TLS handshake timed out")),
            {"ok": True},
        ]

        result = comparison.ComparisonClient("device").progress(
            {"runId": "run", "leaseToken": "lease"}, "enhancing", 26, 40)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(call.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_fixed_benchmark_renders_forty_images_and_uploads_twenty_pairs(self):
        prompts = [{"id": f"prompt-{i:02d}", "rawPrompt": f"prompt {i}",
                    "llmSeed": 100 + i, "diffusionSeed": 200 + i} for i in range(1, 21)]
        benchmark = json.dumps({"version": "v1", "render": {"width": 512, "height": 512, "steps": 9},
                                "prompts": prompts}, separators=(",", ":")).encode()
        old, current = adapter("old"), adapter("current")
        lease = {"runId": "run", "leaseToken": "token", "benchmarkVersion": "v1",
                 "benchmarkDigest": hashlib.sha256(benchmark).hexdigest(),
                 "comparisonAdapter": {"version": "old", "artifactDigest": hashlib.sha256(old).hexdigest()},
                 "currentAdapter": {"version": "current", "artifactDigest": hashlib.sha256(current).hexdigest()}}

        class Client:
            def benchmark(self, _lease): return benchmark
            def adapter(self, _lease, side): return current if side == "current" else old
            def progress(self, *_args): pass
            def submit(self, _lease, artifact, manifest): self.submitted = artifact, manifest
            def fail(self, *_args): self.failed = True
        class Enhancer:
            def __init__(self, adapter_path): self.adapter_path = adapter_path
            def warm(self): pass
            def unload(self): pass
            def _generate_text(self, messages, max_new_tokens, seed): return f"enhanced {seed}"
        image = io.BytesIO(); Image.new("RGB", (512, 512), "blue").save(image, "JPEG")
        renderer = mock.Mock(); renderer.render.return_value = image.getvalue()
        client = Client()

        with mock.patch.object(comparison, "PromptEnhancer", Enhancer):
            self.assertTrue(comparison.run_comparison(client, lease, renderer))

        self.assertEqual(renderer.render.call_count, 40)
        self.assertTrue(all(call.args[0]["steps"] == 9 for call in renderer.render.call_args_list))
        artifact, manifest = client.submitted
        self.assertEqual(len(manifest["outputs"]), 20)
        with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
            self.assertEqual(len(archive.namelist()), 21)

    def test_renderer_is_unloaded_before_enhancement_and_after_comparison_failure(self):
        prompts = [{"id": "prompt-01", "rawPrompt": "fox", "llmSeed": 101,
                    "diffusionSeed": 201}]
        benchmark = json.dumps({"version": "v1", "render": {"width": 512,
            "height": 512, "steps": 9}, "prompts": prompts}, separators=(",", ":")).encode()
        old, current = adapter("old"), adapter("current")
        lease = {"runId": "run", "leaseToken": "token", "benchmarkVersion": "v1",
                 "benchmarkDigest": hashlib.sha256(benchmark).hexdigest(),
                 "comparisonAdapter": {"version": "old", "artifactDigest": hashlib.sha256(old).hexdigest()},
                 "currentAdapter": {"version": "current", "artifactDigest": hashlib.sha256(current).hexdigest()}}

        class Client:
            def benchmark(self, _lease): return benchmark
            def adapter(self, _lease, side): return current if side == "current" else old
            def progress(self, *_args): pass
            def fail(self, _lease, reason): self.reason = reason
        class Enhancer:
            def __init__(self, adapter_path): self.adapter_path = adapter_path
            def warm(self): pass
            def unload(self): pass
            def _generate_text(self, messages, max_new_tokens, seed): return "enhanced"
        renderer = mock.Mock()
        renderer.render.side_effect = RuntimeError("render failed")
        client = Client()

        with mock.patch.object(comparison, "PromptEnhancer", Enhancer):
            self.assertFalse(comparison.run_comparison(client, lease, renderer))

        self.assertEqual(renderer.unload.call_count, 2)
        self.assertEqual(client.reason, "render failed")


if __name__ == "__main__":
    unittest.main()
