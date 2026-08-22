import io
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from peerpixel.mlx_backend import MLXBackend, progress_steps


class MLXBackendTests(unittest.TestCase):
    def test_carriage_return_progress_becomes_monotonic_diffusion_steps(self):
        text = "\r  2%| 1/50\r 50%| 25/50\r100%| 50/50"
        self.assertEqual(progress_steps(text, 50), [1, 25, 50])

    def test_render_preserves_model_contract_and_returns_jpeg(self):
        backend = MLXBackend(executable="/venv/bin/mlxgen")
        seen = []

        class Process:
            def __init__(_self, command, **kwargs):
                self.assertEqual(command[:2], ["/venv/bin/mlxgen", "generate"])
                self.assertIn("AbstractFramework/flux.2-klein-base-4b-4bit", command)
                self.assertEqual(command[command.index("--steps") + 1], "50")
                self.assertEqual(command[command.index("--width") + 1], "512")
                self.assertEqual(command[command.index("--seed") + 1], "7")
                Path(command[command.index("--output") + 1]).write_bytes(
                    b"\xff\xd8\xffmlx\xff\xd9")
                _self.stderr = io.StringIO("\r  2%| 1/50\r 50%| 25/50\r100%| 50/50")
                _self.returncode = 0
            def wait(_self):
                return _self.returncode

        with mock.patch("peerpixel.mlx_backend.subprocess.Popen", Process):
            jpeg = backend.render(
                prompt="a fox", width=512, height=512, steps=50,
                guidance=4.0, seed=7, on_step=lambda done, total: seen.append((done, total)),
            )
        self.assertTrue(jpeg.startswith(b"\xff\xd8\xff"))
        self.assertEqual(seen, [(1, 50), (25, 50), (50, 50)])

    def test_model_download_uses_explicit_published_q4_package(self):
        backend = MLXBackend(executable="mlxgen")
        with mock.patch("peerpixel.mlx_backend.subprocess.run") as run:
            backend.ensure_model()
        self.assertEqual(run.call_args.args[0], [
            "mlxgen", "download", "--model",
            "AbstractFramework/flux.2-klein-base-4b-4bit",
        ])


if __name__ == "__main__":
    unittest.main()
