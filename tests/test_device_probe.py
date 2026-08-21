"""Naming the hardware must never be able to bring anything down.

Every call in here stands in for a driver that is unhappy: a card whose memory
is already spoken for, a CUDA context that will not initialise, a torch build
that raises from somewhere unexpected. The machine is still the machine, and
pairing, benchmarking and the dashboard all have to survive being told so.
"""
import sys
import types
import unittest
import subprocess
from unittest.mock import patch

from peerpixel import render


class FakeCuda:
    def __init__(self, *, available=True, name=None, memory=None):
        self._available = available
        self._name = name
        self._memory = memory

    def is_available(self):
        if isinstance(self._available, Exception):
            raise self._available
        return self._available

    def get_device_name(self, _index):
        if isinstance(self._name, Exception):
            raise self._name
        return self._name

    def mem_get_info(self):
        if isinstance(self._memory, Exception):
            raise self._memory
        return (self._memory // 2, self._memory)


def fake_torch(cuda):
    module = types.ModuleType("torch")
    module.cuda = cuda
    module.backends = types.SimpleNamespace(mps=None)
    module.bfloat16 = "bfloat16"
    module.float32 = "float32"
    return module


class DeviceProbeTests(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("torch")

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved

    def install(self, cuda):
        sys.modules["torch"] = fake_torch(cuda)

    def test_a_healthy_card_is_named_with_its_size(self):
        self.install(FakeCuda(name="NVIDIA GeForce RTX 4080", memory=16_000_000_000))
        answer = types.SimpleNamespace(stdout="14000, 16384\n")
        with patch("peerpixel.render.subprocess.run", return_value=answer):
            device, _, label, total = render.pick_device()
        self.assertEqual(device, "cuda")
        self.assertEqual(label, "NVIDIA GeForce RTX 4080 (17 GB)")
        self.assertEqual(total, 16384 * 1024 ** 2)

    def test_device_sizing_never_enters_the_cuda_runtime(self):
        self.install(FakeCuda(
            name="NVIDIA GeForce RTX 5080",
            memory=AssertionError("must not call torch.cuda.mem_get_info"),
        ))
        answer = types.SimpleNamespace(stdout="14000, 16303\n")

        with patch("peerpixel.render.subprocess.run", return_value=answer):
            device, _, label, total = render.pick_device()

        self.assertEqual(device, "cuda")
        self.assertEqual(label, "NVIDIA GeForce RTX 5080 (17 GB)")
        self.assertGreater(total, 16e9)

    def test_a_card_that_is_out_of_memory_is_still_the_card_this_machine_has(self):
        # The exact failure reported from Linux: torch.AcceleratorError out of
        # this method, taking the whole pairing request with it.
        self.install(FakeCuda(
            name="NVIDIA GeForce RTX 4080",
            memory=RuntimeError("CUDA error: out of memory"),
        ))
        device, _, label, total = render.pick_device()
        self.assertEqual(device, "cuda")
        self.assertEqual(label, "NVIDIA GeForce RTX 4080 (size unknown)")
        self.assertEqual(total, 0)

    def test_a_context_that_will_not_initialise_at_all_still_answers(self):
        self.install(FakeCuda(
            name=RuntimeError("CUDA error: out of memory"),
            memory=RuntimeError("CUDA error: out of memory"),
        ))
        device, _, label, _ = render.pick_device()
        self.assertEqual(device, "cuda")
        self.assertEqual(label, "NVIDIA GPU (size unknown)")

    def test_an_availability_check_that_raises_falls_back_to_the_cpu(self):
        self.install(FakeCuda(available=RuntimeError("no driver")))
        device, _, label, total = render.pick_device()
        self.assertEqual((device, label, total), ("cpu", "CPU", 0))

    def test_describing_the_accelerator_never_raises(self):
        self.install(FakeCuda(
            name=RuntimeError("boom"), memory=RuntimeError("boom")))
        self.assertEqual(render.describe_accelerator(), "NVIDIA GPU (size unknown)")

        sys.modules["torch"] = None   # an import that yields nothing usable
        self.assertEqual(render.describe_accelerator(), "unknown")


class OffloadPolicyTests(unittest.TestCase):
    """A card whose size could not be read is treated as a small one.

    That is the safe direction: putting the whole 4B model on a busy 16 GB card
    is precisely how the crash this guards against happens.
    """

    @staticmethod
    def offloads(device, total):
        return device == "cuda" and (not total or total < 24e9)

    def test_a_small_or_unreadable_card_offloads_and_a_large_one_does_not(self):
        self.assertTrue(self.offloads("cuda", 16e9))
        self.assertTrue(self.offloads("cuda", 0), "unknown counts as small")
        self.assertFalse(self.offloads("cuda", 48e9))
        self.assertFalse(self.offloads("mps", 0))

    def test_cuda_memory_strategy_uses_fast_group_offload_on_consumer_cards(self):
        from peerpixel.render import cuda_group_options, cuda_memory_mode

        self.assertEqual(cuda_memory_mode(total=16e9, free=14e9), "group")
        self.assertEqual(cuda_memory_mode(total=12e9, free=10e9), "group")
        self.assertEqual(cuda_memory_mode(total=48e9, free=40e9), "resident")
        self.assertEqual(cuda_group_options(total=16e9, free=14e9), {
            "non_blocking": False, "use_stream": False, "record_stream": False,
            "exclude_modules": ["vae"],
        })
        self.assertEqual(cuda_group_options(total=48e9, free=19e9)["use_stream"], False)

    def test_group_offload_keeps_the_decoder_on_cuda(self):
        options = render.cuda_group_options(total=16e9, free=14e9)
        self.assertIn("vae", options["exclude_modules"],
                      "CPU VAE weights cannot decode CUDA latents")


class VramDiagnosisTests(unittest.TestCase):
    def test_low_vram_names_the_apps_to_close_with_pid_and_usage(self):
        xml = """<nvidia_smi_log><gpu><processes>
          <process_info><pid>5172</pid><type>G</type>
            <process_name>/usr/lib/chromium/chromium</process_name>
            <used_memory>15273 MiB</used_memory></process_info>
          <process_info><pid>1172</pid><type>G</type>
            <process_name>/usr/bin/kwin_wayland</process_name>
            <used_memory>14 MiB</used_memory></process_info>
        </processes></gpu></nvidia_smi_log>"""
        answer = subprocess.CompletedProcess([], 0, xml, "")

        with patch("peerpixel.render.subprocess.run", return_value=answer):
            processes = render.nvidia_processes()
        with self.assertRaises(RuntimeError) as caught:
            render.require_cuda_headroom(194 * 1024 ** 2, 16303 * 1024 ** 2,
                                         processes=processes)

        message = str(caught.exception)
        self.assertIn("chromium (PID 5172): 14.9 GB", message)
        self.assertIn("kwin_wayland (PID 1172): 14 MB", message)
        self.assertIn("Close or restart", message)
        self.assertIn("run PeerPixel again", message)

    def test_normal_free_vram_needs_no_process_scan_or_warning(self):
        render.require_cuda_headroom(12 * 1024 ** 3, 16 * 1024 ** 3,
                                     processes=[("anything", 1, 15 * 1024 ** 3)])


if __name__ == "__main__":
    unittest.main()
