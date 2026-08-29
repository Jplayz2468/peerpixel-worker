import hashlib
import io
import unittest

from PIL import Image

from peerpixel.upscale import UpscaleJob, Upscaler


def jpeg(size=(8, 6)):
    output = io.BytesIO()
    Image.new("RGB", size, "navy").save(output, "JPEG")
    return output.getvalue()


class Backend:
    def __init__(self):
        self.unloaded = False

    def upscale(self, image, on_tile=None):
        if on_tile:
            on_tile(0, 4)
            for done in range(1, 5):
                on_tile(done, 4)
        return image.resize((image.width * 4, image.height * 4))

    def unload(self):
        self.unloaded = True


class UpscaleTests(unittest.TestCase):
    def test_job_accepts_only_aurasr_v2_at_exactly_four_times(self):
        source = jpeg()
        payload = {"operation": "upscale", "model": "aurasr-v2", "sourceWidth": 8,
                   "sourceHeight": 6, "width": 32, "height": 24,
                   "sourceDigest": hashlib.sha256(source).hexdigest()}
        self.assertEqual(UpscaleJob.from_payload(payload).width, 32)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            UpscaleJob.from_payload({**payload, "width": 64})

    def test_upscaler_verifies_source_reports_tiles_and_unloads(self):
        source = jpeg()
        job = UpscaleJob.from_payload({"operation": "upscale", "model": "aurasr-v2",
            "sourceWidth": 8, "sourceHeight": 6, "width": 32, "height": 24,
            "sourceDigest": hashlib.sha256(source).hexdigest()})
        backend = Backend()
        tiles = []
        result = Upscaler(backend_factory=lambda: backend).run(job, source,
            on_tile=lambda done, total: tiles.append((done, total)))
        self.assertEqual(Image.open(io.BytesIO(result)).size, (32, 24))
        self.assertEqual(tiles[-1], (4, 4))
        self.assertTrue(backend.unloaded)


if __name__ == "__main__":
    unittest.main()
