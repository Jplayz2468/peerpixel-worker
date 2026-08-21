import io
import unittest
from PIL import Image

from peerpixel.upscale import Upscaler


class FourTimes:
    def upscale_4x_overlapped(self, image):
        return image.resize((image.width * 4, image.height * 4))


class UpscaleTests(unittest.TestCase):
    def test_upscale_is_exactly_four_times_and_returns_only_jpeg_bytes(self):
        source = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(source, "JPEG")
        upscaler = Upscaler()
        upscaler.model = FourTimes()
        result = upscaler.upscale(source.getvalue())
        image = Image.open(io.BytesIO(result))
        self.assertEqual(image.size, (12, 8))
        self.assertTrue(result.startswith(b"\xff\xd8"))
