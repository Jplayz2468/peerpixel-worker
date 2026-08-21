"""What the comparison actually measures.

The numbers this produces decide whether somebody keeps their account, so the
properties that matter are pinned down against real images rather than assumed:
identical pictures score zero, a re-encode or a small shift stays near zero,
and an unrelated picture or a solid colour lands a long way off.
"""
import io
import math
import unittest

from peerpixel import compare


def image(draw, size=(256, 256)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, (20, 30, 40))
    draw(ImageDraw.Draw(img), size)
    return img


def jpeg(img, quality=92) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def harbour(shift=0, size=(256, 256)):
    def paint(d, s):
        w, h = s
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=(int(40 + 120 * (1 - t)), int(60 + 120 * (1 - t)), int(110 + 80 * (1 - t))))
        d.rectangle([0, int(h * 0.7), w, h], fill=(18, 24, 30))
        d.ellipse([w * 0.66 + shift, h * 0.15, w * 0.82 + shift, h * 0.31], fill=(250, 240, 210))
        for x in (0.15, 0.35, 0.52):
            d.polygon([(w * x + shift, h * 0.7), (w * (x + 0.05) + shift, h * 0.52),
                       (w * (x + 0.1) + shift, h * 0.7)], fill=(90, 110, 130))
    return image(paint, size)


def portrait():
    def paint(d, s):
        w, h = s
        d.rectangle([0, 0, w, h], fill=(180, 140, 120))
        d.ellipse([w * 0.25, h * 0.2, w * 0.75, h * 0.8], fill=(60, 40, 35))
        d.rectangle([w * 0.1, h * 0.05, w * 0.2, h * 0.95], fill=(240, 230, 200))
    return image(paint)


class CompareTests(unittest.TestCase):
    def test_the_same_bytes_are_identical(self):
        data = jpeg(harbour())
        result = compare.compare(data, data)
        self.assertEqual(result["distance"], 0)
        self.assertEqual(result["rmse"], 0.0)

    def test_a_re_encode_at_a_different_quality_stays_near_zero(self):
        # Two machines never agree on JPEG settings. That must not read as a
        # different picture.
        picture = harbour()
        result = compare.compare(jpeg(picture, 92), jpeg(picture, 60))
        self.assertLessEqual(result["distance"], 4, f"re-encode moved the hash to {result['distance']}")
        self.assertLess(result["rmse"], 12)

    def test_a_different_resolution_of_the_same_picture_stays_near_zero(self):
        result = compare.compare(jpeg(harbour(size=(1024, 1024))), jpeg(harbour(size=(256, 256))))
        self.assertLessEqual(result["distance"], 6)

    def test_a_small_shift_is_still_the_same_composition(self):
        # Roughly what two GPUs disagreeing on the last bits looks like: the
        # same scene, very slightly moved.
        result = compare.compare(jpeg(harbour(shift=0)), jpeg(harbour(shift=3)))
        self.assertLessEqual(result["distance"], 12,
                             "an honest hardware difference must stay inside the pass band")

    def test_an_unrelated_picture_lands_a_long_way_off(self):
        result = compare.compare(jpeg(harbour()), jpeg(portrait()))
        self.assertGreaterEqual(result["distance"], 20,
                                f"unrelated images only differed by {result['distance']}")

    def test_a_solid_colour_is_caught_by_pixel_error_whatever_its_hash_does(self):
        from PIL import Image
        flat = jpeg(Image.new("RGB", (256, 256), (128, 128, 128)))
        result = compare.compare(flat, jpeg(harbour()))
        # A flat image has no gradients, so its hash is degenerate and may land
        # anywhere. The pixel error is what has to catch it.
        self.assertGreater(result["rmse"], 30)

    def test_a_picture_that_will_not_decode_raises_rather_than_scoring_zero(self):
        # The server reads a missing number as "no comparison happened". A zero
        # here would be read as a perfect match and clear a bad render.
        with self.assertRaises(Exception):
            compare.compare(b"not an image at all", jpeg(harbour()))

    def test_the_hash_is_sixty_four_bits_and_symmetric(self):
        first, second = harbour(), portrait()
        self.assertEqual(compare.hamming(compare.dhash(first), compare.dhash(first)), 0)
        self.assertEqual(
            compare.hamming(compare.dhash(first), compare.dhash(second)),
            compare.hamming(compare.dhash(second), compare.dhash(first)),
        )
        self.assertLessEqual(compare.hamming(compare.dhash(first), compare.dhash(second)), 64)

    def test_rmse_is_symmetric_and_bounded(self):
        first, second = harbour(), portrait()
        self.assertAlmostEqual(compare.rmse(first, second), compare.rmse(second, first), places=6)
        self.assertLessEqual(compare.rmse(first, second), 255)


if __name__ == "__main__":
    unittest.main()
