import io
import unittest

from PIL import Image

from peerpixel import render


def png(mode="RGBA", color=(0, 0, 0, 0), size=(8, 6)):
    output = io.BytesIO()
    Image.new(mode, size, color).save(output, "PNG")
    return output.getvalue()


class EditContractTests(unittest.TestCase):
    def test_variation_has_a_restrained_strength_and_needs_a_source(self):
        self.assertEqual(render.edit_spec({
            "editMode": "vary", "editStrength": .25, "sourceImageId": "source",
        }), {"mode": "vary", "strength": .25})
        with self.assertRaisesRegex(ValueError, "edit_source_required"):
            render.edit_spec({"editMode": "vary", "editStrength": .25})
        with self.assertRaisesRegex(ValueError, "invalid_edit_strength"):
            render.edit_spec({
                "editMode": "vary", "editStrength": .8, "sourceImageId": "source",
            })

    def test_inpaint_requires_a_mask_while_vary_builds_a_full_white_mask(self):
        source = png(color=(10, 20, 30, 255))
        image, mask = render.prepare_edit_images(
            source, None, mode="vary", width=16, height=12,
        )
        self.assertEqual(image.size, (16, 12))
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getextrema(), (255, 255))
        with self.assertRaisesRegex(ValueError, "edit_mask_required"):
            render.prepare_edit_images(source, None, mode="inpaint", width=8, height=6)

    def test_inpaint_uses_the_submitted_alpha_as_the_binary_edit_region(self):
        source = png(color=(10, 20, 30, 255))
        mask_source = Image.new("RGBA", (8, 6), (255, 255, 255, 0))
        mask_source.putpixel((3, 2), (255, 255, 255, 255))
        output = io.BytesIO(); mask_source.save(output, "PNG")
        _, mask = render.prepare_edit_images(
            source, output.getvalue(), mode="inpaint", width=8, height=6,
        )
        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertEqual(mask.getpixel((3, 2)), 255)

    def test_only_cuda_advertises_editing(self):
        renderer = render.Renderer.__new__(render.Renderer)
        renderer._device = "cuda"
        self.assertTrue(renderer.supports_editing)
        renderer._device = "mps"
        self.assertFalse(renderer.supports_editing)


if __name__ == "__main__":
    unittest.main()
