import unittest

from peerpixel import z_image


class ZImageContractTests(unittest.TestCase):
    def test_model_artifacts_are_revision_pinned(self):
        self.assertEqual(z_image.BASE_MODEL, "Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(z_image.QUANT_MODEL, "unsloth/Z-Image-Turbo-FP8")
        self.assertEqual(z_image.QUANT_REVISION,
                         "055a19b7ab875e80e6463a8d458228b26ff55915")
        self.assertEqual(z_image.TRANSFORMER_FILE, "Z-Image-Turbo-INT8.pt")
        self.assertEqual(z_image.TEXT_ENCODER_FILE,
                         "Z-Image-Turbo-text_encoder-FP8.pt")

    def test_turbo_contract_uses_native_schedule_without_cfg(self):
        self.assertEqual(z_image.SCHEDULER_STEPS, 9)
        self.assertEqual(z_image.DIT_FORWARDS, 8)
        self.assertEqual(z_image.GUIDANCE, 0.0)

    def test_checkpoint_validation_rejects_wrong_identity_or_scheme(self):
        valid = {"format": z_image.TRANSFORMER_FORMAT, "state_dict": {"weight": object()},
                 "metadata": {"scheme": "int8", "family": "z-image",
                              "base_model_id": z_image.BASE_MODEL}}
        self.assertIs(z_image.validated_state_dict(valid, component="transformer"),
                      valid["state_dict"])
        for key, value in (("scheme", "fp8"), ("family", "flux"),
                           ("base_model_id", "other/model")):
            broken = {**valid, "metadata": {**valid["metadata"], key: value}}
            with self.assertRaisesRegex(RuntimeError, "invalid_z_image_checkpoint"):
                z_image.validated_state_dict(broken, component="transformer")

    def test_text_encoder_checkpoint_requires_fp8_and_exact_component(self):
        valid = {"format": z_image.TEXT_ENCODER_FORMAT,
                 "state_dict": {"weight": object()},
                 "metadata": {"scheme": "fp8", "family": "z-image",
                              "component": "text_encoder",
                              "base_model_id": z_image.BASE_MODEL}}
        self.assertIs(z_image.validated_state_dict(valid, component="text_encoder"),
                      valid["state_dict"])
        invalid = {**valid, "metadata": {**valid["metadata"], "component": "other"}}
        with self.assertRaisesRegex(RuntimeError, "invalid_z_image_checkpoint"):
            z_image.validated_state_dict(invalid, component="text_encoder")


if __name__ == "__main__":
    unittest.main()
