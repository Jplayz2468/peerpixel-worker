import pathlib
import unittest


class DependencyTests(unittest.TestCase):
    def test_lora_runtime_declares_peft_backend(self):
        project = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
        self.assertIn('"peft>=', project)


if __name__ == "__main__":
    unittest.main()
