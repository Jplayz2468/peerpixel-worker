import pathlib
import unittest


class DependencyTests(unittest.TestCase):
    def test_prompt_only_worker_does_not_ship_lora_runtime(self):
        project = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
        self.assertNotIn('"peft>=', project)


if __name__ == "__main__":
    unittest.main()
