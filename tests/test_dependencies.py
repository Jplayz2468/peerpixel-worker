import pathlib
import tomllib
import unittest


class DependencyTests(unittest.TestCase):
    def test_prompt_only_worker_does_not_ship_lora_runtime(self):
        project = tomllib.loads(
            (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text())
        runtime = project["project"]["dependencies"]
        self.assertFalse(any(dependency.startswith(("peft", "trl", "datasets"))
                             for dependency in runtime))
        self.assertTrue(any(dependency.startswith("peft") for dependency in
                            project["project"]["optional-dependencies"]["train"]))


if __name__ == "__main__":
    unittest.main()
