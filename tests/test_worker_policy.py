import unittest

from peerpixel.worker import should_unload_model


class WorkerPolicyTests(unittest.TestCase):
    def test_model_stays_loaded_until_two_idle_hours(self):
        self.assertFalse(should_unload_model(100, 100 + 7199, loaded=True))
        self.assertTrue(should_unload_model(100, 100 + 7200, loaded=True))
        self.assertFalse(should_unload_model(100, 100 + 9000, loaded=False))


if __name__ == "__main__":
    unittest.main()
