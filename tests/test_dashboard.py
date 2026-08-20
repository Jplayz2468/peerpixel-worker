import unittest

from peerpixel.dashboard import DashboardApp, PAGE, request_allowed


class DashboardTests(unittest.TestCase):
    def test_local_requests_need_the_launch_token_and_local_origin(self):
        self.assertTrue(request_allowed("127.0.0.1:8765", "http://127.0.0.1:8765", "secret", "secret"))
        self.assertFalse(request_allowed("attacker.example", "http://attacker.example", "secret", "secret"))
        self.assertFalse(request_allowed("127.0.0.1:8765", "http://attacker.example", "secret", "secret"))
        self.assertFalse(request_allowed("127.0.0.1:8765", "http://127.0.0.1:8765", "wrong", "secret"))

    def test_pair_endpoint_normalizes_code_and_returns_new_identity(self):
        calls = []

        app = DashboardApp(
            pair=lambda code: calls.append(code) or {"deviceId": "dev-1"},
            state=lambda: {"paired": True, "deviceId": "dev-1"},
            start=lambda command: {"running": command},
            stop=lambda: {"running": None},
        )

        status, payload = app.handle("POST", "/api/pair", {"code": " ab12cd "})

        self.assertEqual(status, 200)
        self.assertEqual(calls, ["AB12CD"])
        self.assertEqual(payload, {"paired": True, "deviceId": "dev-1"})

    def test_pair_endpoint_rejects_an_empty_code(self):
        app = DashboardApp(
            pair=lambda code: self.fail("empty code reached pair service"),
            state=lambda: {},
            start=lambda command: {},
            stop=lambda: {},
        )

        status, payload = app.handle("POST", "/api/pair", {"code": "  "})

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "Pairing code required")

    def test_download_is_an_allowed_dashboard_command(self):
        calls = []
        app = DashboardApp(state=lambda: {}, start=lambda command: calls.append(command) or {}, stop=lambda: {})
        status, _ = app.handle("POST", "/api/start", {"command": "download"})
        self.assertEqual(status, 200)
        self.assertEqual(calls, ["download"])

    def test_free_toggle_updates_this_device(self):
        calls = []
        app = DashboardApp(state=lambda: {"allowFree": True}, set_free=lambda allow: calls.append(allow))
        status, payload = app.handle("POST", "/api/free", {"allowFree": True})
        self.assertEqual(status, 200)
        self.assertEqual(calls, [True])
        self.assertTrue(payload["allowFree"])

    def test_progress_is_above_both_views_and_rendering_opens_running(self):
        self.assertLess(PAGE.index('id="progressPanel"'), PAGE.index('id="setupView"'))
        self.assertIn("if(s.phase==='rendering')tab('run')", PAGE)


if __name__ == "__main__":
    unittest.main()
