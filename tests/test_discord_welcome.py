import json
import tempfile
import unittest
from pathlib import Path

from peerpixel.discord_welcome import WelcomeState, welcome_message


class DiscordWelcomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp.name) / "welcomed-members.json"
        self.sent = []
        self.state = WelcomeState(
            guild_id="guild",
            channel_id="imagine",
            state_file=self.state_file,
            send_dm=lambda user_id, message: self.sent.append((user_id, message)),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_first_guild_snapshot_bootstraps_without_messaging_existing_members(self):
        self.state.handle_dispatch("GUILD_CREATE", {
            "id": "guild",
            "members": [{"user": {"id": "existing", "bot": False}}],
        })

        self.assertEqual(self.sent, [])
        self.assertEqual(json.loads(self.state_file.read_text()), ["existing"])

    def test_new_human_member_receives_one_server_channel_instruction(self):
        self.state.handle_dispatch("GUILD_CREATE", {"id": "guild", "members": []})
        member = {"guild_id": "guild", "user": {"id": "new-user", "bot": False}}

        self.state.handle_dispatch("GUILD_MEMBER_ADD", member)
        self.state.handle_dispatch("GUILD_MEMBER_ADD", member)

        self.assertEqual(len(self.sent), 1)
        user_id, message = self.sent[0]
        self.assertEqual(user_id, "new-user")
        self.assertIn("https://discord.com/channels/guild/imagine", message)
        self.assertIn("type `/imagine`", message)
        self.assertIn("inside the PeerPixel server", message)
        self.assertIn("Don't send `/imagine` in this DM", message)

    def test_bots_and_other_guilds_are_ignored(self):
        self.state.handle_dispatch("GUILD_CREATE", {"id": "guild", "members": []})
        self.state.handle_dispatch("GUILD_MEMBER_ADD", {
            "guild_id": "guild", "user": {"id": "bot", "bot": True},
        })
        self.state.handle_dispatch("GUILD_MEMBER_ADD", {
            "guild_id": "somewhere-else", "user": {"id": "human", "bot": False},
        })

        self.assertEqual(self.sent, [])

    def test_a_failed_dm_is_recorded_and_does_not_crash_or_repeat(self):
        attempts = []

        def closed_dms(user_id, _message):
            attempts.append(user_id)
            raise RuntimeError("discord_http_403")

        state = WelcomeState("guild", "imagine", self.state_file, closed_dms)
        state.handle_dispatch("GUILD_CREATE", {"id": "guild", "members": []})
        member = {"guild_id": "guild", "user": {"id": "closed", "bot": False}}
        state.handle_dispatch("GUILD_MEMBER_ADD", member)
        state.handle_dispatch("GUILD_MEMBER_ADD", member)

        self.assertEqual(attempts, ["closed"])


class WelcomeCopyTests(unittest.TestCase):
    def test_copy_never_tells_people_to_run_the_command_in_the_dm(self):
        message = welcome_message("guild", "imagine")
        self.assertIn("open **#imagine in the PeerPixel server**", message)
        self.assertIn("Don't send `/imagine` in this DM", message)


if __name__ == "__main__":
    unittest.main()
