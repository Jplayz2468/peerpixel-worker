import json
import struct
import unittest

from peerpixel import relay


class RelayFrameTests(unittest.TestCase):
    def test_a_frame_carries_its_header_and_bytes_back_unchanged(self):
        payload = bytes([0xFF, 0xD8, 0xFF, 1, 2, 3, 0xFF, 0xD9])
        header, body = relay.decode(
            relay.encode({"type": "probe_result", "jobId": "abc"}, payload)
        )
        self.assertEqual(header, {"type": "probe_result", "jobId": "abc"})
        self.assertEqual(body, payload)

    def test_an_empty_payload_round_trips(self):
        header, body = relay.decode(relay.encode({"type": "ping"}))
        self.assertEqual(header, {"type": "ping"})
        self.assertEqual(body, b"")

    def test_the_wire_format_is_a_big_endian_length_then_json(self):
        # The server side reads this with DataView.getUint32(0, false). If this
        # assertion ever changes, public/relay-frame.mjs changes with it.
        frame = relay.encode({"type": "x"}, b"payload")
        (length,) = struct.unpack(">I", frame[:4])
        self.assertEqual(json.loads(frame[4:4 + length].decode()), {"type": "x"})
        self.assertEqual(frame[4 + length:], b"payload")

    def test_a_malformed_frame_is_dropped_rather_than_raised(self):
        self.assertIsNone(relay.decode(b"\x00\x01"))
        self.assertIsNone(relay.decode(b"\x00\x00\x00\x00"))
        self.assertIsNone(relay.decode(b"\x00\x00\x00\x5a\x01\x02"))
        self.assertIsNone(relay.decode(struct.pack(">I", 3) + b"{{{"))
        self.assertIsNone(relay.decode(struct.pack(">I", 2) + b"[]"))
        self.assertIsNone(relay.decode("a text message"))
        self.assertIsNone(relay.decode(None))

    def test_a_payload_cannot_be_smuggled_through_the_header(self):
        with self.assertRaises(ValueError):
            relay.encode({"type": "x", "pad": "y" * relay.MAX_HEADER_BYTES})
        self.assertIsNone(relay.decode(struct.pack(">I", relay.MAX_HEADER_BYTES + 1) + b"x"))

    def test_generation_attestations_fit_in_the_authenticated_header(self):
        header = {
            "type": "probe_result", "jobId": "probe-1",
            "enhancedPrompt": "richly described scene " * 100,
            "moderation": {"label": "normal", "nsfwScore": 0.00001},
            "manifestVersion": "2026-08-23.1", "recipeId": "photoreal-v1",
            "attestations": [{
                "operation": name, "inputDigest": "a" * 64,
                "outputDigest": "b" * 64, "runtimeVersion": "peerpixel-worker/0.8.5",
            } for name in ("prompt", "render", "moderation")],
        }
        encoded = relay.encode(header, b"jpeg")
        self.assertEqual(relay.decode(encoded), (header, b"jpeg"))


if __name__ == "__main__":
    unittest.main()
