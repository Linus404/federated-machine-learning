import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from src.flower_config import ensure_local_superlink_profile, main


class LocalFlowerProfileTests(unittest.TestCase):
    def test_missing_profile_is_appended_without_changing_existing_profiles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            existing = (
                "[superlink.existing]\n"
                'address = "example.internal:9093"\n'
                "insecure = true\n"
            )
            config_path.write_text(existing, encoding="utf-8")

            ensure_local_superlink_profile(config_path)

            content = config_path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith(existing))
            profile = tomllib.loads(content)["superlink"]["local-docker"]
            self.assertEqual(
                profile,
                {"address": "127.0.0.1:9093", "insecure": True},
            )

    def test_correct_existing_profile_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            original = (
                "[superlink.local-docker]\n"
                'address = "127.0.0.1:9093"\n'
                "insecure = true\n"
            )
            config_path.write_text(original, encoding="utf-8")

            ensure_local_superlink_profile(config_path)

            self.assertEqual(config_path.read_text(encoding="utf-8"), original)

    def test_stale_existing_profile_fails_without_modifying_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            stale = (
                "[superlink.local-docker]\n"
                'address = "remote.example:9093"\n'
                "insecure = false\n"
            )
            config_path.write_text(stale, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                ensure_local_superlink_profile(config_path)

            self.assertEqual(config_path.read_text(encoding="utf-8"), stale)

    def test_invalid_toml_fails_without_modifying_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            invalid = "[superlink.local-docker\n"
            config_path.write_text(invalid, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "valid TOML"):
                ensure_local_superlink_profile(config_path)

            self.assertEqual(config_path.read_text(encoding="utf-8"), invalid)

    def test_invalid_superlink_table_fails_without_modifying_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            invalid = 'superlink = "not-a-table"\n'
            config_path.write_text(invalid, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid superlink table"):
                ensure_local_superlink_profile(config_path)

            self.assertEqual(config_path.read_text(encoding="utf-8"), invalid)

    def test_main_waits_for_four_online_supernodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            with patch("src.flower_config.wait_for_online_supernodes") as wait:
                main(
                    [
                        "--config",
                        str(config_path),
                        "--readiness-timeout",
                        "7.5",
                    ]
                )

            wait.assert_called_once_with(
                address="127.0.0.1:9093",
                expected_online=4,
                timeout_seconds=7.5,
            )


if __name__ == "__main__":
    unittest.main()
