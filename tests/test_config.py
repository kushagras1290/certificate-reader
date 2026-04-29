from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from certificate_reader.config import ConfigError, RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_rejects_non_zip_input(self) -> None:
        with self.assertRaisesRegex(ConfigError, "input_zip must be a .zip file"):
            RuntimeConfig.from_values(input_zip="input.pdf", output_csv="out.csv")

    def test_requires_openai_api_key_for_live_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = RuntimeConfig.from_values(input_zip="input.zip", output_csv="out.csv")
            with self.assertRaisesRegex(ConfigError, "OPENAI_API_KEY"):
                config.validate_for_openai()

    def test_reads_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CERTIFICATE_OPENAI_MODEL": "gpt-test",
                "CERTIFICATE_MAX_FILES": "12",
                "CERTIFICATE_IMAGE_DETAIL": "auto",
            },
            clear=True,
        ):
            config = RuntimeConfig.from_values(input_zip="input.zip", output_csv="out.csv")
        self.assertEqual(config.model, "gpt-test")
        self.assertEqual(config.max_files, 12)
        self.assertEqual(config.image_detail, "auto")


if __name__ == "__main__":
    unittest.main()
