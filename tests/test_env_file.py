from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from certificate_reader.env_file import EnvFileError, load_env_file


class EnvFileTests(unittest.TestCase):
    def test_loads_env_file_without_overriding_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "# ignored",
                        "OPENAI_API_KEY=from-file",
                        'CERTIFICATE_OPENAI_MODEL="gpt-test"',
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "from-shell"}, clear=True):
                loaded = load_env_file(env_file)
                self.assertEqual(loaded, 1)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-shell")
                self.assertEqual(os.environ["CERTIFICATE_OPENAI_MODEL"], "gpt-test")

    def test_rejects_malformed_env_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("OPENAI_API_KEY\n", encoding="utf-8")

            with self.assertRaisesRegex(EnvFileError, "missing '='"):
                load_env_file(env_file)


if __name__ == "__main__":
    unittest.main()
