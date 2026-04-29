from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from certificate_reader.zip_reader import UnsafeArchiveError, ZipLimits, read_certificate_zip


class ZipReaderTests(unittest.TestCase):
    def test_reads_supported_files_and_reports_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_zip = Path(temp_dir) / "certs.zip"
            with zipfile.ZipFile(input_zip, "w") as archive:
                archive.writestr("nested/cert1.jpg", b"jpeg bytes")
                archive.writestr("nested/readme.txt", b"ignore")

            result = read_certificate_zip(
                input_zip,
                ZipLimits(max_files=10, max_file_bytes=1024, max_total_uncompressed_bytes=4096),
            )

            self.assertEqual(len(result.documents), 1)
            self.assertEqual(result.documents[0].source_name, "nested/cert1.jpg")
            self.assertEqual(result.documents[0].media_type, "image/jpeg")
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(result.errors[0].source_name, "nested/readme.txt")

    def test_rejects_path_traversal_archive_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_zip = Path(temp_dir) / "certs.zip"
            with zipfile.ZipFile(input_zip, "w") as archive:
                archive.writestr("../escape.pdf", b"%PDF")

            with self.assertRaises(UnsafeArchiveError):
                read_certificate_zip(
                    input_zip,
                    ZipLimits(max_files=10, max_file_bytes=1024, max_total_uncompressed_bytes=4096),
                )

    def test_rejects_uncompressed_size_above_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_zip = Path(temp_dir) / "certs.zip"
            with zipfile.ZipFile(input_zip, "w") as archive:
                archive.writestr("big.pdf", b"x" * 2048)

            with self.assertRaises(UnsafeArchiveError):
                read_certificate_zip(
                    input_zip,
                    ZipLimits(max_files=10, max_file_bytes=4096, max_total_uncompressed_bytes=1024),
                )


if __name__ == "__main__":
    unittest.main()
