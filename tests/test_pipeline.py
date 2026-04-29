from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from certificate_reader.domain import CertificateDocument, CertificateExtraction, FieldValue
from certificate_reader.pipeline import extract_zip_to_csv
from certificate_reader.zip_reader import ZipLimits


class FakeExtractor:
    def extract(self, document: CertificateDocument) -> CertificateExtraction:
        if document.source_name.endswith("bad.png"):
            raise RuntimeError("vision failure")
        return CertificateExtraction(
            certificate_number="ABC-123",
            laboratory="Example Lab",
            gemstone="Ruby",
            weight_carats="2.10 ct",
            confidence=0.91,
            needs_review=False,
            additional_fields=(FieldValue(name="Refractive Index", value="1.76"),),
            raw_text_excerpt="Example Lab ABC-123 Ruby",
        )


class PipelineTests(unittest.TestCase):
    def test_extract_zip_to_csv_writes_success_and_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_zip = root / "certificates.zip"
            output_csv = root / "out" / "certificates.csv"
            with zipfile.ZipFile(input_zip, "w") as archive:
                archive.writestr("good.pdf", b"%PDF-1.4 fake")
                archive.writestr("bad.png", b"\x89PNG\r\n\x1a\nfake")
                archive.writestr("notes.txt", b"not a cert")

            with self.assertLogs("certificate_reader.pipeline", level="ERROR"):
                summary = extract_zip_to_csv(
                    input_zip=input_zip,
                    output_csv=output_csv,
                    limits=ZipLimits(max_files=10, max_file_bytes=1024, max_total_uncompressed_bytes=4096),
                    extractor=FakeExtractor(),
                )

            self.assertEqual(summary.total_rows, 3)
            self.assertEqual(summary.successful_rows, 1)
            self.assertEqual(summary.error_rows, 2)

            with output_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual([row["source_file"] for row in rows], ["bad.png", "good.pdf", "notes.txt"])
            good = rows[1]
            self.assertEqual(good["status"], "success")
            self.assertEqual(good["certificate_number"], "ABC-123")
            self.assertEqual(good["needs_review"], "false")
            self.assertIn("Refractive Index", good["additional_fields_json"])
            self.assertEqual(rows[0]["status"], "error")
            self.assertIn("vision failure", rows[0]["error_message"])
            self.assertIn("unsupported file type", rows[2]["error_message"])


if __name__ == "__main__":
    unittest.main()
