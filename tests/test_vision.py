from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from certificate_reader.domain import CertificateDocument
from certificate_reader.vision import OpenAIVisionExtractor


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "certificate_number": "RPT-1",
                    "laboratory": "Lab",
                    "report_type": "Gemstone Report",
                    "issue_date": None,
                    "gemstone": "Emerald",
                    "species": None,
                    "variety": None,
                    "weight_carats": "1.20 ct",
                    "measurements": None,
                    "shape_cut": None,
                    "color": "Green",
                    "transparency": None,
                    "origin": None,
                    "treatment": "None",
                    "comments": None,
                    "confidence": 0.88,
                    "needs_review": False,
                    "additional_fields": [],
                    "raw_text_excerpt": "Lab RPT-1 Emerald",
                }
            )
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class VisionTests(unittest.TestCase):
    def test_extract_builds_response_api_image_request(self) -> None:
        client = FakeClient()
        extractor = OpenAIVisionExtractor(
            model="gpt-test",
            timeout_seconds=10,
            max_retries=0,
            image_detail="high",
            client=client,
        )
        document = CertificateDocument(
            source_name="certs/cert.jpg",
            suffix=".jpg",
            media_type="image/jpeg",
            content=b"image-bytes",
        )

        extraction = extractor.extract(document)

        self.assertEqual(extraction.certificate_number, "RPT-1")
        self.assertEqual(extraction.gemstone, "Emerald")
        kwargs = client.responses.kwargs
        self.assertIsNotNone(kwargs)
        assert kwargs is not None
        self.assertEqual(kwargs["model"], "gpt-test")
        self.assertIn("text", kwargs)
        input_messages = kwargs["input"]
        self.assertIsInstance(input_messages, list)
        user_content = input_messages[1]["content"]
        image_part = user_content[1]
        self.assertEqual(image_part["type"], "input_image")
        self.assertTrue(image_part["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(image_part["detail"], "high")

    def test_extract_builds_response_api_pdf_request(self) -> None:
        client = FakeClient()
        extractor = OpenAIVisionExtractor(
            model="gpt-test",
            timeout_seconds=10,
            max_retries=0,
            image_detail="auto",
            client=client,
        )
        document = CertificateDocument(
            source_name="nested/cert.pdf",
            suffix=".pdf",
            media_type="application/pdf",
            content=b"%PDF-1.4",
        )

        extractor.extract(document)

        kwargs = client.responses.kwargs
        self.assertIsNotNone(kwargs)
        assert kwargs is not None
        input_messages = kwargs["input"]
        user_content = input_messages[1]["content"]
        file_part = user_content[1]
        self.assertEqual(file_part["type"], "input_file")
        self.assertEqual(file_part["filename"], "cert.pdf")
        self.assertTrue(file_part["file_data"].startswith("data:application/pdf;base64,"))


if __name__ == "__main__":
    unittest.main()
