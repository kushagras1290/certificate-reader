from __future__ import annotations

import base64
import json
import logging
import random
import time
from typing import Any, Protocol

from certificate_reader.domain import CertificateDocument, CertificateExtraction

LOGGER = logging.getLogger(__name__)

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "certificate_number": {
            "type": ["string", "null"],
            "description": "Certificate, report, memo, or identification number exactly as printed.",
        },
        "laboratory": {
            "type": ["string", "null"],
            "description": "Issuing gem laboratory or authority name.",
        },
        "report_type": {
            "type": ["string", "null"],
            "description": "Certificate/report type, for example Gemstone Report or Identification Report.",
        },
        "issue_date": {
            "type": ["string", "null"],
            "description": "Issue date exactly as printed. Do not reformat ambiguous dates.",
        },
        "gemstone": {
            "type": ["string", "null"],
            "description": "Gemstone or product name shown on the certificate.",
        },
        "species": {
            "type": ["string", "null"],
            "description": "Gem species if explicitly printed.",
        },
        "variety": {
            "type": ["string", "null"],
            "description": "Gem variety if explicitly printed.",
        },
        "weight_carats": {
            "type": ["string", "null"],
            "description": "Weight with unit, usually carats, exactly as visible.",
        },
        "measurements": {
            "type": ["string", "null"],
            "description": "Dimensions or measurements exactly as visible.",
        },
        "shape_cut": {
            "type": ["string", "null"],
            "description": "Shape, cut, style, or cutting style exactly as visible.",
        },
        "color": {
            "type": ["string", "null"],
            "description": "Color or hue description exactly as visible.",
        },
        "transparency": {
            "type": ["string", "null"],
            "description": "Transparency/clarity value if explicitly printed.",
        },
        "origin": {
            "type": ["string", "null"],
            "description": "Origin/geographic origin if explicitly printed.",
        },
        "treatment": {
            "type": ["string", "null"],
            "description": "Treatment/enhancement status exactly as visible.",
        },
        "comments": {
            "type": ["string", "null"],
            "description": "Important comments, conclusions, remarks, or footnotes.",
        },
        "confidence": {
            "type": ["number", "null"],
            "description": "0.0 to 1.0 confidence in the extraction quality.",
        },
        "needs_review": {
            "type": "boolean",
            "description": "True when any important field is unreadable, missing, ambiguous, or low confidence.",
        },
        "additional_fields": {
            "type": "array",
            "description": "Other printed certificate fields not covered above.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": ["string", "null"]},
                },
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        },
        "raw_text_excerpt": {
            "type": ["string", "null"],
            "description": "Short OCR-like excerpt of the most relevant visible text.",
        },
    },
    "required": [
        "certificate_number",
        "laboratory",
        "report_type",
        "issue_date",
        "gemstone",
        "species",
        "variety",
        "weight_carats",
        "measurements",
        "shape_cut",
        "color",
        "transparency",
        "origin",
        "treatment",
        "comments",
        "confidence",
        "needs_review",
        "additional_fields",
        "raw_text_excerpt",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You extract structured data from gem, jewellery, and laboratory certificates. "
    "Use only visible information from the document. Never infer a value from context. "
    "If text is missing, cropped, blurred, or ambiguous, return null for that field and set needs_review=true. "
    "Preserve original spelling, capitalization, symbols, dates, report numbers, units, and lab wording."
)


class VisionExtractionError(RuntimeError):
    """Raised when the vision provider cannot return usable structured data."""


class VisionExtractor(Protocol):
    def extract(self, document: CertificateDocument) -> CertificateExtraction:
        """Extract certificate data from one document."""


class OpenAIVisionExtractor:
    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        image_detail: str,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.image_detail = image_detail
        self._client = client

    def extract(self, document: CertificateDocument) -> CertificateExtraction:
        client = self._get_client()
        request_payload = self._build_input(document)
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                started_at = time.monotonic()
                response = client.responses.create(
                    model=self.model,
                    input=request_payload,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "certificate_extraction",
                            "strict": True,
                            "schema": EXTRACTION_SCHEMA,
                        }
                    },
                    timeout=self.timeout_seconds,
                )
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                LOGGER.info(
                    "certificate_extraction_completed",
                    extra={"source_file": document.source_name, "elapsed_ms": elapsed_ms},
                )
                return CertificateExtraction.from_model_dict(_parse_response_json(response))
            except _transient_openai_errors() as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                sleep_seconds = _retry_delay_seconds(attempt)
                LOGGER.warning(
                    "certificate_extraction_retry",
                    extra={
                        "source_file": document.source_name,
                        "attempt": attempt,
                        "sleep_seconds": round(sleep_seconds, 3),
                        "error_type": type(exc).__name__,
                    },
                )
                time.sleep(sleep_seconds)
            except _status_openai_errors() as exc:
                if not _is_retryable_status_error(exc):
                    raise VisionExtractionError(f"OpenAI rejected {document.source_name}: {exc}") from exc
                last_error = exc
                if attempt >= attempts:
                    break
                sleep_seconds = _retry_delay_seconds(attempt)
                LOGGER.warning(
                    "certificate_extraction_retry",
                    extra={
                        "source_file": document.source_name,
                        "attempt": attempt,
                        "sleep_seconds": round(sleep_seconds, 3),
                        "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    },
                )
                time.sleep(sleep_seconds)
            except _permanent_openai_errors() as exc:
                raise VisionExtractionError(f"OpenAI rejected {document.source_name}: {exc}") from exc
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise VisionExtractionError(f"Model returned invalid JSON for {document.source_name}") from exc

        raise VisionExtractionError(f"OpenAI extraction failed for {document.source_name}: {last_error}") from last_error

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VisionExtractionError("The openai package is not installed. Run: py -m pip install -r requirements.txt") from exc
        self._client = OpenAI(timeout=self.timeout_seconds, max_retries=0)
        return self._client

    def _build_input(self, document: CertificateDocument) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Extract certificate data from this file: {document.source_name}. "
                    "Return one JSON object that matches the schema."
                ),
            }
        ]
        encoded = base64.b64encode(document.content).decode("ascii")
        if document.media_type == "application/pdf":
            content.append(
                {
                    "type": "input_file",
                    "filename": _safe_prompt_filename(document.source_name),
                    "file_data": f"data:application/pdf;base64,{encoded}",
                }
            )
        else:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{document.media_type};base64,{encoded}",
                    "detail": self.image_detail,
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]


def _parse_response_json(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("response did not include output_text")
    parsed = json.loads(output_text)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON root is not an object")
    return parsed


def _safe_prompt_filename(source_name: str) -> str:
    filename = source_name.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return filename or "certificate.pdf"


def _retry_delay_seconds(attempt: int) -> float:
    base = min(8.0, 0.75 * (2 ** (attempt - 1)))
    return base + random.uniform(0.0, 0.25)


def _transient_openai_errors() -> tuple[type[BaseException], ...]:
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:
        return (TimeoutError,)
    return (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError)


def _status_openai_errors() -> tuple[type[BaseException], ...]:
    try:
        from openai import APIStatusError
    except ImportError:
        return ()
    return (APIStatusError,)


def _permanent_openai_errors() -> tuple[type[BaseException], ...]:
    try:
        from openai import BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError
    except ImportError:
        return ()
    return (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)


def _is_retryable_status_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500
