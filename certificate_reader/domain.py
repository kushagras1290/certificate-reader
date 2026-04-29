from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

CSV_COLUMNS: tuple[str, ...] = (
    "source_file",
    "status",
    "error_message",
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
    "additional_fields_json",
    "raw_text_excerpt",
)

EXTRACTION_FIELDS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class CertificateDocument:
    source_name: str
    suffix: str
    media_type: str
    content: bytes


@dataclass(frozen=True)
class ArchiveMemberError:
    source_name: str
    message: str


@dataclass(frozen=True)
class ArchiveReadResult:
    documents: tuple[CertificateDocument, ...]
    errors: tuple[ArchiveMemberError, ...] = ()


@dataclass(frozen=True)
class FieldValue:
    name: str
    value: str | None


@dataclass(frozen=True)
class CertificateExtraction:
    certificate_number: str | None = None
    laboratory: str | None = None
    report_type: str | None = None
    issue_date: str | None = None
    gemstone: str | None = None
    species: str | None = None
    variety: str | None = None
    weight_carats: str | None = None
    measurements: str | None = None
    shape_cut: str | None = None
    color: str | None = None
    transparency: str | None = None
    origin: str | None = None
    treatment: str | None = None
    comments: str | None = None
    confidence: float | None = None
    needs_review: bool = True
    additional_fields: tuple[FieldValue, ...] = field(default_factory=tuple)
    raw_text_excerpt: str | None = None

    @classmethod
    def from_model_dict(cls, payload: dict[str, Any]) -> CertificateExtraction:
        additional_fields = tuple(_parse_additional_fields(payload.get("additional_fields")))
        confidence = _parse_confidence(payload.get("confidence"))
        return cls(
            certificate_number=_nullable_str(payload.get("certificate_number")),
            laboratory=_nullable_str(payload.get("laboratory")),
            report_type=_nullable_str(payload.get("report_type")),
            issue_date=_nullable_str(payload.get("issue_date")),
            gemstone=_nullable_str(payload.get("gemstone")),
            species=_nullable_str(payload.get("species")),
            variety=_nullable_str(payload.get("variety")),
            weight_carats=_nullable_str(payload.get("weight_carats")),
            measurements=_nullable_str(payload.get("measurements")),
            shape_cut=_nullable_str(payload.get("shape_cut")),
            color=_nullable_str(payload.get("color")),
            transparency=_nullable_str(payload.get("transparency")),
            origin=_nullable_str(payload.get("origin")),
            treatment=_nullable_str(payload.get("treatment")),
            comments=_nullable_str(payload.get("comments")),
            confidence=confidence,
            needs_review=bool(payload.get("needs_review", True)),
            additional_fields=additional_fields,
            raw_text_excerpt=_nullable_str(payload.get("raw_text_excerpt")),
        )

    def to_csv_row(self, source_file: str, status: str, error_message: str = "") -> dict[str, str]:
        additional_fields_json = json.dumps(
            [{"name": field_value.name, "value": field_value.value} for field_value in self.additional_fields],
            ensure_ascii=False,
            sort_keys=True,
        )
        return {
            "source_file": source_file,
            "status": status,
            "error_message": error_message,
            "certificate_number": self.certificate_number or "",
            "laboratory": self.laboratory or "",
            "report_type": self.report_type or "",
            "issue_date": self.issue_date or "",
            "gemstone": self.gemstone or "",
            "species": self.species or "",
            "variety": self.variety or "",
            "weight_carats": self.weight_carats or "",
            "measurements": self.measurements or "",
            "shape_cut": self.shape_cut or "",
            "color": self.color or "",
            "transparency": self.transparency or "",
            "origin": self.origin or "",
            "treatment": self.treatment or "",
            "comments": self.comments or "",
            "confidence": "" if self.confidence is None else f"{self.confidence:.3f}",
            "needs_review": "true" if self.needs_review else "false",
            "additional_fields_json": additional_fields_json,
            "raw_text_excerpt": self.raw_text_excerpt or "",
        }


@dataclass(frozen=True)
class ExtractionResult:
    source_file: str
    status: str
    extraction: CertificateExtraction | None = None
    error_message: str = ""

    @classmethod
    def success(cls, source_file: str, extraction: CertificateExtraction) -> ExtractionResult:
        return cls(source_file=source_file, status="success", extraction=extraction)

    @classmethod
    def error(cls, source_file: str, message: str) -> ExtractionResult:
        return cls(source_file=source_file, status="error", error_message=message)

    def to_csv_row(self) -> dict[str, str]:
        extraction = self.extraction or CertificateExtraction()
        return extraction.to_csv_row(
            source_file=self.source_file,
            status=self.status,
            error_message=self.error_message,
        )


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _parse_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _parse_additional_fields(value: Any) -> list[FieldValue]:
    if not isinstance(value, list):
        return []
    fields: list[FieldValue] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _nullable_str(item.get("name"))
        if not name:
            continue
        fields.append(FieldValue(name=name, value=_nullable_str(item.get("value"))))
    return fields
