from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from certificate_reader.csv_writer import write_results_csv
from certificate_reader.domain import ArchiveMemberError, ExtractionResult
from certificate_reader.vision import VisionExtractor
from certificate_reader.zip_reader import ZipLimits, read_certificate_zip

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSummary:
    output_csv: Path
    total_rows: int
    successful_rows: int
    error_rows: int


def extract_zip_to_csv(
    *,
    input_zip: Path,
    output_csv: Path,
    limits: ZipLimits,
    extractor: VisionExtractor,
    fail_fast: bool = False,
) -> PipelineSummary:
    archive_result = read_certificate_zip(input_zip, limits)
    results: list[ExtractionResult] = [_archive_error_to_result(error) for error in archive_result.errors]

    for document in archive_result.documents:
        LOGGER.info("certificate_extraction_started", extra={"source_file": document.source_name})
        try:
            extraction = extractor.extract(document)
        except Exception as exc:
            if fail_fast:
                raise
            LOGGER.exception(
                "certificate_extraction_failed",
                extra={"source_file": document.source_name, "error_type": type(exc).__name__},
            )
            results.append(ExtractionResult.error(document.source_name, str(exc)))
            continue
        results.append(ExtractionResult.success(document.source_name, extraction))

    results.sort(key=lambda result: result.source_file.lower())
    write_results_csv(results, output_csv)

    successful_rows = sum(1 for result in results if result.status == "success")
    error_rows = len(results) - successful_rows
    LOGGER.info(
        "certificate_csv_written",
        extra={
            "output_csv": str(output_csv),
            "total_rows": len(results),
            "successful_rows": successful_rows,
            "error_rows": error_rows,
        },
    )
    return PipelineSummary(
        output_csv=output_csv,
        total_rows=len(results),
        successful_rows=successful_rows,
        error_rows=error_rows,
    )


def _archive_error_to_result(error: ArchiveMemberError) -> ExtractionResult:
    return ExtractionResult.error(error.source_name, error.message)
