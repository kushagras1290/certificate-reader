from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from certificate_reader.config import (
    DEFAULT_IMAGE_DETAIL,
    DEFAULT_MAX_FILE_MB,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOTAL_MB,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    ConfigError,
    RuntimeConfig,
)
from certificate_reader.env_file import EnvFileError, load_env_file
from certificate_reader.pipeline import extract_zip_to_csv
from certificate_reader.vision import OpenAIVisionExtractor
from certificate_reader.zip_reader import CertificateArchiveError, ZipLimits


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        load_env_file(args.env_file)
        config = RuntimeConfig.from_values(
            input_zip=args.input,
            output_csv=args.output,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            max_files=args.max_files,
            max_file_mb=args.max_file_mb,
            max_total_mb=args.max_total_mb,
            image_detail=args.image_detail,
            fail_fast=args.fail_fast,
        )
        config.validate_for_openai()
        limits = ZipLimits.from_megabytes(
            max_files=config.max_files,
            max_file_mb=config.max_file_mb,
            max_total_mb=config.max_total_mb,
        )
        extractor = OpenAIVisionExtractor(
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            image_detail=config.image_detail,
        )
        summary = extract_zip_to_csv(
            input_zip=config.input_zip,
            output_csv=config.output_csv,
            limits=limits,
            extractor=extractor,
            fail_fast=config.fail_fast,
        )
    except (ConfigError, EnvFileError, CertificateArchiveError) as exc:
        logging.getLogger(__name__).error("certificate_reader_failed", extra={"error": str(exc)})
        return 1
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("certificate_reader_interrupted")
        return 130

    logging.getLogger(__name__).info(
        "certificate_reader_finished",
        extra={
            "output_csv": str(summary.output_csv),
            "total_rows": summary.total_rows,
            "successful_rows": summary.successful_rows,
            "error_rows": summary.error_rows,
        },
    )
    return 2 if summary.error_rows else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract certificate data from a ZIP of PDFs/images and write a CSV.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the input .zip file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output .csv file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI vision-capable model to use.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request OpenAI timeout.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries for transient OpenAI/network errors.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="Maximum files allowed inside the ZIP archive.",
    )
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=DEFAULT_MAX_FILE_MB,
        help="Maximum size per ZIP member before it is skipped.",
    )
    parser.add_argument(
        "--max-total-mb",
        type=int,
        default=DEFAULT_MAX_TOTAL_MB,
        help="Maximum total uncompressed ZIP size.",
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default=DEFAULT_IMAGE_DETAIL,
        help="OpenAI image detail level for image certificates.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first extraction failure instead of writing error rows.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv-style config file to load before reading environment variables.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
