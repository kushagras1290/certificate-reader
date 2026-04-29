from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_FILES = 500
DEFAULT_MAX_FILE_MB = 25
DEFAULT_MAX_TOTAL_MB = 512
DEFAULT_IMAGE_DETAIL = "high"


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    input_zip: Path
    output_csv: Path
    model: str
    timeout_seconds: float
    max_retries: int
    max_files: int
    max_file_mb: int
    max_total_mb: int
    image_detail: str
    fail_fast: bool

    @classmethod
    def from_values(
        cls,
        *,
        input_zip: str | Path,
        output_csv: str | Path,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_files: int | None = None,
        max_file_mb: int | None = None,
        max_total_mb: int | None = None,
        image_detail: str | None = None,
        fail_fast: bool = False,
    ) -> RuntimeConfig:
        resolved_model = model or os.getenv("CERTIFICATE_OPENAI_MODEL") or DEFAULT_MODEL
        resolved_timeout = _positive_float(
            timeout_seconds,
            os.getenv("CERTIFICATE_OPENAI_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
            "timeout_seconds",
        )
        resolved_retries = _positive_int(
            max_retries,
            os.getenv("CERTIFICATE_OPENAI_MAX_RETRIES"),
            DEFAULT_MAX_RETRIES,
            "max_retries",
            allow_zero=True,
        )
        resolved_max_files = _positive_int(
            max_files,
            os.getenv("CERTIFICATE_MAX_FILES"),
            DEFAULT_MAX_FILES,
            "max_files",
        )
        resolved_max_file_mb = _positive_int(
            max_file_mb,
            os.getenv("CERTIFICATE_MAX_FILE_MB"),
            DEFAULT_MAX_FILE_MB,
            "max_file_mb",
        )
        resolved_max_total_mb = _positive_int(
            max_total_mb,
            os.getenv("CERTIFICATE_MAX_TOTAL_MB"),
            DEFAULT_MAX_TOTAL_MB,
            "max_total_mb",
        )
        resolved_detail = image_detail or os.getenv("CERTIFICATE_IMAGE_DETAIL") or DEFAULT_IMAGE_DETAIL
        if resolved_detail not in {"low", "high", "auto"}:
            raise ConfigError("image_detail must be one of: low, high, auto")

        input_path = Path(input_zip).expanduser()
        output_path = Path(output_csv).expanduser()
        if input_path.suffix.lower() != ".zip":
            raise ConfigError("input_zip must be a .zip file")
        if output_path.suffix.lower() != ".csv":
            raise ConfigError("output_csv must be a .csv file")

        return cls(
            input_zip=input_path,
            output_csv=output_path,
            model=resolved_model.strip(),
            timeout_seconds=resolved_timeout,
            max_retries=resolved_retries,
            max_files=resolved_max_files,
            max_file_mb=resolved_max_file_mb,
            max_total_mb=resolved_max_total_mb,
            image_detail=resolved_detail,
            fail_fast=fail_fast,
        )

    def validate_for_openai(self) -> None:
        if not self.model:
            raise ConfigError("OpenAI model cannot be empty")
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigError("OPENAI_API_KEY is required for live extraction")


def _positive_int(
    explicit: int | None,
    env_value: str | None,
    default: int,
    field_name: str,
    *,
    allow_zero: bool = False,
) -> int:
    raw_value: int | str = explicit if explicit is not None else env_value or default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc
    if value < 0 or (value == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field_name} must be {comparator}")
    return value


def _positive_float(
    explicit: float | None,
    env_value: str | None,
    default: float,
    field_name: str,
) -> float:
    raw_value: float | str = explicit if explicit is not None else env_value or default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc
    if value <= 0:
        raise ConfigError(f"{field_name} must be positive")
    return value
