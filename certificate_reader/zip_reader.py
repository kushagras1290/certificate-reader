from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from certificate_reader.domain import (
    SUPPORTED_MEDIA_TYPES,
    ArchiveMemberError,
    ArchiveReadResult,
    CertificateDocument,
)

BYTES_PER_MB = 1024 * 1024


class CertificateArchiveError(Exception):
    """Base error for unsafe or unreadable certificate archives."""


class UnsafeArchiveError(CertificateArchiveError):
    """Raised when an archive contains unsafe paths or zip-bomb-like sizes."""


@dataclass(frozen=True)
class ZipLimits:
    max_files: int
    max_file_bytes: int
    max_total_uncompressed_bytes: int

    @classmethod
    def from_megabytes(cls, *, max_files: int, max_file_mb: int, max_total_mb: int) -> ZipLimits:
        return cls(
            max_files=max_files,
            max_file_bytes=max_file_mb * BYTES_PER_MB,
            max_total_uncompressed_bytes=max_total_mb * BYTES_PER_MB,
        )


def read_certificate_zip(path: Path, limits: ZipLimits) -> ArchiveReadResult:
    if not path.exists():
        raise CertificateArchiveError(f"ZIP file does not exist: {path}")
    if not path.is_file():
        raise CertificateArchiveError(f"ZIP path is not a file: {path}")

    documents: list[CertificateDocument] = []
    errors: list[ArchiveMemberError] = []
    total_uncompressed = 0

    try:
        with zipfile.ZipFile(path) as archive:
            file_infos = [info for info in archive.infolist() if not info.is_dir()]
            if len(file_infos) > limits.max_files:
                raise UnsafeArchiveError(
                    f"Archive contains {len(file_infos)} files, above limit {limits.max_files}"
                )

            for info in file_infos:
                safe_name = _safe_member_name(info.filename)
                total_uncompressed += info.file_size
                if total_uncompressed > limits.max_total_uncompressed_bytes:
                    raise UnsafeArchiveError("Archive uncompressed size exceeds configured limit")
                suffix = PurePosixPath(safe_name).suffix.lower()
                media_type = SUPPORTED_MEDIA_TYPES.get(suffix)
                if not media_type:
                    errors.append(ArchiveMemberError(safe_name, f"unsupported file type: {suffix or '<none>'}"))
                    continue
                if info.file_size <= 0:
                    errors.append(ArchiveMemberError(safe_name, "empty file"))
                    continue
                if info.file_size > limits.max_file_bytes:
                    errors.append(
                        ArchiveMemberError(
                            safe_name,
                            f"file exceeds configured per-file limit ({info.file_size} bytes)",
                        )
                    )
                    continue
                with archive.open(info) as member_file:
                    content = member_file.read()
                if len(content) != info.file_size:
                    errors.append(ArchiveMemberError(safe_name, "archive member read size mismatch"))
                    continue
                documents.append(
                    CertificateDocument(
                        source_name=safe_name,
                        suffix=suffix,
                        media_type=media_type,
                        content=content,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise CertificateArchiveError("Input is not a valid ZIP archive") from exc

    return ArchiveReadResult(documents=tuple(documents), errors=tuple(errors))


def _safe_member_name(name: str) -> str:
    if "\x00" in name:
        raise UnsafeArchiveError("Archive member contains a NUL byte")
    normalized = name.replace("\\", "/").strip()
    if not normalized:
        raise UnsafeArchiveError("Archive member has an empty path")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise UnsafeArchiveError(f"Archive member uses an absolute path: {name}")
    parts = path.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchiveError(f"Archive member path traversal is not allowed: {name}")
    if any(":" in part for part in parts):
        raise UnsafeArchiveError(f"Archive member path contains an unsafe drive separator: {name}")
    return str(path)
