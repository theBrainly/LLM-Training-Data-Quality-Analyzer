"""Ingestion configuration and supported-format detection.

This module owns the file-system-facing parts of the Ingestion_Engine: resolving
the effective maximum file size from a (possibly invalid or unset) configured
value, mapping a file extension to a :class:`SupportedFormat`, and resolving a
submitted path (a single file or a non-recursive directory) into a stream of
:class:`RawRecordUnit` objects while enforcing existence, size, and
supported-format constraints.

Size limit resolution (Requirement 1.7): the effective maximum file size is the
configured value clamped into the inclusive range ``[1 MiB, 50 GiB]``. When no
value is configured the default of 5 GiB is used. The design specifies binary
units (MiB/GiB), so those are used here.

Path resolution (Requirements 1.1-1.6): :class:`IngestionEngine` reads raw file
content into :class:`RawRecordUnit` streams. A missing path, a directory with no
Supported_Format files, and (for a single submitted file) an oversize file are
*fail-fast* conditions surfaced through ``IngestionResult.error`` with no
records streamed. Unsupported-extension files are skipped and recorded in
``skipped_files`` (Requirement 1.5). Within a directory, oversize files are
excluded from the stream and recorded in ``oversize_files`` so that the
remaining supported files still stream (Requirement 1.4, "from that file").
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from analyzer.errors import (
    FileSizeError,
    NoSupportedFilesError,
    PathNotFoundError,
)
from analyzer.models import RecordLocation, SupportedFormat
from analyzer.parsers import RawRecordUnit

__all__ = [
    "MIN_FILE_SIZE_BYTES",
    "MAX_FILE_SIZE_BYTES",
    "DEFAULT_FILE_SIZE_BYTES",
    "IngestionConfig",
    "IngestionResult",
    "IngestionEngine",
    "clamp_file_size",
    "detect_format",
]

# Inclusive bounds and default for the maximum file size (Requirement 1.7).
MIN_FILE_SIZE_BYTES: int = 1 * 1024 * 1024            # 1 MiB
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024 * 1024    # 50 GiB
DEFAULT_FILE_SIZE_BYTES: int = 5 * 1024 * 1024 * 1024  # 5 GiB

# Map of lowercased file extensions (including the leading dot) to the
# Supported_Format they denote.
_EXTENSION_TO_FORMAT: dict[str, SupportedFormat] = {
    ".json": SupportedFormat.JSON,
    ".jsonl": SupportedFormat.JSONL,
    ".csv": SupportedFormat.CSV,
    ".parquet": SupportedFormat.PARQUET,
}


def clamp_file_size(value: int | None) -> int:
    """Resolve the effective maximum file size in bytes (Requirement 1.7).

    An unset value (``None``) resolves to the default of 5 GiB. Any other value
    is clamped into the inclusive range ``[1 MiB, 50 GiB]``: values below the
    minimum (including zero and negatives) become 1 MiB, and values above the
    maximum become 50 GiB.
    """
    if value is None:
        return DEFAULT_FILE_SIZE_BYTES
    if value < MIN_FILE_SIZE_BYTES:
        return MIN_FILE_SIZE_BYTES
    if value > MAX_FILE_SIZE_BYTES:
        return MAX_FILE_SIZE_BYTES
    return value


@dataclass
class IngestionConfig:
    """Configuration for the Ingestion_Engine.

    ``max_file_size_bytes`` is always stored as the *effective* value: the
    configured value clamped into ``[1 MiB, 50 GiB]``, defaulting to 5 GiB when
    unset. Construct with ``IngestionConfig()`` for the default, or pass a
    ``configured_max_file_size_bytes`` (which may be any integer or ``None``)
    to have it clamped on construction.
    """

    max_file_size_bytes: int = field(default=DEFAULT_FILE_SIZE_BYTES)

    def __init__(self, configured_max_file_size_bytes: int | None = None) -> None:
        self.max_file_size_bytes = clamp_file_size(configured_max_file_size_bytes)


def detect_format(path: str) -> SupportedFormat | None:
    """Return the :class:`SupportedFormat` for ``path`` by file extension.

    Matching is case-insensitive on the extension (``.JSON`` and ``.json`` both
    map to :class:`SupportedFormat.JSON`). Returns ``None`` when the extension
    does not correspond to a Supported_Format, so the caller can skip the file
    and record it (Requirement 1.5).
    """
    _, ext = os.path.splitext(path)
    return _EXTENSION_TO_FORMAT.get(ext.lower())


@dataclass
class IngestionResult:
    """The outcome of resolving a submitted path into raw record units.

    ``units`` is a lazy iterator of :class:`RawRecordUnit` objects, one per
    Supported_Format file that passed the existence and size checks; each unit
    carries the raw byte content of its file so the Parser can expand it into
    :class:`Record` objects. ``skipped_files`` lists unsupported-extension files
    that were skipped (Requirement 1.5). ``error`` carries a *fail-fast* error
    (missing path, oversize single file, or empty/no-supported-files directory)
    and, when set, ``units`` yields nothing (Requirements 1.3, 1.4, 1.6).

    ``oversize_files`` records Supported_Format files that were excluded from a
    *directory* stream because they exceeded the configured size limit; the
    remaining supported files in that directory still stream (Requirement 1.4,
    "stream no Records from that file").
    """

    units: Iterator[RawRecordUnit]
    skipped_files: list[str] = field(default_factory=list)
    error: PathNotFoundError | FileSizeError | NoSupportedFilesError | None = None
    oversize_files: list[str] = field(default_factory=list)


def _empty_units() -> Iterator[RawRecordUnit]:
    """Return an iterator that yields no record units."""
    return iter(())


def _read_unit(path: str) -> RawRecordUnit:
    """Read the raw byte content of ``path`` into a :class:`RawRecordUnit`.

    The unit carries the file's raw bytes as its payload and a
    :class:`RecordLocation` stamped with the source file, leaving format-aware
    expansion (array elements, lines, rows) to the Parser.
    """
    with open(path, "rb") as handle:
        payload = handle.read()
    return RawRecordUnit(
        source_file=path,
        payload=payload,
        location=RecordLocation(source_file=path),
    )


class IngestionEngine:
    """Resolves a submitted path into a stream of raw record units.

    For a single file the engine enforces existence, supported-format, and size
    constraints. For a directory it enumerates only the files located *directly*
    in the directory (non-recursive, Requirement 1.2), skipping
    unsupported-extension files and excluding oversize files while streaming the
    remaining Supported_Format files in a deterministic (name-sorted) order.
    """

    def ingest(
        self, path: str, config: IngestionConfig | None = None
    ) -> IngestionResult:
        """Resolve ``path`` into an :class:`IngestionResult`.

        A non-existent path short-circuits to a :class:`PathNotFoundError` with
        no records streamed (Requirement 1.3). Directories and single files are
        dispatched to their respective handlers.
        """
        if config is None:
            config = IngestionConfig()

        if not os.path.exists(path):
            return IngestionResult(units=_empty_units(), error=PathNotFoundError(path))

        if os.path.isdir(path):
            return self._ingest_directory(path, config)
        return self._ingest_file(path, config)

    def _ingest_file(self, path: str, config: IngestionConfig) -> IngestionResult:
        """Resolve a single submitted file.

        An unsupported extension is skipped and recorded (Requirement 1.5); an
        oversize file is a fail-fast error that streams no records (Requirement
        1.4); otherwise a single unit carrying the file's raw content is
        streamed (Requirement 1.1).
        """
        if detect_format(path) is None:
            return IngestionResult(units=_empty_units(), skipped_files=[path])

        actual_bytes = os.path.getsize(path)
        if actual_bytes > config.max_file_size_bytes:
            return IngestionResult(
                units=_empty_units(),
                error=FileSizeError(
                    path,
                    limit_bytes=config.max_file_size_bytes,
                    actual_bytes=actual_bytes,
                ),
            )

        return IngestionResult(units=iter([_read_unit(path)]))

    def _ingest_directory(self, path: str, config: IngestionConfig) -> IngestionResult:
        """Resolve a directory by enumerating its direct file entries.

        Files are visited in name-sorted order for determinism. Subdirectories
        are ignored (non-recursive). Unsupported-extension files are recorded in
        ``skipped_files`` (Requirement 1.5); oversize Supported_Format files are
        recorded in ``oversize_files`` and excluded from the stream (Requirement
        1.4). If no Supported_Format files exist in the directory at all, a
        :class:`NoSupportedFilesError` is returned (Requirement 1.6).
        """
        skipped_files: list[str] = []
        oversize_files: list[str] = []
        readable_files: list[str] = []
        supported_found = False

        for name in sorted(os.listdir(path)):
            entry = os.path.join(path, name)
            if not os.path.isfile(entry):
                continue

            if detect_format(entry) is None:
                skipped_files.append(entry)
                continue

            supported_found = True
            if os.path.getsize(entry) > config.max_file_size_bytes:
                oversize_files.append(entry)
                continue
            readable_files.append(entry)

        if not supported_found:
            return IngestionResult(
                units=_empty_units(),
                skipped_files=skipped_files,
                error=NoSupportedFilesError(path),
            )

        units = (_read_unit(file_path) for file_path in readable_files)
        return IngestionResult(
            units=units,
            skipped_files=skipped_files,
            oversize_files=oversize_files,
        )
