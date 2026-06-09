"""Unit tests for fail-fast ingestion errors (task 2.5).

The design's Error Handling section classifies three ingestion conditions as
*fail-fast*: each stops the operation and returns a typed error (carried on
``IngestionResult.error``) with no records streamed.

* Missing file path -> :class:`PathNotFoundError`, no records (Requirement 1.3).
* File exceeds the size limit -> :class:`FileSizeError`, no records from that
  file (Requirement 1.4). For a *single* submitted file this is the fail-fast
  error; within a *directory* the oversize file is recorded in
  ``oversize_files`` and excluded while the remaining files still stream.
* Directory with no supported files -> :class:`NoSupportedFilesError`
  (Requirement 1.6).
"""

from __future__ import annotations

from analyzer.errors import (
    FileSizeError,
    NoSupportedFilesError,
    PathNotFoundError,
)
from analyzer.ingestion import (
    MIN_FILE_SIZE_BYTES,
    IngestionConfig,
    IngestionEngine,
)


def _write(path, content: bytes = b"[]") -> str:
    path.write_bytes(content)
    return str(path)


class TestMissingPath:
    """Requirement 1.3: a missing path errors and streams no records."""

    def test_missing_file_returns_error_and_no_records(self, tmp_path):
        missing = str(tmp_path / "absent.json")

        result = IngestionEngine().ingest(missing)

        assert isinstance(result.error, PathNotFoundError)
        assert result.error.path == missing
        assert list(result.units) == []

    def test_missing_directory_returns_error_and_no_records(self, tmp_path):
        missing = str(tmp_path / "no_such_dir")

        result = IngestionEngine().ingest(missing)

        assert isinstance(result.error, PathNotFoundError)
        assert result.error.path == missing
        assert list(result.units) == []


class TestOversizeFile:
    """Requirement 1.4: an oversize file yields an error/issue and no records."""

    def test_single_oversize_file_errors_and_streams_nothing(self, tmp_path):
        # IngestionConfig(0) clamps the limit up to 1 MiB; write one byte over.
        target = _write(tmp_path / "big.json", b"x" * (MIN_FILE_SIZE_BYTES + 1))

        result = IngestionEngine().ingest(target, IngestionConfig(0))

        assert isinstance(result.error, FileSizeError)
        assert result.error.path == target
        assert result.error.limit_bytes == MIN_FILE_SIZE_BYTES
        assert result.error.actual_bytes == MIN_FILE_SIZE_BYTES + 1
        assert list(result.units) == []

    def test_oversize_file_in_directory_is_recorded_and_excluded(self, tmp_path):
        small = _write(tmp_path / "small.json", b"[]")
        big = _write(tmp_path / "big.json", b"x" * (MIN_FILE_SIZE_BYTES + 1))

        result = IngestionEngine().ingest(str(tmp_path), IngestionConfig(0))
        units = list(result.units)

        # The oversize file streams no records but does not abort the directory.
        assert result.error is None
        assert result.oversize_files == [big]
        assert [u.source_file for u in units] == [small]

    def test_all_files_oversize_streams_no_records(self, tmp_path):
        big = _write(tmp_path / "big.json", b"x" * (MIN_FILE_SIZE_BYTES + 1))

        result = IngestionEngine().ingest(str(tmp_path), IngestionConfig(0))

        assert result.error is None
        assert result.oversize_files == [big]
        assert list(result.units) == []


class TestNoSupportedFiles:
    """Requirement 1.6: a directory with no supported files errors."""

    def test_unsupported_only_directory_returns_error(self, tmp_path):
        _write(tmp_path / "notes.txt", b"hello")
        _write(tmp_path / "image.png", b"\x89PNG")

        result = IngestionEngine().ingest(str(tmp_path))

        assert isinstance(result.error, NoSupportedFilesError)
        assert result.error.directory == str(tmp_path)
        assert list(result.units) == []

    def test_empty_directory_returns_error(self, tmp_path):
        result = IngestionEngine().ingest(str(tmp_path))

        assert isinstance(result.error, NoSupportedFilesError)
        assert result.error.directory == str(tmp_path)
        assert list(result.units) == []
