"""Unit tests for path resolution, directory enumeration, and size/existence
checks in the Ingestion_Engine (task 2.2).

Covers Requirements 1.1 (stream a supported file), 1.2 (non-recursive directory
combination), 1.3 (missing path error), 1.4 (oversize file), 1.5
(unsupported-extension files recorded), and 1.6 (directory with no supported
files).
"""

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


class TestSingleFile:
    def test_supported_file_streams_one_unit_with_raw_content(self, tmp_path):
        target = _write(tmp_path / "data.json", b'[{"a": 1}]')
        result = IngestionEngine().ingest(target)

        units = list(result.units)
        assert result.error is None
        assert result.skipped_files == []
        assert len(units) == 1
        assert units[0].source_file == target
        assert units[0].payload == b'[{"a": 1}]'
        assert units[0].location is not None
        assert units[0].location.source_file == target

    def test_unsupported_extension_recorded_and_no_units(self, tmp_path):
        target = _write(tmp_path / "notes.txt", b"hello")
        result = IngestionEngine().ingest(target)

        assert result.error is None
        assert result.skipped_files == [target]
        assert list(result.units) == []

    def test_missing_path_returns_error_and_no_units(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.json")
        result = IngestionEngine().ingest(missing)

        assert isinstance(result.error, PathNotFoundError)
        assert result.error.path == missing
        assert list(result.units) == []

    def test_oversize_file_returns_error_and_no_units(self, tmp_path):
        # Limit clamps up to 1 MiB; write a file just over that.
        target = _write(tmp_path / "big.json", b"x" * (MIN_FILE_SIZE_BYTES + 1))
        result = IngestionEngine().ingest(target, IngestionConfig(0))

        assert isinstance(result.error, FileSizeError)
        assert result.error.path == target
        assert result.error.limit_bytes == MIN_FILE_SIZE_BYTES
        assert result.error.actual_bytes == MIN_FILE_SIZE_BYTES + 1
        assert list(result.units) == []

    def test_file_at_limit_is_streamed(self, tmp_path):
        target = _write(tmp_path / "ok.json", b"x" * MIN_FILE_SIZE_BYTES)
        result = IngestionEngine().ingest(target, IngestionConfig(0))

        assert result.error is None
        assert len(list(result.units)) == 1


class TestDirectory:
    def test_combines_supported_files_in_order_and_records_skipped(self, tmp_path):
        a = _write(tmp_path / "a.json", b'[{"x": 1}]')
        c = _write(tmp_path / "c.csv", b"h\n1\n")
        b = _write(tmp_path / "b.jsonl", b'{"y": 2}\n')
        skipped = _write(tmp_path / "z.txt", b"ignore me")

        result = IngestionEngine().ingest(str(tmp_path))
        units = list(result.units)

        assert result.error is None
        # Deterministic, name-sorted order: a.json, b.jsonl, c.csv
        assert [u.source_file for u in units] == [a, b, c]
        assert result.skipped_files == [skipped]
        assert result.oversize_files == []

    def test_no_supported_files_returns_error(self, tmp_path):
        _write(tmp_path / "notes.txt", b"hello")
        _write(tmp_path / "image.png", b"\x89PNG")

        result = IngestionEngine().ingest(str(tmp_path))

        assert isinstance(result.error, NoSupportedFilesError)
        assert result.error.directory == str(tmp_path)
        assert list(result.units) == []
        # Unsupported files are still recorded even when erroring.
        assert sorted(result.skipped_files) == sorted(
            [str(tmp_path / "notes.txt"), str(tmp_path / "image.png")]
        )

    def test_empty_directory_returns_error(self, tmp_path):
        result = IngestionEngine().ingest(str(tmp_path))

        assert isinstance(result.error, NoSupportedFilesError)
        assert list(result.units) == []

    def test_enumeration_is_non_recursive(self, tmp_path):
        top = _write(tmp_path / "top.json", b"[]")
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        _write(nested_dir / "inner.json", b"[]")

        result = IngestionEngine().ingest(str(tmp_path))
        units = list(result.units)

        assert [u.source_file for u in units] == [top]

    def test_oversize_file_excluded_but_others_stream(self, tmp_path):
        small = _write(tmp_path / "small.json", b"[]")
        big = _write(tmp_path / "big.json", b"x" * (MIN_FILE_SIZE_BYTES + 1))

        result = IngestionEngine().ingest(str(tmp_path), IngestionConfig(0))
        units = list(result.units)

        assert result.error is None
        assert [u.source_file for u in units] == [small]
        assert result.oversize_files == [big]

    def test_all_supported_files_oversize_is_not_a_no_supported_error(self, tmp_path):
        big = _write(tmp_path / "big.json", b"x" * (MIN_FILE_SIZE_BYTES + 1))

        result = IngestionEngine().ingest(str(tmp_path), IngestionConfig(0))

        assert result.error is None
        assert list(result.units) == []
        assert result.oversize_files == [big]
