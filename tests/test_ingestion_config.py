"""Unit tests for ingestion config, size clamping, and extension detection.

Covers task 2.1: IngestionConfig default + clamping (Requirement 1.7) and
extension -> SupportedFormat mapping used for unsupported-file skipping
(Requirement 1.5).
"""

from analyzer.ingestion import (
    DEFAULT_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    IngestionConfig,
    clamp_file_size,
    detect_format,
)
from analyzer.models import SupportedFormat


class TestSizeBounds:
    def test_bound_constants(self):
        assert MIN_FILE_SIZE_BYTES == 1 * 1024 * 1024
        assert MAX_FILE_SIZE_BYTES == 50 * 1024 * 1024 * 1024
        assert DEFAULT_FILE_SIZE_BYTES == 5 * 1024 * 1024 * 1024
        assert MIN_FILE_SIZE_BYTES <= DEFAULT_FILE_SIZE_BYTES <= MAX_FILE_SIZE_BYTES


class TestClampFileSize:
    def test_unset_defaults_to_5_gib(self):
        assert clamp_file_size(None) == DEFAULT_FILE_SIZE_BYTES

    def test_value_in_range_is_unchanged(self):
        in_range = 2 * 1024 * 1024 * 1024  # 2 GiB
        assert clamp_file_size(in_range) == in_range

    def test_below_minimum_clamps_up(self):
        assert clamp_file_size(0) == MIN_FILE_SIZE_BYTES
        assert clamp_file_size(-1) == MIN_FILE_SIZE_BYTES
        assert clamp_file_size(MIN_FILE_SIZE_BYTES - 1) == MIN_FILE_SIZE_BYTES

    def test_above_maximum_clamps_down(self):
        assert clamp_file_size(MAX_FILE_SIZE_BYTES + 1) == MAX_FILE_SIZE_BYTES
        assert clamp_file_size(10 ** 15) == MAX_FILE_SIZE_BYTES

    def test_exact_bounds_are_kept(self):
        assert clamp_file_size(MIN_FILE_SIZE_BYTES) == MIN_FILE_SIZE_BYTES
        assert clamp_file_size(MAX_FILE_SIZE_BYTES) == MAX_FILE_SIZE_BYTES


class TestIngestionConfig:
    def test_default_is_5_gib(self):
        assert IngestionConfig().max_file_size_bytes == DEFAULT_FILE_SIZE_BYTES

    def test_configured_value_is_clamped(self):
        assert IngestionConfig(0).max_file_size_bytes == MIN_FILE_SIZE_BYTES
        assert (
            IngestionConfig(MAX_FILE_SIZE_BYTES + 1).max_file_size_bytes
            == MAX_FILE_SIZE_BYTES
        )

    def test_in_range_value_retained(self):
        in_range = 3 * 1024 * 1024 * 1024
        assert IngestionConfig(in_range).max_file_size_bytes == in_range

    def test_explicit_none_uses_default(self):
        assert IngestionConfig(None).max_file_size_bytes == DEFAULT_FILE_SIZE_BYTES


class TestDetectFormat:
    def test_supported_extensions(self):
        assert detect_format("data.json") == SupportedFormat.JSON
        assert detect_format("data.jsonl") == SupportedFormat.JSONL
        assert detect_format("data.csv") == SupportedFormat.CSV
        assert detect_format("data.parquet") == SupportedFormat.PARQUET

    def test_case_insensitive(self):
        assert detect_format("DATA.JSON") == SupportedFormat.JSON
        assert detect_format("Data.CsV") == SupportedFormat.CSV

    def test_with_directory_components(self):
        assert detect_format("/a/b/c/train.jsonl") == SupportedFormat.JSONL

    def test_dotted_filename_uses_final_extension(self):
        assert detect_format("archive.2024.csv") == SupportedFormat.CSV

    def test_unsupported_extension_returns_none(self):
        assert detect_format("notes.txt") is None
        assert detect_format("image.png") is None
        assert detect_format("data.json.gz") is None

    def test_no_extension_returns_none(self):
        assert detect_format("README") is None
        assert detect_format("data.") is None
