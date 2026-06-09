"""Unit tests for combining per-file parse output into a single ordered
Dataset (task 4.1).

Covers Requirement 1.1 (a supported file's records are parsed into the dataset)
and Requirement 1.2 (records from every supported file in a directory are
combined into a single Dataset, in file-listing order then within-file parse
order). Also covers carrying ``skipped_files`` from ingestion (Requirement 1.5)
onto the combined Dataset.

Tests import directly from :mod:`analyzer.combine` and drive it with the real
:class:`IngestionEngine` and :class:`Parser` over temporary files.
"""

from analyzer.combine import CombinedDataset, combine
from analyzer.ingestion import IngestionEngine, IngestionResult
from analyzer.models import IssueCategory
from analyzer.parsers import Parser, RawRecordUnit


def _ingest(path):
    return IngestionEngine().ingest(str(path))


def test_single_file_records_parsed_into_dataset(tmp_path):
    target = tmp_path / "data.json"
    target.write_bytes(b'[{"a": 1}, {"a": 2}]')

    combined = combine(_ingest(target), Parser())

    assert isinstance(combined, CombinedDataset)
    assert [r.fields for r in combined.dataset.records] == [{"a": 1}, {"a": 2}]
    assert combined.dataset.source_files == [str(target)]
    assert combined.dataset.skipped_files == []
    assert combined.issues == []


def test_directory_combines_files_in_listing_then_parse_order(tmp_path):
    # Name-sorted directory order is a.json then b.jsonl.
    (tmp_path / "a.json").write_bytes(b'[{"v": "a0"}, {"v": "a1"}]')
    (tmp_path / "b.jsonl").write_bytes(b'{"v": "b0"}\n{"v": "b1"}\n')

    combined = combine(_ingest(tmp_path), Parser())

    # File-listing order (a before b) then within-file parse order.
    assert [r.fields["v"] for r in combined.dataset.records] == [
        "a0",
        "a1",
        "b0",
        "b1",
    ]
    assert combined.dataset.source_files == [
        str(tmp_path / "a.json"),
        str(tmp_path / "b.jsonl"),
    ]


def test_skipped_files_are_carried_onto_dataset(tmp_path):
    (tmp_path / "data.json").write_bytes(b'[{"a": 1}]')
    (tmp_path / "notes.txt").write_bytes(b"not a supported format")

    result = _ingest(tmp_path)
    combined = combine(result, Parser())

    assert combined.dataset.skipped_files == [str(tmp_path / "notes.txt")]
    # The unsupported file contributes no records and is not a source file.
    assert combined.dataset.source_files == [str(tmp_path / "data.json")]
    assert [r.fields for r in combined.dataset.records] == [{"a": 1}]


def test_parse_issues_are_collected_across_files(tmp_path):
    # A malformed JSONL line yields a parse issue but the valid line still parses.
    (tmp_path / "good.json").write_bytes(b'[{"a": 1}]')
    (tmp_path / "mixed.jsonl").write_bytes(b'{"b": 2}\nnot-json\n')

    combined = combine(_ingest(tmp_path), Parser())

    assert [r.fields for r in combined.dataset.records] == [{"a": 1}, {"b": 2}]
    assert len(combined.issues) == 1
    assert combined.issues[0].category is IssueCategory.PARSE_ERROR


def test_format_detected_per_unit_preserves_unit_order(tmp_path):
    # Drive combine with a hand-built result whose units span multiple formats,
    # confirming each unit is parsed by its own detected format in stream order.
    csv_path = str(tmp_path / "rows.csv")
    json_path = str(tmp_path / "arr.json")
    units = iter(
        [
            RawRecordUnit(source_file=csv_path, payload=b"name\nalice\nbob\n"),
            RawRecordUnit(source_file=json_path, payload=b'[{"name": "carol"}]'),
        ]
    )
    result = IngestionResult(units=units, skipped_files=["x.bin"])

    combined = combine(result, Parser())

    assert [r.fields for r in combined.dataset.records] == [
        {"name": "alice"},
        {"name": "bob"},
        {"name": "carol"},
    ]
    assert combined.dataset.source_files == [csv_path, json_path]
    assert combined.dataset.skipped_files == ["x.bin"]


def test_ingestion_error_yields_empty_dataset_with_skipped_files(tmp_path):
    # A directory with only unsupported files: ingestion errors and streams
    # nothing, so the combined dataset is empty but still carries skip info.
    (tmp_path / "notes.txt").write_bytes(b"nope")

    result = _ingest(tmp_path)
    combined = combine(result, Parser())

    assert combined.dataset.records == []
    assert combined.dataset.source_files == []
    assert combined.dataset.skipped_files == [str(tmp_path / "notes.txt")]
