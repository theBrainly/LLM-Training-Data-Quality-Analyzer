"""Integration tests for directory ingestion against temporary directories
(task 4.3).

Exercises the full ingestion -> parse -> combine path over a real directory on
disk that mixes all four Supported_Formats (JSON, JSONL, CSV, Parquet) plus an
unsupported-extension file. Asserts the combined :class:`Dataset` holds every
supported file's records in file-listing order (Requirements 1.1, 1.2) and that
the unsupported file is recorded in ``skipped_files`` while contributing no
records (Requirement 1.5).

Parquet fixtures are written with :mod:`pyarrow`; the text formats are written
as plain files. The temporary directory is provided by pytest's ``tmp_path``
fixture (safe for an example-based test, unlike inside a Hypothesis ``@given``).
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.combine import combine
from analyzer.ingestion import IngestionEngine
from analyzer.parsers import Parser


def _ingest_and_combine(path):
    return combine(IngestionEngine().ingest(str(path)), Parser())


def test_mixed_format_directory_with_unsupported_file(tmp_path):
    """A directory mixing all four formats plus an unsupported file produces
    the expected combined Dataset and skip list (Requirements 1.1, 1.2, 1.5)."""
    # Name-sorted listing order: a.json, b.jsonl, c.csv, d.parquet, e.txt.
    json_path = tmp_path / "a.json"
    jsonl_path = tmp_path / "b.jsonl"
    csv_path = tmp_path / "c.csv"
    parquet_path = tmp_path / "d.parquet"
    txt_path = tmp_path / "e.txt"

    json_path.write_text(json.dumps([{"text": "a0"}, {"text": "a1"}]), encoding="utf-8")
    jsonl_path.write_text('{"text": "b0"}\n{"text": "b1"}\n', encoding="utf-8")
    csv_path.write_text("text\nc0\nc1\n", encoding="utf-8")
    pq.write_table(pa.table({"text": ["d0", "d1"]}), str(parquet_path))
    txt_path.write_text("not a supported format", encoding="utf-8")

    combined = _ingest_and_combine(tmp_path)

    # Records appear in file-listing order then within-file parse order.
    assert [r.fields["text"] for r in combined.dataset.records] == [
        "a0",
        "a1",
        "b0",
        "b1",
        "c0",
        "c1",
        "d0",
        "d1",
    ]
    # Only the four supported files are sources, in name-sorted order.
    assert combined.dataset.source_files == [
        str(json_path),
        str(jsonl_path),
        str(csv_path),
        str(parquet_path),
    ]
    # The unsupported-extension file is recorded as skipped (Requirement 1.5).
    assert combined.dataset.skipped_files == [str(txt_path)]
    # Well-formed fixtures yield no parse issues.
    assert combined.issues == []


def test_unsupported_files_contribute_no_records(tmp_path):
    """Unsupported files are skipped and never add records (Requirement 1.5)."""
    (tmp_path / "data.json").write_text(json.dumps([{"text": "x"}]), encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "readme.md").write_text("# notes", encoding="utf-8")

    combined = _ingest_and_combine(tmp_path)

    assert [r.fields for r in combined.dataset.records] == [{"text": "x"}]
    assert combined.dataset.source_files == [str(tmp_path / "data.json")]
    assert combined.dataset.skipped_files == [
        str(tmp_path / "image.png"),
        str(tmp_path / "readme.md"),
    ]


def test_directory_of_only_unsupported_files_yields_empty_dataset(tmp_path):
    """A directory with no supported files yields an empty Dataset that still
    records the skipped files (Requirements 1.2, 1.5)."""
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    (tmp_path / "pic.jpg").write_bytes(b"\xff\xd8\xff")

    combined = _ingest_and_combine(tmp_path)

    assert combined.dataset.records == []
    assert combined.dataset.source_files == []
    assert combined.dataset.skipped_files == [
        str(tmp_path / "notes.txt"),
        str(tmp_path / "pic.jpg"),
    ]
