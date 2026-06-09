"""Unit tests for the Parquet parser strategy (Task 3.4).

Covers:
* Requirement 2.4 - every row across all row groups becomes a Record, in
  document order, one Record per row, located by ``(row_group, row_index)``.
* Requirement 2.5 - a corrupt/unreadable Parquet file becomes a located
  file-level Quality_Issue and parsing continues with the remaining units.

Parquet payload convention: a unit carries the raw Parquet file bytes as its
payload (the ingestion engine reads files as bytes). Parquet bytes are built
in-memory with ``pyarrow`` so the tests exercise the real I/O boundary.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.models import IssueCategory, SupportedFormat
from analyzer.parsers import Parser, RawRecordUnit


def _parquet_bytes(table: pa.Table, *, row_group_size: int | None = None) -> bytes:
    """Serialize an Arrow table to Parquet bytes, optionally chunked by row group."""
    sink = pa.BufferOutputStream()
    if row_group_size is None:
        pq.write_table(table, sink)
    else:
        pq.write_table(table, sink, row_group_size=row_group_size)
    return sink.getvalue().to_pybytes()


def _single_group_bytes(rows: list[dict]) -> bytes:
    return _parquet_bytes(pa.Table.from_pylist(rows))


def _parquet_unit(source_file: str, payload: bytes) -> RawRecordUnit:
    return RawRecordUnit(source_file=source_file, payload=payload)


class TestDispatch:
    def test_parquet_is_registered(self):
        assert SupportedFormat.PARQUET in Parser._STRATEGIES


class TestSingleRowGroup:
    def test_one_record_per_row_in_order(self):
        # Requirement 2.4
        rows = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Carol", "age": 41},
        ]
        records, issues = Parser().parse(
            [_parquet_unit("people.parquet", _single_group_bytes(rows))],
            SupportedFormat.PARQUET,
        )

        assert issues == []
        assert [r.fields for r in records] == rows

    def test_location_carries_row_group_and_row_index(self):
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        records, _ = Parser().parse(
            [_parquet_unit("nums.parquet", _single_group_bytes(rows))],
            SupportedFormat.PARQUET,
        )

        assert [(r.location.row_group, r.location.row_index) for r in records] == [
            (0, 0),
            (0, 1),
            (0, 2),
        ]
        assert all(r.location.source_file == "nums.parquet" for r in records)
        # Parquet locations do not use the line/array coordinates.
        assert all(r.location.line_number is None for r in records)
        assert all(r.location.array_index is None for r in records)

    def test_preserves_canonical_value_types(self):
        rows = [{"s": "txt", "i": 7, "f": 1.5, "b": True, "n": None}]
        records, issues = Parser().parse(
            [_parquet_unit("t.parquet", _single_group_bytes(rows))],
            SupportedFormat.PARQUET,
        )

        assert issues == []
        assert records[0].fields == {"s": "txt", "i": 7, "f": 1.5, "b": True, "n": None}


class TestMultipleRowGroups:
    def test_iterates_all_row_groups_in_order(self):
        # Requirement 2.4 - rows span multiple row groups but stay in order.
        rows = [{"i": value} for value in range(10)]
        data = _parquet_bytes(pa.Table.from_pylist(rows), row_group_size=4)

        # Sanity: the writer actually produced multiple row groups.
        meta = pq.ParquetFile(pa.BufferReader(data)).num_row_groups
        assert meta > 1

        records, issues = Parser().parse(
            [_parquet_unit("multi.parquet", data)], SupportedFormat.PARQUET
        )

        assert issues == []
        assert [r.fields["i"] for r in records] == list(range(10))

    def test_row_index_is_relative_to_row_group(self):
        rows = [{"i": value} for value in range(10)]
        data = _parquet_bytes(pa.Table.from_pylist(rows), row_group_size=4)

        records, _ = Parser().parse(
            [_parquet_unit("multi.parquet", data)], SupportedFormat.PARQUET
        )

        coords = [(r.location.row_group, r.location.row_index) for r in records]
        # 10 rows in groups of 4 -> groups 0,1 of size 4 and group 2 of size 2.
        assert coords == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
        ]


class TestEmptyAndMultiUnit:
    def test_empty_table_yields_no_records_and_no_issue(self):
        data = _single_group_bytes([{"x": 1}])
        # Build a genuinely empty table preserving schema.
        empty = pa.Table.from_pylist([], schema=pa.schema([("x", pa.int64())]))
        records, issues = Parser().parse(
            [_parquet_unit("empty.parquet", _parquet_bytes(empty))],
            SupportedFormat.PARQUET,
        )
        assert records == []
        assert issues == []
        # The non-empty companion still parses on its own.
        records2, _ = Parser().parse(
            [_parquet_unit("d.parquet", data)], SupportedFormat.PARQUET
        )
        assert len(records2) == 1

    def test_multiple_units_parsed_in_order(self):
        a = _single_group_bytes([{"v": "a0"}, {"v": "a1"}])
        b = _single_group_bytes([{"v": "b0"}])
        records, issues = Parser().parse(
            [_parquet_unit("a.parquet", a), _parquet_unit("b.parquet", b)],
            SupportedFormat.PARQUET,
        )

        assert issues == []
        assert [r.fields["v"] for r in records] == ["a0", "a1", "b0"]
        assert [r.location.source_file for r in records] == [
            "a.parquet",
            "a.parquet",
            "b.parquet",
        ]


class TestParseFaults:
    def test_corrupt_file_produces_file_level_issue_and_no_records(self):
        # Requirement 2.5 - corrupt Parquet bytes -> located file-level issue.
        records, issues = Parser().parse(
            [_parquet_unit("bad.parquet", b"not a parquet file at all")],
            SupportedFormat.PARQUET,
        )

        assert records == []
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        assert issue.location.source_file == "bad.parquet"
        assert issue.location.row_group is None
        assert issue.location.row_index is None

    def test_corrupt_file_does_not_abort_remaining_units(self):
        # Requirement 2.5 - a bad unit is isolated; later units still parse.
        good = _single_group_bytes([{"v": 1}, {"v": 2}])
        records, issues = Parser().parse(
            [
                _parquet_unit("bad.parquet", b"garbage"),
                _parquet_unit("good.parquet", good),
            ],
            SupportedFormat.PARQUET,
        )

        assert [r.fields["v"] for r in records] == [1, 2]
        assert len(issues) == 1
        assert issues[0].location.source_file == "bad.parquet"

    def test_non_bytes_payload_produces_file_level_issue(self):
        records, issues = Parser().parse(
            [RawRecordUnit(source_file="x.parquet", payload="not bytes")],
            SupportedFormat.PARQUET,
        )

        assert records == []
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.PARSE_ERROR
        assert issues[0].location.source_file == "x.parquet"
