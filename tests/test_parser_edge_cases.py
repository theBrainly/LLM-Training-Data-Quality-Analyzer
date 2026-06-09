"""Unit tests for empty-file and CSV header-defect parsing (task 3.7).

Covers Requirements 2.6 and 2.7:

* **Requirement 2.7** - a CSV with a missing header row, or with duplicate field
  names in the header, produces a file-level Quality_Issue identifying the
  affected file (or column) and emits no Records for that file. This is fully
  implemented and asserted here.

* **Requirement 2.6** - a file containing zero parseable records returns an
  empty list of Records.

  Note on observed behavior: the current Parser distinguishes a *legitimately
  empty* source (an empty JSON array, an empty JSONL document, a CSV with a
  valid header but no data rows, or an empty Parquet table) from a source whose
  content is *all malformed*. A legitimately empty source returns an empty
  Record list with no issue, whereas an all-malformed source returns an empty
  Record list plus one located parse issue per malformed unit. These tests
  document that actual behavior rather than asserting a dedicated file-level
  "produced no records" issue for genuinely empty files (which the Parser does
  not currently emit).

CSV/JSON/JSONL units carry the whole file text as payload; Parquet units carry
raw Parquet bytes built in-memory via ``pyarrow``.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.models import IssueCategory, SupportedFormat
from analyzer.parsers import Parser, RawRecordUnit


# --------------------------------------------------------------------------- #
# Requirement 2.6 - zero parseable records yields an empty Record list.
# --------------------------------------------------------------------------- #

class TestEmptyFilesYieldNoRecords:
    def test_empty_json_array_yields_empty_list(self):
        records, issues = Parser().parse(
            [RawRecordUnit("empty.json", json.dumps([]))], SupportedFormat.JSON
        )
        assert records == []
        assert issues == []

    def test_empty_jsonl_yields_empty_list(self):
        records, issues = Parser().parse(
            [RawRecordUnit("empty.jsonl", "")], SupportedFormat.JSONL
        )
        assert records == []
        assert issues == []

    def test_whitespace_only_jsonl_yields_empty_list(self):
        # Whitespace-only lines are ignored entirely (Requirement 2.2), so a
        # document of only blank lines parses to zero records and no issues.
        records, issues = Parser().parse(
            [RawRecordUnit("blanks.jsonl", "\n  \n\t\n")], SupportedFormat.JSONL
        )
        assert records == []
        assert issues == []

    def test_header_only_csv_yields_empty_list(self):
        # A valid header with no data rows is a legitimately empty source.
        records, issues = Parser().parse(
            [RawRecordUnit("hdr.csv", "a,b,c\n")], SupportedFormat.CSV
        )
        assert records == []
        assert issues == []

    def test_empty_parquet_table_yields_empty_list(self):
        empty = pa.Table.from_pylist([], schema=pa.schema([("x", pa.int64())]))
        sink = pa.BufferOutputStream()
        pq.write_table(empty, sink)
        records, issues = Parser().parse(
            [RawRecordUnit("empty.parquet", sink.getvalue().to_pybytes())],
            SupportedFormat.PARQUET,
        )
        assert records == []
        assert issues == []


class TestAllMalformedFilesYieldNoRecords:
    def test_json_array_of_only_non_objects_yields_empty_list_and_issues(self):
        # Every element is unparseable into fields, so no Records are produced
        # and one located parse issue is recorded per element (Requirement 2.5),
        # leaving the Record list empty (Requirement 2.6).
        elements = [1, "two", [3], None]
        records, issues = Parser().parse(
            [RawRecordUnit("bad.json", json.dumps(elements))], SupportedFormat.JSON
        )
        assert records == []
        assert len(issues) == len(elements)
        assert all(i.category == IssueCategory.PARSE_ERROR for i in issues)
        assert sorted(i.location.array_index for i in issues) == [0, 1, 2, 3]

    def test_jsonl_of_only_malformed_lines_yields_empty_list_and_issues(self):
        records, issues = Parser().parse(
            [RawRecordUnit("bad.jsonl", "{not json\n42\n[1, 2")], SupportedFormat.JSONL
        )
        assert records == []
        assert len(issues) == 3
        assert all(i.category == IssueCategory.PARSE_ERROR for i in issues)
        assert sorted(i.location.line_number for i in issues) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Requirement 2.7 - CSV missing/duplicate header: file-level issue, no records.
# --------------------------------------------------------------------------- #

class TestCsvMissingHeader:
    def test_empty_csv_is_missing_header_issue_with_no_records(self):
        records, issues = Parser().parse(
            [RawRecordUnit("empty.csv", "")], SupportedFormat.CSV
        )
        assert records == []
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        # File-level location (no record coordinate) identifies the file.
        assert issue.location is not None
        assert issue.location.source_file == "empty.csv"
        assert issue.location.line_number is None

    def test_missing_header_suppresses_all_records(self):
        # Even though data-shaped rows follow, a file whose header row is empty
        # produces no Records (the first row is always the header).
        records, issues = Parser().parse(
            [RawRecordUnit("nohdr.csv", "\n1,2\n3,4\n")], SupportedFormat.CSV
        )
        assert records == []
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.PARSE_ERROR
        assert issues[0].location.source_file == "nohdr.csv"


class TestCsvDuplicateHeader:
    def test_duplicate_header_names_suppress_all_records(self):
        records, issues = Parser().parse(
            [RawRecordUnit("dup.csv", "a,b,a\n1,2,3\n4,5,6\n")], SupportedFormat.CSV
        )
        assert records == []
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        # The offending duplicate column is identified (Requirement 2.7).
        assert issue.field_name == "a"
        assert issue.location.source_file == "dup.csv"

    def test_multiple_duplicate_columns_still_single_file_level_issue(self):
        records, issues = Parser().parse(
            [RawRecordUnit("dup2.csv", "x,x,y,y\n1,2,3,4\n")], SupportedFormat.CSV
        )
        assert records == []
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.PARSE_ERROR
        # The first duplicate (sorted) is named.
        assert issues[0].field_name == "x"

    def test_duplicate_header_does_not_block_other_files(self):
        # A bad CSV file is isolated; a following well-formed file still parses.
        units = [
            RawRecordUnit("dup.csv", "a,a\n1,2\n"),
            RawRecordUnit("ok.csv", "a,b\n1,2\n"),
        ]
        records, issues = Parser().parse(units, SupportedFormat.CSV)
        assert [r.fields for r in records] == [{"a": "1", "b": "2"}]
        assert len(issues) == 1
        assert issues[0].location.source_file == "dup.csv"
