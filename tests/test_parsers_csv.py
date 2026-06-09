"""Unit tests for the CSV parser strategy (Task 3.3).

Covers:
* Requirement 2.3 - first row is the header; each subsequent row becomes a
  Record mapping every header name to the cell at that column position.
* Requirement 2.5 - a row that cannot be parsed (column-count mismatch)
  becomes a located Quality_Issue and parsing continues.
* Requirement 2.7 - a missing header row or duplicate header field names
  produce a file-level Quality_Issue and no Records for that file.

CSV payload convention: a unit carries the entire file text as its payload
(mirroring the whole-file JSON/JSONL convention). CSV cells are textual, so
every parsed field value is a string.
"""

from analyzer.models import IssueCategory, SupportedFormat
from analyzer.parsers import Parser, RawRecordUnit


def _csv_unit(source_file: str, text: str) -> RawRecordUnit:
    return RawRecordUnit(source_file=source_file, payload=text)


class TestDispatch:
    def test_csv_is_registered(self):
        assert SupportedFormat.CSV in Parser._STRATEGIES


class TestHeaderMapping:
    def test_maps_cells_to_header_names_by_position(self):
        # Requirement 2.3
        text = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        records, issues = Parser().parse(
            [_csv_unit("people.csv", text)], SupportedFormat.CSV
        )

        assert issues == []
        assert [r.fields for r in records] == [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]

    def test_record_order_and_line_numbers_preserved(self):
        text = "a,b\n1,2\n3,4\n5,6\n"
        records, issues = Parser().parse(
            [_csv_unit("nums.csv", text)], SupportedFormat.CSV
        )

        assert issues == []
        # Header is line 1; data rows start at line 2.
        assert [r.location.line_number for r in records] == [2, 3, 4]
        assert all(r.location.source_file == "nums.csv" for r in records)

    def test_cells_remain_strings(self):
        text = "n,flag\n42,true\n"
        records, _ = Parser().parse(
            [_csv_unit("t.csv", text)], SupportedFormat.CSV
        )
        assert records[0].fields == {"n": "42", "flag": "true"}

    def test_quoted_field_with_comma_and_newline(self):
        text = 'name,note\n"Smith, J.","line one\nline two"\n'
        records, issues = Parser().parse(
            [_csv_unit("q.csv", text)], SupportedFormat.CSV
        )
        assert issues == []
        assert records[0].fields == {
            "name": "Smith, J.",
            "note": "line one\nline two",
        }

    def test_header_only_file_yields_no_records_and_no_issues(self):
        text = "a,b,c\n"
        records, issues = Parser().parse(
            [_csv_unit("hdr.csv", text)], SupportedFormat.CSV
        )
        assert records == []
        assert issues == []

    def test_empty_cells_preserved_as_empty_strings(self):
        text = "a,b,c\n1,,3\n"
        records, issues = Parser().parse(
            [_csv_unit("e.csv", text)], SupportedFormat.CSV
        )
        assert issues == []
        assert records[0].fields == {"a": "1", "b": "", "c": "3"}

    def test_blank_lines_between_rows_are_ignored(self):
        text = "a,b\n1,2\n\n3,4\n"
        records, issues = Parser().parse(
            [_csv_unit("blank.csv", text)], SupportedFormat.CSV
        )
        assert issues == []
        assert [r.fields for r in records] == [
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]

    def test_bytes_payload_is_decoded(self):
        text = "a,b\n1,2\n".encode("utf-8")
        records, issues = Parser().parse(
            [RawRecordUnit(source_file="b.csv", payload=text)], SupportedFormat.CSV
        )
        assert issues == []
        assert [r.fields for r in records] == [{"a": "1", "b": "2"}]


class TestMissingHeader:
    def test_empty_file_is_missing_header_issue_with_no_records(self):
        # Requirement 2.7: missing header row -> file-level issue, no records.
        records, issues = Parser().parse(
            [_csv_unit("empty.csv", "")], SupportedFormat.CSV
        )
        assert records == []
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        assert issue.location is not None
        assert issue.location.source_file == "empty.csv"
        assert issue.location.line_number is None


class TestDuplicateHeader:
    def test_duplicate_header_names_suppress_records(self):
        # Requirement 2.7: duplicate header field names -> file-level issue,
        # no records, and the offending column is identified.
        text = "a,b,a\n1,2,3\n4,5,6\n"
        records, issues = Parser().parse(
            [_csv_unit("dup.csv", text)], SupportedFormat.CSV
        )
        assert records == []
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        assert issue.field_name == "a"
        assert issue.location.source_file == "dup.csv"


class TestFailSoftRows:
    def test_column_count_mismatch_is_located_issue_and_parsing_continues(self):
        # Requirement 2.5: a malformed row is skipped with a located issue and
        # the surrounding valid rows are still parsed.
        text = "a,b\n1,2\n3\n4,5\n"
        records, issues = Parser().parse(
            [_csv_unit("mixed.csv", text)], SupportedFormat.CSV
        )

        assert [r.fields for r in records] == [
            {"a": "1", "b": "2"},
            {"a": "4", "b": "5"},
        ]
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        assert issue.location.line_number == 3
        assert issue.location.source_file == "mixed.csv"

    def test_extra_column_row_is_located_issue(self):
        text = "a,b\n1,2,3\n4,5\n"
        records, issues = Parser().parse(
            [_csv_unit("extra.csv", text)], SupportedFormat.CSV
        )
        assert [r.fields for r in records] == [{"a": "4", "b": "5"}]
        assert [i.location.line_number for i in issues] == [2]


class TestMultipleUnits:
    def test_units_tracked_independently_per_file(self):
        units = [
            _csv_unit("a.csv", "x,y\n1,2\n"),
            _csv_unit("b.csv", "x,y\n3,4\n5,6\n"),
        ]
        records, issues = Parser().parse(units, SupportedFormat.CSV)
        assert issues == []
        assert [
            (r.location.source_file, r.location.line_number) for r in records
        ] == [
            ("a.csv", 2),
            ("b.csv", 2),
            ("b.csv", 3),
        ]

    def test_one_bad_file_does_not_block_others(self):
        units = [
            _csv_unit("dup.csv", "a,a\n1,2\n"),
            _csv_unit("ok.csv", "a,b\n1,2\n"),
        ]
        records, issues = Parser().parse(units, SupportedFormat.CSV)
        assert [r.fields for r in records] == [{"a": "1", "b": "2"}]
        assert len(issues) == 1
        assert issues[0].location.source_file == "dup.csv"
