"""Unit tests for the JSONL parser strategy (Task 3.2).

Covers Requirement 2.2 (one Record per non-whitespace line, whitespace-only
lines ignored) and Requirement 2.5 (located Quality_Issue for unparseable
lines, parsing continues).

JSONL payload convention: a unit carries the entire file text as its payload
(mirroring the whole-file JSON convention). The strategy splits the text into
physical lines, tracks 1-based line numbers across every line, ignores
whitespace-only lines, and parses each remaining line as a standalone JSON
object.
"""

import json

from analyzer.models import IssueCategory, SupportedFormat
from analyzer.parsers import Parser, RawRecordUnit


def _jsonl_unit(source_file: str, *objects) -> RawRecordUnit:
    """Build a JSONL unit whose payload is one JSON object per line."""
    text = "\n".join(json.dumps(obj) for obj in objects)
    return RawRecordUnit(source_file=source_file, payload=text)


class TestDispatch:
    def test_jsonl_is_registered(self):
        assert SupportedFormat.JSONL in Parser._STRATEGIES


class TestJsonlParsing:
    def test_one_record_per_non_whitespace_line_in_order(self):
        # Requirement 2.2: one Record per non-empty line, in order.
        objects = [{"text": "a"}, {"text": "b", "n": 3}, {"text": "c"}]
        records, issues = Parser().parse(
            [_jsonl_unit("data.jsonl", *objects)], SupportedFormat.JSONL
        )

        assert issues == []
        assert [r.fields for r in records] == objects
        assert [r.location.line_number for r in records] == [1, 2, 3]
        assert all(r.location.source_file == "data.jsonl" for r in records)

    def test_whitespace_only_lines_are_ignored(self):
        # Requirement 2.2: lines containing only whitespace are ignored and
        # produce neither a Record nor an issue. Line numbers still count the
        # blank lines so locations stay accurate.
        text = (
            '{"text": "a"}\n'
            "\n"
            "   \n"
            "\t \n"
            '{"text": "b"}\n'
        )
        unit = RawRecordUnit(source_file="gaps.jsonl", payload=text)
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)

        assert issues == []
        assert [r.fields for r in records] == [{"text": "a"}, {"text": "b"}]
        # 'a' is line 1; 'b' is line 5 (lines 2-4 are blank/whitespace).
        assert [r.location.line_number for r in records] == [1, 5]

    def test_trailing_and_leading_blank_lines_ignored(self):
        text = '\n\n{"text": "only"}\n\n'
        unit = RawRecordUnit(source_file="pad.jsonl", payload=text)
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)

        assert issues == []
        assert [r.fields for r in records] == [{"text": "only"}]
        assert records[0].location.line_number == 3

    def test_empty_payload_yields_no_records_and_no_issues(self):
        unit = RawRecordUnit(source_file="empty.jsonl", payload="")
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)
        assert records == []
        assert issues == []

    def test_whitespace_only_payload_yields_nothing(self):
        unit = RawRecordUnit(source_file="ws.jsonl", payload="   \n\t\n  ")
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)
        assert records == []
        assert issues == []

    def test_canonical_value_types_preserved(self):
        obj = {
            "s": "x",
            "i": 1,
            "f": 1.5,
            "b": True,
            "null": None,
            "list": [1, 2],
            "obj": {"k": "v"},
        }
        records, issues = Parser().parse(
            [_jsonl_unit("v.jsonl", obj)], SupportedFormat.JSONL
        )
        assert issues == []
        assert records[0].fields == obj

    def test_malformed_line_is_located_issue_and_parsing_continues(self):
        # Requirement 2.5: an unparseable line becomes a located issue and the
        # surrounding valid lines are still parsed.
        text = '{"text": "ok1"}\n{not valid json\n{"text": "ok2"}\n'
        unit = RawRecordUnit(source_file="mixed.jsonl", payload=text)
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)

        assert [r.fields for r in records] == [{"text": "ok1"}, {"text": "ok2"}]
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        assert issue.location is not None
        assert issue.location.line_number == 2
        assert issue.location.source_file == "mixed.jsonl"

    def test_non_object_line_is_located_issue(self):
        # A valid JSON value that is not an object cannot map to record fields.
        text = '{"text": "ok"}\n42\n[1, 2]\n"str"\n'
        unit = RawRecordUnit(source_file="scalar.jsonl", payload=text)
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)

        assert [r.fields for r in records] == [{"text": "ok"}]
        assert sorted(i.location.line_number for i in issues) == [2, 3, 4]
        assert all(i.category == IssueCategory.PARSE_ERROR for i in issues)

    def test_line_numbers_account_for_blank_lines_among_bad_lines(self):
        # Blank lines between a good and a bad line keep numbering accurate.
        text = '{"ok": 1}\n\nbad line\n'
        unit = RawRecordUnit(source_file="n.jsonl", payload=text)
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)

        assert [r.fields for r in records] == [{"ok": 1}]
        assert records[0].location.line_number == 1
        assert len(issues) == 1
        assert issues[0].location.line_number == 3

    def test_bytes_payload_is_decoded(self):
        text = '{"t": "a"}\n{"t": "b"}\n'.encode("utf-8")
        unit = RawRecordUnit(source_file="b.jsonl", payload=text)
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)
        assert issues == []
        assert [r.fields for r in records] == [{"t": "a"}, {"t": "b"}]

    def test_invalid_bytes_payload_is_file_level_issue(self):
        unit = RawRecordUnit(source_file="bad.jsonl", payload=b"\xff\xfe")
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)
        assert records == []
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.PARSE_ERROR
        assert issues[0].location.source_file == "bad.jsonl"
        assert issues[0].location.line_number is None

    def test_mapping_payload_is_single_record(self):
        unit = RawRecordUnit(source_file="map.jsonl", payload={"text": "x"})
        records, issues = Parser().parse([unit], SupportedFormat.JSONL)
        assert issues == []
        assert [r.fields for r in records] == [{"text": "x"}]
        assert records[0].location.line_number == 1

    def test_multiple_units_track_lines_independently_per_file(self):
        units = [
            _jsonl_unit("a.jsonl", {"t": "a0"}, {"t": "a1"}),
            _jsonl_unit("b.jsonl", {"t": "b0"}),
        ]
        records, issues = Parser().parse(units, SupportedFormat.JSONL)
        assert issues == []
        assert [(r.location.source_file, r.location.line_number) for r in records] == [
            ("a.jsonl", 1),
            ("a.jsonl", 2),
            ("b.jsonl", 1),
        ]
