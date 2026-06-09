"""Unit tests for the parser dispatch and the JSON array parser (Task 3.1).

Covers Requirements 2.1 (one Record per top-level array element) and 2.5
(located Quality_Issue for unparseable elements, parsing continues).
"""

import json

from analyzer.models import IssueCategory, SupportedFormat
from analyzer.parsers import Parser, RawRecordUnit


def _json_unit(source_file: str, value) -> RawRecordUnit:
    """Build a JSON unit whose payload is the serialized ``value``."""
    return RawRecordUnit(source_file=source_file, payload=json.dumps(value))


class TestDispatch:
    def test_all_supported_formats_are_registered(self):
        # Every SupportedFormat now has a parse strategy, including Parquet
        # (registered by Task 3.4).
        parser = Parser()
        for fmt in SupportedFormat:
            assert fmt in parser._STRATEGIES

    def test_json_is_registered(self):
        assert SupportedFormat.JSON in Parser._STRATEGIES


class TestJsonArrayParsing:
    def test_one_record_per_array_element_in_order(self):
        # Requirement 2.1: exactly one Record per top-level array element.
        elements = [{"text": "a"}, {"text": "b", "n": 3}, {"text": "c"}]
        records, issues = Parser().parse(
            [_json_unit("data.json", elements)], SupportedFormat.JSON
        )

        assert issues == []
        assert len(records) == len(elements)
        assert [r.fields for r in records] == elements
        # Locations carry the array index and source file.
        assert [r.location.array_index for r in records] == [0, 1, 2]
        assert all(r.location.source_file == "data.json" for r in records)

    def test_empty_array_yields_no_records_and_no_issues(self):
        records, issues = Parser().parse(
            [_json_unit("empty.json", [])], SupportedFormat.JSON
        )
        assert records == []
        assert issues == []

    def test_canonical_value_types_preserved(self):
        element = {
            "s": "x",
            "i": 1,
            "f": 1.5,
            "b": True,
            "null": None,
            "list": [1, 2],
            "obj": {"k": "v"},
        }
        records, issues = Parser().parse(
            [_json_unit("v.json", [element])], SupportedFormat.JSON
        )
        assert issues == []
        assert records[0].fields == element

    def test_non_object_element_is_located_issue_and_parsing_continues(self):
        # Requirement 2.5: an element that cannot map to fields becomes a
        # located issue; surrounding valid elements are still parsed.
        elements = [{"text": "ok1"}, 42, {"text": "ok2"}]
        records, issues = Parser().parse(
            [_json_unit("mixed.json", elements)], SupportedFormat.JSON
        )

        assert [r.fields for r in records] == [{"text": "ok1"}, {"text": "ok2"}]
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category == IssueCategory.PARSE_ERROR
        assert issue.location is not None
        assert issue.location.array_index == 1
        assert issue.location.source_file == "mixed.json"

    def test_multiple_bad_elements_each_get_an_issue(self):
        elements = ["bad", {"ok": 1}, ["also-bad"], 7]
        records, issues = Parser().parse(
            [_json_unit("m.json", elements)], SupportedFormat.JSON
        )
        assert [r.fields for r in records] == [{"ok": 1}]
        assert sorted(i.location.array_index for i in issues) == [0, 2, 3]
        assert all(i.category == IssueCategory.PARSE_ERROR for i in issues)

    def test_malformed_json_payload_is_file_level_issue(self):
        unit = RawRecordUnit(source_file="broken.json", payload="{not valid json")
        records, issues = Parser().parse([unit], SupportedFormat.JSON)
        assert records == []
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.PARSE_ERROR
        assert issues[0].location.source_file == "broken.json"
        assert issues[0].location.array_index is None

    def test_non_array_top_level_is_issue(self):
        # A top-level object (not an array) cannot be parsed under Requirement 2.1.
        records, issues = Parser().parse(
            [_json_unit("obj.json", {"text": "a"})], SupportedFormat.JSON
        )
        assert records == []
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.PARSE_ERROR
        assert issues[0].location.array_index is None

    def test_bytes_payload_is_decoded(self):
        unit = RawRecordUnit(
            source_file="b.json", payload=json.dumps([{"t": "a"}]).encode("utf-8")
        )
        records, issues = Parser().parse([unit], SupportedFormat.JSON)
        assert issues == []
        assert [r.fields for r in records] == [{"t": "a"}]

    def test_multiple_units_index_independently_per_file(self):
        units = [
            _json_unit("a.json", [{"t": "a0"}, {"t": "a1"}]),
            _json_unit("b.json", [{"t": "b0"}]),
        ]
        records, issues = Parser().parse(units, SupportedFormat.JSON)
        assert issues == []
        assert [(r.location.source_file, r.location.array_index) for r in records] == [
            ("a.json", 0),
            ("a.json", 1),
            ("b.json", 0),
        ]
