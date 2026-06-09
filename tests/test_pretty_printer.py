"""Unit tests for the Pretty_Printer's JSON/JSONL serialization and
representability checks (Requirements 3.1, 3.3).

CSV/Parquet serialization and empty-list handling are covered separately;
these tests focus on the JSON and JSONL paths implemented in task 6.1.
"""

import json

from analyzer.errors import UnrepresentableValueError
from analyzer.models import Record, RecordLocation, SupportedFormat
from analyzer.pretty_printer import PrettyPrinter, PrintResult


def _loc(name: str = "f.jsonl") -> RecordLocation:
    return RecordLocation(source_file=name, line_number=1)


def _rec(fields: dict) -> Record:
    return Record(fields=fields, location=_loc())


class TestJsonSerialization:
    def test_serializes_array_preserving_record_order(self):
        records = [
            _rec({"text": "first"}),
            _rec({"text": "second"}),
            _rec({"text": "third"}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.JSON)

        assert result.error is None
        assert result.text is not None
        parsed = json.loads(result.text)
        assert parsed == [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ]

    def test_preserves_field_order_within_record(self):
        records = [_rec({"b": 1, "a": 2, "c": 3})]
        result = PrettyPrinter().print(records, SupportedFormat.JSON)

        # The serialized object keys appear in insertion order.
        assert result.text is not None
        assert result.text.index('"b"') < result.text.index('"a"') < result.text.index('"c"')

    def test_round_trips_canonical_value_types(self):
        fields = {
            "s": "hello",
            "i": 42,
            "f": 3.5,
            "b": True,
            "n": None,
            "list": [1, "two", [True, None]],
            "obj": {"nested": {"x": 1}, "arr": [1.0, 2.0]},
        }
        records = [_rec(fields)]
        result = PrettyPrinter().print(records, SupportedFormat.JSON)

        assert result.error is None
        assert json.loads(result.text) == [fields]

    def test_empty_list_yields_empty_array(self):
        result = PrettyPrinter().print([], SupportedFormat.JSON)
        assert result.error is None
        assert json.loads(result.text) == []


class TestJsonlSerialization:
    def test_one_object_per_line_in_order(self):
        records = [
            _rec({"text": "a", "n": 1}),
            _rec({"text": "b", "n": 2}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.JSONL)

        assert result.error is None
        lines = result.text.split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"text": "a", "n": 1}
        assert json.loads(lines[1]) == {"text": "b", "n": 2}

    def test_empty_list_yields_empty_string(self):
        result = PrettyPrinter().print([], SupportedFormat.JSONL)
        assert result.error is None
        assert result.text == ""


class TestRepresentabilityErrors:
    def test_non_finite_float_halts_with_located_error_json(self):
        records = [
            _rec({"ok": "value"}),
            _rec({"good": 1, "bad": float("nan")}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.JSON)

        assert result.text is None
        assert isinstance(result.error, UnrepresentableValueError)
        assert result.error.record_index == 1
        assert result.error.field_name == "bad"
        assert result.error.fmt == "json"

    def test_infinity_in_nested_container_is_unrepresentable(self):
        records = [_rec({"vals": [1, 2, float("inf")]})]
        result = PrettyPrinter().print(records, SupportedFormat.JSONL)

        assert result.text is None
        assert isinstance(result.error, UnrepresentableValueError)
        assert result.error.record_index == 0
        assert result.error.field_name == "vals"

    def test_reports_first_offending_record(self):
        records = [
            _rec({"a": float("nan")}),
            _rec({"b": float("inf")}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.JSON)

        assert result.text is None
        assert result.error.record_index == 0
        assert result.error.field_name == "a"

    def test_returns_print_result_type(self):
        result = PrettyPrinter().print([_rec({"x": 1})], SupportedFormat.JSON)
        assert isinstance(result, PrintResult)
