"""Unit tests for the Pretty_Printer's CSV and Parquet serialization and
empty-list handling (Requirements 3.1, 3.3, 3.4).

These tests focus on the CSV (scalar cells; nested values unrepresentable) and
Parquet (Arrow table) paths added in task 6.2, plus the valid empty-list
representation each format must produce without error.
"""

import csv
import io

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.errors import UnrepresentableValueError
from analyzer.models import Record, RecordLocation, SupportedFormat
from analyzer.pretty_printer import PrettyPrinter, PrintResult


def _loc(name: str = "f.csv") -> RecordLocation:
    return RecordLocation(source_file=name, line_number=1)


def _rec(fields: dict) -> Record:
    return Record(fields=fields, location=_loc())


def _read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _read_parquet(text: str) -> list[dict]:
    data = text.encode("latin-1")
    return pq.read_table(pa.BufferReader(data)).to_pylist()


class TestCsvSerialization:
    def test_writes_header_and_one_row_per_record_in_order(self):
        records = [
            _rec({"id": "1", "text": "first"}),
            _rec({"id": "2", "text": "second"}),
            _rec({"id": "3", "text": "third"}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.CSV)

        assert result.error is None
        assert result.text is not None
        rows = _read_csv(result.text)
        assert rows == [
            {"id": "1", "text": "first"},
            {"id": "2", "text": "second"},
            {"id": "3", "text": "third"},
        ]

    def test_header_preserves_field_order(self):
        result = PrettyPrinter().print([_rec({"b": "1", "a": "2", "c": "3"})], SupportedFormat.CSV)

        assert result.text is not None
        header = result.text.splitlines()[0]
        assert header == "b,a,c"

    def test_scalar_values_render_as_cells(self):
        records = [_rec({"s": "hi", "i": 42, "f": 3.5, "b": True, "n": None})]
        result = PrettyPrinter().print(records, SupportedFormat.CSV)

        assert result.error is None
        rows = _read_csv(result.text)
        # CSV cells are textual; None becomes an empty cell.
        assert rows == [{"s": "hi", "i": "42", "f": "3.5", "b": "True", "n": ""}]

    def test_union_of_field_names_across_records(self):
        records = [
            _rec({"a": "1", "b": "2"}),
            _rec({"a": "3", "c": "4"}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.CSV)

        assert result.error is None
        assert result.text.splitlines()[0] == "a,b,c"
        rows = _read_csv(result.text)
        # Missing columns become empty cells.
        assert rows == [
            {"a": "1", "b": "2", "c": ""},
            {"a": "3", "b": "", "c": "4"},
        ]

    def test_nested_list_value_is_unrepresentable_with_located_error(self):
        records = [
            _rec({"ok": "value"}),
            _rec({"good": "1", "bad": [1, 2, 3]}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.CSV)

        assert result.text is None
        assert isinstance(result.error, UnrepresentableValueError)
        assert result.error.record_index == 1
        assert result.error.field_name == "bad"
        assert result.error.fmt == "csv"

    def test_nested_dict_value_is_unrepresentable(self):
        records = [_rec({"meta": {"k": "v"}})]
        result = PrettyPrinter().print(records, SupportedFormat.CSV)

        assert result.text is None
        assert isinstance(result.error, UnrepresentableValueError)
        assert result.error.record_index == 0
        assert result.error.field_name == "meta"

    def test_empty_list_yields_empty_string(self):
        result = PrettyPrinter().print([], SupportedFormat.CSV)
        assert result.error is None
        assert result.text == ""


class TestParquetSerialization:
    def test_round_trips_records_in_order(self):
        records = [
            _rec({"id": 1, "text": "first"}),
            _rec({"id": 2, "text": "second"}),
        ]
        result = PrettyPrinter().print(records, SupportedFormat.PARQUET)

        assert result.error is None
        assert result.text is not None
        assert _read_parquet(result.text) == [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
        ]

    def test_nested_values_are_representable(self):
        fields = {
            "s": "hello",
            "i": 7,
            "f": 1.5,
            "b": False,
            "list": [1, 2, 3],
            "obj": {"nested": 1},
        }
        result = PrettyPrinter().print([_rec(fields)], SupportedFormat.PARQUET)

        assert result.error is None
        assert _read_parquet(result.text) == [fields]

    def test_empty_list_yields_valid_empty_table(self):
        result = PrettyPrinter().print([], SupportedFormat.PARQUET)

        assert result.error is None
        assert result.text is not None
        table = pq.read_table(pa.BufferReader(result.text.encode("latin-1")))
        assert table.num_rows == 0
        assert _read_parquet(result.text) == []


class TestReturnType:
    def test_csv_returns_print_result(self):
        result = PrettyPrinter().print([_rec({"x": "1"})], SupportedFormat.CSV)
        assert isinstance(result, PrintResult)

    def test_parquet_returns_print_result(self):
        result = PrettyPrinter().print([_rec({"x": 1})], SupportedFormat.PARQUET)
        assert isinstance(result, PrintResult)
