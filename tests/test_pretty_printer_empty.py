"""Unit tests for empty-list serialization per format (Requirement 3.4).

Serializing an empty record list must yield a *valid empty document* for every
Supported_Format and must never return an error. Each test asserts both the
expected empty representation and that parsing the empty document back yields
zero records, so the empty output is genuinely well-formed for its format.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.models import SupportedFormat
from analyzer.parsers import Parser, RawRecordUnit
from analyzer.pretty_printer import PrettyPrinter


def _print_empty(fmt: SupportedFormat):
    return PrettyPrinter().print([], fmt)


class TestEmptyListSerialization:
    def test_json_empty_list_is_empty_array(self):
        result = _print_empty(SupportedFormat.JSON)
        assert result.error is None
        assert result.text == "[]"

    def test_jsonl_empty_list_is_empty_string(self):
        result = _print_empty(SupportedFormat.JSONL)
        assert result.error is None
        assert result.text == ""

    def test_csv_empty_list_is_empty_string(self):
        result = _print_empty(SupportedFormat.CSV)
        assert result.error is None
        assert result.text == ""

    def test_parquet_empty_list_is_valid_empty_table(self):
        result = _print_empty(SupportedFormat.PARQUET)
        assert result.error is None
        assert result.text is not None
        table = pq.read_table(pa.BufferReader(result.text.encode("latin-1")))
        assert table.num_rows == 0


class TestEmptyDocumentParsesToZeroRecords:
    """The empty document each format produces parses back to zero records."""

    def test_json_empty_array_parses_to_no_records(self):
        result = _print_empty(SupportedFormat.JSON)
        records, _ = Parser().parse(
            [RawRecordUnit(source_file="empty.json", payload=result.text)],
            SupportedFormat.JSON,
        )
        assert records == []

    def test_jsonl_empty_parses_to_no_records(self):
        result = _print_empty(SupportedFormat.JSONL)
        records, _ = Parser().parse(
            [RawRecordUnit(source_file="empty.jsonl", payload=result.text)],
            SupportedFormat.JSONL,
        )
        assert records == []

    def test_parquet_empty_parses_to_no_records(self):
        result = _print_empty(SupportedFormat.PARQUET)
        records, _ = Parser().parse(
            [
                RawRecordUnit(
                    source_file="empty.parquet",
                    payload=result.text.encode("latin-1"),
                )
            ],
            SupportedFormat.PARQUET,
        )
        assert records == []
