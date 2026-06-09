"""Unit tests for the canonical value model and record/dataset types."""

from analyzer.models import (
    Dataset,
    Record,
    RecordLocation,
    Span,
    SupportedFormat,
    fields_equivalent,
    records_equivalent,
    values_equal,
)


def _loc(name: str = "f.jsonl") -> RecordLocation:
    return RecordLocation(source_file=name, line_number=1)


class TestSupportedFormat:
    def test_format_values(self):
        assert SupportedFormat.JSON.value == "json"
        assert SupportedFormat.JSONL.value == "jsonl"
        assert SupportedFormat.CSV.value == "csv"
        assert SupportedFormat.PARQUET.value == "parquet"


class TestRecordLocation:
    def test_defaults_are_none(self):
        loc = RecordLocation(source_file="a.json")
        assert loc.source_file == "a.json"
        assert loc.line_number is None
        assert loc.array_index is None
        assert loc.row_group is None
        assert loc.row_index is None

    def test_is_frozen_and_hashable(self):
        loc = RecordLocation(source_file="a.parquet", row_group=0, row_index=3)
        # frozen dataclasses are hashable
        assert hash(loc) == hash(
            RecordLocation(source_file="a.parquet", row_group=0, row_index=3)
        )


class TestSpan:
    def test_span_offsets(self):
        span = Span(start=2, end=7)
        assert span.start == 2
        assert span.end == 7

    def test_span_equality(self):
        assert Span(0, 1) == Span(0, 1)
        assert Span(0, 1) != Span(0, 2)


class TestDataset:
    def test_defaults(self):
        ds = Dataset(records=[])
        assert ds.records == []
        assert ds.source_files == []
        assert ds.skipped_files == []

    def test_holds_records_in_order(self):
        r1 = Record(fields={"text": "a"}, location=_loc())
        r2 = Record(fields={"text": "b"}, location=_loc())
        ds = Dataset(records=[r1, r2], source_files=["f.jsonl"], skipped_files=["x.txt"])
        assert ds.records == [r1, r2]
        assert ds.skipped_files == ["x.txt"]


class TestValuesEqual:
    def test_scalars(self):
        assert values_equal("a", "a")
        assert not values_equal("a", "b")
        assert values_equal(1, 1)
        assert values_equal(1.5, 1.5)

    def test_none(self):
        assert values_equal(None, None)
        assert not values_equal(None, 0)
        assert not values_equal(0, None)

    def test_bool_distinct_from_int(self):
        # bool must not be conflated with the equivalent int value.
        assert not values_equal(True, 1)
        assert not values_equal(False, 0)
        assert values_equal(True, True)
        assert values_equal(False, False)

    def test_int_float_distinct(self):
        assert not values_equal(1, 1.0)

    def test_nested_list(self):
        assert values_equal([1, "a", [True]], [1, "a", [True]])
        assert not values_equal([1, 2], [1, 2, 3])
        assert not values_equal([1, 2], [2, 1])

    def test_nested_dict(self):
        assert values_equal({"a": 1, "b": [2]}, {"b": [2], "a": 1})
        assert not values_equal({"a": 1}, {"a": 2})
        assert not values_equal({"a": 1}, {"a": 1, "b": 2})

    def test_container_vs_scalar(self):
        assert not values_equal([1], 1)
        assert not values_equal({"a": 1}, "a")
        assert not values_equal([1], {"0": 1})


class TestFieldsEquivalent:
    def test_identical(self):
        assert fields_equivalent({"a": 1, "b": "x"}, {"a": 1, "b": "x"})

    def test_key_order_irrelevant(self):
        assert fields_equivalent({"a": 1, "b": 2}, {"b": 2, "a": 1})

    def test_different_key_sets(self):
        assert not fields_equivalent({"a": 1}, {"a": 1, "b": 2})
        assert not fields_equivalent({"a": 1, "b": 2}, {"a": 1})

    def test_different_values(self):
        assert not fields_equivalent({"a": 1}, {"a": 2})


class TestRecordEquivalence:
    def test_equivalent_ignores_location_and_metadata(self):
        r1 = Record(
            fields={"text": "hello", "n": 3},
            location=RecordLocation(source_file="a.json", array_index=0),
            metadata={"src": "a"},
        )
        r2 = Record(
            fields={"n": 3, "text": "hello"},
            location=RecordLocation(source_file="b.jsonl", line_number=9),
            metadata={"src": "b"},
        )
        assert r1.equivalent(r2)
        assert r2.equivalent(r1)

    def test_not_equivalent_when_fields_differ(self):
        r1 = Record(fields={"text": "hello"}, location=_loc())
        r2 = Record(fields={"text": "world"}, location=_loc())
        assert not r1.equivalent(r2)

    def test_not_equivalent_when_keys_differ(self):
        r1 = Record(fields={"text": "hello"}, location=_loc())
        r2 = Record(fields={"text": "hello", "extra": 1}, location=_loc())
        assert not r1.equivalent(r2)


class TestRecordsEquivalent:
    def test_same_order_same_fields(self):
        left = [
            Record(fields={"t": "a"}, location=_loc()),
            Record(fields={"t": "b"}, location=_loc()),
        ]
        right = [
            Record(fields={"t": "a"}, location=RecordLocation(source_file="z")),
            Record(fields={"t": "b"}, location=RecordLocation(source_file="z")),
        ]
        assert records_equivalent(left, right)

    def test_different_length(self):
        left = [Record(fields={"t": "a"}, location=_loc())]
        right = []
        assert not records_equivalent(left, right)

    def test_order_matters(self):
        left = [
            Record(fields={"t": "a"}, location=_loc()),
            Record(fields={"t": "b"}, location=_loc()),
        ]
        right = [
            Record(fields={"t": "b"}, location=_loc()),
            Record(fields={"t": "a"}, location=_loc()),
        ]
        assert not records_equivalent(left, right)
