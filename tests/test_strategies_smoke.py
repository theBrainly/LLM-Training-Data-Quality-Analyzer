"""Smoke tests for the shared Hypothesis strategies (task 1.4).

These tests do not validate any product requirement; they confirm that each
shared strategy in ``tests/strategies.py`` produces well-formed values so the
property tests built on top of them (in later tasks) start from a sound base.
Where cheap, they cross-check a strategy against the already-implemented
Pretty_Printer (JSON/JSONL) to confirm the representable/unrepresentable
parameterization behaves as intended.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from analyzer.models import Record, SupportedFormat, Value
from analyzer.pretty_printer import PrettyPrinter
from tests.strategies import (
    ALL_FORMATS,
    PII_CATEGORIES,
    PIISample,
    StubToxicityModel,
    field_values,
    format_sources,
    invalid_thresholds,
    pii_free_text,
    pii_text,
    record_lists,
    records,
    thresholds,
)

_SETTINGS = settings(max_examples=50, deadline=None)


def _is_canonical_value(value) -> bool:
    """Structural check that ``value`` inhabits the canonical ``Value`` type."""
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (str, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_canonical_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_canonical_value(val)
            for key, val in value.items()
        )
    return False


class TestValueAndRecordStrategies:
    @_SETTINGS
    @given(field_values())
    def test_field_values_are_canonical(self, value: Value):
        assert _is_canonical_value(value)

    @_SETTINGS
    @given(records())
    def test_records_are_well_formed(self, record: Record):
        assert isinstance(record, Record)
        assert isinstance(record.fields, dict)
        assert all(isinstance(name, str) for name in record.fields)
        assert all(_is_canonical_value(v) for v in record.fields.values())
        assert record.location.source_file

    @_SETTINGS
    @given(record_lists())
    def test_record_lists_are_lists_of_records(self, recs):
        assert isinstance(recs, list)
        assert all(isinstance(r, Record) for r in recs)


class TestFormatParameterization:
    @_SETTINGS
    @given(record_lists(fmt=SupportedFormat.JSON, representable=True))
    def test_representable_json_records_serialize(self, recs):
        result = PrettyPrinter().print(recs, SupportedFormat.JSON)
        assert result.error is None
        assert result.text is not None

    @_SETTINGS
    @given(record_lists(fmt=SupportedFormat.JSONL, representable=True))
    def test_representable_jsonl_records_serialize(self, recs):
        result = PrettyPrinter().print(recs, SupportedFormat.JSONL)
        assert result.error is None

    @_SETTINGS
    @given(record_lists(fmt=SupportedFormat.JSON, representable=False))
    def test_unrepresentable_json_records_fail_to_serialize(self, recs):
        # At least one record carries a non-finite float, so printing halts.
        result = PrettyPrinter().print(recs, SupportedFormat.JSON)
        assert result.text is None
        assert result.error is not None

    @_SETTINGS
    @given(record_lists(fmt=SupportedFormat.CSV, representable=False))
    def test_unrepresentable_csv_records_contain_nested_value(self, recs):
        # CSV's unrepresentable case is a nested container in some field.
        has_nested = any(
            isinstance(v, (list, dict))
            for rec in recs
            for v in rec.fields.values()
        )
        assert has_nested

    @_SETTINGS
    @given(format_sources())
    def test_format_sources_yields_supported_format(self, fmt):
        assert fmt in ALL_FORMATS
        assert isinstance(fmt, SupportedFormat)


class TestThresholdStrategies:
    @_SETTINGS
    @given(thresholds())
    def test_thresholds_in_unit_interval(self, value: float):
        assert 0.0 <= value <= 1.0

    @_SETTINGS
    @given(invalid_thresholds())
    def test_invalid_thresholds_out_of_range(self, value: float):
        assert value < 0.0 or value > 1.0


class TestPiiStrategies:
    @_SETTINGS
    @given(pii_text())
    def test_pii_text_spans_are_accurate(self, sample: PIISample):
        assert isinstance(sample, PIISample)
        assert len(sample.occurrences) >= 1
        for occ in sample.occurrences:
            assert occ.category in PII_CATEGORIES
            assert sample.text[occ.start : occ.end] == occ.value
            assert occ.start < occ.end

    @_SETTINGS
    @given(pii_free_text())
    def test_pii_free_text_has_no_obvious_pii(self, text: str):
        assert "@" not in text
        assert not any(ch.isdigit() for ch in text)


class TestStubToxicityModel:
    def test_fixed_score_is_returned(self):
        model = StubToxicityModel(score=0.42)
        assert model.score("anything") == 0.42
        assert model.calls == ["anything"]

    def test_scores_are_clamped_by_default(self):
        assert StubToxicityModel(score=2.0).score("x") == 1.0
        assert StubToxicityModel(score=-1.0).score("x") == 0.0

    def test_clamp_can_be_disabled(self):
        assert StubToxicityModel(score=2.0, clamp=False).score("x") == 2.0

    def test_per_input_mapping(self):
        model = StubToxicityModel(scores={"a": 0.1, "b": 0.9})
        assert model.score("a") == 0.1
        assert model.score("b") == 0.9

    def test_score_fn_is_used(self):
        model = StubToxicityModel(score_fn=lambda t: len(t) / 10.0)
        assert model.score("abcde") == 0.5

    def test_fail_raises(self):
        model = StubToxicityModel(fail=True)
        try:
            model.score("x")
        except RuntimeError:
            pass
        else:  # pragma: no cover - explicit failure if no exception
            raise AssertionError("expected RuntimeError from failing stub")
