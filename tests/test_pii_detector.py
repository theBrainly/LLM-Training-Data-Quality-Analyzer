"""Unit tests for per-category PII detection (Requirements 6.1-6.4)."""

import pytest

from analyzer.detectors.pii import (
    PIIDetector,
    find_pii_in_text,
    luhn_valid,
)
from analyzer.models import IssueCategory, Record, RecordLocation


def _record(text: str, field: str = "text") -> Record:
    return Record(
        fields={field: text},
        location=RecordLocation(source_file="data.jsonl", line_number=1),
    )


# Known Luhn-valid card (test card number used widely as a sample).
_VALID_CARD = "4111111111111111"


class TestLuhn:
    def test_valid_card_passes(self):
        assert luhn_valid(_VALID_CARD) is True

    def test_invalid_card_fails(self):
        # Flip a digit so the checksum no longer holds.
        assert luhn_valid("4111111111111112") is False

    def test_non_digits_fail(self):
        assert luhn_valid("") is False
        assert luhn_valid("4111-1111") is False


class TestPerCategoryDetection:
    @pytest.mark.parametrize(
        "text, category",
        [
            ("contact me at john.doe@example.com please", "email"),
            ("call 415-555-0199 now", "phone"),
            ("ssn 123-45-6789 on file", "government_id"),
            ("123 Market Street is the place", "physical_address"),
            (f"card {_VALID_CARD} expires soon", "credit_card"),
        ],
    )
    def test_single_occurrence_category_and_span(self, text, category):
        detector = PIIDetector()
        issues = detector.detect(_record(text))
        assert len(issues) == 1
        issue = issues[0]
        assert issue.category is IssueCategory.PII
        assert issue.pii_category == category
        # The recorded span slices back to the exact PII instance.
        assert text[issue.span.start : issue.span.end] in text
        assert issue.span.start < issue.span.end

    def test_invalid_credit_card_not_flagged(self):
        detector = PIIDetector()
        issues = detector.detect(_record("card 4111111111111112 here"))
        assert issues == []

    def test_phone_and_government_id_are_distinguished(self):
        detector = PIIDetector()
        phone_issues = detector.detect(_record("415-555-0199"))
        gov_issues = detector.detect(_record("123-45-6789"))
        assert [i.pii_category for i in phone_issues] == ["phone"]
        assert [i.pii_category for i in gov_issues] == ["government_id"]


class TestMultipleOccurrences:
    def test_repeated_category_each_recorded(self):
        detector = PIIDetector()
        text = "a@b.com talk to c@d.org and e@f.net"
        issues = detector.detect(_record(text))
        assert len(issues) == 3
        assert all(i.pii_category == "email" for i in issues)
        # Spans are exact for every occurrence.
        for issue in issues:
            sliced = text[issue.span.start : issue.span.end]
            assert "@" in sliced

    def test_mixed_categories_in_one_text(self):
        detector = PIIDetector()
        text = "mail x@y.com phone 415-555-0199 ssn 123-45-6789"
        issues = detector.detect(_record(text))
        cats = sorted(i.pii_category for i in issues)
        assert cats == ["email", "government_id", "phone"]

    def test_spans_are_ordered_by_position(self):
        detector = PIIDetector()
        text = "x@y.com then 415-555-0199"
        issues = detector.detect(_record(text))
        starts = [i.span.start for i in issues]
        assert starts == sorted(starts)


class TestNoPII:
    def test_pii_free_text_records_no_issue(self):
        detector = PIIDetector()
        issues = detector.detect(_record("the quick brown fox jumps over"))
        assert issues == []

    def test_non_string_fields_are_ignored(self):
        detector = PIIDetector()
        record = Record(
            fields={"count": 5, "ratio": 0.5, "flag": True, "nothing": None},
            location=RecordLocation(source_file="data.json", array_index=0),
        )
        assert detector.detect(record) == []


class TestMultiField:
    def test_each_field_scanned_with_field_name(self):
        detector = PIIDetector()
        record = Record(
            fields={"a": "x@y.com", "b": "no pii here", "c": "415-555-0199"},
            location=RecordLocation(source_file="data.jsonl", line_number=2),
        )
        issues = detector.detect(record)
        by_field = {i.field_name: i.pii_category for i in issues}
        assert by_field == {"a": "email", "c": "phone"}
        # Spans are relative to the field's own text.
        email_issue = next(i for i in issues if i.field_name == "a")
        assert record.fields["a"][email_issue.span.start : email_issue.span.end] == "x@y.com"


class TestAnalysisFailure:
    def test_failure_records_analysis_failure_and_preserves_content(self, monkeypatch):
        detector = PIIDetector()
        record = _record("x@y.com")
        original_fields = dict(record.fields)

        def boom(_text):
            raise RuntimeError("scan exploded")

        monkeypatch.setattr(detector, "_scan_text", boom)
        issues = detector.detect(record)

        assert len(issues) == 1
        assert issues[0].category is IssueCategory.ANALYSIS_FAILURE
        assert issues[0].location == record.location
        # Record content is left unchanged (Requirement 6.4).
        assert record.fields == original_fields


class TestFindPiiHelper:
    def test_returns_exact_substrings(self):
        text = "reach me: john@example.com or 123 Market Street"
        matches = find_pii_in_text(text)
        for match in matches:
            assert text[match.start : match.end] == match.value
        cats = sorted(m.category for m in matches)
        assert cats == ["email", "physical_address"]
