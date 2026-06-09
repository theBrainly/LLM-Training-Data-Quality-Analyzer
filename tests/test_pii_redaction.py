"""Unit tests for PII redaction (Requirement 6.5, task 8.2).

Redaction returns a *copy* of a record in which each detected PII span is
replaced by a placeholder for that span's PII category, while leaving the
original record (and its field mapping) unchanged.
"""

from analyzer.detectors.pii import PIIDetector
from analyzer.models import Record, RecordLocation


def _record(text: str, field: str = "text") -> Record:
    return Record(
        fields={field: text},
        location=RecordLocation(source_file="data.jsonl", line_number=1),
    )


# Known Luhn-valid sample card number.
_VALID_CARD = "4111111111111111"


class TestPlaceholderReplacement:
    def test_email_replaced_with_placeholder(self):
        redacted = PIIDetector().redact(_record("contact john.doe@example.com please"))
        assert redacted.fields["text"] == "contact [EMAIL] please"

    def test_phone_replaced_with_placeholder(self):
        redacted = PIIDetector().redact(_record("call 415-555-0199 now"))
        assert redacted.fields["text"] == "call [PHONE] now"

    def test_government_id_replaced_with_placeholder(self):
        redacted = PIIDetector().redact(_record("ssn 123-45-6789 on file"))
        assert redacted.fields["text"] == "ssn [GOVERNMENT_ID] on file"

    def test_physical_address_replaced_with_placeholder(self):
        redacted = PIIDetector().redact(_record("123 Market Street is here"))
        assert redacted.fields["text"] == "[PHYSICAL_ADDRESS] is here"

    def test_credit_card_replaced_with_placeholder(self):
        redacted = PIIDetector().redact(_record(f"card {_VALID_CARD} expires"))
        assert redacted.fields["text"] == "card [CREDIT_CARD] expires"


class TestMultipleSpans:
    def test_multiple_occurrences_all_replaced(self):
        text = "a@b.com talk to c@d.org and e@f.net"
        redacted = PIIDetector().redact(_record(text))
        assert redacted.fields["text"] == "[EMAIL] talk to [EMAIL] and [EMAIL]"

    def test_mixed_categories_replaced_with_correct_placeholders(self):
        text = "mail x@y.com phone 415-555-0199 ssn 123-45-6789"
        redacted = PIIDetector().redact(_record(text))
        assert redacted.fields["text"] == (
            "mail [EMAIL] phone [PHONE] ssn [GOVERNMENT_ID]"
        )


class TestNoDetectablePIIRemains:
    def test_redacted_text_has_no_remaining_pii(self):
        detector = PIIDetector()
        text = "mail x@y.com phone 415-555-0199 ssn 123-45-6789 card " + _VALID_CARD
        redacted = detector.redact(_record(text))
        # The redacted copy must contain no detectable PII (Requirement 6.5).
        assert detector.detect(redacted) == []


class TestOriginalUnchanged:
    def test_original_record_and_fields_untouched(self):
        original_text = "contact john.doe@example.com please"
        record = _record(original_text)
        snapshot = dict(record.fields)

        redacted = PIIDetector().redact(record)

        # Original record object and its field mapping are unchanged.
        assert record.fields == snapshot
        assert record.fields["text"] == original_text
        # The returned record is a distinct object with a distinct mapping.
        assert redacted is not record
        assert redacted.fields is not record.fields

    def test_location_is_preserved_on_copy(self):
        record = _record("x@y.com")
        redacted = PIIDetector().redact(record)
        assert redacted.location == record.location


class TestNonPIIContent:
    def test_pii_free_field_is_copied_verbatim(self):
        redacted = PIIDetector().redact(_record("the quick brown fox"))
        assert redacted.fields["text"] == "the quick brown fox"

    def test_non_string_fields_preserved(self):
        record = Record(
            fields={"count": 5, "ratio": 0.5, "flag": True, "nothing": None},
            location=RecordLocation(source_file="data.json", array_index=0),
        )
        redacted = PIIDetector().redact(record)
        assert redacted.fields == {
            "count": 5,
            "ratio": 0.5,
            "flag": True,
            "nothing": None,
        }


class TestMultiField:
    def test_each_string_field_redacted_independently(self):
        record = Record(
            fields={"a": "x@y.com", "b": "no pii here", "c": "415-555-0199"},
            location=RecordLocation(source_file="data.jsonl", line_number=2),
        )
        redacted = PIIDetector().redact(record)
        assert redacted.fields == {
            "a": "[EMAIL]",
            "b": "no pii here",
            "c": "[PHONE]",
        }
