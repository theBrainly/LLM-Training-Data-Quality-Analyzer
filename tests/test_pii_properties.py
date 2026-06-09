"""Property-based tests for the PII_Detector (Requirements 6.1-6.3, 6.5).

Each test validates exactly one Correctness Property from the design's
32-property numbering and references the requirement clause it covers:

* Property 13 - all PII occurrences are detected (6.1).
* Property 14 - PII-free text produces no PII issues (6.2).
* Property 15 - PII issue spans are accurate (6.3).
* Property 16 - redaction removes PII and preserves the original (6.5).

The PII samples are drawn from :func:`tests.strategies.pii_text`, which embeds
a known multiset of PII instances into PII-free filler such that, for every
occurrence, ``text[start:end] == value``. That ground truth lets each property
compare the detector's output against the exact set of planted occurrences.
The PII categories the strategy plants (email, phone, government id, physical
address, Luhn-valid credit card) are mutually unambiguous - each requires a
distinguishing marker (``@``, a 3-3-4 vs 3-2-4 dash grouping, a street suffix,
or a 13-19 digit Luhn-valid run) - so a planted occurrence is recognized as its
own category and nothing else.
"""

from __future__ import annotations

from hypothesis import given, settings

from analyzer.detectors.pii import PIIDetector
from analyzer.models import IssueCategory, Record, RecordLocation
from tests.strategies import pii_free_text, pii_text

_SETTINGS = settings(max_examples=100, deadline=None)

_FIELD = "text"


def _record(text: str) -> Record:
    """A single-field Record carrying ``text`` in the scanned field."""
    return Record(
        fields={_FIELD: text},
        location=RecordLocation(source_file="data.jsonl", line_number=1),
    )


def _pii_issues(detector: PIIDetector, record: Record):
    return [i for i in detector.detect(record) if i.category is IssueCategory.PII]


# --------------------------------------------------------------------------- #
# Property 13: All PII occurrences are detected
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 13: All PII occurrences are detected
class TestAllOccurrencesDetected:
    @_SETTINGS
    @given(pii_text())
    def test_one_issue_per_embedded_occurrence(self, sample):
        """**Validates: Requirements 6.1**

        For text built by embedding a known multiset of PII instances
        (including repeated categories) into PII-free filler, the detector
        records exactly one PII issue per embedded occurrence, and the multiset
        of detected categories equals the planted multiset.
        """
        detector = PIIDetector()
        issues = _pii_issues(detector, _record(sample.text))

        # Exactly one issue per planted occurrence.
        assert len(issues) == len(sample.occurrences)

        # The multiset of detected categories matches the planted ground truth.
        expected = sorted(occ.category for occ in sample.occurrences)
        actual = sorted(issue.pii_category for issue in issues)
        assert actual == expected


# --------------------------------------------------------------------------- #
# Property 14: PII-free text produces no PII issues
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 14: PII-free text produces no PII issues
class TestPiiFreeTextProducesNoIssues:
    @_SETTINGS
    @given(pii_free_text())
    def test_no_pii_issue_for_pii_free_text(self, text):
        """**Validates: Requirements 6.2**

        Text containing no PII is analyzed to completion without recording any
        PII Quality_Issue for the record.
        """
        detector = PIIDetector()
        issues = detector.detect(_record(text))
        assert not any(i.category is IssueCategory.PII for i in issues)


# --------------------------------------------------------------------------- #
# Property 15: PII issue spans are accurate
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 15: PII issue spans are accurate
class TestSpansAreAccurate:
    @_SETTINGS
    @given(pii_text())
    def test_each_issue_span_matches_a_planted_occurrence(self, sample):
        """**Validates: Requirements 6.3**

        Every PII issue identifies the field and a ``[start, end)`` span such
        that the substring at that span equals the matched PII value, and the
        set of reported spans equals the set of planted occurrence spans
        (with matching categories).
        """
        detector = PIIDetector()
        issues = _pii_issues(detector, _record(sample.text))

        # Ground truth: the exact substring planted at each occurrence span.
        by_span = {
            (occ.category, occ.start, occ.end): occ.value
            for occ in sample.occurrences
        }

        # Each issue identifies the field and a span whose substring equals the
        # planted PII value for that exact (category, start, end) occurrence.
        for issue in issues:
            assert issue.field_name == _FIELD
            assert issue.span.start < issue.span.end
            key = (issue.pii_category, issue.span.start, issue.span.end)
            assert key in by_span
            assert sample.text[issue.span.start : issue.span.end] == by_span[key]

        # The set of reported spans equals the set of planted occurrence spans.
        actual = {
            (issue.pii_category, issue.span.start, issue.span.end)
            for issue in issues
        }
        assert actual == set(by_span)


# --------------------------------------------------------------------------- #
# Property 16: Redaction removes PII and preserves the original
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 16: Redaction removes PII and preserves the original
class TestRedactionRemovesPiiAndPreservesOriginal:
    @_SETTINGS
    @given(pii_text())
    def test_redaction_removes_pii_and_leaves_original_unchanged(self, sample):
        """**Validates: Requirements 6.5**

        Redaction returns a copy in which each detected PII span is replaced by
        a placeholder for that span's category; the redacted copy contains no
        detectable PII; and the original record (and its field mapping) is left
        unchanged.
        """
        detector = PIIDetector()
        record = _record(sample.text)
        snapshot = dict(record.fields)

        redacted = detector.redact(record)

        # Original record is untouched and the copy is a distinct object.
        assert record.fields == snapshot
        assert record.fields[_FIELD] == sample.text
        assert redacted is not record

        # The redacted copy contains no detectable PII.
        assert not any(
            i.category is IssueCategory.PII for i in detector.detect(redacted)
        )

        # Each planted category's placeholder appears in the redacted text, and
        # none of the original PII values survive verbatim.
        redacted_text = redacted.fields[_FIELD]
        for occ in sample.occurrences:
            assert f"[{occ.category.upper()}]" in redacted_text
            assert occ.value not in redacted_text
