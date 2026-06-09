"""Unit test for PII analysis failure handling (Requirement 6.4, task 8.7).

When a Record cannot be analyzed for PII, the PII_Detector records a single
``ANALYSIS_FAILURE`` Quality_Issue that identifies the Record and leaves the
original Record content unchanged. The failure is simulated by monkeypatching
the detector's internal ``_scan_text`` primitive to raise.
"""

from __future__ import annotations

import pytest

from analyzer.detectors.pii import PIIDetector
from analyzer.models import IssueCategory, Record, RecordLocation


def _record(text: str, field: str = "text") -> Record:
    return Record(
        fields={field: text},
        location=RecordLocation(source_file="data.jsonl", line_number=7),
    )


def test_analysis_failure_records_issue_and_preserves_content(monkeypatch):
    """**Validates: Requirements 6.4**

    A failing analysis produces exactly one ANALYSIS_FAILURE issue identifying
    the record, and the record's content is left unchanged.
    """
    detector = PIIDetector()
    record = _record("contact john.doe@example.com please")
    original_fields = dict(record.fields)

    def boom(_text):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(detector, "_scan_text", boom)

    issues = detector.detect(record)

    # Exactly one issue, and it indicates an analysis failure for this record.
    assert len(issues) == 1
    failure = issues[0]
    assert failure.category is IssueCategory.ANALYSIS_FAILURE
    assert failure.location == record.location
    # No PII issues are produced when analysis fails.
    assert not any(i.category is IssueCategory.PII for i in issues)

    # The original record content is left unchanged (Requirement 6.4).
    assert record.fields == original_fields
