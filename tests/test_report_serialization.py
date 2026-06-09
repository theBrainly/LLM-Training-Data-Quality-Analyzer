"""Unit tests for Report_Generator serialization (Requirements 10.3, 10.4,
10.7, 10.8).

Covers JSON serialization (parses back, carries metrics/grouped issues/summary
counts), Markdown serialization (contains the same sections), rejection of
unsupported output formats, and the no-partial-output behaviour on a
serialization failure.

Imports come directly from ``analyzer.report`` to avoid coupling to the
package ``__init__`` exports.
"""

import json

import pytest

from analyzer.errors import SerializationError, UnsupportedFormatError
from analyzer.models import (
    Dataset,
    IssueCategory,
    Metrics,
    QualityIssue,
    Record,
    RecordLocation,
    Span,
)
from analyzer.report import (
    OutputFormat,
    ReportGenerator,
    SerializeResult,
)


def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data", array_index=index)


def _rec(index: int) -> Record:
    return Record(fields={"text": f"record {index}"}, location=_loc(index))


def _dataset(n: int) -> Dataset:
    return Dataset(records=[_rec(i) for i in range(n)], source_files=["data"])


def _metrics(record_count: int) -> Metrics:
    return Metrics(
        record_count=record_count,
        mean_tokens=2,
        min_tokens=1,
        max_tokens=3,
        issue_record_proportion=0.5,
        quality_score=0.5,
    )


def _report_with_issues() -> object:
    issues = [
        QualityIssue(category=IssueCategory.PII, location=_loc(0),
                     pii_category="email", span=Span(0, 5), detail="found email"),
        QualityIssue(category=IssueCategory.PII, location=_loc(1),
                     detail="found phone"),
        QualityIssue(category=IssueCategory.DUPLICATE, location=_loc(2),
                     related_location=_loc(0), detail="dup of 0"),
    ]
    return ReportGenerator().build(_dataset(3), _metrics(3), issues)


class TestJsonSerialization:
    def test_json_parses_without_error(self):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, OutputFormat.JSON)
        assert result.error is None
        # A single document that parses without error (Req 10.3).
        json.loads(result.text)

    def test_json_contains_metrics_issues_and_counts(self):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, OutputFormat.JSON)
        payload = json.loads(result.text)

        # Every computed metric by name and value (Req 10.1/10.3).
        assert payload["metrics"]["record_count"] == 3
        assert payload["metrics"]["quality_score"] == 0.5

        # Summary counts.
        assert payload["summary"]["total_records"] == 3
        assert payload["summary"]["total_issues"] == 3

        # Issues grouped by category over the full enum, with counts.
        grouped = payload["issues_by_category"]
        assert set(grouped) == {c.value for c in IssueCategory}
        assert grouped["pii"]["count"] == 2
        assert len(grouped["pii"]["issues"]) == 2
        assert grouped["duplicate"]["count"] == 1
        # A zero-count category is present with an empty issue list.
        assert grouped["toxicity"]["count"] == 0
        assert grouped["toxicity"]["issues"] == []

    def test_json_accepts_string_format(self):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, "json")
        assert result.error is None
        json.loads(result.text)


class TestMarkdownSerialization:
    def test_markdown_contains_required_sections(self):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, OutputFormat.MARKDOWN)
        assert result.error is None
        text = result.text

        # Summary counts (Req 10.4 / 10.5).
        assert "Total records: 3" in text
        assert "Total issues: 3" in text

        # Metrics by name and value.
        assert "## Metrics" in text
        assert "record_count: 3" in text
        assert "quality_score: 0.5" in text

        # Grouped issues: every category present, including a zero-count one.
        assert "## Issues by Category" in text
        assert "pii (count: 2)" in text
        assert "duplicate (count: 1)" in text
        assert "toxicity (count: 0)" in text
        assert "found email" in text

    def test_markdown_accepts_string_format(self):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, "markdown")
        assert result.error is None
        assert "# Data Quality Report" in result.text


class TestEmptyReport:
    def test_empty_report_serializes_to_both_formats(self):
        report = ReportGenerator().build(_dataset(0), _metrics(0), [])

        json_result = ReportGenerator().serialize(report, OutputFormat.JSON)
        payload = json.loads(json_result.text)
        assert payload["summary"]["total_records"] == 0
        assert payload["summary"]["total_issues"] == 0

        md_result = ReportGenerator().serialize(report, OutputFormat.MARKDOWN)
        assert "Total records: 0" in md_result.text
        assert "Total issues: 0" in md_result.text


class TestUnsupportedFormat:
    @pytest.mark.parametrize("fmt", ["csv", "yaml", "xml", "JSONL", ""])
    def test_unsupported_string_format_rejected(self, fmt):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, fmt)
        assert result.text is None
        assert isinstance(result.error, UnsupportedFormatError)
        assert result.error.fmt == fmt

    def test_unsupported_non_string_format_rejected(self):
        report = _report_with_issues()
        result = ReportGenerator().serialize(report, 123)
        assert result.text is None
        assert isinstance(result.error, UnsupportedFormatError)


class TestSerializationFailure:
    def test_non_finite_metric_yields_serialization_error_and_no_output(self):
        # A non-finite metric cannot be expressed in JSON; serialization must
        # fail with no partial output and an error naming the format (Req 10.7).
        report = ReportGenerator().build(_dataset(1), _metrics(1), [])
        report.metrics.quality_score = float("nan")

        result = ReportGenerator().serialize(report, OutputFormat.JSON)
        assert result.text is None
        assert isinstance(result.error, SerializationError)
        assert result.error.fmt == "json"
