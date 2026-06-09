"""Property-based tests for the Report_Generator (Requirement 10).

Each test below validates exactly one Correctness Property from the design's
32-property numbering (Properties 27-32), covering report assembly
(metric inclusion, category grouping, summary totals) and serialization
(JSON validity, Markdown sections, unsupported-format rejection).

Strategies build real :class:`Report` objects via
:meth:`ReportGenerator.build` over generated datasets, metrics, and issue
lists, so the properties exercise the production assembly/serialization paths
without mocking. Generated :class:`Metrics` carry only finite values so that a
supported-format serialization is expected to succeed (the serialization
*failure* path is covered by the dedicated unit test).
"""

from __future__ import annotations

import dataclasses
import json

from hypothesis import given, settings
from hypothesis import strategies as st

from analyzer.errors import UnsupportedFormatError
from analyzer.models import (
    Dataset,
    IssueCategory,
    Metrics,
    QualityIssue,
    Record,
    RecordLocation,
)
from analyzer.report import OutputFormat, ReportGenerator

# The metric field names every report must expose by name (Requirement 10.1).
METRIC_NAMES: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(Metrics)
)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data", array_index=index)


@st.composite
def metrics_st(draw) -> Metrics:
    """A :class:`Metrics` with finite, serializable values."""
    return Metrics(
        record_count=draw(st.integers(min_value=0, max_value=10_000)),
        mean_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        min_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        max_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        issue_record_proportion=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        quality_score=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def issue_st(draw) -> QualityIssue:
    """A :class:`QualityIssue` of an arbitrary category with a simple location."""
    category = draw(st.sampled_from(list(IssueCategory)))
    index = draw(st.integers(min_value=0, max_value=1000))
    detail = draw(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=30))
    return QualityIssue(category=category, location=_loc(index), detail=detail)


@st.composite
def built_reports(draw):
    """Build a real Report plus the inputs that produced it.

    Returns ``(report, dataset, metrics, issues)`` so properties can compare
    the assembled report against the ground-truth inputs.
    """
    n_records = draw(st.integers(min_value=0, max_value=25))
    dataset = Dataset(
        records=[Record(fields={"text": f"r{i}"}, location=_loc(i)) for i in range(n_records)],
        source_files=["data"],
    )
    metrics = draw(metrics_st())
    issues = draw(st.lists(issue_st(), max_size=25))
    report = ReportGenerator().build(dataset, metrics, issues)
    return report, dataset, metrics, issues


# --------------------------------------------------------------------------- #
# Property 27
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 27: Report includes every metric
@settings(max_examples=200)
@given(built_reports())
def test_report_includes_every_metric(built):
    """Validates: Requirements 10.1

    The assembled report carries every computed metric by name and value, and
    the serialized JSON document exposes each metric name with its exact value.
    """
    report, _dataset, metrics, _issues = built

    # The report carries the Metrics object verbatim.
    assert report.metrics is metrics

    # Every metric is present by name and value in the serialized document.
    result = ReportGenerator().serialize(report, OutputFormat.JSON)
    assert result.error is None
    payload = json.loads(result.text)
    assert set(payload["metrics"]) == set(METRIC_NAMES)
    assert payload["metrics"] == dataclasses.asdict(metrics)


# --------------------------------------------------------------------------- #
# Property 28
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 28: Issues are grouped by category with zero-count categories present
@settings(max_examples=200)
@given(built_reports())
def test_issues_grouped_with_zero_count_categories_present(built):
    """Validates: Requirements 10.2

    Every IssueCategory appears in the grouping; each category's count equals
    the number of its issues (zero-count categories present with count 0 and an
    empty list); and the per-category counts sum to ``total_issues``.
    """
    report, _dataset, _metrics, issues = built

    # Every category is present in both maps.
    assert set(report.issues_by_category) == set(IssueCategory)
    assert set(report.category_counts) == set(IssueCategory)

    expected = {category: 0 for category in IssueCategory}
    for issue in issues:
        expected[issue.category] += 1

    for category in IssueCategory:
        grouped = report.issues_by_category[category]
        assert report.category_counts[category] == expected[category]
        assert len(grouped) == expected[category]
        if expected[category] == 0:
            assert grouped == []
        # Issues are filed under their own category.
        assert all(issue.category is category for issue in grouped)

    assert sum(report.category_counts.values()) == report.total_issues


# --------------------------------------------------------------------------- #
# Property 29
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 29: JSON report is a single valid document with required sections
@settings(max_examples=200)
@given(built_reports())
def test_json_report_is_single_valid_document_with_sections(built):
    """Validates: Requirements 10.3

    JSON serialization yields a single document that parses without error and
    contains the metrics, the issues grouped by category, and the summary
    counts.
    """
    report, _dataset, metrics, _issues = built

    result = ReportGenerator().serialize(report, OutputFormat.JSON)
    assert result.error is None
    assert result.text is not None

    # A single document that parses back without error.
    payload = json.loads(result.text)

    # Metrics section.
    assert payload["metrics"] == dataclasses.asdict(metrics)

    # Grouped issues over the full category enum, each with a count + list.
    grouped = payload["issues_by_category"]
    assert set(grouped) == {c.value for c in IssueCategory}
    for category in IssueCategory:
        entry = grouped[category.value]
        assert entry["count"] == report.category_counts[category]
        assert len(entry["issues"]) == report.category_counts[category]

    # Summary counts section.
    assert payload["summary"]["total_records"] == report.total_records
    assert payload["summary"]["total_issues"] == report.total_issues


# --------------------------------------------------------------------------- #
# Property 30
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 30: Markdown report contains required sections
@settings(max_examples=200)
@given(built_reports())
def test_markdown_report_contains_required_sections(built):
    """Validates: Requirements 10.4

    Markdown serialization produces a single document containing every metric
    name, every category grouping with its count, and the summary counts.
    """
    report, _dataset, _metrics, _issues = built

    result = ReportGenerator().serialize(report, OutputFormat.MARKDOWN)
    assert result.error is None
    text = result.text
    assert text is not None

    # Summary counts.
    assert f"Total records: {report.total_records}" in text
    assert f"Total issues: {report.total_issues}" in text

    # Every metric name appears.
    assert "## Metrics" in text
    for name in METRIC_NAMES:
        assert f"{name}:" in text

    # Every category grouping appears with its count.
    assert "## Issues by Category" in text
    for category in IssueCategory:
        count = report.category_counts[category]
        assert f"{category.value} (count: {count})" in text


# --------------------------------------------------------------------------- #
# Property 31
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 31: Report totals match actual counts
@settings(max_examples=200)
@given(built_reports())
def test_report_totals_match_actual_counts(built):
    """Validates: Requirements 10.5, 10.6

    ``total_records`` equals the number of records analyzed and
    ``total_issues`` equals the number of detected issues; both are
    non-negative and the per-category counts sum to ``total_issues``.
    """
    report, dataset, _metrics, issues = built

    assert report.total_records == len(dataset.records)
    assert report.total_issues == len(issues)
    assert report.total_records >= 0
    assert report.total_issues >= 0
    assert sum(report.category_counts.values()) == report.total_issues


# --------------------------------------------------------------------------- #
# Property 32
# --------------------------------------------------------------------------- #

def _unsupported_format_strings() -> st.SearchStrategy[str]:
    """Random strings that are not a supported output format value."""
    return st.text(max_size=12).filter(
        lambda s: s.lower() not in {fmt.value for fmt in OutputFormat}
    )


# Feature: llm-training-data-quality-analyzer, Property 32: Unsupported report format is rejected
@settings(max_examples=200)
@given(built_reports(), _unsupported_format_strings())
def test_unsupported_format_is_rejected(built, fmt):
    """Validates: Requirements 10.8

    A requested output format other than JSON or Markdown is rejected: no
    output is produced and an error names the unsupported format.
    """
    report, _dataset, _metrics, _issues = built

    result = ReportGenerator().serialize(report, fmt)
    assert result.text is None
    assert isinstance(result.error, UnsupportedFormatError)
    assert result.error.fmt == fmt
