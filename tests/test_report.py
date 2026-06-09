"""Unit tests for the Report_Generator assembly step (Requirements 10.1, 10.2,
10.5, 10.6).

These example-based tests cover grouping issues over the full
:class:`IssueCategory` enum (zero-count categories present with count 0),
carrying the computed :class:`Metrics`, and the summary counts
``total_records``/``total_issues`` including the zero-record case.

Imports come directly from ``analyzer.report`` to avoid coupling to the
package ``__init__`` exports.
"""

from analyzer.models import (
    Dataset,
    IssueCategory,
    Metrics,
    QualityIssue,
    Record,
    RecordLocation,
)
from analyzer.report import ReportGenerator


def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data", array_index=index)


def _rec(index: int) -> Record:
    return Record(fields={"text": f"record {index}"}, location=_loc(index))


def _dataset(n: int) -> Dataset:
    return Dataset(records=[_rec(i) for i in range(n)], source_files=["data"])


def _issue(category: IssueCategory, index: int) -> QualityIssue:
    return QualityIssue(category=category, location=_loc(index))


def _metrics(record_count: int) -> Metrics:
    return Metrics(
        record_count=record_count,
        mean_tokens=2,
        min_tokens=2,
        max_tokens=2,
        issue_record_proportion=0.0,
        quality_score=1.0,
    )


class TestGrouping:
    def test_all_categories_present_even_with_no_issues(self):
        report = ReportGenerator().build(_dataset(3), _metrics(3), [])
        assert set(report.issues_by_category) == set(IssueCategory)
        assert set(report.category_counts) == set(IssueCategory)

    def test_zero_count_categories_report_zero(self):
        issues = [_issue(IssueCategory.PII, 0)]
        report = ReportGenerator().build(_dataset(1), _metrics(1), issues)
        # The populated category counts its issue; every other category is 0.
        assert report.category_counts[IssueCategory.PII] == 1
        for category in IssueCategory:
            if category is not IssueCategory.PII:
                assert report.category_counts[category] == 0
                assert report.issues_by_category[category] == []

    def test_issues_grouped_under_their_category(self):
        issues = [
            _issue(IssueCategory.DUPLICATE, 0),
            _issue(IssueCategory.DUPLICATE, 1),
            _issue(IssueCategory.TOXICITY, 2),
        ]
        report = ReportGenerator().build(_dataset(3), _metrics(3), issues)
        assert report.category_counts[IssueCategory.DUPLICATE] == 2
        assert report.category_counts[IssueCategory.TOXICITY] == 1
        assert report.issues_by_category[IssueCategory.DUPLICATE] == issues[:2]
        assert report.issues_by_category[IssueCategory.TOXICITY] == [issues[2]]

    def test_counts_sum_to_total_issues(self):
        issues = [
            _issue(IssueCategory.PII, 0),
            _issue(IssueCategory.PII, 1),
            _issue(IssueCategory.EMPTY_RECORD, 2),
        ]
        report = ReportGenerator().build(_dataset(3), _metrics(3), issues)
        assert sum(report.category_counts.values()) == report.total_issues


class TestMetricsCarried:
    def test_report_carries_metrics_object(self):
        metrics = _metrics(2)
        report = ReportGenerator().build(_dataset(2), metrics, [])
        assert report.metrics is metrics


class TestSummaryCounts:
    def test_total_records_matches_dataset(self):
        report = ReportGenerator().build(_dataset(5), _metrics(5), [])
        assert report.total_records == 5

    def test_total_issues_matches_issue_count(self):
        issues = [_issue(IssueCategory.PII, i) for i in range(4)]
        report = ReportGenerator().build(_dataset(4), _metrics(4), issues)
        assert report.total_issues == 4

    def test_zero_record_dataset_has_zero_counts(self):
        report = ReportGenerator().build(_dataset(0), _metrics(0), [])
        assert report.total_records == 0
        assert report.total_issues == 0

    def test_counts_are_non_negative(self):
        report = ReportGenerator().build(_dataset(0), _metrics(0), [])
        assert report.total_records >= 0
        assert report.total_issues >= 0
