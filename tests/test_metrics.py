"""Unit tests for the Metrics_Engine (Requirements 4.1-4.7).

These example-based tests cover record counting, token-count mean/min/max,
the issue-record proportion (with issues mapped to records by location), the
quality-score formula, and the empty-dataset zero values. Property-based
coverage lives in separate tasks (13.2, 13.3).

Imports come directly from ``analyzer.metrics`` to avoid coupling to the
package ``__init__`` exports.
"""

from analyzer.metrics import MetricsEngine, token_count
from analyzer.models import (
    Dataset,
    IssueCategory,
    QualityIssue,
    Record,
    RecordLocation,
)


def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data", array_index=index)


def _rec(fields: dict, index: int) -> Record:
    return Record(fields=fields, location=_loc(index))


def _dataset(records: list[Record]) -> Dataset:
    return Dataset(records=records, source_files=["data"])


def _issue(index: int) -> QualityIssue:
    return QualityIssue(category=IssueCategory.PARSE_ERROR, location=_loc(index))


class TestTokenCount:
    def test_counts_whitespace_delimited_tokens(self):
        assert token_count(_rec({"text": "hello world foo"}, 0)) == 3

    def test_collapses_runs_of_whitespace(self):
        assert token_count(_rec({"text": "  a\t\n  b   "}, 0)) == 2

    def test_empty_text_is_zero_tokens(self):
        assert token_count(_rec({"text": ""}, 0)) == 0

    def test_includes_numbers_and_nested_containers(self):
        rec = _rec({"a": "one two", "b": 42, "c": ["x", {"d": "y z"}]}, 0)
        # tokens: one, two, 42, x, y, z
        assert token_count(rec) == 6

    def test_bool_and_none_contribute_no_tokens(self):
        assert token_count(_rec({"flag": True, "missing": None}, 0)) == 0


class TestRecordCount:
    def test_counts_records(self):
        metrics = MetricsEngine().compute(
            _dataset([_rec({"t": "a"}, 0), _rec({"t": "b"}, 1)]), []
        )
        assert metrics.record_count == 2


class TestTokenStatistics:
    def test_mean_min_max(self):
        records = [
            _rec({"t": "a"}, 0),            # 1 token
            _rec({"t": "a b c"}, 1),        # 3 tokens
            _rec({"t": "a b c d e"}, 2),    # 5 tokens
        ]
        metrics = MetricsEngine().compute(_dataset(records), [])
        assert metrics.min_tokens == 1
        assert metrics.max_tokens == 5
        assert metrics.mean_tokens == 3  # (1+3+5)/3 == 3

    def test_mean_is_rounded_integer_within_bounds(self):
        records = [
            _rec({"t": "a"}, 0),        # 1
            _rec({"t": "a b c c"}, 1),  # 4
        ]
        metrics = MetricsEngine().compute(_dataset(records), [])
        # mean of 1 and 4 is 2.5 -> rounds to 2 (banker's rounding), in [1, 4]
        assert metrics.min_tokens <= metrics.mean_tokens <= metrics.max_tokens
        assert isinstance(metrics.mean_tokens, int)


class TestIssueProportionAndScore:
    def test_proportion_counts_records_with_issues(self):
        records = [_rec({"t": "a"}, 0), _rec({"t": "b"}, 1), _rec({"t": "c"}, 2)]
        # One record (index 0) has issues; the rest are clean.
        issues = [_issue(0), _issue(0)]
        metrics = MetricsEngine().compute(_dataset(records), issues)
        assert metrics.issue_record_proportion == 1 / 3
        assert metrics.quality_score == 1.0 - 1 / 3

    def test_multiple_issues_on_one_record_count_once(self):
        records = [_rec({"t": "a"}, 0), _rec({"t": "b"}, 1)]
        issues = [_issue(0), _issue(0), _issue(0)]
        metrics = MetricsEngine().compute(_dataset(records), issues)
        assert metrics.issue_record_proportion == 0.5
        assert metrics.quality_score == 0.5

    def test_dataset_level_issues_do_not_map_to_records(self):
        records = [_rec({"t": "a"}, 0), _rec({"t": "b"}, 1)]
        dataset_issue = QualityIssue(
            category=IssueCategory.SCHEMA_INFERENCE_FAILED, location=None
        )
        metrics = MetricsEngine().compute(_dataset(records), [dataset_issue])
        assert metrics.issue_record_proportion == 0.0
        assert metrics.quality_score == 1.0

    def test_all_records_flagged_yields_zero_score(self):
        records = [_rec({"t": "a"}, 0), _rec({"t": "b"}, 1)]
        issues = [_issue(0), _issue(1)]
        metrics = MetricsEngine().compute(_dataset(records), issues)
        assert metrics.issue_record_proportion == 1.0
        assert metrics.quality_score == 0.0

    def test_no_issues_yields_perfect_score(self):
        records = [_rec({"t": "a"}, 0)]
        metrics = MetricsEngine().compute(_dataset(records), [])
        assert metrics.issue_record_proportion == 0.0
        assert metrics.quality_score == 1.0


class TestEmptyDataset:
    def test_all_metrics_are_zero(self):
        metrics = MetricsEngine().compute(_dataset([]), [])
        assert metrics.record_count == 0
        assert metrics.mean_tokens == 0
        assert metrics.min_tokens == 0
        assert metrics.max_tokens == 0
        assert metrics.issue_record_proportion == 0.0
        assert metrics.quality_score == 0.0
