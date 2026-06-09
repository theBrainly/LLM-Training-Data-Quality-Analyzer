"""Unit tests for the Duplicate_Detector (Requirements 5.1-5.6).

These example-based tests cover exact-duplicate detection, near-duplicate pair
detection against a threshold, paired references to the original record, the
default threshold, invalid-threshold rejection, and the trivial 0/1-record
datasets. Property-based coverage lives in separate tasks (7.2-7.5).
"""

from analyzer.detectors.duplicate import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateDetector,
)
from analyzer.models import (
    Dataset,
    IssueCategory,
    Record,
    RecordLocation,
)


def _rec(fields: dict, index: int) -> Record:
    return Record(
        fields=fields,
        location=RecordLocation(source_file="data", array_index=index),
    )


def _dataset(records: list[Record]) -> Dataset:
    return Dataset(records=records, source_files=["data"])


class TestExactDuplicates:
    def test_flags_byte_for_byte_identical_record(self):
        records = [
            _rec({"text": "hello world"}, 0),
            _rec({"text": "hello world"}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records))

        dupes = [i for i in issues if i.category is IssueCategory.DUPLICATE]
        assert len(dupes) == 1
        issue = dupes[0]
        # References both the duplicate and the original record (Req 5.3).
        assert issue.location == records[1].location
        assert issue.related_location == records[0].location

    def test_identity_ignores_field_order(self):
        records = [
            _rec({"a": "x", "b": "y"}, 0),
            _rec({"b": "y", "a": "x"}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records))
        assert sum(i.category is IssueCategory.DUPLICATE for i in issues) == 1

    def test_three_identical_records_yield_two_issues_against_first(self):
        records = [
            _rec({"text": "same"}, 0),
            _rec({"text": "same"}, 1),
            _rec({"text": "same"}, 2),
        ]
        issues = DuplicateDetector().detect(_dataset(records))
        dupes = [i for i in issues if i.category is IssueCategory.DUPLICATE]
        assert len(dupes) == 2
        # Every duplicate references the first occurrence as the original.
        for issue in dupes:
            assert issue.related_location == records[0].location

    def test_bool_and_int_are_not_identical(self):
        records = [
            _rec({"v": True}, 0),
            _rec({"v": 1}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records))
        assert not any(i.category is IssueCategory.DUPLICATE for i in issues)

    def test_distinct_records_yield_no_exact_duplicate(self):
        records = [
            _rec({"text": "alpha beta"}, 0),
            _rec({"text": "completely different content here"}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records))
        assert not any(i.category is IssueCategory.DUPLICATE for i in issues)


class TestNearDuplicates:
    def test_flags_pair_at_or_above_threshold(self):
        # Shared {the,quick,brown,fox}=4, union 6 -> Jaccard 4/6 ~= 0.667.
        records = [
            _rec({"text": "the quick brown fox jumps"}, 0),
            _rec({"text": "the quick brown fox leaps"}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records), threshold=0.6)

        near = [i for i in issues if i.category is IssueCategory.NEAR_DUPLICATE]
        assert len(near) == 1
        issue = near[0]
        assert issue.location == records[1].location
        assert issue.related_location == records[0].location
        assert issue.score is not None and issue.score >= 0.6

    def test_pair_below_threshold_is_not_flagged(self):
        records = [
            _rec({"text": "the quick brown fox jumps"}, 0),
            _rec({"text": "the quick brown fox leaps"}, 1),
        ]
        # Similarity is 0.8 < 0.9 default threshold.
        issues = DuplicateDetector().detect(_dataset(records))
        assert not any(
            i.category is IssueCategory.NEAR_DUPLICATE for i in issues
        )

    def test_exact_duplicate_is_not_also_near_duplicate(self):
        records = [
            _rec({"text": "identical content"}, 0),
            _rec({"text": "identical content"}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records), threshold=0.5)
        assert sum(i.category is IssueCategory.DUPLICATE for i in issues) == 1
        assert not any(
            i.category is IssueCategory.NEAR_DUPLICATE for i in issues
        )

    def test_contentless_records_get_zero_similarity(self):
        # Distinct field names, no word tokens -> empty-union similarity 0.0,
        # so they are not near-duplicates under any positive threshold.
        records = [
            _rec({"a": ""}, 0),
            _rec({"b": ""}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records), threshold=0.5)
        assert not any(
            i.category is IssueCategory.NEAR_DUPLICATE for i in issues
        )


class TestThresholdConfiguration:
    def test_default_threshold_is_point_nine(self):
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.9

    def test_threshold_above_one_is_rejected_and_defaulted(self):
        records = [
            _rec({"text": "the quick brown fox jumps over"}, 0),
            _rec({"text": "the quick brown fox jumps under"}, 1),
        ]
        issues = DuplicateDetector().detect(_dataset(records), threshold=1.5)

        config_issues = [
            i for i in issues if i.category is IssueCategory.CONFIG_ERROR
        ]
        assert len(config_issues) == 1
        assert "1.5" in config_issues[0].detail
        # Similarity here is 5/7 ~= 0.71, below the retained default 0.9.
        assert not any(
            i.category is IssueCategory.NEAR_DUPLICATE for i in issues
        )

    def test_negative_threshold_is_rejected(self):
        records = [_rec({"text": "a"}, 0), _rec({"text": "b"}, 1)]
        issues = DuplicateDetector().detect(_dataset(records), threshold=-0.2)
        assert sum(
            i.category is IssueCategory.CONFIG_ERROR for i in issues
        ) == 1

    def test_nan_threshold_is_rejected(self):
        records = [_rec({"text": "a"}, 0), _rec({"text": "b"}, 1)]
        issues = DuplicateDetector().detect(
            _dataset(records), threshold=float("nan")
        )
        assert sum(
            i.category is IssueCategory.CONFIG_ERROR for i in issues
        ) == 1

    def test_boundary_thresholds_are_accepted(self):
        records = [_rec({"text": "a"}, 0), _rec({"text": "b"}, 1)]
        for value in (0.0, 1.0):
            issues = DuplicateDetector().detect(
                _dataset(records), threshold=value
            )
            assert not any(
                i.category is IssueCategory.CONFIG_ERROR for i in issues
            )


class TestTrivialDatasets:
    def test_empty_dataset_yields_no_issues(self):
        assert DuplicateDetector().detect(_dataset([])) == []

    def test_single_record_dataset_yields_no_issues(self):
        records = [_rec({"text": "only one"}, 0)]
        assert DuplicateDetector().detect(_dataset(records)) == []

    def test_single_record_with_invalid_threshold_still_reports_config_error(self):
        # Threshold validation happens before the trivial-size short-circuit.
        records = [_rec({"text": "only one"}, 0)]
        issues = DuplicateDetector().detect(_dataset(records), threshold=2.0)
        assert sum(
            i.category is IssueCategory.CONFIG_ERROR for i in issues
        ) == 1
