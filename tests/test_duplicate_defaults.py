"""Unit tests for Duplicate_Detector defaults and trivial datasets (task 7.5).

These example-based tests cover the two clauses not exercised by the property
tests:

* Requirement 5.4 - the default similarity threshold of 0.9 is used when none
  is configured, so an unconfigured ``detect`` behaves identically to an
  explicit ``threshold=0.9`` run and flags a pair whose similarity is exactly
  0.9 while leaving a below-default pair unflagged.
* Requirement 5.6 - datasets of zero or exactly one Record complete with zero
  Quality_Issues.
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


# Nine shared word tokens; record two adds one extra token. Token-set Jaccard
# is therefore 9 / 10 = 0.9 exactly, the default threshold boundary.
_COMMON = [f"word{i}" for i in range(9)]


def _point_nine_pair() -> list[Record]:
    text_a = " ".join(_COMMON)
    text_b = " ".join(_COMMON + ["extra"])
    return [_rec({"text": text_a}, 0), _rec({"text": text_b}, 1)]


def _below_default_pair() -> list[Record]:
    # Share 8 tokens; each side carries distinct extras -> Jaccard 8/11 ~= 0.73.
    text_a = " ".join(_COMMON)  # 9 tokens (word0..word8)
    text_b = " ".join(_COMMON[:8] + ["x", "y"])  # word0..word7, x, y
    return [_rec({"text": text_a}, 0), _rec({"text": text_b}, 1)]


class TestDefaultThreshold:
    def test_default_constant_is_point_nine(self):
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.9

    def test_unconfigured_detect_matches_explicit_default(self):
        dataset = _dataset(_point_nine_pair())
        assert DuplicateDetector().detect(dataset) == DuplicateDetector().detect(
            dataset, threshold=0.9
        )

    def test_default_flags_pair_at_exactly_point_nine(self):
        records = _point_nine_pair()
        issues = DuplicateDetector().detect(_dataset(records))

        near = [i for i in issues if i.category is IssueCategory.NEAR_DUPLICATE]
        assert len(near) == 1
        issue = near[0]
        assert issue.location == records[1].location
        assert issue.related_location == records[0].location
        assert issue.score is not None
        assert abs(issue.score - 0.9) < 1e-9

    def test_default_does_not_flag_pair_below_point_nine(self):
        issues = DuplicateDetector().detect(_dataset(_below_default_pair()))
        assert not any(
            i.category is IssueCategory.NEAR_DUPLICATE for i in issues
        )


class TestTrivialDatasets:
    def test_empty_dataset_records_zero_issues(self):
        assert DuplicateDetector().detect(_dataset([])) == []

    def test_single_record_records_zero_issues(self):
        records = [_rec({"text": "only one record here"}, 0)]
        assert DuplicateDetector().detect(_dataset(records)) == []
