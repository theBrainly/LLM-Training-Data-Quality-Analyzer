"""Property-based tests for the Duplicate_Detector (Requirements 5.1-5.5).

Each test validates exactly one Correctness Property from the design's
32-property numbering and references the requirement clauses it covers:

* Property 10 - exact duplicates are flagged against their original (5.1, 5.3).
* Property 11 - near-duplicates are flagged exactly at or above the configured
  threshold (5.2, 5.3).
* Property 12 - an invalid duplicate threshold is rejected and defaulted (5.5).

The exact-duplicate and near-duplicate datasets are planted so the ground
truth is known: exact duplicates are produced by repeating value-identical
content, and near-duplicate similarity is constructed over controlled
lowercase word-token sets whose token-set Jaccard coefficient can be computed
directly with the detector's normalization (lowercase ``\\w+`` tokens).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

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
from tests.strategies import invalid_thresholds, record_lists, thresholds

_SETTINGS = settings(max_examples=100, deadline=None)

# A small vocabulary of single lowercase-letter word tokens. Each entry is a
# valid normalized token on its own, so a space-joined list of these words has
# a normalized token set equal to the set of the words themselves.
_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data", array_index=index)


def _rec(fields: dict, index: int) -> Record:
    return Record(fields=fields, location=_loc(index))


def _dataset(records: list[Record]) -> Dataset:
    return Dataset(records=records, source_files=["data"])


def _expected_jaccard(left: set[str], right: set[str]) -> float:
    """Token-set Jaccard with the detector's empty-union convention (0.0)."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


# --------------------------------------------------------------------------- #
# Property 10: Exact duplicates are flagged against their original
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 10: Exact duplicates are flagged against their original
class TestExactDuplicatesAgainstOriginal:
    @_SETTINGS
    @given(st.lists(st.integers(min_value=0, max_value=4), min_size=0, max_size=12))
    def test_every_exact_duplicate_references_its_first_occurrence(self, values):
        """**Validates: Requirements 5.1, 5.3**

        Records sharing an integer identity carry byte-for-byte identical
        content, so a Record is an exact duplicate iff its identity appeared at
        an earlier index. Each such Record must yield exactly one DUPLICATE
        issue whose ``location`` is the duplicate and whose ``related_location``
        is the first (original) occurrence of that identity.
        """
        records = [
            _rec({"text": f"content number {value}"}, index)
            for index, value in enumerate(values)
        ]
        issues = DuplicateDetector().detect(_dataset(records))

        # Ground truth: map each identity to its first occurrence's index.
        first_seen: dict[int, int] = {}
        expected_pairs: list[tuple[RecordLocation, RecordLocation]] = []
        for index, value in enumerate(values):
            if value in first_seen:
                expected_pairs.append(
                    (records[index].location, records[first_seen[value]].location)
                )
            else:
                first_seen[value] = index

        dupes = [i for i in issues if i.category is IssueCategory.DUPLICATE]
        actual_pairs = [(i.location, i.related_location) for i in dupes]

        # Locations are unique per record, so multiset equality reduces to set
        # equality plus a length check.
        assert len(actual_pairs) == len(expected_pairs)
        assert set(actual_pairs) == set(expected_pairs)


# --------------------------------------------------------------------------- #
# Property 11: Near-duplicates are flagged exactly at or above threshold
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 11: Near-duplicates are flagged exactly at or above threshold
class TestNearDuplicateThreshold:
    @_SETTINGS
    @given(
        st.lists(st.sampled_from(_VOCAB), min_size=1, max_size=6),
        st.lists(st.sampled_from(_VOCAB), min_size=1, max_size=6),
        thresholds(),
    )
    def test_pair_is_flagged_iff_similarity_meets_threshold(
        self, words_a, words_b, threshold
    ):
        """**Validates: Requirements 5.2, 5.3**

        For two non-identical Records, a NEAR_DUPLICATE issue is recorded iff
        their token-set Jaccard similarity meets or exceeds the threshold, and
        the issue references both the duplicate and the original Record. Two
        value-identical Records are exact duplicates instead (no near-dup
        issue), which the detector reports under the DUPLICATE category.
        """
        text_a = " ".join(words_a)
        text_b = " ".join(words_b)
        records = [_rec({"text": text_a}, 0), _rec({"text": text_b}, 1)]

        issues = DuplicateDetector().detect(_dataset(records), threshold=threshold)
        # A valid in-range threshold never produces a configuration error.
        assert not any(i.category is IssueCategory.CONFIG_ERROR for i in issues)

        near = [i for i in issues if i.category is IssueCategory.NEAR_DUPLICATE]
        exact = [i for i in issues if i.category is IssueCategory.DUPLICATE]

        if text_a == text_b:
            # Byte-for-byte identical content => exact duplicate, not near-dup.
            assert len(exact) == 1
            assert len(near) == 0
            return

        assert len(exact) == 0
        similarity = _expected_jaccard(set(words_a), set(words_b))
        if similarity >= threshold:
            assert len(near) == 1
            issue = near[0]
            assert issue.location == records[1].location
            assert issue.related_location == records[0].location
            assert issue.score is not None
            assert issue.score >= threshold
            assert abs(issue.score - similarity) < 1e-9
        else:
            assert len(near) == 0


# --------------------------------------------------------------------------- #
# Property 12: Invalid duplicate threshold is rejected and defaulted
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 12: Invalid duplicate threshold is rejected and defaulted
class TestInvalidThresholdDefaulted:
    @_SETTINGS
    @given(record_lists(min_size=0, max_size=6), invalid_thresholds())
    def test_invalid_threshold_is_rejected_and_default_retained(
        self, records, invalid
    ):
        """**Validates: Requirements 5.5**

        An out-of-range threshold is rejected with exactly one CONFIG_ERROR
        issue identifying the invalid value, and detection proceeds with the
        retained default of 0.9 - so the non-config issues are identical to
        those produced by an explicit default-threshold run.
        """
        dataset = _dataset(records)
        issues = DuplicateDetector().detect(dataset, threshold=invalid)

        config = [i for i in issues if i.category is IssueCategory.CONFIG_ERROR]
        assert len(config) == 1
        # The error indication identifies the invalid value.
        assert repr(invalid) in config[0].detail

        non_config = [i for i in issues if i.category is not IssueCategory.CONFIG_ERROR]
        defaulted = DuplicateDetector().detect(
            dataset, threshold=DEFAULT_SIMILARITY_THRESHOLD
        )
        assert non_config == defaulted
