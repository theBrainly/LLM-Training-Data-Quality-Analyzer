"""Property-based tests for the Toxicity_Detector (Requirements 7.1-7.5).

Each test validates exactly one Correctness Property from the design's
32-property numbering and references the requirement clauses it covers:

* Property 17 - toxicity scores are bounded in ``[0.0, 1.0]`` (7.1).
* Property 18 - toxicity flagging matches the threshold with a ``>=``
  comparison (7.2, 7.3).
* Property 19 - an invalid toxicity threshold is rejected and defaulted (7.5).

Scoring is delegated to the deterministic :class:`StubToxicityModel` so the
score a record receives is generated alongside the record and the
threshold-driven business logic is exercised independently of any real
classifier.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from analyzer.detectors.toxicity import (
    DEFAULT_TOXICITY_THRESHOLD,
    ToxicityDetector,
)
from analyzer.models import IssueCategory
from tests.strategies import (
    StubToxicityModel,
    invalid_thresholds,
    records,
    thresholds,
)

_SETTINGS = settings(max_examples=100, deadline=None)


# --------------------------------------------------------------------------- #
# Property 17: Toxicity scores are bounded
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 17: Toxicity scores are bounded
class TestToxicityScoresBounded:
    @_SETTINGS
    @given(
        records(),
        st.floats(
            min_value=-1.0e6,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_assigned_score_lies_in_unit_interval(self, record, raw_score):
        """**Validates: Requirements 7.1**

        Whatever finite numeric value the model produces - including values far
        outside the unit interval - the score the Toxicity_Detector assigns to
        a Record is a numeric value in ``[0.0, 1.0]``. ``clamp=False`` lets the
        raw out-of-range value reach the detector so its own bounding is
        exercised.
        """
        model = StubToxicityModel(score=raw_score, clamp=False)
        score = ToxicityDetector(model).score(record)

        assert score is not None
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Property 18: Toxicity flagging matches the threshold
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 18: Toxicity flagging matches the threshold
class TestToxicityThresholdFlagging:
    @_SETTINGS
    @given(records(), thresholds(), thresholds())
    def test_flagged_iff_score_meets_or_exceeds_threshold(
        self, record, score, threshold
    ):
        """**Validates: Requirements 7.2, 7.3**

        Given a score and threshold both in ``[0.0, 1.0]``, the detector records
        exactly one TOXICITY issue carrying that score if and only if the score
        is greater than or equal to the threshold; otherwise it records none.
        """
        model = StubToxicityModel(score=score)
        issues = ToxicityDetector(model).detect(record, threshold=threshold)

        # A valid in-range threshold never produces a configuration error.
        assert not any(i.category is IssueCategory.CONFIG_ERROR for i in issues)

        toxicity = [i for i in issues if i.category is IssueCategory.TOXICITY]
        if score >= threshold:
            assert len(toxicity) == 1
            assert toxicity[0].score == score
            assert toxicity[0].location == record.location
        else:
            assert toxicity == []


# --------------------------------------------------------------------------- #
# Property 19: Invalid toxicity threshold is rejected and defaulted
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 19: Invalid toxicity threshold is rejected and defaulted
class TestInvalidToxicityThresholdDefaulted:
    @_SETTINGS
    @given(records(), thresholds(), invalid_thresholds())
    def test_invalid_threshold_is_rejected_and_default_retained(
        self, record, score, invalid
    ):
        """**Validates: Requirements 7.5**

        An out-of-range threshold is rejected with exactly one CONFIG_ERROR
        issue identifying the invalid value, and detection proceeds with the
        retained default of 0.8 - so the non-config issues are identical to
        those produced by an explicit default-threshold run.
        """
        model = StubToxicityModel(score=score)
        detector = ToxicityDetector(model)
        issues = detector.detect(record, threshold=invalid)

        config = [i for i in issues if i.category is IssueCategory.CONFIG_ERROR]
        assert len(config) == 1
        assert config[0].field_name == "toxicity_threshold"
        # The error indication identifies the invalid value.
        assert repr(invalid) in config[0].detail

        non_config = [
            i for i in issues if i.category is not IssueCategory.CONFIG_ERROR
        ]
        defaulted = ToxicityDetector(StubToxicityModel(score=score)).detect(
            record, threshold=DEFAULT_TOXICITY_THRESHOLD
        )
        assert non_config == defaulted
