"""Unit tests for threshold-driven toxicity detection (Requirements 7.1-7.6)."""

import pytest

from analyzer.detectors.toxicity import (
    DEFAULT_TOXICITY_THRESHOLD,
    StubToxicityModel,
    ToxicityDetector,
    ToxicityModel,
)
from analyzer.models import IssueCategory, Record, RecordLocation


def _record(text: str, field: str = "text") -> Record:
    return Record(
        fields={field: text},
        location=RecordLocation(source_file="data.jsonl", line_number=1),
    )


class TestStubToxicityModel:
    def test_protocol_membership(self):
        assert isinstance(StubToxicityModel(), ToxicityModel)

    def test_is_deterministic(self):
        model = StubToxicityModel(score_fn=lambda text: len(text) / 100.0)
        assert model.score("hello") == model.score("hello")

    def test_clamps_into_unit_interval(self):
        assert StubToxicityModel(score=5.0).score("x") == 1.0
        assert StubToxicityModel(score=-2.0).score("x") == 0.0

    def test_fail_raises(self):
        with pytest.raises(RuntimeError):
            StubToxicityModel(fail=True).score("x")


class TestScoreBounds:
    def test_score_within_unit_interval(self):
        # Requirement 7.1: assigned score is numeric in [0.0, 1.0].
        detector = ToxicityDetector(StubToxicityModel(score=0.42))
        assert detector.score(_record("anything")) == 0.42

    def test_out_of_range_model_value_is_clamped(self):
        detector = ToxicityDetector(StubToxicityModel(score=0.0, score_fn=lambda t: 9.0))
        assert detector.score(_record("x")) == 1.0


class TestThresholdFlagging:
    def test_score_at_threshold_is_flagged(self):
        # Requirement 7.2: >= threshold flags (boundary equality included).
        detector = ToxicityDetector(StubToxicityModel(score=0.8))
        issues = detector.detect(_record("toxic"), threshold=0.8)
        assert len(issues) == 1
        assert issues[0].category is IssueCategory.TOXICITY
        assert issues[0].score == 0.8

    def test_score_above_threshold_is_flagged(self):
        detector = ToxicityDetector(StubToxicityModel(score=0.95))
        issues = detector.detect(_record("toxic"), threshold=0.8)
        assert len(issues) == 1
        assert issues[0].category is IssueCategory.TOXICITY

    def test_score_below_threshold_not_flagged(self):
        # Requirement 7.3: below threshold produces no issue.
        detector = ToxicityDetector(StubToxicityModel(score=0.79))
        assert detector.detect(_record("mild"), threshold=0.8) == []

    def test_default_threshold_is_point_eight(self):
        # Requirement 7.4: default threshold 0.8.
        assert DEFAULT_TOXICITY_THRESHOLD == 0.8
        detector = ToxicityDetector(StubToxicityModel(score=0.8))
        # No explicit threshold -> default 0.8 applies, so 0.8 is flagged.
        assert len(detector.detect(_record("toxic"))) == 1
        detector_low = ToxicityDetector(StubToxicityModel(score=0.5))
        assert detector_low.detect(_record("mild")) == []


class TestInvalidThresholdConfig:
    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, float("nan"), "0.5", None, True])
    def test_invalid_threshold_rejected_default_retained(self, bad):
        # Requirement 7.5: reject invalid threshold, keep 0.8, record CONFIG_ERROR.
        # Use a score of 0.8 so detection reflects the retained default of 0.8.
        detector = ToxicityDetector(StubToxicityModel(score=0.8))
        issues = detector.detect(_record("toxic"), threshold=bad)

        config_issues = [i for i in issues if i.category is IssueCategory.CONFIG_ERROR]
        assert len(config_issues) == 1
        assert config_issues[0].field_name == "toxicity_threshold"

        # The retained default 0.8 governs detection: score 0.8 is flagged.
        toxicity_issues = [i for i in issues if i.category is IssueCategory.TOXICITY]
        assert len(toxicity_issues) == 1

    def test_invalid_threshold_with_low_score_not_flagged(self):
        detector = ToxicityDetector(StubToxicityModel(score=0.5))
        issues = detector.detect(_record("mild"), threshold=-3.0)
        assert any(i.category is IssueCategory.CONFIG_ERROR for i in issues)
        assert not any(i.category is IssueCategory.TOXICITY for i in issues)


class TestScoringFailure:
    def test_model_raises_leaves_unscored_and_records_issue(self):
        # Requirement 7.6: scoring failure -> unscored + scoring-failure issue.
        detector = ToxicityDetector(StubToxicityModel(fail=True))
        assert detector.score(_record("x")) is None

        issues = detector.detect(_record("x"))
        assert len(issues) == 1
        assert issues[0].category is IssueCategory.ANALYSIS_FAILURE
        assert not any(i.category is IssueCategory.TOXICITY for i in issues)

    def test_model_returns_none_is_a_failure(self):
        detector = ToxicityDetector(StubToxicityModel(score_fn=lambda t: None))
        assert detector.score(_record("x")) is None
        issues = detector.detect(_record("x"))
        assert [i.category for i in issues] == [IssueCategory.ANALYSIS_FAILURE]

    def test_non_finite_model_value_is_a_failure(self):
        # A model that bypasses the stub's clamping and returns a raw
        # non-finite value is treated as a failure to produce a score.
        class _NonFiniteModel:
            def score(self, text):
                return float("inf")

        detector = ToxicityDetector(_NonFiniteModel())
        assert detector.score(_record("x")) is None

    def test_non_numeric_model_value_is_a_failure(self):
        class _StringModel:
            def score(self, text):
                return "very toxic"

        detector = ToxicityDetector(_StringModel())
        assert detector.score(_record("x")) is None
