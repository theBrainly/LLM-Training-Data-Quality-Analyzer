"""Unit tests for toxicity default threshold and scoring-failure paths.

Covers the edge-case requirements that complement the Toxicity_Detector
properties:

* Requirement 7.4 - the default threshold ``0.8`` is used when none is
  configured.
* Requirement 7.6 - a scoring failure leaves the Record unscored and records a
  single scoring-failure issue, with no toxicity issue produced.
"""

from __future__ import annotations

from analyzer.detectors.toxicity import (
    DEFAULT_TOXICITY_THRESHOLD,
    ToxicityDetector,
)
from analyzer.models import IssueCategory, Record, RecordLocation
from tests.strategies import StubToxicityModel


def _record(text: str = "some content") -> Record:
    return Record(
        fields={"text": text},
        location=RecordLocation(source_file="data.jsonl", line_number=1),
    )


class TestDefaultThreshold:
    """Requirement 7.4: default toxicity threshold is 0.8 when unconfigured."""

    def test_default_constant_is_point_eight(self):
        assert DEFAULT_TOXICITY_THRESHOLD == 0.8

    def test_score_at_default_boundary_is_flagged_when_unconfigured(self):
        # No explicit threshold => default 0.8 applies; 0.8 >= 0.8 flags.
        detector = ToxicityDetector(StubToxicityModel(score=0.8))
        issues = detector.detect(_record())
        toxicity = [i for i in issues if i.category is IssueCategory.TOXICITY]
        assert len(toxicity) == 1
        assert toxicity[0].score == 0.8

    def test_score_just_below_default_not_flagged_when_unconfigured(self):
        # 0.79 < default 0.8 => no toxicity issue.
        detector = ToxicityDetector(StubToxicityModel(score=0.79))
        assert detector.detect(_record()) == []

    def test_unconfigured_matches_explicit_default(self):
        # Omitting the threshold is equivalent to passing the documented default.
        for score in (0.0, 0.5, 0.8, 1.0):
            implicit = ToxicityDetector(StubToxicityModel(score=score)).detect(
                _record()
            )
            explicit = ToxicityDetector(StubToxicityModel(score=score)).detect(
                _record(), threshold=DEFAULT_TOXICITY_THRESHOLD
            )
            assert implicit == explicit


class TestScoringFailureLeavesRecordUnscored:
    """Requirement 7.6: scoring failure leaves the record unscored with an issue."""

    def test_model_raising_leaves_unscored(self):
        detector = ToxicityDetector(StubToxicityModel(fail=True))
        assert detector.score(_record()) is None

    def test_failure_records_single_scoring_failure_issue(self):
        detector = ToxicityDetector(StubToxicityModel(fail=True))
        issues = detector.detect(_record())

        assert len(issues) == 1
        assert issues[0].category is IssueCategory.ANALYSIS_FAILURE
        assert issues[0].location == _record().location

    def test_failure_produces_no_toxicity_issue(self):
        detector = ToxicityDetector(StubToxicityModel(fail=True))
        issues = detector.detect(_record())
        assert not any(i.category is IssueCategory.TOXICITY for i in issues)

    def test_model_returning_none_is_a_failure(self):
        detector = ToxicityDetector(StubToxicityModel(score_fn=lambda _t: None))
        assert detector.score(_record()) is None
        issues = detector.detect(_record())
        assert [i.category for i in issues] == [IssueCategory.ANALYSIS_FAILURE]
