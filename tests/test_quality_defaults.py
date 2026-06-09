"""Unit tests for Quality_Detector default thresholds (task 10.6).

These example-based tests confirm that, when the quality thresholds are *not*
configured, the Quality_Detector applies the documented defaults of a minimum
token threshold of 3 and a gibberish threshold of 0.5 (Requirement 8.4):

* the exported constants and the :class:`QualityConfig` defaults carry those
  values;
* :meth:`QualityConfig.resolve` with no arguments yields them and no errors;
* an unconfigured ``detect`` (no ``cfg``) behaves identically to an explicit
  default :class:`QualityConfig`, flagging exactly at the default boundaries.
"""

from analyzer.detectors.quality import (
    DEFAULT_GIBBERISH_THRESHOLD,
    DEFAULT_MIN_TOKEN_THRESHOLD,
    QualityConfig,
    QualityDetector,
)
from analyzer.models import (
    IssueCategory,
    Record,
    RecordLocation,
)


def _rec(fields: dict, index: int = 0) -> Record:
    return Record(
        fields=fields,
        location=RecordLocation(source_file="data", array_index=index),
    )


def _categories(issues) -> list[IssueCategory]:
    return [i.category for i in issues]


class TestDefaultConstantsAndConfig:
    def test_default_constants(self):
        assert DEFAULT_MIN_TOKEN_THRESHOLD == 3
        assert DEFAULT_GIBBERISH_THRESHOLD == 0.5

    def test_config_defaults(self):
        cfg = QualityConfig()
        assert cfg.min_token_threshold == 3
        assert cfg.gibberish_threshold == 0.5

    def test_resolve_with_no_args_uses_defaults(self):
        resolution = QualityConfig.resolve()
        assert resolution.errors == []
        assert resolution.config.min_token_threshold == 3
        assert resolution.config.gibberish_threshold == 0.5


class TestDefaultMinTokenAppliedWhenUnconfigured:
    def test_two_tokens_flagged_short_at_default(self):
        # 2 tokens < default 3 -> too short.
        issues = QualityDetector().detect(_rec({"text": "hello world"}))
        assert IssueCategory.LOW_QUALITY_SHORT in _categories(issues)

    def test_three_tokens_not_flagged_at_default(self):
        # Exactly 3 tokens; "strictly less than" -> not flagged at default.
        issues = QualityDetector().detect(_rec({"text": "one two three"}))
        assert IssueCategory.LOW_QUALITY_SHORT not in _categories(issues)


class TestDefaultGibberishAppliedWhenUnconfigured:
    def test_proportion_at_default_boundary_flagged(self):
        # "a!" -> 1 of 2 chars non-alnum -> proportion 0.5 == default 0.5 -> flagged.
        issues = QualityDetector().detect(_rec({"text": "a!"}))
        assert IssueCategory.LOW_QUALITY_GIBBERISH in _categories(issues)

    def test_proportion_below_default_not_flagged(self):
        # "abc!" -> 1 of 4 chars non-alnum -> proportion 0.25 < default 0.5.
        issues = QualityDetector().detect(_rec({"text": "abc!"}))
        assert IssueCategory.LOW_QUALITY_GIBBERISH not in _categories(issues)


class TestUnconfiguredMatchesExplicitDefault:
    def test_detect_without_cfg_matches_explicit_default_config(self):
        records = [
            _rec({"text": "hello world"}),                 # short
            _rec({"text": "a!"}),                           # gibberish + short
            _rec({"a": "", "b": "   "}),                    # empty + short
            _rec({"text": "one two three four five"}),      # clean
        ]
        detector = QualityDetector()
        for record in records:
            assert detector.detect(record) == detector.detect(
                record, QualityConfig()
            )
