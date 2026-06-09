"""Unit tests for the Quality_Detector (Requirements 8.1-8.5).

These example-based tests cover the too-short, gibberish, and empty-record
checks, the exactly-one-issue-per-category and record-unmodified guarantees,
the documented defaults, and fail-safe rejection of invalid configuration that
retains the previously applied default. Property-based coverage lives in a
separate task.
"""

import copy

from analyzer.detectors.quality import (
    DEFAULT_GIBBERISH_THRESHOLD,
    DEFAULT_MIN_TOKEN_THRESHOLD,
    MIN_TOKEN_UPPER_BOUND,
    QualityConfig,
    QualityDetector,
)
from analyzer.errors import ConfigError
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


class TestTooShort:
    def test_below_threshold_is_flagged_once(self):
        # Two tokens, default threshold 3 -> too short.
        record = _rec({"text": "hello world"})
        issues = QualityDetector().detect(record)
        short = [i for i in issues if i.category is IssueCategory.LOW_QUALITY_SHORT]
        assert len(short) == 1
        assert short[0].location == record.location

    def test_at_threshold_is_not_flagged(self):
        # Exactly three tokens; "strictly less than" means not flagged.
        record = _rec({"text": "one two three"})
        issues = QualityDetector().detect(record)
        assert IssueCategory.LOW_QUALITY_SHORT not in _categories(issues)

    def test_above_threshold_is_not_flagged(self):
        record = _rec({"text": "one two three four five"})
        issues = QualityDetector().detect(record)
        assert IssueCategory.LOW_QUALITY_SHORT not in _categories(issues)

    def test_token_count_spans_multiple_fields(self):
        record = _rec({"a": "alpha beta", "b": "gamma delta"})
        issues = QualityDetector().detect(record)
        assert IssueCategory.LOW_QUALITY_SHORT not in _categories(issues)

    def test_custom_threshold(self):
        record = _rec({"text": "one two three four"})
        cfg = QualityConfig.resolve(min_token_threshold=5).config
        issues = QualityDetector().detect(record, cfg)
        assert IssueCategory.LOW_QUALITY_SHORT in _categories(issues)


class TestGibberish:
    def test_high_non_alnum_proportion_is_flagged_once(self):
        # All punctuation -> proportion 1.0 >= 0.5.
        record = _rec({"text": "!@#$%^&*()"})
        issues = QualityDetector().detect(record)
        gib = [
            i for i in issues if i.category is IssueCategory.LOW_QUALITY_GIBBERISH
        ]
        assert len(gib) == 1
        assert gib[0].score is not None and gib[0].score >= 0.5

    def test_clean_text_is_not_flagged(self):
        # Letters only, no separators -> proportion 0.0.
        record = _rec({"text": "wellformedalphanumericcontent123"})
        issues = QualityDetector().detect(record)
        assert IssueCategory.LOW_QUALITY_GIBBERISH not in _categories(issues)

    def test_at_threshold_is_flagged(self):
        # Two chars, one non-alnum -> proportion exactly 0.5 -> flagged (>=).
        record = _rec({"text": "a!"})
        cfg = QualityConfig()
        issues = QualityDetector().detect(record, cfg)
        assert IssueCategory.LOW_QUALITY_GIBBERISH in _categories(issues)

    def test_empty_text_has_zero_proportion(self):
        # No characters -> proportion defined as 0.0, no gibberish issue.
        record = _rec({"text": ""})
        issues = QualityDetector().detect(record)
        assert IssueCategory.LOW_QUALITY_GIBBERISH not in _categories(issues)


class TestEmpty:
    def test_all_blank_fields_flagged_once(self):
        record = _rec({"a": "", "b": "   ", "c": "\t\n"})
        issues = QualityDetector().detect(record)
        empties = [i for i in issues if i.category is IssueCategory.EMPTY_RECORD]
        assert len(empties) == 1
        assert empties[0].location == record.location

    def test_one_nonblank_field_prevents_empty(self):
        record = _rec({"a": "", "b": "content"})
        issues = QualityDetector().detect(record)
        assert IssueCategory.EMPTY_RECORD not in _categories(issues)

    def test_none_value_is_blank(self):
        record = _rec({"a": None, "b": "   "})
        issues = QualityDetector().detect(record)
        assert IssueCategory.EMPTY_RECORD in _categories(issues)

    def test_numeric_field_is_not_blank(self):
        record = _rec({"a": "", "b": 0})
        issues = QualityDetector().detect(record)
        assert IssueCategory.EMPTY_RECORD not in _categories(issues)

    def test_required_fields_subset(self):
        # Only "a" is required; it is blank even though "b" has content.
        record = _rec({"a": "  ", "b": "content here please"})
        cfg = QualityConfig.resolve(required_fields=("a",)).config
        issues = QualityDetector().detect(record, cfg)
        assert IssueCategory.EMPTY_RECORD in _categories(issues)

    def test_missing_required_field_counts_as_blank(self):
        record = _rec({"b": "content here please"})
        cfg = QualityConfig.resolve(required_fields=("a",)).config
        issues = QualityDetector().detect(record, cfg)
        assert IssueCategory.EMPTY_RECORD in _categories(issues)


class TestRecordUnmodifiedAndOnePerCategory:
    def test_record_is_left_unmodified(self):
        record = _rec({"text": "!@"})
        before = copy.deepcopy(record)
        QualityDetector().detect(record)
        assert record == before

    def test_at_most_one_issue_per_category(self):
        # A short, all-punctuation, single blank-ish field can trip several
        # checks; each category must still appear at most once.
        record = _rec({"text": "!!"})
        issues = QualityDetector().detect(record)
        cats = _categories(issues)
        for category in (
            IssueCategory.LOW_QUALITY_SHORT,
            IssueCategory.LOW_QUALITY_GIBBERISH,
            IssueCategory.EMPTY_RECORD,
        ):
            assert cats.count(category) <= 1

    def test_independent_checks_can_co_occur(self):
        # "!!" -> 1 token (short), proportion 1.0 (gibberish). Not empty
        # because the field has non-whitespace content.
        record = _rec({"text": "!!"})
        issues = QualityDetector().detect(record)
        cats = _categories(issues)
        assert IssueCategory.LOW_QUALITY_SHORT in cats
        assert IssueCategory.LOW_QUALITY_GIBBERISH in cats
        assert IssueCategory.EMPTY_RECORD not in cats


class TestDefaults:
    def test_documented_defaults(self):
        assert DEFAULT_MIN_TOKEN_THRESHOLD == 3
        assert DEFAULT_GIBBERISH_THRESHOLD == 0.5
        cfg = QualityConfig()
        assert cfg.min_token_threshold == 3
        assert cfg.gibberish_threshold == 0.5

    def test_resolve_with_defaults_has_no_errors(self):
        resolution = QualityConfig.resolve()
        assert resolution.errors == []
        assert resolution.config.min_token_threshold == 3
        assert resolution.config.gibberish_threshold == 0.5


class TestConfigValidation:
    def test_valid_overrides_accepted(self):
        resolution = QualityConfig.resolve(
            min_token_threshold=10, gibberish_threshold=0.25
        )
        assert resolution.errors == []
        assert resolution.config.min_token_threshold == 10
        assert resolution.config.gibberish_threshold == 0.25

    def test_boundary_values_accepted(self):
        for mt in (1, MIN_TOKEN_UPPER_BOUND):
            assert QualityConfig.resolve(min_token_threshold=mt).errors == []
        for gt in (0.0, 1.0):
            assert QualityConfig.resolve(gibberish_threshold=gt).errors == []

    def test_min_token_zero_rejected_default_retained(self):
        resolution = QualityConfig.resolve(min_token_threshold=0)
        assert len(resolution.errors) == 1
        assert isinstance(resolution.errors[0], ConfigError)
        assert resolution.errors[0].parameter == "min_token_threshold"
        # Previously applied default retained (Requirement 8.5).
        assert resolution.config.min_token_threshold == DEFAULT_MIN_TOKEN_THRESHOLD

    def test_min_token_above_upper_bound_rejected(self):
        resolution = QualityConfig.resolve(
            min_token_threshold=MIN_TOKEN_UPPER_BOUND + 1
        )
        assert len(resolution.errors) == 1
        assert resolution.config.min_token_threshold == DEFAULT_MIN_TOKEN_THRESHOLD

    def test_min_token_non_integer_rejected(self):
        resolution = QualityConfig.resolve(min_token_threshold=3.5)
        assert len(resolution.errors) == 1
        assert resolution.config.min_token_threshold == DEFAULT_MIN_TOKEN_THRESHOLD

    def test_min_token_bool_rejected(self):
        resolution = QualityConfig.resolve(min_token_threshold=True)
        assert len(resolution.errors) == 1
        assert resolution.config.min_token_threshold == DEFAULT_MIN_TOKEN_THRESHOLD

    def test_gibberish_out_of_range_rejected_default_retained(self):
        resolution = QualityConfig.resolve(gibberish_threshold=1.5)
        assert len(resolution.errors) == 1
        assert resolution.errors[0].parameter == "gibberish_threshold"
        assert (
            resolution.config.gibberish_threshold == DEFAULT_GIBBERISH_THRESHOLD
        )

    def test_gibberish_negative_rejected(self):
        resolution = QualityConfig.resolve(gibberish_threshold=-0.1)
        assert len(resolution.errors) == 1
        assert (
            resolution.config.gibberish_threshold == DEFAULT_GIBBERISH_THRESHOLD
        )

    def test_gibberish_nan_rejected(self):
        resolution = QualityConfig.resolve(gibberish_threshold=float("nan"))
        assert len(resolution.errors) == 1
        assert (
            resolution.config.gibberish_threshold == DEFAULT_GIBBERISH_THRESHOLD
        )

    def test_both_invalid_yields_two_errors(self):
        resolution = QualityConfig.resolve(
            min_token_threshold=-5, gibberish_threshold=2.0
        )
        assert len(resolution.errors) == 2
        params = {e.parameter for e in resolution.errors}
        assert params == {"min_token_threshold", "gibberish_threshold"}

    def test_invalid_value_retains_prior_override_not_default(self):
        # A previously applied (non-default) valid config is the base; an
        # invalid new value retains the prior applied value, not the default.
        base = QualityConfig.resolve(
            min_token_threshold=7, gibberish_threshold=0.2
        ).config
        resolution = QualityConfig.resolve(
            min_token_threshold=0, gibberish_threshold=0.3, base=base
        )
        assert len(resolution.errors) == 1
        # Retains the prior applied 7, not the documented default 3.
        assert resolution.config.min_token_threshold == 7
        # Valid new gibberish value applied.
        assert resolution.config.gibberish_threshold == 0.3
