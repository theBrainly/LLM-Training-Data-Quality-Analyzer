"""Property-based tests for the Quality_Detector (Requirements 8.1-8.5).

Each test validates exactly one Correctness Property from the design's
32-property numbering and references the requirement clause it covers:

* Property 20 - too-short detection (Requirement 8.1).
* Property 21 - gibberish detection (Requirement 8.2).
* Property 22 - empty-record detection (Requirement 8.3).
* Property 23 - invalid quality config is rejected and the prior default
  retained (Requirement 8.5).

For the metric-driven properties (20-22) the test recomputes the relevant
metric from the Record exactly as the detector defines it - text is flattened
over the fields (strings verbatim, numbers as their textual form, booleans and
``None`` contributing nothing, lists/dicts flattened over their elements) and
joined with single spaces - to form an independent ground truth. The detector
must flag the Record for a category if and only if that ground-truth metric
crosses the configured threshold, must emit at most one issue per category, and
must leave the Record unmodified.
"""

from __future__ import annotations

import copy
import string

from hypothesis import given, settings
from hypothesis import strategies as st

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
    Value,
)
from tests.strategies import field_names, records

_SETTINGS = settings(max_examples=100, deadline=None)


# --------------------------------------------------------------------------- #
# Independent ground-truth recomputation (mirrors the spec's definitions)
# --------------------------------------------------------------------------- #

def _flatten(value: Value, out: list[str]) -> None:
    """Collect the textual content of ``value`` recursively (Requirement 8).

    Strings contribute verbatim; numbers contribute ``str(value)``; booleans
    and ``None`` contribute nothing; lists and dicts are flattened over their
    elements / values.
    """
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (int, float)):
        out.append(str(value))
    elif isinstance(value, list):
        for item in value:
            _flatten(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten(item, out)


def _record_text(record: Record) -> str:
    parts: list[str] = []
    for key in record.fields:
        _flatten(record.fields[key], parts)
    return " ".join(parts)


def _token_count(text: str) -> int:
    return len(text.split())


def _non_alnum_proportion(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    return sum(1 for ch in text if not ch.isalnum()) / total


def _is_blank(value: Value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _categories(issues) -> list[IssueCategory]:
    return [i.category for i in issues]


# --------------------------------------------------------------------------- #
# Empty-record generators (blank vs. content fields toggling the empty check)
# --------------------------------------------------------------------------- #

_BLANK_VALUES = st.one_of(
    st.just(""),
    st.text(alphabet=" \t\n", min_size=1, max_size=4),  # whitespace-only
    st.none(),
    st.just([]),
    st.just({}),
)

_CONTENT_VALUES = st.one_of(
    st.text(alphabet=string.ascii_letters, min_size=1, max_size=6),
    st.integers(),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.integers(), min_size=1, max_size=3),
    st.dictionaries(field_names(), st.integers(), min_size=1, max_size=2),
)


@st.composite
def _blankable_records(draw) -> Record:
    """A Record whose fields are a mix of blank and content-bearing values."""
    names = draw(st.lists(field_names(), min_size=0, max_size=5, unique=True))
    fields: dict[str, Value] = {
        name: draw(st.one_of(_BLANK_VALUES, _CONTENT_VALUES)) for name in names
    }
    index = draw(st.integers(min_value=0, max_value=100))
    return Record(
        fields=fields,
        location=RecordLocation(source_file="data", array_index=index),
    )


# --------------------------------------------------------------------------- #
# Invalid-config generators for Property 23
# --------------------------------------------------------------------------- #

def _invalid_min_tokens() -> st.SearchStrategy[object]:
    """A minimum-token value outside the valid integer range [1, 1_000_000]."""
    return st.one_of(
        st.integers(max_value=0),
        st.integers(
            min_value=MIN_TOKEN_UPPER_BOUND + 1,
            max_value=MIN_TOKEN_UPPER_BOUND + 10_000,
        ),
        st.floats(allow_nan=False, allow_infinity=False),  # any float is non-int
        st.booleans(),  # bool is rejected even though it subclasses int
    )


def _invalid_gibberish() -> st.SearchStrategy[object]:
    """A gibberish value outside the valid numeric range [0.0, 1.0]."""
    return st.one_of(
        st.floats(
            min_value=1.0 + 1e-6, max_value=1e6,
            allow_nan=False, allow_infinity=False,
        ),
        st.floats(
            min_value=-1e6, max_value=-1e-6,
            allow_nan=False, allow_infinity=False,
        ),
        st.text(max_size=4),  # non-numeric
        st.none(),
    )


# --------------------------------------------------------------------------- #
# Property 20: Too-short detection
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 20: Too-short detection
class TestTooShortDetection:
    @_SETTINGS
    @given(records(), st.integers(min_value=1, max_value=8))
    def test_flagged_iff_token_count_below_threshold(self, record, threshold):
        """**Validates: Requirements 8.1**

        The Quality_Detector records exactly one ``LOW_QUALITY_SHORT`` issue if
        and only if the Record's whitespace-delimited token count is strictly
        less than the configured minimum token threshold, and the Record is
        returned unmodified.
        """
        before = copy.deepcopy(record)
        cfg = QualityConfig.resolve(min_token_threshold=threshold).config

        issues = QualityDetector().detect(record, cfg)
        short = [
            i for i in issues if i.category is IssueCategory.LOW_QUALITY_SHORT
        ]

        expected_tokens = _token_count(_record_text(record))
        if expected_tokens < threshold:
            assert len(short) == 1
            assert short[0].location == record.location
            assert short[0].score == float(expected_tokens)
        else:
            assert short == []

        # The Record is never mutated.
        assert record == before


# --------------------------------------------------------------------------- #
# Property 21: Gibberish detection
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 21: Gibberish detection
class TestGibberishDetection:
    @_SETTINGS
    @given(
        records(),
        st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_flagged_iff_non_alnum_proportion_meets_threshold(
        self, record, threshold
    ):
        """**Validates: Requirements 8.2**

        The Quality_Detector records exactly one ``LOW_QUALITY_GIBBERISH`` issue
        if and only if the proportion of non-alphanumeric characters (count of
        non-alphanumeric characters divided by total character count) meets or
        exceeds the configured gibberish threshold, and the Record is returned
        unmodified.
        """
        before = copy.deepcopy(record)
        cfg = QualityConfig.resolve(gibberish_threshold=threshold).config

        issues = QualityDetector().detect(record, cfg)
        gib = [
            i for i in issues if i.category is IssueCategory.LOW_QUALITY_GIBBERISH
        ]

        proportion = _non_alnum_proportion(_record_text(record))
        if proportion >= threshold:
            assert len(gib) == 1
            assert gib[0].location == record.location
            assert gib[0].score == proportion
        else:
            assert gib == []

        assert record == before


# --------------------------------------------------------------------------- #
# Property 22: Empty-record detection
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 22: Empty-record detection
class TestEmptyRecordDetection:
    @_SETTINGS
    @given(_blankable_records())
    def test_flagged_iff_all_required_fields_blank(self, record):
        """**Validates: Requirements 8.3**

        With the default config (every present field is required), the
        Quality_Detector records exactly one ``EMPTY_RECORD`` issue if and only
        if every field of the Record is zero-length or whitespace-only, and the
        Record is returned unmodified.
        """
        before = copy.deepcopy(record)

        issues = QualityDetector().detect(record)
        empty = [i for i in issues if i.category is IssueCategory.EMPTY_RECORD]

        all_blank = all(_is_blank(v) for v in record.fields.values())
        if all_blank:
            assert len(empty) == 1
            assert empty[0].location == record.location
        else:
            assert empty == []

        assert record == before


# --------------------------------------------------------------------------- #
# Property 23: Invalid quality config is rejected and the prior default retained
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 23: Invalid quality config is rejected and the prior default retained
class TestInvalidConfigRejectedDefaultRetained:
    @_SETTINGS
    @given(
        # An optional prior-applied (valid) config establishing the value to
        # retain; ``None`` means the documented defaults are the prior values.
        st.one_of(
            st.none(),
            st.tuples(
                st.integers(min_value=1, max_value=MIN_TOKEN_UPPER_BOUND),
                st.floats(
                    min_value=0.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                ),
            ),
        ),
        _invalid_min_tokens(),
        _invalid_gibberish(),
    )
    def test_invalid_values_rejected_and_prior_retained(
        self, prior, invalid_min, invalid_gib
    ):
        """**Validates: Requirements 8.5**

        When both thresholds are configured with values outside their valid
        ranges, :meth:`QualityConfig.resolve` rejects each one, retains the
        previously applied value (a prior valid override when supplied, else the
        documented default), and records a :class:`ConfigError` per rejected
        value identifying the invalid value and the retained default.
        """
        if prior is None:
            base = None
            expected_min = DEFAULT_MIN_TOKEN_THRESHOLD
            expected_gib = DEFAULT_GIBBERISH_THRESHOLD
        else:
            base = QualityConfig.resolve(
                min_token_threshold=prior[0], gibberish_threshold=prior[1]
            ).config
            assert base.min_token_threshold == prior[0]
            assert base.gibberish_threshold == prior[1]
            expected_min = prior[0]
            expected_gib = prior[1]

        resolution = QualityConfig.resolve(
            min_token_threshold=invalid_min,
            gibberish_threshold=invalid_gib,
            base=base,
        )

        # The invalid values never take effect; the prior values are retained.
        assert resolution.config.min_token_threshold == expected_min
        assert resolution.config.gibberish_threshold == expected_gib

        # Exactly one error per rejected parameter, each a ConfigError naming
        # the parameter, the invalid value, and the retained default.
        assert len(resolution.errors) == 2
        assert all(isinstance(e, ConfigError) for e in resolution.errors)
        by_param = {e.parameter: e for e in resolution.errors}
        assert set(by_param) == {"min_token_threshold", "gibberish_threshold"}

        min_err = by_param["min_token_threshold"]
        assert min_err.invalid_value == invalid_min
        assert min_err.retained_default == expected_min

        gib_err = by_param["gibberish_threshold"]
        assert gib_err.invalid_value == invalid_gib
        assert gib_err.retained_default == expected_gib
