"""Low-quality content detection (Requirement 8).

The :class:`QualityDetector` inspects a single :class:`~analyzer.models.Record`
and emits :class:`~analyzer.models.QualityIssue`\\ s for three independent,
configurable heuristics. None of the checks mutate the Record - each returns
the Record untouched and merely reports issues:

* **Too short** - the Record's whitespace-delimited token count is strictly
  less than the configured minimum token threshold. Emits exactly one
  ``LOW_QUALITY_SHORT`` issue (Requirement 8.1).
* **Gibberish** - the proportion of non-alphanumeric characters in the Record's
  text (count of non-alphanumeric characters divided by the total character
  count, yielding a value in ``[0.0, 1.0]``) meets or exceeds the configured
  gibberish threshold. Emits exactly one ``LOW_QUALITY_GIBBERISH`` issue
  (Requirement 8.2).
* **Empty** - every required field of the Record is zero-length or contains
  only whitespace characters. Emits exactly one ``EMPTY_RECORD`` issue
  (Requirement 8.3).

The three checks are independent: a Record may satisfy more than one and thus
receive more than one issue, but never more than one issue *per category*.

Configuration is *fail-safe* (Requirements 8.4, 8.5). The documented defaults
are a minimum token threshold of ``3`` (a positive integer in ``[1, 1_000_000]``)
and a gibberish threshold of ``0.5`` (a value in ``[0.0, 1.0]``). A configured
value outside its valid range is rejected via :meth:`QualityConfig.resolve`:
the previously applied value (the default, or a prior valid override) is
retained and a :class:`~analyzer.errors.ConfigError` is reported, rather than
the invalid value ever taking effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from analyzer.errors import ConfigError
from analyzer.models import (
    IssueCategory,
    QualityIssue,
    Record,
    Value,
)

__all__ = [
    "QualityConfig",
    "QualityDetector",
    "ConfigResolution",
    "DEFAULT_MIN_TOKEN_THRESHOLD",
    "DEFAULT_GIBBERISH_THRESHOLD",
    "MIN_TOKEN_LOWER_BOUND",
    "MIN_TOKEN_UPPER_BOUND",
]

# Documented defaults (Requirement 8.4).
DEFAULT_MIN_TOKEN_THRESHOLD: int = 3
DEFAULT_GIBBERISH_THRESHOLD: float = 0.5

# Valid range for the minimum token threshold: a positive integer in
# [1, 1_000_000] (Requirements 8.4, 8.5).
MIN_TOKEN_LOWER_BOUND: int = 1
MIN_TOKEN_UPPER_BOUND: int = 1_000_000


def _flatten_text(value: Value, out: list[str]) -> None:
    """Collect the textual content of ``value`` into ``out`` recursively.

    Strings contribute verbatim; numbers contribute their textual form;
    booleans and ``None`` contribute nothing (they carry no character content);
    lists and dicts are flattened over their elements/values.
    """
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (int, float)):
        out.append(str(value))
    elif isinstance(value, list):
        for item in value:
            _flatten_text(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten_text(item, out)


def _record_text(record: Record) -> str:
    """Return the Record's textual content as a single space-joined string.

    The textual values of every field are flattened and joined with single
    spaces, giving a deterministic representation over which token and
    character statistics are computed.
    """
    parts: list[str] = []
    for key in record.fields:
        _flatten_text(record.fields[key], parts)
    return " ".join(parts)


def _token_count(text: str) -> int:
    """Number of whitespace-delimited tokens in ``text`` (Requirement 8.1)."""
    return len(text.split())


def _non_alnum_proportion(text: str) -> float:
    """Proportion of non-alphanumeric characters in ``text`` (Requirement 8.2).

    Computed as the count of non-alphanumeric characters divided by the total
    character count, yielding a value in ``[0.0, 1.0]``. Empty text has no
    characters and is defined to have a proportion of ``0.0``.
    """
    total = len(text)
    if total == 0:
        return 0.0
    non_alnum = sum(1 for ch in text if not ch.isalnum())
    return non_alnum / total


def _is_blank(value: Value) -> bool:
    """Return True iff ``value`` is zero-length or whitespace-only content.

    ``None``, the empty string, and whitespace-only strings are blank; empty
    lists and dicts are zero-length and thus blank. Numbers, booleans,
    non-blank strings, and non-empty containers carry content and are not
    blank.
    """
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


@dataclass(frozen=True)
class ConfigResolution:
    """The outcome of validating quality-detector configuration.

    ``config`` is always a usable :class:`QualityConfig` (invalid values never
    take effect); ``errors`` lists a :class:`~analyzer.errors.ConfigError` for
    every rejected value, each identifying the parameter, the invalid value,
    and the retained default (Requirement 8.5).
    """

    config: "QualityConfig"
    errors: list[ConfigError] = field(default_factory=list)


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds governing the low-quality content checks.

    The defaults match Requirement 8.4. Instances are validated up front via
    :meth:`resolve`; constructing a :class:`QualityConfig` directly assumes the
    supplied values are already valid.
    """

    min_token_threshold: int = DEFAULT_MIN_TOKEN_THRESHOLD
    gibberish_threshold: float = DEFAULT_GIBBERISH_THRESHOLD
    # Field names treated as "required" for the empty-record check. ``None``
    # means every field present on the Record is required.
    required_fields: tuple[str, ...] | None = None

    @classmethod
    def resolve(
        cls,
        min_token_threshold: object = DEFAULT_MIN_TOKEN_THRESHOLD,
        gibberish_threshold: object = DEFAULT_GIBBERISH_THRESHOLD,
        required_fields: tuple[str, ...] | None = None,
        base: "QualityConfig | None" = None,
    ) -> ConfigResolution:
        """Validate the supplied thresholds, retaining defaults on rejection.

        Each threshold is validated independently. An invalid value is
        rejected and the previously applied value is retained - the
        corresponding field of ``base`` when provided, otherwise the documented
        default (Requirements 8.4, 8.5). A :class:`~analyzer.errors.ConfigError`
        is recorded for every rejected value. The returned
        :class:`ConfigResolution` always carries a usable config.
        """
        prior = base if base is not None else cls()
        errors: list[ConfigError] = []

        resolved_min = cls._resolve_min_token(
            min_token_threshold, prior.min_token_threshold, errors
        )
        resolved_gibberish = cls._resolve_gibberish(
            gibberish_threshold, prior.gibberish_threshold, errors
        )

        config = replace(
            prior,
            min_token_threshold=resolved_min,
            gibberish_threshold=resolved_gibberish,
            required_fields=required_fields
            if required_fields is not None
            else prior.required_fields,
        )
        return ConfigResolution(config=config, errors=errors)

    @staticmethod
    def _resolve_min_token(
        value: object, retained: int, errors: list[ConfigError]
    ) -> int:
        """Return a valid minimum token threshold, retaining ``retained`` on
        rejection.

        Valid values are integers (booleans excluded) within
        ``[MIN_TOKEN_LOWER_BOUND, MIN_TOKEN_UPPER_BOUND]``.
        """
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and MIN_TOKEN_LOWER_BOUND <= value <= MIN_TOKEN_UPPER_BOUND
        )
        if valid:
            return int(value)
        errors.append(
            ConfigError(
                parameter="min_token_threshold",
                invalid_value=value,
                retained_default=retained,
            )
        )
        return retained

    @staticmethod
    def _resolve_gibberish(
        value: object, retained: float, errors: list[ConfigError]
    ) -> float:
        """Return a valid gibberish threshold, retaining ``retained`` on
        rejection.

        Valid values are real numbers (booleans excluded, NaN excluded) within
        the inclusive range ``[0.0, 1.0]``.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            valid = False
        else:
            numeric = float(value)
            valid = not math.isnan(numeric) and 0.0 <= numeric <= 1.0
        if valid:
            return float(value)
        errors.append(
            ConfigError(
                parameter="gibberish_threshold",
                invalid_value=value,
                retained_default=retained,
            )
        )
        return retained


class QualityDetector:
    """Detects too-short, gibberish, and empty records (Requirement 8)."""

    def detect(
        self, record: Record, cfg: QualityConfig | None = None
    ) -> list[QualityIssue]:
        """Return the low-quality issues for ``record`` under ``cfg``.

        ``cfg`` defaults to a :class:`QualityConfig` with the documented
        thresholds. Each of the three checks contributes at most one issue of
        its own category, and the Record is never modified.
        """
        config = cfg if cfg is not None else QualityConfig()
        issues: list[QualityIssue] = []

        text = _record_text(record)

        # Too short (Requirement 8.1).
        tokens = _token_count(text)
        if tokens < config.min_token_threshold:
            issues.append(
                QualityIssue(
                    category=IssueCategory.LOW_QUALITY_SHORT,
                    location=record.location,
                    detail=(
                        f"Record has {tokens} whitespace-delimited token(s), "
                        f"fewer than the minimum threshold of "
                        f"{config.min_token_threshold}"
                    ),
                    score=float(tokens),
                )
            )

        # Gibberish (Requirement 8.2).
        proportion = _non_alnum_proportion(text)
        if proportion >= config.gibberish_threshold:
            issues.append(
                QualityIssue(
                    category=IssueCategory.LOW_QUALITY_GIBBERISH,
                    location=record.location,
                    detail=(
                        f"Non-alphanumeric proportion {proportion:.3f} meets or "
                        f"exceeds the gibberish threshold of "
                        f"{config.gibberish_threshold}"
                    ),
                    score=proportion,
                )
            )

        # Empty (Requirement 8.3).
        if self._is_empty(record, config):
            issues.append(
                QualityIssue(
                    category=IssueCategory.EMPTY_RECORD,
                    location=record.location,
                    detail=(
                        "Every required field is zero-length or whitespace-only"
                    ),
                )
            )

        return issues

    @staticmethod
    def _is_empty(record: Record, config: QualityConfig) -> bool:
        """Return True iff every required field of ``record`` is blank.

        When ``config.required_fields`` is ``None`` every field present on the
        Record is required. A field named in ``required_fields`` but absent
        from the Record is treated as blank (zero-length). A Record with no
        required fields is considered empty (vacuously, no field carries
        content).
        """
        if config.required_fields is None:
            fields = record.fields
            return all(_is_blank(value) for value in fields.values())

        return all(
            _is_blank(record.fields.get(name)) for name in config.required_fields
        )
