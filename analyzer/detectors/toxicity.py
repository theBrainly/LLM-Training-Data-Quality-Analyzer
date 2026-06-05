"""Toxicity detection behind a pluggable model interface (Requirement 7).

Toxicity scoring is abstracted behind the :class:`ToxicityModel` interface so
that the *threshold-driven detection logic* implemented here can be tested with
a deterministic stub, independent of any real (and non-deterministic) external
classifier. The detector itself owns only the surrounding business rules:

* It assigns each :class:`~analyzer.models.Record` a single toxicity score that
  is a numeric value in ``[0.0, 1.0]`` (Requirement 7.1). The score returned by
  the model is validated and clamped into that range so the bound always holds.
* It flags a record with exactly one ``TOXICITY``
  :class:`~analyzer.models.QualityIssue` carrying the score when, and only when,
  the score is *greater than or equal to* the configured threshold
  (Requirements 7.2, 7.3) - a ``>=`` comparison.
* The default toxicity threshold is ``0.8`` (Requirement 7.4).
* Configuration is *fail-safe*: a threshold that is non-numeric or outside the
  inclusive range ``[0.0, 1.0]`` is rejected, the default ``0.8`` is retained,
  and a ``CONFIG_ERROR`` issue carrying the invalid value is recorded
  (Requirement 7.5).
* Scoring is *fail-soft*: if the model fails to produce a usable score (it
  raises, returns ``None``, or returns a non-numeric / non-finite value), the
  record is left *unscored* and a single scoring-failure issue
  (``ANALYSIS_FAILURE``) identifying the record is recorded (Requirement 7.6).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from analyzer.errors import ConfigError
from analyzer.models import (
    IssueCategory,
    QualityIssue,
    Record,
    Value,
)

__all__ = [
    "ToxicityModel",
    "StubToxicityModel",
    "ToxicityDetector",
    "DEFAULT_TOXICITY_THRESHOLD",
]

# Default toxicity threshold used when none is configured or when a configured
# value is rejected as invalid (Requirements 7.4, 7.5).
DEFAULT_TOXICITY_THRESHOLD: float = 0.8


@runtime_checkable
class ToxicityModel(Protocol):
    """Interface for a pluggable toxicity classifier.

    Implementations receive the textual content of a record and return a
    toxicity score. A return value of ``None`` (or a raised exception) signals
    that the model could not produce a score for the given text, which the
    :class:`ToxicityDetector` surfaces as a scoring failure (Requirement 7.6).
    """

    def score(self, text: str) -> float | None:
        """Return a toxicity score for ``text`` or ``None`` if unscorable."""
        ...


class StubToxicityModel:
    """A deterministic stand-in toxicity model for wiring and testing.

    This product-side stub lets the analysis pipeline be assembled and exercised
    without a real classifier. It is fully deterministic: the same input always
    yields the same score. It can be configured to

    * return a single fixed ``score`` for every input (the default),
    * compute a score from the text via a ``score_fn`` callable, or
    * simulate a scoring failure by setting ``fail=True`` (Requirement 7.6).

    Returned scores are clamped into ``[0.0, 1.0]`` so the stub honours the
    bound the detector relies on (Requirement 7.1).
    """

    def __init__(
        self,
        *,
        score: float = 0.0,
        score_fn=None,
        fail: bool = False,
    ) -> None:
        self._score = score
        self._score_fn = score_fn
        self.fail = fail

    def score(self, text: str) -> float | None:
        """Return a deterministic toxicity score for ``text``."""
        if self.fail:
            raise RuntimeError("toxicity scoring failed")

        value = self._score_fn(text) if self._score_fn is not None else self._score
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))


def _record_text(record: Record) -> str:
    """Concatenate the textual content of a record's string fields.

    Only string field values contribute to the text handed to the model; other
    value types are ignored. Fields are joined in declaration order with single
    spaces so the model sees a stable, deterministic input for a given record.
    """
    parts: list[str] = []
    for value in record.fields.values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


class ToxicityDetector:
    """Scores records for toxicity and flags those at or above a threshold."""

    def __init__(self, model: ToxicityModel) -> None:
        self._model = model

    def score(self, record: Record) -> float | None:
        """Return ``record``'s toxicity score in ``[0.0, 1.0]`` or ``None``.

        The record's textual content is passed to the configured
        :class:`ToxicityModel`. A valid, finite numeric result is clamped into
        ``[0.0, 1.0]`` and returned (Requirement 7.1). If the model raises,
        returns ``None``, or returns a non-numeric / non-finite value, the
        record is treated as *unscored* and ``None`` is returned
        (Requirement 7.6).
        """
        try:
            raw = self._model.score(_record_text(record))
        except Exception:  # noqa: BLE001 - fail-soft per Requirement 7.6
            return None
        return _coerce_score(raw)

    def detect(
        self, record: Record, threshold: float = DEFAULT_TOXICITY_THRESHOLD
    ) -> list[QualityIssue]:
        """Return the Quality_Issues for ``record`` under ``threshold``.

        Behaviour:

        * An invalid ``threshold`` (non-numeric or outside ``[0.0, 1.0]``) is
          rejected: the default ``0.8`` is used instead and a ``CONFIG_ERROR``
          issue carrying the invalid value is prepended (Requirement 7.5).
        * If scoring fails, a single ``ANALYSIS_FAILURE`` scoring-failure issue
          identifying the record is recorded and no toxicity issue is produced
          (Requirement 7.6).
        * Otherwise exactly one ``TOXICITY`` issue carrying the score is
          recorded iff the score is ``>=`` the effective threshold
          (Requirements 7.2, 7.3).
        """
        issues: list[QualityIssue] = []

        effective_threshold = self._resolve_threshold(threshold, issues)

        score = self.score(record)

        if score is None:
            issues.append(
                QualityIssue(
                    category=IssueCategory.ANALYSIS_FAILURE,
                    location=record.location,
                    detail="Toxicity scoring failed; record left unscored",
                )
            )
            return issues

        if score >= effective_threshold:
            issues.append(
                QualityIssue(
                    category=IssueCategory.TOXICITY,
                    location=record.location,
                    score=score,
                    detail=(
                        f"Toxicity score {score:.3f} >= threshold "
                        f"{effective_threshold:.3f}"
                    ),
                )
            )

        return issues

    @staticmethod
    def _resolve_threshold(threshold: float, issues: list[QualityIssue]) -> float:
        """Validate ``threshold`` and return the effective value to use.

        On rejection the default ``0.8`` is returned and a ``CONFIG_ERROR``
        issue identifying the invalid value is appended to ``issues``
        (Requirement 7.5).
        """
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            valid = False
        else:
            numeric = float(threshold)
            valid = math.isfinite(numeric) and 0.0 <= numeric <= 1.0

        if valid:
            return float(threshold)

        error = ConfigError(
            parameter="toxicity_threshold",
            invalid_value=threshold,
            retained_default=DEFAULT_TOXICITY_THRESHOLD,
        )
        issues.append(
            QualityIssue(
                category=IssueCategory.CONFIG_ERROR,
                location=None,
                detail=str(error),
                field_name="toxicity_threshold",
            )
        )
        return DEFAULT_TOXICITY_THRESHOLD


def _coerce_score(raw: Value | None) -> float | None:
    """Return a finite score clamped to ``[0.0, 1.0]`` or ``None`` if unusable.

    A usable score is a real, finite number (booleans are rejected, since a
    boolean is not a meaningful toxicity score). Anything else - ``None``, a
    string, ``NaN``/``inf``, or a boolean - is treated as a failure to produce a
    score (Requirement 7.6).
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    numeric = float(raw)
    if not math.isfinite(numeric):
        return None
    return max(0.0, min(1.0, numeric))
