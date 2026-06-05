"""Duplicate and near-duplicate detection (Requirement 5).

The :class:`DuplicateDetector` identifies two distinct kinds of redundancy in a
:class:`~analyzer.models.Dataset`:

* **Exact duplicates** - a Record that is byte-for-byte identical to a prior
  Record. Identity is decided over a deterministic *canonical content key*
  derived from the Record's ``fields`` (independent of key ordering, matching
  the canonical-value equivalence the rest of the system uses), so two Records
  whose fields are value-equal share a content key. One issue is recorded per
  duplicate Record, referencing the first (original) occurrence
  (Requirements 5.1, 5.3).
* **Near-duplicates** - a *pair* of Records whose normalized textual similarity
  meets or exceeds the configured threshold without being byte-for-byte
  identical (the ``Near_Duplicate`` glossary definition). Similarity is the
  token-set Jaccard coefficient over the Records' normalized word tokens. One
  issue is recorded per near-duplicate pair, referencing both Records
  (Requirements 5.2, 5.3).

Configuration is *fail-safe*: the default similarity threshold is ``0.9``
(Requirement 5.4), and a configured threshold outside the inclusive range
``[0.0, 1.0]`` is rejected, the default ``0.9`` is retained, and a
``CONFIG_ERROR`` :class:`~analyzer.models.QualityIssue` carrying the invalid
value is emitted (Requirement 5.5).

Datasets containing zero or one Record always complete with zero issues
(Requirement 5.6).
"""

from __future__ import annotations

import json
import re

from analyzer.errors import ConfigError
from analyzer.models import (
    Dataset,
    IssueCategory,
    QualityIssue,
    Record,
    Value,
)

__all__ = [
    "DuplicateDetector",
    "DEFAULT_SIMILARITY_THRESHOLD",
]

# Default similarity threshold used when none is configured or when a configured
# value is rejected as out of range (Requirements 5.4, 5.5).
DEFAULT_SIMILARITY_THRESHOLD: float = 0.9

# Word-token pattern used to normalize record text for similarity comparison.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _canonical_value(value: Value) -> object:
    """Return a JSON-canonical form of ``value`` for content hashing.

    Dictionaries are emitted with sorted keys so that two Records whose fields
    are value-equal but differ only in key insertion order produce the same
    canonical content key (matching ``fields_equivalent`` semantics). Booleans
    are tagged distinctly from integers so ``True`` and ``1`` never collide.
    """
    if isinstance(value, bool):
        return ["__bool__", value]
    if isinstance(value, dict):
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def _content_key(record: Record) -> str:
    """A deterministic content key identifying a Record's exact content.

    Two Records share a content key iff their ``fields`` are value-equal (key
    ordering aside). The key is derived only from ``fields`` - not from
    ``location`` or ``metadata`` - so byte-for-byte content identity is decided
    independently of where each Record originated.
    """
    canonical = {key: _canonical_value(record.fields[key]) for key in sorted(record.fields)}
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=repr)


def _flatten_text(value: Value, out: list[str]) -> None:
    """Collect the textual content of ``value`` into ``out`` recursively."""
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


def _normalized_tokens(record: Record) -> frozenset[str]:
    """Return the set of normalized (lowercased) word tokens of a Record.

    The Record's textual content (string and numeric field values, flattened
    recursively through nested lists/dicts) is lowercased and split into word
    tokens. The token *set* drives a Jaccard similarity comparison.
    """
    parts: list[str] = []
    for key in record.fields:
        _flatten_text(record.fields[key], parts)
    text = " ".join(parts).lower()
    return frozenset(_TOKEN_RE.findall(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Token-set Jaccard similarity in ``[0.0, 1.0]``.

    An empty union (both Records carry no word tokens) yields ``0.0`` so that
    contentless Records are not spuriously reported as near-duplicates; exact
    content identity among such Records is still handled by exact detection.
    """
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class DuplicateDetector:
    """Detects exact duplicates and near-duplicate pairs within a Dataset."""

    def detect(
        self, dataset: Dataset, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[QualityIssue]:
        """Return the Quality_Issues for exact and near duplicates in ``dataset``.

        ``threshold`` is the similarity threshold for near-duplicate pairs. A
        value outside ``[0.0, 1.0]`` (or a non-numeric value) is rejected: the
        default ``0.9`` is used instead and a ``CONFIG_ERROR`` issue is prepended
        to the result (Requirement 5.5).
        """
        issues: list[QualityIssue] = []

        effective_threshold = self._resolve_threshold(threshold, issues)

        records = dataset.records
        # Datasets with zero or one record can have no duplicates (Req 5.6).
        if len(records) < 2:
            return issues

        keys = [_content_key(record) for record in records]
        token_sets = [_normalized_tokens(record) for record in records]

        # Map each content key to the index of its first (original) occurrence.
        first_seen: dict[str, int] = {}

        for i in range(len(records)):
            key = keys[i]
            original_index = first_seen.get(key)

            if original_index is not None:
                # Exact duplicate of an earlier record (Requirements 5.1, 5.3).
                issues.append(
                    QualityIssue(
                        category=IssueCategory.DUPLICATE,
                        location=records[i].location,
                        related_location=records[original_index].location,
                        detail=(
                            f"Record at index {i} is byte-for-byte identical to the "
                            f"record at index {original_index}"
                        ),
                    )
                )
                # An exact duplicate is, by definition, not a near-duplicate; the
                # pair is fully described by the exact-duplicate issue.
                continue

            first_seen[key] = i

            # Near-duplicate pairs: compare against every prior record that is
            # not byte-for-byte identical (Requirements 5.2, 5.3).
            for j in range(i):
                if keys[j] == key:
                    continue
                similarity = _jaccard(token_sets[i], token_sets[j])
                if similarity >= effective_threshold:
                    issues.append(
                        QualityIssue(
                            category=IssueCategory.NEAR_DUPLICATE,
                            location=records[i].location,
                            related_location=records[j].location,
                            detail=(
                                f"Record at index {i} is a near-duplicate of the record "
                                f"at index {j} (similarity {similarity:.3f} >= "
                                f"threshold {effective_threshold:.3f})"
                            ),
                            score=similarity,
                        )
                    )

        return issues

    @staticmethod
    def _resolve_threshold(
        threshold: float, issues: list[QualityIssue]
    ) -> float:
        """Validate ``threshold`` and return the effective value to use.

        On rejection the default ``0.9`` is returned and a ``CONFIG_ERROR``
        issue identifying the invalid value is appended to ``issues``
        (Requirement 5.5).
        """
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            valid = False
        else:
            # Reject NaN (which fails both comparisons) and out-of-range values.
            valid = 0.0 <= float(threshold) <= 1.0

        if valid:
            return float(threshold)

        error = ConfigError(
            parameter="similarity_threshold",
            invalid_value=threshold,
            retained_default=DEFAULT_SIMILARITY_THRESHOLD,
        )
        issues.append(
            QualityIssue(
                category=IssueCategory.CONFIG_ERROR,
                location=None,
                detail=str(error),
                field_name="similarity_threshold",
            )
        )
        return DEFAULT_SIMILARITY_THRESHOLD
