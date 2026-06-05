"""Deterministic per-category PII detection.

The :class:`PIIDetector` scans the text fields of a :class:`Record` for the PII
categories the Analyzer recognizes (Requirement 6.1):

* ``email``            - email addresses
* ``phone``            - phone numbers (``ddd-ddd-dddd``)
* ``physical_address`` - street addresses (``<number> <Name> <Suffix>``)
* ``government_id``    - government identifiers (SSN-style ``ddd-dd-dddd``)
* ``credit_card``      - credit card numbers (digit runs validated with Luhn)

Detection is deterministic and based on regular expressions plus a Luhn check
for credit cards. For every PII occurrence found (including repeated
occurrences of the same category) the detector records exactly one
:class:`QualityIssue` carrying the PII category and the matched ``[start, end)``
character offsets within the field's text (Requirement 6.3). Text containing no
PII produces no issues (Requirement 6.2).

If a record cannot be analyzed, the detector records a single
``ANALYSIS_FAILURE`` issue identifying the record and leaves the original record
content unchanged (Requirement 6.4).

The class is structured so that PII redaction (Requirement 6.5, task 8.2) can be
layered on top of the same match-finding logic: :func:`find_pii_in_text` is the
shared primitive that both detection and a future ``redact`` method build on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from analyzer.models import (
    IssueCategory,
    QualityIssue,
    Record,
    Span,
    Value,
)

__all__ = [
    "PII_CATEGORIES",
    "PIIMatch",
    "PIIDetector",
    "find_pii_in_text",
    "luhn_valid",
]


# Category labels for the PII kinds the detector recognizes (Requirement 6.1).
# These match the labels emitted by the test strategies' ground truth.
PII_CATEGORIES: tuple[str, ...] = (
    "email",
    "phone",
    "physical_address",
    "government_id",
    "credit_card",
)


@dataclass(frozen=True)
class PIIMatch:
    """A single PII occurrence located within a piece of text.

    ``text[start:end] == value`` always holds for the text the match was found
    in, so callers can rely on the span to slice the exact PII instance.
    """

    category: str
    value: str
    start: int                          # inclusive char offset
    end: int                            # exclusive char offset


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

# Email: a local part, an ``@``, a domain, and a dotted TLD of >= 2 letters.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# Phone: ``ddd-ddd-dddd`` (a 3-digit middle group distinguishes it from an SSN).
_PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")

# Government identifier (SSN-style): ``ddd-dd-dddd`` (2-digit middle group).
_GOV_ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Physical address: ``<number> <Capitalized name> <Street suffix>``.
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][a-zA-Z]*\s+"
    r"(?:Street|Avenue|Road|Boulevard|Lane|Drive)\b"
)

# Credit card candidate: an unbroken run of 13-19 digits (validated with Luhn).
_CREDIT_CARD_RE = re.compile(r"\b\d{13,19}\b")

# Order matters only for tie-breaking when two candidate spans overlap; the
# more specific dashed patterns and the Luhn-validated card take precedence.
_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("email", _EMAIL_RE),
    ("government_id", _GOV_ID_RE),
    ("phone", _PHONE_RE),
    ("physical_address", _ADDRESS_RE),
    ("credit_card", _CREDIT_CARD_RE),
)


def luhn_valid(number: str) -> bool:
    """Return True iff ``number`` (a string of digits) passes the Luhn check."""
    if not number or not number.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(number)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def find_pii_in_text(text: str) -> list[PIIMatch]:
    """Find every PII occurrence in ``text`` as non-overlapping matches.

    Candidate matches are gathered from each category's pattern (credit-card
    candidates additionally validated with Luhn), then overlapping candidates
    are resolved by preferring the earliest start and, on ties, the longest
    match. The returned matches are ordered by their start offset, so the
    caller sees one match per real occurrence including repeats
    (Requirement 6.1/6.3).
    """
    candidates: list[PIIMatch] = []
    for category, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            value = m.group()
            if category == "credit_card" and not luhn_valid(value):
                continue
            candidates.append(
                PIIMatch(
                    category=category,
                    value=value,
                    start=m.start(),
                    end=m.end(),
                )
            )

    # Resolve overlaps deterministically: earliest start first, then longest.
    candidates.sort(key=lambda c: (c.start, -(c.end - c.start)))
    selected: list[PIIMatch] = []
    last_end = -1
    for candidate in candidates:
        if candidate.start >= last_end:
            selected.append(candidate)
            last_end = candidate.end
    selected.sort(key=lambda c: c.start)
    return selected


class PIIDetector:
    """Detects personally identifiable information within record text."""

    def detect(self, record: Record) -> list[QualityIssue]:
        """Return one :class:`QualityIssue` per PII occurrence in ``record``.

        Each issue identifies the record (via its location and the field name),
        the PII category, and the matched ``[start, end)`` character offsets
        within that field's text (Requirement 6.3). A record whose text
        contains no PII yields an empty list (Requirement 6.2).

        If analysis of the record fails for any reason, a single
        ``ANALYSIS_FAILURE`` issue is recorded for the record and the record's
        content is left unchanged (Requirement 6.4).
        """
        try:
            issues: list[QualityIssue] = []
            for field_name, value in record.fields.items():
                if not isinstance(value, str):
                    continue
                for match in self._scan_text(value):
                    issues.append(
                        QualityIssue(
                            category=IssueCategory.PII,
                            location=record.location,
                            field_name=field_name,
                            pii_category=match.category,
                            span=Span(start=match.start, end=match.end),
                            detail=f"{match.category} detected in field '{field_name}'",
                        )
                    )
            return issues
        except Exception as exc:  # noqa: BLE001 - fail-soft per Requirement 6.4
            return [
                QualityIssue(
                    category=IssueCategory.ANALYSIS_FAILURE,
                    location=record.location,
                    detail=f"PII analysis failed: {exc}",
                )
            ]

    def redact(self, record: Record) -> Record:
        """Return a redacted copy of ``record`` with PII spans replaced.

        Each detected PII span in every string field is replaced by a
        placeholder corresponding to that span's PII category (e.g. ``email``
        becomes ``[EMAIL]``, ``physical_address`` becomes
        ``[PHYSICAL_ADDRESS]``). The redacted copy therefore contains none of
        the original PII instances, while the original ``record`` (and its
        ``fields`` mapping) is left unchanged (Requirement 6.5).

        Replacement is performed right-to-left within each field so that the
        ``[start, end)`` offsets of earlier matches remain valid as later
        matches are substituted.
        """
        redacted_fields: dict[str, Value] = dict(record.fields)
        for field_name, value in record.fields.items():
            if not isinstance(value, str):
                continue
            redacted_fields[field_name] = self._redact_text(value)
        return Record(
            fields=redacted_fields,
            location=record.location,
            metadata=dict(record.metadata),
        )

    def _scan_text(self, text: str) -> list[PIIMatch]:
        """Locate PII occurrences within a single text value.

        Isolated as a method so detection (and future redaction) share the same
        match-finding logic and so failure paths can be exercised in tests.
        """
        return find_pii_in_text(text)

    def _redact_text(self, text: str) -> str:
        """Replace every detected PII span in ``text`` with its placeholder.

        Matches are substituted right-to-left so that the offsets of matches
        earlier in the string stay correct while later ones are replaced.
        """
        result = text
        for match in sorted(self._scan_text(text), key=lambda m: m.start, reverse=True):
            placeholder = _placeholder_for(match.category)
            result = result[: match.start] + placeholder + result[match.end :]
        return result


def _placeholder_for(category: str) -> str:
    """Return the redaction placeholder for a PII ``category``.

    The placeholder is the upper-cased category name wrapped in brackets, e.g.
    ``email`` -> ``[EMAIL]`` and ``physical_address`` -> ``[PHYSICAL_ADDRESS]``.
    """
    return f"[{category.upper()}]"
