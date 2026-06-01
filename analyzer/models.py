"""Core data models for the LLM Training Data Quality Analyzer.

This module defines the canonical value model and the record/dataset types
that the rest of the system operates on. Everything downstream of the Parser
works with these types rather than raw bytes, so all analysis and
serialization logic is format-agnostic.

The canonical ``Value`` type is the linchpin of round-trip fidelity: parsers
normalize every format's native types into ``Value``, and the Pretty_Printer
only needs to check representability of ``Value`` in a target format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

# Canonical value model: the set of value types that all formats round-trip
# through. Parsers normalize into these; the Pretty_Printer checks
# representability against them.
Value = Union[str, int, float, bool, None, list["Value"], dict[str, "Value"]]


class SupportedFormat(Enum):
    """A data serialization format the Analyzer can parse."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


@dataclass(frozen=True)
class RecordLocation:
    """Format-specific coordinates identifying where a record originated.

    Carries enough information for parse errors and detector issues to point
    precisely at the offending record (line number for JSONL/CSV, array index
    for JSON, or row group and row index for Parquet).
    """

    source_file: str
    line_number: int | None = None      # JSONL, CSV
    array_index: int | None = None      # JSON
    row_group: int | None = None        # Parquet
    row_index: int | None = None        # Parquet


@dataclass(frozen=True)
class Span:
    """A half-open character offset range ``[start, end)`` within text."""

    start: int                          # inclusive char offset
    end: int                            # exclusive char offset


@dataclass(frozen=True)
class Record:
    """A single training example: ordered text fields plus optional metadata.

    Two records are *equivalent* iff their ``fields`` have identical key sets
    and value-equal entries. The order of records (not keys) is what callers
    track for round-trip checks.
    """

    fields: dict[str, Value]            # ordered; field name -> value
    location: RecordLocation
    metadata: dict[str, Value] = field(default_factory=dict)

    def equivalent(self, other: "Record") -> bool:
        """Return True iff this record's fields are equivalent to ``other``'s.

        Equivalence is defined over ``fields`` only (identical key sets and
        value-equal entries) and is independent of ``location`` and
        ``metadata``. This is the comparison used by round-trip fidelity
        checks (Requirement 3.2).
        """
        if not isinstance(other, Record):
            return NotImplemented
        return fields_equivalent(self.fields, other.fields)


@dataclass
class Dataset:
    """An ordered collection of records combined across input files."""

    records: list[Record]               # ordered, combined across input files
    source_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)


def fields_equivalent(left: dict[str, Value], right: dict[str, Value]) -> bool:
    """Return True iff two field mappings have identical key sets and
    value-equal entries.

    Field ordering does not affect equivalence; key sets must match exactly
    and each shared key's values must be equal under canonical ``Value``
    equality.
    """
    if left.keys() != right.keys():
        return False
    return all(values_equal(left[key], right[key]) for key in left)


def values_equal(left: Value, right: Value) -> bool:
    """Structural equality over the canonical ``Value`` type.

    Booleans are treated as distinct from numbers (``True`` is not equal to
    ``1``) so that round-trip checks do not silently collapse types. Lists and
    dicts are compared recursively; dict equality ignores key ordering but
    requires identical key sets.
    """
    # bool is a subclass of int in Python; keep them distinct.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right

    if left is None or right is None:
        return left is None and right is None

    if isinstance(left, dict) and isinstance(right, dict):
        return fields_equivalent(left, right)

    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )

    # Disallow cross-type comparisons between containers and scalars.
    if isinstance(left, (list, dict)) != isinstance(right, (list, dict)):
        return False

    return type(left) is type(right) and left == right


def records_equivalent(left: list[Record], right: list[Record]) -> bool:
    """Return True iff two record lists are equivalent for round-trip checks.

    Equivalent means: the same number of records in the same order, and each
    record contains the identical set of field names with field values equal
    to the corresponding original field values (Requirement 3.2).
    """
    if len(left) != len(right):
        return False
    return all(a.equivalent(b) for a, b in zip(left, right))


class IssueCategory(Enum):
    """The category of a detected :class:`QualityIssue`.

    The full set of categories is enumerated so that reports can group issues
    over every category, reporting zero-count categories with a count of 0
    (Requirement 10.2).
    """

    PARSE_ERROR = "parse_error"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    PII = "pii"
    TOXICITY = "toxicity"
    LOW_QUALITY_SHORT = "low_quality_short"
    LOW_QUALITY_GIBBERISH = "low_quality_gibberish"
    EMPTY_RECORD = "empty_record"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    FIELD_TYPE_MISMATCH = "field_type_mismatch"
    SCHEMA_INFERENCE_FAILED = "schema_inference_failed"
    ANALYSIS_FAILURE = "analysis_failure"
    CONFIG_ERROR = "config_error"


@dataclass(frozen=True)
class QualityIssue:
    """A detected problem associated with a specific record or the dataset.

    A ``location`` of ``None`` denotes a dataset-level issue (e.g. a file that
    produced no records or a failed schema inference). ``related_location``
    points at the original record for duplicate pairs (Requirement 5.3), while
    ``pii_category`` and ``span`` carry the PII category and matched character
    offsets for PII issues (Requirement 6.3). ``score`` carries a numeric score
    such as a toxicity score (Requirement 7.2).
    """

    category: IssueCategory
    location: RecordLocation | None     # None => dataset-level issue
    field_name: str | None = None
    related_location: RecordLocation | None = None   # original record for dup pairs
    detail: str = ""                    # human-readable explanation
    pii_category: str | None = None
    span: Span | None = None
    score: float | None = None          # e.g., toxicity score


class FieldType(Enum):
    """The declared type of a schema field, mapped from canonical ``Value``."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    NULL = "null"
    LIST = "list"
    OBJECT = "object"


@dataclass
class FieldSpec:
    """A single field declaration within a :class:`Schema`."""

    name: str
    type: FieldType
    required: bool


@dataclass
class Schema:
    """A declared or inferred description of the fields a record should have.

    ``inferred`` is ``True`` when the schema was inferred from the first record
    of a dataset rather than supplied by the caller (Requirement 9.3).
    """

    fields: list[FieldSpec]
    inferred: bool = False              # True when inferred from first record


@dataclass
class Metrics:
    """Quantitative quality metrics computed over a dataset.

    For empty datasets the token statistics and proportions are reported as
    ``0``/``0.0`` and the quality score is ``0.0`` (Requirements 4.5-4.7).
    """

    record_count: int                   # >= 0
    mean_tokens: int                    # 0 when empty
    min_tokens: int                     # 0 when empty
    max_tokens: int                     # 0 when empty
    issue_record_proportion: float      # [0.0, 1.0]; 0.0 when empty
    quality_score: float                # [0.0, 1.0]; 0.0 when empty


@dataclass
class Report:
    """A structured report assembled from metrics, issues, and summary counts.

    ``issues_by_category`` and ``category_counts`` are keyed over the full
    :class:`IssueCategory` enum so that categories with no issues appear with a
    count of 0 (Requirement 10.2). ``total_records`` and ``total_issues`` are
    non-negative summary counts (Requirement 10.5).
    """

    metrics: Metrics
    issues_by_category: dict[IssueCategory, list[QualityIssue]]  # all categories present
    category_counts: dict[IssueCategory, int]                    # zero categories => 0
    total_records: int                  # >= 0
    total_issues: int                   # >= 0
