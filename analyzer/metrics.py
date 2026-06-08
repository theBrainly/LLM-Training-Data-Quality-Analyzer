"""The Metrics_Engine: quantitative quality metrics over a Dataset.

This module computes the aggregate metrics defined in Requirement 4: the total
record count, the per-record token-count mean/min/max, the proportion of
records carrying at least one :class:`QualityIssue`, and the overall
``Quality_Score``.

Token counting convention
--------------------------
A record's token count is the number of **whitespace-delimited tokens** in its
textual content. The textual content is produced by flattening every field
value into text (strings as-is, numbers stringified, nested lists/dicts walked
recursively) and joining the parts with a single space, then splitting on
runs of whitespace via :meth:`str.split`. This matches the Quality_Detector's
notion of a token count (Requirement 8.1, "number of whitespace-delimited
tokens") and the text-flattening used by the Duplicate_Detector, keeping a
single, consistent definition of "the text of a record" across the system.

Quality score
--------------
``quality_score = 1.0 - issue_record_proportion``. Because the proportion lies
in ``[0.0, 1.0]``, the score is likewise bounded in ``[0.0, 1.0]``. For an
empty dataset the proportion, the token statistics, and the score are all
forced to their documented zero values (Requirements 4.5-4.7).
"""

from __future__ import annotations

from analyzer.models import (
    Dataset,
    Metrics,
    QualityIssue,
    Record,
    RecordLocation,
    Value,
)


def _flatten_text(value: Value, out: list[str]) -> None:
    """Collect the textual content of ``value`` into ``out`` recursively.

    Booleans and ``None`` contribute no text (they carry no tokens); strings
    are taken verbatim, numbers are stringified, and lists/dicts are walked so
    that nested textual content is included.
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


def token_count(record: Record) -> int:
    """Return the number of whitespace-delimited tokens in ``record``.

    The record's field values are flattened into text and split on whitespace.
    A record with no textual content yields ``0``.
    """
    parts: list[str] = []
    for key in record.fields:
        _flatten_text(record.fields[key], parts)
    text = " ".join(parts)
    return len(text.split())


class MetricsEngine:
    """Computes aggregate quality metrics for a :class:`Dataset`."""

    def compute(self, dataset: Dataset, issues: list[QualityIssue]) -> Metrics:
        """Compute :class:`Metrics` for ``dataset`` given its ``issues``.

        ``record_count`` is the number of records (>= 0). Token statistics are
        the mean (rounded to a non-negative integer), minimum, and maximum of
        the per-record token counts. ``issue_record_proportion`` is the share
        of records that carry at least one issue, and ``quality_score`` is
        ``1.0 - issue_record_proportion``. Empty datasets yield the documented
        zero/0.0 values (Requirements 4.5-4.7).
        """
        records = dataset.records
        record_count = len(records)

        # Empty dataset: every metric collapses to its documented zero value.
        if record_count == 0:
            return Metrics(
                record_count=0,
                mean_tokens=0,
                min_tokens=0,
                max_tokens=0,
                issue_record_proportion=0.0,
                quality_score=0.0,
            )

        token_counts = [token_count(record) for record in records]
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        # Mean reported as a non-negative integer; rounding a value within
        # [min, max] keeps the result within [min, max] (Requirement 4.2).
        mean_tokens = round(sum(token_counts) / record_count)

        issue_locations = {
            issue.location for issue in issues if issue.location is not None
        }
        records_with_issues = self._count_records_with_issues(
            records, issue_locations
        )
        issue_record_proportion = records_with_issues / record_count
        quality_score = 1.0 - issue_record_proportion

        return Metrics(
            record_count=record_count,
            mean_tokens=mean_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            issue_record_proportion=issue_record_proportion,
            quality_score=quality_score,
        )

    @staticmethod
    def _count_records_with_issues(
        records: list[Record], issue_locations: set[RecordLocation]
    ) -> int:
        """Count records whose location is referenced by at least one issue.

        Issues are mapped to records by their :class:`RecordLocation`;
        dataset-level issues (``location is None``) reference no record and are
        excluded from ``issue_locations`` by the caller.
        """
        return sum(
            1 for record in records if record.location in issue_locations
        )
