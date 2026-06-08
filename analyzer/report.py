"""The Report_Generator: assembles analysis results into a structured report.

This module covers report *assembly* (Requirement 10 acceptance criteria
10.1, 10.2, 10.5, 10.6): grouping detected :class:`QualityIssue` objects by
category over the full :class:`IssueCategory` enum, carrying the computed
:class:`Metrics`, and reporting the summary counts ``total_records`` and
``total_issues``.

Serialization to JSON/Markdown and rejection of unsupported output formats
(Requirements 10.3, 10.4, 10.7, 10.8) are layered onto the same generator via
:meth:`ReportGenerator.serialize`, which is independent of ``build``.

Serialization model
-------------------
``serialize`` produces a single document for one of two output formats,
identified by :class:`OutputFormat` (``json`` / ``markdown``); a string
``"json"``/``"markdown"`` is also accepted for convenience. The result is a
:class:`SerializeResult` carrying exactly one of ``text`` (the complete
document) or ``error``:

* JSON output is a single document that parses without error and contains the
  quality metrics, the issues grouped by category, and the summary counts
  (Requirement 10.3).
* Markdown output is a single document containing the metrics, the grouped
  issues, and the summary counts (Requirement 10.4).
* A requested format other than JSON or Markdown is rejected with an
  :class:`UnsupportedFormatError` naming the format and no output is produced
  (Requirement 10.8).
* If serialization of a supported format fails, no partial output is produced
  and a :class:`SerializationError` naming the format is returned (Requirement
  10.7).

Grouping convention
-------------------
``issues_by_category`` and ``category_counts`` are keyed over **every** member
of :class:`IssueCategory`. Categories with no detected issues are present with
an empty list and a count of ``0`` (Requirement 10.2). The per-category counts
therefore always sum to ``total_issues``.

Summary counts
--------------
``total_records`` is the number of records analyzed (the length of the
dataset's record list, ``>= 0``) and ``total_issues`` is the number of detected
issues (``>= 0``). A zero-record dataset yields ``total_records == 0``; when no
issues are detected ``total_issues == 0`` (Requirements 10.5, 10.6).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from enum import Enum

from analyzer.errors import SerializationError, UnsupportedFormatError
from analyzer.models import (
    Dataset,
    IssueCategory,
    Metrics,
    QualityIssue,
    Report,
)


class OutputFormat(Enum):
    """A serialization target for a :class:`Report` (Requirements 10.3, 10.4)."""

    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class SerializeResult:
    """The outcome of a serialization request.

    Exactly one of ``text`` / ``error`` is populated. On success ``text`` holds
    the complete serialized document and ``error`` is ``None``. On failure
    ``text`` is ``None`` (no partial output) and ``error`` identifies what went
    wrong: an :class:`UnsupportedFormatError` for a rejected format
    (Requirement 10.8) or a :class:`SerializationError` for a serialization
    failure (Requirement 10.7).
    """

    text: str | None
    error: SerializationError | UnsupportedFormatError | None


class ReportGenerator:
    """Assembles metrics, grouped issues, and summary counts into a Report."""

    def build(
        self,
        dataset: Dataset,
        metrics: Metrics,
        issues: list[QualityIssue],
    ) -> Report:
        """Assemble a :class:`Report` from analysis results.

        Issues are grouped by :attr:`QualityIssue.category` over the full
        :class:`IssueCategory` enum, so every category is present and
        categories with no issues report an empty list and a count of ``0``
        (Requirement 10.2). The :class:`Metrics` object is carried verbatim so
        the report exposes every computed metric by name and value
        (Requirement 10.1). ``total_records`` is the number of records analyzed
        and ``total_issues`` is the number of detected issues, both ``>= 0``;
        a zero-record dataset yields zero counts (Requirements 10.5, 10.6).
        """
        # Seed every category so zero-count categories are present (Req 10.2).
        issues_by_category: dict[IssueCategory, list[QualityIssue]] = {
            category: [] for category in IssueCategory
        }
        for issue in issues:
            issues_by_category[issue.category].append(issue)

        category_counts: dict[IssueCategory, int] = {
            category: len(grouped)
            for category, grouped in issues_by_category.items()
        }

        total_records = len(dataset.records)
        total_issues = len(issues)

        return Report(
            metrics=metrics,
            issues_by_category=issues_by_category,
            category_counts=category_counts,
            total_records=total_records,
            total_issues=total_issues,
        )

    def serialize(
        self,
        report: Report,
        fmt: OutputFormat | str,
    ) -> SerializeResult:
        """Serialize ``report`` into ``fmt`` as a single document.

        ``fmt`` may be an :class:`OutputFormat` member or the string
        ``"json"``/``"markdown"`` (case-insensitive). Any other value is
        rejected with an :class:`UnsupportedFormatError` and produces no output
        (Requirement 10.8). For a supported format the report is rendered to a
        single JSON (Requirement 10.3) or Markdown (Requirement 10.4) document
        containing the metrics, the issues grouped by category, and the summary
        counts. If rendering fails for any reason, no partial output is
        produced and a :class:`SerializationError` naming the format is
        returned (Requirement 10.7).
        """
        resolved = _resolve_format(fmt)
        if resolved is None:
            requested = fmt.value if isinstance(fmt, OutputFormat) else str(fmt)
            return SerializeResult(text=None, error=UnsupportedFormatError(requested))

        try:
            if resolved is OutputFormat.JSON:
                text = _serialize_json(report)
            else:
                text = _serialize_markdown(report)
        except Exception as exc:  # noqa: BLE001 - any failure => no partial output
            return SerializeResult(
                text=None,
                error=SerializationError(resolved.value, detail=str(exc)),
            )

        return SerializeResult(text=text, error=None)


def _resolve_format(fmt: OutputFormat | str) -> OutputFormat | None:
    """Map ``fmt`` to an :class:`OutputFormat`, or ``None`` if unsupported.

    Accepts an :class:`OutputFormat` member directly or a case-insensitive
    string matching one of its values; everything else (other strings, other
    types) is unsupported and yields ``None`` so the caller can reject it.
    """
    if isinstance(fmt, OutputFormat):
        return fmt
    if isinstance(fmt, str):
        try:
            return OutputFormat(fmt.lower())
        except ValueError:
            return None
    return None


def _location_to_dict(location) -> dict | None:
    """Render a :class:`RecordLocation` (or ``None``) as a JSON-safe dict."""
    if location is None:
        return None
    return dataclasses.asdict(location)


def _issue_to_dict(issue: QualityIssue) -> dict:
    """Render a :class:`QualityIssue` as a JSON-safe dict.

    Enums become their string value, nested locations and spans become plain
    dicts, so the whole structure is serializable without a custom encoder.
    """
    return {
        "category": issue.category.value,
        "location": _location_to_dict(issue.location),
        "field_name": issue.field_name,
        "related_location": _location_to_dict(issue.related_location),
        "detail": issue.detail,
        "pii_category": issue.pii_category,
        "span": (
            {"start": issue.span.start, "end": issue.span.end}
            if issue.span is not None
            else None
        ),
        "score": issue.score,
    }


def _serialize_json(report: Report) -> str:
    """Serialize ``report`` as a single JSON document (Requirement 10.3).

    The document carries the quality metrics by name and value, the issues
    grouped by category (with per-category counts), and the summary counts.
    ``allow_nan=False`` makes a non-finite metric value raise rather than emit
    invalid JSON, so the caller surfaces a serialization failure.
    """
    payload = {
        "metrics": dataclasses.asdict(report.metrics),
        "summary": {
            "total_records": report.total_records,
            "total_issues": report.total_issues,
        },
        "issues_by_category": {
            category.value: {
                "count": report.category_counts[category],
                "issues": [
                    _issue_to_dict(issue)
                    for issue in report.issues_by_category[category]
                ],
            }
            for category in IssueCategory
        },
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)


def _serialize_markdown(report: Report) -> str:
    """Serialize ``report`` as a single Markdown document (Requirement 10.4).

    Sections cover the summary counts, every computed metric by name and value,
    and the issues grouped by category (every category present, zero-count
    categories noted explicitly).
    """
    lines: list[str] = ["# Data Quality Report", ""]

    lines.append("## Summary")
    lines.append(f"- Total records: {report.total_records}")
    lines.append(f"- Total issues: {report.total_issues}")
    lines.append("")

    lines.append("## Metrics")
    for name, value in dataclasses.asdict(report.metrics).items():
        lines.append(f"- {name}: {value}")
    lines.append("")

    lines.append("## Issues by Category")
    for category in IssueCategory:
        count = report.category_counts[category]
        lines.append(f"### {category.value} (count: {count})")
        issues = report.issues_by_category[category]
        if not issues:
            lines.append("_No issues._")
        else:
            for issue in issues:
                detail = issue.detail or "(no detail)"
                lines.append(f"- {detail}")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"
