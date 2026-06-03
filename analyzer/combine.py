"""Combine per-file parse output into a single ordered :class:`Dataset`.

This module bridges the Ingestion_Engine and the Parser. The Ingestion_Engine
produces an :class:`~analyzer.ingestion.IngestionResult` whose ``units`` stream
carries one :class:`~analyzer.parsers.RawRecordUnit` per Supported_Format file
(in file-listing order). The Parser converts a unit into :class:`Record`
objects according to that file's :class:`SupportedFormat`.

:func:`combine` walks the unit stream in order, determines each file's format
from its path (via :func:`~analyzer.ingestion.detect_format`), parses the unit,
and concatenates the resulting records so that the combined :class:`Dataset`
holds records in *file-listing order then within-file parse order*
(Requirements 1.1, 1.2). The parse :class:`QualityIssue` objects raised across
all files are collected alongside the dataset (they are dataset-level findings,
not part of the immutable record collection). The ``skipped_files`` recorded by
ingestion (unsupported-extension files, Requirement 1.5) are carried onto the
``Dataset`` unchanged, and ``source_files`` is set to the list of files that
were parsed, in the order they were streamed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analyzer.ingestion import IngestionResult, detect_format
from analyzer.models import Dataset, QualityIssue, Record
from analyzer.parsers import Parser

__all__ = ["CombinedDataset", "combine"]


@dataclass
class CombinedDataset:
    """The result of combining per-file parse output.

    ``dataset`` holds the records concatenated in file-listing order then
    within-file parse order, with ``skipped_files`` carried from ingestion and
    ``source_files`` listing the parsed files in stream order. ``issues`` holds
    every parse :class:`QualityIssue` raised while parsing the files, in the
    same file-then-within-file order; these are surfaced separately because the
    :class:`Dataset` itself carries only records.
    """

    dataset: Dataset
    issues: list[QualityIssue] = field(default_factory=list)


def combine(result: IngestionResult, parser: Parser) -> CombinedDataset:
    """Combine an :class:`IngestionResult`'s units into one :class:`Dataset`.

    For each streamed :class:`RawRecordUnit` (in file-listing order), the file's
    :class:`SupportedFormat` is determined from its source path and the unit is
    parsed with ``parser``. Records are concatenated in file order then in the
    within-file parse order the Parser returns, and the parse issues are
    accumulated in the same order (Requirements 1.1, 1.2).

    ``skipped_files`` from the ingestion result is carried onto the resulting
    :class:`Dataset` unchanged (Requirement 1.5), and ``source_files`` is the
    list of parsed file paths in stream order (each parsed file appears once, in
    the order first seen). A unit whose path does not map to a Supported_Format
    is skipped defensively (the Ingestion_Engine only streams supported files).
    """
    records: list[Record] = []
    issues: list[QualityIssue] = []
    source_files: list[str] = []
    seen_files: set[str] = set()

    for unit in result.units:
        source_file = _unit_source_file(unit)
        fmt = detect_format(source_file)
        if fmt is None:
            # Defensive: ingestion streams only Supported_Format files, so a
            # unit whose extension is unsupported cannot be parsed; skip it.
            continue

        unit_records, unit_issues = parser.parse([unit], fmt)
        records.extend(unit_records)
        issues.extend(unit_issues)

        if source_file not in seen_files:
            seen_files.add(source_file)
            source_files.append(source_file)

    dataset = Dataset(
        records=records,
        source_files=source_files,
        skipped_files=list(result.skipped_files),
    )
    return CombinedDataset(dataset=dataset, issues=issues)


def _unit_source_file(unit) -> str:
    """Return the source-file path for ``unit``.

    Prefers the unit's own ``source_file`` and falls back to the source file
    carried on its :class:`RecordLocation` when present, mirroring the
    provenance handling the Parser strategies use.
    """
    source_file = getattr(unit, "source_file", None)
    if source_file:
        return source_file
    if unit.location is not None:
        return unit.location.source_file
    return ""
