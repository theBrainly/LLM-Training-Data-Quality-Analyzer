"""Format parsers that convert raw record units into :class:`Record` objects.

Everything downstream of the Parser operates on :class:`Record` objects rather
than raw bytes, so this module is the single place where format syntax is
turned into the canonical in-memory representation.

The :class:`Parser` dispatches by :class:`SupportedFormat` to a per-format
strategy. Parsing is *fail-soft* at the record level: an individual unparseable
unit becomes a located :class:`QualityIssue` and parsing continues with the
remaining units (Requirement 2.5). File-level structural failures are handled
by the relevant per-format strategies.

This module implements the dispatch structure, the JSON array strategy, the
JSONL strategy, the CSV strategy, and the Parquet strategy.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Union

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.models import (
    IssueCategory,
    QualityIssue,
    Record,
    RecordLocation,
    SupportedFormat,
    Value,
)

# A raw, not-yet-parsed unit handed to the Parser by the Ingestion_Engine.
#
# For JSON a unit carries the entire file content (the top-level array) as its
# ``payload``; the JSON strategy expands that array into one Record per
# element. JSONL follows the same whole-file convention: a unit carries the
# entire file text, and the JSONL strategy splits it into physical lines,
# producing one Record per non-whitespace line. For the remaining row oriented
# formats a unit carries a single row. ``payload`` therefore spans raw
# text/bytes as well as already-decoded mappings, mirroring the canonical
# ``Value`` inputs the parsers normalize.
Payload = Union[bytes, str, Mapping[str, Value]]


@dataclass(frozen=True)
class RawRecordUnit:
    """A raw record unit produced by ingestion and consumed by the Parser.

    ``source_file`` and ``location`` carry the provenance used to build precise
    :class:`RecordLocation` coordinates on both successful records and parse
    issues. ``location`` is optional: when omitted the strategy derives a
    location from ``source_file`` and the appropriate format-specific index.
    """

    source_file: str
    payload: Payload
    location: RecordLocation | None = None


# A per-format strategy: consumes the raw units for a single format and returns
# the parsed records plus any located parse issues.
ParseStrategy = Callable[
    ["Parser", list[RawRecordUnit]], tuple[list[Record], list[QualityIssue]]
]


class Parser:
    """Converts raw record units into :class:`Record` objects per format.

    The public :meth:`parse` entry point dispatches on the declared
    :class:`SupportedFormat`. Each format is handled by a strategy registered in
    :data:`Parser._STRATEGIES`; adding a new format is a matter of implementing
    a strategy method and registering it there.
    """

    def parse(
        self, units: Iterable[RawRecordUnit], fmt: SupportedFormat
    ) -> tuple[list[Record], list[QualityIssue]]:
        """Parse ``units`` according to ``fmt``.

        Returns a tuple ``(records, issues)`` where ``records`` holds one
        :class:`Record` per successfully parsed source unit (in source order)
        and ``issues`` holds the located :class:`QualityIssue` objects raised
        for unparseable units. A single bad unit never aborts the parse
        (Requirement 2.5).
        """
        strategy = self._STRATEGIES.get(fmt)
        if strategy is None:
            raise ValueError(f"Unsupported format for parsing: {fmt!r}")
        return strategy(self, list(units))

    # -- JSON ---------------------------------------------------------------

    def _parse_json(
        self, units: list[RawRecordUnit]
    ) -> tuple[list[Record], list[QualityIssue]]:
        """Parse JSON units whose top-level structure is an array.

        Each unit's payload is decoded as JSON and is expected to be a
        top-level array; one :class:`Record` is produced per array element
        (Requirement 2.1). An element that is not a JSON object cannot map to a
        record's fields, and a unit whose payload is not valid JSON or is not a
        top-level array cannot be parsed; in every such case a located
        :class:`QualityIssue` is appended and parsing continues (Requirement
        2.5).
        """
        records: list[Record] = []
        issues: list[QualityIssue] = []

        for unit in units:
            decoded = self._decode_json_payload(unit, issues)
            if decoded is _DECODE_FAILED:
                continue

            if not isinstance(decoded, list):
                issues.append(
                    QualityIssue(
                        category=IssueCategory.PARSE_ERROR,
                        location=self._json_location(unit, array_index=None),
                        detail=(
                            "JSON top-level structure is not an array "
                            f"(found {type(decoded).__name__})"
                        ),
                    )
                )
                continue

            for index, element in enumerate(decoded):
                location = self._json_location(unit, array_index=index)
                if isinstance(element, dict):
                    records.append(Record(fields=dict(element), location=location))
                else:
                    issues.append(
                        QualityIssue(
                            category=IssueCategory.PARSE_ERROR,
                            location=location,
                            detail=(
                                "JSON array element is not an object "
                                f"(found {type(element).__name__})"
                            ),
                        )
                    )

        return records, issues

    @staticmethod
    def _decode_json_payload(
        unit: RawRecordUnit, issues: list[QualityIssue]
    ) -> object:
        """Decode a unit's payload into a Python object.

        A payload that is already a mapping is returned as-is (treated as a
        single-object payload). Text/bytes payloads are decoded with
        :func:`json.loads`. On any decode failure a located parse issue is
        appended and the sentinel :data:`_DECODE_FAILED` is returned.
        """
        payload = unit.payload
        if isinstance(payload, Mapping):
            return dict(payload)

        try:
            if isinstance(payload, (bytes, bytearray)):
                return json.loads(payload.decode("utf-8"))
            return json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            issues.append(
                QualityIssue(
                    category=IssueCategory.PARSE_ERROR,
                    location=Parser._json_location(unit, array_index=None),
                    detail=f"Malformed JSON payload: {exc}",
                )
            )
            return _DECODE_FAILED

    @staticmethod
    def _json_location(unit: RawRecordUnit, array_index: int | None) -> RecordLocation:
        """Build a JSON :class:`RecordLocation` for ``unit`` and ``array_index``.

        Preserves the source file from the unit's supplied location when
        present, and stamps the array index that identifies the element within
        the top-level array.
        """
        source_file = (
            unit.location.source_file if unit.location is not None else unit.source_file
        )
        return RecordLocation(source_file=source_file, array_index=array_index)

    # -- JSONL --------------------------------------------------------------

    def _parse_jsonl(
        self, units: list[RawRecordUnit]
    ) -> tuple[list[Record], list[QualityIssue]]:
        """Parse JSONL units, one :class:`Record` per non-whitespace line.

        Each unit carries the entire file text as its payload (mirroring the
        whole-file convention used by the JSON strategy). The text is split
        into physical lines and 1-based line numbers are tracked across *every*
        line so that locations stay accurate; lines containing only whitespace
        are ignored entirely, producing neither a record nor an issue
        (Requirement 2.2). Every remaining (non-whitespace) line is decoded as a
        standalone JSON value: a JSON object becomes a :class:`Record`, while a
        line that is not valid JSON, or whose value is not a JSON object,
        becomes a located :class:`QualityIssue` and parsing continues with the
        following lines (Requirement 2.5).

        A payload that is already a mapping is treated as a single record on
        line 1, and a payload whose bytes cannot be decoded as UTF-8 yields a
        single file-scoped parse issue.
        """
        records: list[Record] = []
        issues: list[QualityIssue] = []

        for unit in units:
            lines = self._jsonl_lines(unit, issues)
            if lines is _DECODE_FAILED:
                continue

            for line_number, line in lines:
                if line.strip() == "":
                    # Whitespace-only line: ignored, not a record nor an issue.
                    continue

                location = self._jsonl_location(unit, line_number=line_number)
                try:
                    decoded = json.loads(line)
                except ValueError as exc:
                    issues.append(
                        QualityIssue(
                            category=IssueCategory.PARSE_ERROR,
                            location=location,
                            detail=f"Malformed JSON on line {line_number}: {exc}",
                        )
                    )
                    continue

                if isinstance(decoded, dict):
                    records.append(Record(fields=dict(decoded), location=location))
                else:
                    issues.append(
                        QualityIssue(
                            category=IssueCategory.PARSE_ERROR,
                            location=location,
                            detail=(
                                "JSONL line is not an object "
                                f"(found {type(decoded).__name__})"
                            ),
                        )
                    )

        return records, issues

    @staticmethod
    def _jsonl_lines(
        unit: RawRecordUnit, issues: list[QualityIssue]
    ) -> object:
        """Yield ``(line_number, line)`` pairs for a unit's payload.

        Text/bytes payloads are split into physical lines with 1-based line
        numbers that count every line (including blank ones) so locations stay
        faithful to the source file. A mapping payload is surfaced as a single
        line-1 record by re-encoding it as JSON. A payload whose bytes cannot be
        decoded as UTF-8 appends a single file-scoped parse issue and returns
        the :data:`_DECODE_FAILED` sentinel.
        """
        payload = unit.payload
        if isinstance(payload, Mapping):
            return [(1, json.dumps(dict(payload)))]

        if isinstance(payload, (bytes, bytearray)):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                issues.append(
                    QualityIssue(
                        category=IssueCategory.PARSE_ERROR,
                        location=Parser._jsonl_location(unit, line_number=None),
                        detail=f"Malformed JSONL payload: {exc}",
                    )
                )
                return _DECODE_FAILED
        else:
            text = payload

        # Split only on real JSONL line terminators. ``str.splitlines()`` is
        # avoided because it breaks on every Unicode line boundary (NEL,
        # U+2028/U+2029, vertical tab, etc.), which would corrupt JSON string
        # values that legitimately contain those characters. The Pretty_Printer
        # joins records with "\n", so splitting on "\n" (and tolerating a
        # trailing "\r" from CRLF inputs) is the correct inverse.
        lines = text.split("\n")
        return [
            (number, line[:-1] if line.endswith("\r") else line)
            for number, line in enumerate(lines, start=1)
        ]

    @staticmethod
    def _jsonl_location(unit: RawRecordUnit, line_number: int | None) -> RecordLocation:
        """Build a JSONL :class:`RecordLocation` for ``unit`` and ``line_number``.

        Preserves the source file from the unit's supplied location when
        present, and stamps the 1-based line number identifying the line within
        the file.
        """
        source_file = (
            unit.location.source_file if unit.location is not None else unit.source_file
        )
        return RecordLocation(source_file=source_file, line_number=line_number)

    # -- CSV ----------------------------------------------------------------

    def _parse_csv(
        self, units: list[RawRecordUnit]
    ) -> tuple[list[Record], list[QualityIssue]]:
        """Parse CSV units: the first row is the header, each later row a Record.

        Each unit carries the entire file text as its payload (the whole-file
        convention shared with the JSON and JSONL strategies). The first row is
        the header; every subsequent row becomes a :class:`Record` whose fields
        map each header name to the cell value at the corresponding column
        position (Requirement 2.3). CSV cells are textual, so all field values
        are strings.

        File-level structural failures suppress *all* records for that file and
        produce a single file-scoped :class:`QualityIssue` (Requirement 2.7): a
        missing header row (a file with no rows, or whose header row has zero
        columns) and a header containing duplicate field names are both handled
        this way.

        Row-level failures are fail-soft (Requirement 2.5): a data row whose
        column count differs from the header's, or a malformed-CSV error raised
        while reading, appends a located :class:`QualityIssue` and parsing
        continues with the remaining rows. Physically blank lines are ignored
        entirely, producing neither a record nor an issue.
        """
        records: list[Record] = []
        issues: list[QualityIssue] = []

        for unit in units:
            self._parse_csv_unit(unit, records, issues)

        return records, issues

    def _parse_csv_unit(
        self,
        unit: RawRecordUnit,
        records: list[Record],
        issues: list[QualityIssue],
    ) -> None:
        """Parse a single CSV unit, appending to ``records``/``issues``.

        Reads the header and data rows, applies the file-level missing/duplicate
        header checks (which suppress records for the file), then maps each
        well-formed data row onto the header names by column position.
        """
        source_file = (
            unit.location.source_file if unit.location is not None else unit.source_file
        )

        text = self._csv_text(unit, source_file, issues)
        if text is _DECODE_FAILED:
            return

        reader = csv.reader(io.StringIO(text))
        header: list[str] | None = None
        data_rows: list[tuple[int, list[str]]] = []
        try:
            for row in reader:
                if header is None:
                    header = row
                    continue
                if not row:
                    # Physically blank line: ignored, neither record nor issue.
                    continue
                data_rows.append((reader.line_num, row))
        except csv.Error as exc:
            issues.append(
                QualityIssue(
                    category=IssueCategory.PARSE_ERROR,
                    location=RecordLocation(
                        source_file=source_file, line_number=reader.line_num
                    ),
                    detail=f"Malformed CSV on line {reader.line_num}: {exc}",
                )
            )

        # File-level: missing header row (no rows at all, or an empty header
        # row) suppresses every record for this file (Requirement 2.7).
        if header is None or len(header) == 0:
            issues.append(
                QualityIssue(
                    category=IssueCategory.PARSE_ERROR,
                    location=RecordLocation(source_file=source_file, line_number=None),
                    detail="CSV file has no header row",
                )
            )
            return

        # File-level: duplicate header field names suppress every record for
        # this file and identify the offending column (Requirement 2.7).
        duplicates = sorted(
            name for name, count in Counter(header).items() if count > 1
        )
        if duplicates:
            issues.append(
                QualityIssue(
                    category=IssueCategory.PARSE_ERROR,
                    location=RecordLocation(source_file=source_file, line_number=1),
                    field_name=duplicates[0],
                    detail=(
                        "CSV header contains duplicate field names: "
                        + ", ".join(repr(name) for name in duplicates)
                    ),
                )
            )
            return

        for line_number, row in data_rows:
            location = RecordLocation(source_file=source_file, line_number=line_number)
            if len(row) != len(header):
                issues.append(
                    QualityIssue(
                        category=IssueCategory.PARSE_ERROR,
                        location=location,
                        detail=(
                            f"CSV row on line {line_number} has {len(row)} fields "
                            f"but the header declares {len(header)}"
                        ),
                    )
                )
                continue
            fields: dict[str, Value] = {
                name: cell for name, cell in zip(header, row)
            }
            records.append(Record(fields=fields, location=location))

    @staticmethod
    def _csv_text(
        unit: RawRecordUnit, source_file: str, issues: list[QualityIssue]
    ) -> object:
        """Decode a unit's payload into CSV source text.

        ``str`` payloads are used as-is and ``bytes`` payloads are decoded as
        UTF-8. A payload whose bytes cannot be decoded, or one that is not text
        at all, appends a file-scoped parse issue and returns the
        :data:`_DECODE_FAILED` sentinel.
        """
        payload = unit.payload
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            try:
                return payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                issues.append(
                    QualityIssue(
                        category=IssueCategory.PARSE_ERROR,
                        location=RecordLocation(
                            source_file=source_file, line_number=None
                        ),
                        detail=f"Malformed CSV payload: {exc}",
                    )
                )
                return _DECODE_FAILED

        issues.append(
            QualityIssue(
                category=IssueCategory.PARSE_ERROR,
                location=RecordLocation(source_file=source_file, line_number=None),
                detail=(
                    "CSV payload is not text "
                    f"(found {type(payload).__name__})"
                ),
            )
        )
        return _DECODE_FAILED

    # -- Parquet ------------------------------------------------------------

    def _parse_parquet(
        self, units: list[RawRecordUnit]
    ) -> tuple[list[Record], list[QualityIssue]]:
        """Parse Parquet units into one :class:`Record` per row.

        Each unit's payload carries the raw Parquet file bytes. The file is
        read with :mod:`pyarrow.parquet`, iterating every row group in order;
        one :class:`Record` is produced per row across all row groups, in
        document order, and its :class:`RecordLocation` carries the
        ``(row_group, row_index)`` coordinates that identify the row
        (Requirement 2.4). ``row_index`` is the 0-based position of the row
        *within* its row group.

        Failures are reported with located :class:`QualityIssue` objects and
        parsing continues where sensible (Requirement 2.5): a corrupt or
        unreadable Parquet file (or a non-bytes payload) yields a single
        file-scoped issue and no records for that file, while a row group that
        cannot be read yields a row-group-scoped issue and parsing proceeds
        with the remaining row groups.
        """
        records: list[Record] = []
        issues: list[QualityIssue] = []

        for unit in units:
            self._parse_parquet_unit(unit, records, issues)

        return records, issues

    def _parse_parquet_unit(
        self,
        unit: RawRecordUnit,
        records: list[Record],
        issues: list[QualityIssue],
    ) -> None:
        """Parse a single Parquet unit, appending to ``records``/``issues``.

        Opens the Parquet file from the unit's raw bytes and walks each row
        group in order. A file that cannot be opened (corrupt, truncated, or a
        non-bytes payload) produces one file-level issue; a row group that
        cannot be read produces a row-group-located issue and is skipped.
        """
        source_file = (
            unit.location.source_file if unit.location is not None else unit.source_file
        )

        payload = unit.payload
        if not isinstance(payload, (bytes, bytearray)):
            issues.append(
                QualityIssue(
                    category=IssueCategory.PARSE_ERROR,
                    location=self._parquet_location(source_file),
                    detail=(
                        "Parquet payload is not bytes "
                        f"(found {type(payload).__name__})"
                    ),
                )
            )
            return

        try:
            reader = pa.BufferReader(bytes(payload))
            parquet_file = pq.ParquetFile(reader)
        except Exception as exc:  # pyarrow raises a variety of error types
            issues.append(
                QualityIssue(
                    category=IssueCategory.PARSE_ERROR,
                    location=self._parquet_location(source_file),
                    detail=f"Malformed Parquet file: {exc}",
                )
            )
            return

        for row_group in range(parquet_file.num_row_groups):
            try:
                table = parquet_file.read_row_group(row_group)
                rows = table.to_pylist()
            except Exception as exc:
                issues.append(
                    QualityIssue(
                        category=IssueCategory.PARSE_ERROR,
                        location=self._parquet_location(
                            source_file, row_group=row_group
                        ),
                        detail=(
                            f"Unreadable Parquet row group {row_group}: {exc}"
                        ),
                    )
                )
                continue

            for row_index, row in enumerate(rows):
                location = self._parquet_location(
                    source_file, row_group=row_group, row_index=row_index
                )
                records.append(Record(fields=dict(row), location=location))

    @staticmethod
    def _parquet_location(
        source_file: str,
        row_group: int | None = None,
        row_index: int | None = None,
    ) -> RecordLocation:
        """Build a Parquet :class:`RecordLocation`.

        Stamps the ``row_group`` and ``row_index`` coordinates that identify a
        row within the file; both default to ``None`` for file- or
        row-group-scoped issues.
        """
        return RecordLocation(
            source_file=source_file,
            row_group=row_group,
            row_index=row_index,
        )

    # Strategy registry keyed by format. JSONL/CSV/Parquet are registered by
    # their respective tasks; absent entries raise from ``parse``.
    _STRATEGIES: dict[SupportedFormat, ParseStrategy] = {}


# Sentinel marking a payload that failed to decode (distinct from a valid
# ``None`` JSON value).
_DECODE_FAILED: object = object()


Parser._STRATEGIES = {
    SupportedFormat.JSON: Parser._parse_json,
    SupportedFormat.JSONL: Parser._parse_jsonl,
    SupportedFormat.CSV: Parser._parse_csv,
    SupportedFormat.PARQUET: Parser._parse_parquet,
}
