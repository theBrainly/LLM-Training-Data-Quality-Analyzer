"""Property-based tests for the format parsers (tasks 3.5 and 3.6).

These drive the real :class:`~analyzer.parsers.Parser` over sources built for
every :class:`~analyzer.models.SupportedFormat` from generated record data and
assert two design Correctness Properties:

* **Property 3 - Per-format parse counts match source cardinality** (task 3.5):
  for a dataset serialized in any Supported_Format, the Parser produces exactly
  one Record per source unit (JSON array element, JSONL non-whitespace line, CSV
  data row, or Parquet row across all row groups), preserving order; for CSV,
  each Record maps every header name to the cell value at that column's
  position (Requirements 2.1, 2.2, 2.3, 2.4).

* **Property 4 - Parse fault isolation** (task 3.6): for a sequence of source
  units mixing valid and malformed records at arbitrary positions, the Parser
  produces a Record for every valid unit and records exactly one Quality_Issue
  per malformed unit identifying that unit's location, never dropping or
  corrupting a valid record (Requirement 2.5).

Sources are constructed directly per format (JSON array text, JSONL lines with
arbitrary interspersed blank/whitespace lines, CSV header + rows via
:mod:`csv`, and Parquet bytes via :mod:`pyarrow`) so the source cardinality and
each unit's validity are known ground truth.
"""

from __future__ import annotations

import csv
import io
import json
import string

import pyarrow as pa
import pyarrow.parquet as pq
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from analyzer.models import (
    IssueCategory,
    SupportedFormat,
    fields_equivalent,
)
from analyzer.parsers import Parser, RawRecordUnit

_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# --------------------------------------------------------------------------- #
# Shared value strategies (values that survive each format's round-trip).
# --------------------------------------------------------------------------- #

_FIELD_NAMES = st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=6)

# Canonical scalars that survive a JSON/JSONL serialize -> parse round-trip.
_JSON_SCALARS = st.one_of(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=12),
    st.integers(),
    st.booleans(),
    st.none(),
    st.floats(allow_nan=False, allow_infinity=False),
)
# A JSON/JSONL "object" record's fields.
_JSON_OBJECTS = st.dictionaries(_FIELD_NAMES, _JSON_SCALARS, max_size=4)

# CSV cells are textual; restrict to a comma/newline-free alphabet so the
# written table round-trips through csv exactly, keeping the mapping assertion
# precise. Empty cells are allowed for data rows.
_CSV_ALPHABET = string.ascii_letters + string.digits + " "


# --------------------------------------------------------------------------- #
# Property 3 - per-format parse counts match source cardinality.
# --------------------------------------------------------------------------- #

@st.composite
def _json_count_case(draw):
    """A JSON top-level array source and its expected per-element records."""
    objects = draw(st.lists(_JSON_OBJECTS, max_size=8))
    unit = RawRecordUnit(source_file="data.json", payload=json.dumps(objects))
    return SupportedFormat.JSON, [unit], objects


@st.composite
def _jsonl_count_case(draw):
    """A JSONL source with arbitrary interspersed blank/whitespace lines.

    Blank and whitespace-only lines must be ignored entirely (Requirement 2.2),
    so the expected records are exactly the object lines in order.
    """
    objects = draw(st.lists(_JSON_OBJECTS, max_size=8))
    blanks = st.sampled_from(["", "   ", "\t", "  \t "])

    lines: list[str] = []
    # Optional leading blank lines.
    for _ in range(draw(st.integers(0, 2))):
        lines.append(draw(blanks))
    for obj in objects:
        lines.append(json.dumps(obj))
        # Optional blank lines interspersed after each object line.
        for _ in range(draw(st.integers(0, 2))):
            lines.append(draw(blanks))

    unit = RawRecordUnit(source_file="data.jsonl", payload="\n".join(lines))
    return SupportedFormat.JSONL, [unit], objects


@st.composite
def _csv_count_case(draw):
    """A CSV header + data rows source and its expected per-row records.

    The header has at least two unique columns (so an all-empty data row never
    collapses to a blank, ignored line), and every data row has exactly the
    header's column count. Each expected Record maps every header name to the
    cell at that column's position (Requirement 2.3).
    """
    header = draw(
        st.lists(_FIELD_NAMES, min_size=2, max_size=5, unique=True)
    )
    cell = st.text(alphabet=_CSV_ALPHABET, max_size=8)
    rows = draw(
        st.lists(
            st.lists(cell, min_size=len(header), max_size=len(header)),
            max_size=8,
        )
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

    unit = RawRecordUnit(source_file="data.csv", payload=buffer.getvalue())
    expected = [dict(zip(header, row)) for row in rows]
    return SupportedFormat.CSV, [unit], expected


_PARQUET_COLUMN_TYPES = {
    "int": (
        pa.int64(),
        st.one_of(
            st.none(),
            st.integers(min_value=-(2**63), max_value=2**63 - 1),
        ),
    ),
    "float": (
        pa.float64(),
        st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    ),
    "str": (
        pa.string(),
        st.one_of(
            st.none(),
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=10),
        ),
    ),
    "bool": (pa.bool_(), st.one_of(st.none(), st.booleans())),
}


@st.composite
def _parquet_count_case(draw):
    """A Parquet source (often spanning multiple row groups) and its rows.

    Columns carry an explicit homogeneous type so the row dicts round-trip
    through Parquet unchanged; the row-group size is drawn small so multiple row
    groups are exercised when there are enough rows (Requirement 2.4).
    """
    names = draw(st.lists(_FIELD_NAMES, min_size=1, max_size=4, unique=True))
    kinds = [draw(st.sampled_from(list(_PARQUET_COLUMN_TYPES))) for _ in names]
    schema = pa.schema(
        [(name, _PARQUET_COLUMN_TYPES[kind][0]) for name, kind in zip(names, kinds)]
    )

    n_rows = draw(st.integers(min_value=0, max_value=10))
    rows: list[dict] = []
    for _ in range(n_rows):
        row = {
            name: draw(_PARQUET_COLUMN_TYPES[kind][1])
            for name, kind in zip(names, kinds)
        }
        rows.append(row)

    table = pa.Table.from_pylist(rows, schema=schema)
    sink = pa.BufferOutputStream()
    row_group_size = draw(st.integers(min_value=1, max_value=4))
    pq.write_table(table, sink, row_group_size=row_group_size)
    payload = sink.getvalue().to_pybytes()

    unit = RawRecordUnit(source_file="data.parquet", payload=payload)
    # Round-trip the rows through Arrow so the expectation matches Parquet's own
    # normalization of the declared types exactly.
    expected = table.to_pylist()
    return SupportedFormat.PARQUET, [unit], expected


def _count_cases():
    return st.one_of(
        _json_count_case(),
        _jsonl_count_case(),
        _csv_count_case(),
        _parquet_count_case(),
    )


# Feature: llm-training-data-quality-analyzer, Property 3: Per-format parse counts match source cardinality
@_SETTINGS
@given(case=_count_cases())
def test_per_format_parse_counts_match_source_cardinality(case):
    """One Record per source unit, in order; CSV maps header names by position.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4
    """
    fmt, units, expected = case
    records, issues = Parser().parse(units, fmt)

    # No malformed units were planted, so parsing is clean.
    assert issues == []
    # Exactly one Record per source unit (array element / non-whitespace line /
    # data row / Parquet row), with none dropped or duplicated.
    assert len(records) == len(expected)
    # Order is preserved and (for CSV) each header name maps to its column cell.
    for record, expected_fields in zip(records, expected):
        assert fields_equivalent(record.fields, expected_fields)


# --------------------------------------------------------------------------- #
# Property 4 - parse fault isolation.
# --------------------------------------------------------------------------- #

# Malformed JSON array elements / JSONL line payloads: never a JSON object, so
# each becomes exactly one located parse issue.
_MALFORMED_JSON_ELEMENTS = st.one_of(
    st.integers(),
    st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
    st.booleans(),
    st.lists(st.integers(), max_size=3),
    st.none(),
)
# Raw malformed JSONL line text: either invalid JSON or a non-object JSON value.
_MALFORMED_JSONL_LINES = st.sampled_from(
    ["{", "}", "[1, 2", "not json", "12345", '"a string"', "true", "null"]
)


@st.composite
def _json_fault_case(draw):
    """A JSON array mixing object (valid) and non-object (malformed) elements."""
    slots = draw(
        st.lists(
            st.one_of(
                st.tuples(st.just(True), _JSON_OBJECTS),
                st.tuples(st.just(False), _MALFORMED_JSON_ELEMENTS),
            ),
            min_size=1,
            max_size=10,
        )
    )
    array = [value for _, value in slots]
    expected_valid = [value for valid, value in slots if valid]
    malformed_keys = [i for i, (valid, _) in enumerate(slots) if not valid]

    unit = RawRecordUnit(source_file="data.json", payload=json.dumps(array))
    return SupportedFormat.JSON, [unit], expected_valid, malformed_keys


@st.composite
def _jsonl_fault_case(draw):
    """A JSONL document mixing valid object lines and malformed lines."""
    slots = draw(
        st.lists(
            st.one_of(
                st.tuples(st.just(True), _JSON_OBJECTS),
                st.tuples(st.just(False), _MALFORMED_JSONL_LINES),
            ),
            min_size=1,
            max_size=10,
        )
    )
    lines = [
        json.dumps(value) if valid else value for valid, value in slots
    ]
    expected_valid = [value for valid, value in slots if valid]
    # 1-based line numbers; every slot occupies exactly one physical line.
    malformed_keys = [i + 1 for i, (valid, _) in enumerate(slots) if not valid]

    unit = RawRecordUnit(source_file="data.jsonl", payload="\n".join(lines))
    return SupportedFormat.JSONL, [unit], expected_valid, malformed_keys


@st.composite
def _csv_fault_case(draw):
    """A CSV table mixing well-formed rows and column-count-mismatch rows."""
    header = draw(
        st.lists(_FIELD_NAMES, min_size=2, max_size=5, unique=True)
    )
    width = len(header)
    valid_cell = st.text(alphabet=_CSV_ALPHABET, max_size=6)
    # Malformed rows have a different column count; cells are non-empty so a
    # single-column malformed row never collapses to an ignored blank line.
    bad_cell = st.text(alphabet=_CSV_ALPHABET, min_size=1, max_size=6)

    def _valid_row():
        return st.tuples(
            st.just(True),
            st.lists(valid_cell, min_size=width, max_size=width),
        )

    def _malformed_row():
        wrong_width = st.sampled_from([width - 1, width + 1])
        return st.tuples(
            st.just(False),
            wrong_width.flatmap(
                lambda w: st.lists(bad_cell, min_size=w, max_size=w)
            ),
        )

    slots = draw(
        st.lists(
            st.one_of(_valid_row(), _malformed_row()),
            min_size=1,
            max_size=10,
        )
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for _, row in slots:
        writer.writerow(row)

    expected_valid = [
        dict(zip(header, row)) for valid, row in slots if valid
    ]
    # Header is line 1; data row at slot index k is line k + 2.
    malformed_keys = [
        k + 2 for k, (valid, _) in enumerate(slots) if not valid
    ]

    unit = RawRecordUnit(source_file="data.csv", payload=buffer.getvalue())
    return SupportedFormat.CSV, [unit], expected_valid, malformed_keys


def _fault_cases():
    return st.one_of(
        _json_fault_case(),
        _jsonl_fault_case(),
        _csv_fault_case(),
    )


def _issue_key(issue, fmt: SupportedFormat):
    """The locating coordinate an issue should carry for ``fmt``."""
    if fmt is SupportedFormat.JSON:
        return issue.location.array_index
    return issue.location.line_number


# Feature: llm-training-data-quality-analyzer, Property 4: Parse fault isolation
@_SETTINGS
@given(case=_fault_cases())
def test_parse_fault_isolation(case):
    """Every valid unit yields a Record; every malformed unit yields one located
    issue; valid records are never dropped or corrupted.

    Validates: Requirements 2.5
    """
    fmt, units, expected_valid, malformed_keys = case
    records, issues = Parser().parse(units, fmt)

    # A Record is produced for every valid unit, in order, uncorrupted.
    assert len(records) == len(expected_valid)
    for record, expected_fields in zip(records, expected_valid):
        assert fields_equivalent(record.fields, expected_fields)

    # Exactly one located Quality_Issue per malformed unit.
    assert len(issues) == len(malformed_keys)
    assert all(issue.category == IssueCategory.PARSE_ERROR for issue in issues)
    actual_keys = sorted(_issue_key(issue, fmt) for issue in issues)
    assert actual_keys == sorted(malformed_keys)
