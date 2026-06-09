"""Property-based tests for the Pretty_Printer round-trip fidelity and the
located-serialization-error behaviour (Requirements 3.1, 3.2, 3.3).

Two correctness properties are implemented here, each by exactly one
property-based test (Hypothesis, >= 100 examples):

* Property 1 - parse -> print -> parse round-trip fidelity, exercised across
  *every* Supported_Format (JSON, JSONL, CSV, Parquet) inside the single test.
* Property 2 - a value that cannot be represented in the target format halts
  serialization with a located error (record index + field name).

Round-trip mechanics per format:

* JSON   - the printer emits a single top-level array; the JSON parser expects a
  whole-file array payload, so the printed text is fed back as one unit.
* JSONL  - the printer emits one object per line; fed back as one whole-file unit.
* CSV    - the printer emits a header + one row per record; fed back as text.
  CSV cells are scalar text only, so the round-trip domain for CSV is records
  whose fields share a single header and hold string values - exactly the shape
  records parsed *from* a CSV have (Requirement 3.2 is scoped to records parsed
  from a Supported_Format).
* Parquet - the printer emits ``latin-1``-encoded Parquet bytes packed into a
  ``str``; to parse back the text is re-encoded with ``latin-1`` and the bytes
  are handed to the Parquet parser. Parquet columns are typed, so the round-trip
  domain is uniform-key records with a single value type per column - again the
  shape of records parsed from a Parquet table.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from analyzer.models import (
    Record,
    RecordLocation,
    SupportedFormat,
    records_equivalent,
)
from analyzer.parsers import Parser, RawRecordUnit
from analyzer.pretty_printer import PrettyPrinter
from tests.strategies import field_names, record_lists


# --------------------------------------------------------------------------- #
# Format-appropriate round-trippable record strategies
# --------------------------------------------------------------------------- #
#
# JSON and JSONL place no structural constraint on a record list beyond value
# representability, so the shared ``record_lists`` strategy is used directly.
# CSV and Parquet share a single header / typed schema across all rows, so the
# round-trippable domain for them is uniform-key records (the shape produced by
# parsing those formats). These two strategies build exactly that domain.


def _csv_cell_text() -> st.SearchStrategy[str]:
    """Text that survives a CSV cell round-trip (no surrogates, no control
    characters that would split or alter a line)."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        max_size=20,
    )


# Each Parquet column carries a single value type so Arrow does not promote
# (e.g. int+float -> float) or fail to unify the column. ``None`` is mixed in
# because a typed Arrow column is nullable and round-trips ``None`` faithfully.
_PARQUET_COLUMN_VALUES = (
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)


@st.composite
def _csv_table_records(draw) -> list[Record]:
    """A list of uniform-key, string-valued records (CSV round-trip domain)."""
    names = draw(st.lists(field_names(), min_size=1, max_size=4, unique=True))
    n = draw(st.integers(min_value=0, max_value=6))
    return [
        Record(
            fields={name: draw(_csv_cell_text()) for name in names},
            location=RecordLocation(source_file="rt.csv", line_number=row + 2),
        )
        for row in range(n)
    ]


@st.composite
def _parquet_table_records(draw) -> list[Record]:
    """A list of uniform-key records with one value type per column (Parquet
    round-trip domain)."""
    names = draw(st.lists(field_names(), min_size=1, max_size=4, unique=True))
    column_value = {
        name: st.one_of(st.none(), draw(st.sampled_from(_PARQUET_COLUMN_VALUES)))
        for name in names
    }
    n = draw(st.integers(min_value=0, max_value=6))
    return [
        Record(
            fields={name: draw(column_value[name]) for name in names},
            location=RecordLocation(
                source_file="rt.parquet", row_group=0, row_index=row
            ),
        )
        for row in range(n)
    ]


def _records_for(fmt: SupportedFormat, data) -> list[Record]:
    """Draw a round-trippable record list appropriate for ``fmt``."""
    if fmt is SupportedFormat.CSV:
        return data.draw(_csv_table_records())
    if fmt is SupportedFormat.PARQUET:
        return data.draw(_parquet_table_records())
    # JSON / JSONL: any representable record list round-trips.
    return data.draw(record_lists(fmt=fmt, representable=True))


def _parse_back(text: str, fmt: SupportedFormat) -> list[Record]:
    """Parse the Pretty_Printer's output for ``fmt`` back into records.

    JSON/JSONL/CSV are whole-file text payloads; Parquet output is ``latin-1``
    text wrapping the binary document, recovered with ``.encode("latin-1")``.
    """
    if fmt is SupportedFormat.PARQUET:
        payload = text.encode("latin-1")
    else:
        payload = text
    unit = RawRecordUnit(source_file=f"rt.{fmt.value}", payload=payload)
    records, _issues = Parser().parse([unit], fmt)
    return records


# --------------------------------------------------------------------------- #
# Property 1: Round-trip fidelity (parse -> print -> parse)
# --------------------------------------------------------------------------- #

# Feature: llm-training-data-quality-analyzer, Property 1: Round-trip fidelity (parse → print → parse)
@settings(max_examples=100)
@given(data=st.data())
def test_roundtrip_fidelity_across_all_formats(data):
    """Printing a list of representable records and parsing the output back
    yields an equivalent list, for every Supported_Format.

    **Validates: Requirements 3.1, 3.2**
    """
    printer = PrettyPrinter()
    for fmt in SupportedFormat:
        records = _records_for(fmt, data)

        result = printer.print(records, fmt)
        # Every value is representable, so printing must succeed with output.
        assert result.error is None, (fmt, result.error)
        assert result.text is not None

        parsed = _parse_back(result.text, fmt)
        assert records_equivalent(records, parsed), (
            fmt,
            [r.fields for r in records],
            [r.fields for r in parsed],
        )


# --------------------------------------------------------------------------- #
# Property 2: Unrepresentable values halt printing with a located error
# --------------------------------------------------------------------------- #

# Formats in which at least one canonical value is unrepresentable. Parquet is
# excluded: Apache Arrow can represent every canonical Value (including nested
# containers and non-finite floats), so the property has no instances there.
_LOSSY_FORMATS = (
    SupportedFormat.JSON,
    SupportedFormat.JSONL,
    SupportedFormat.CSV,
)


# Feature: llm-training-data-quality-analyzer, Property 2: Unrepresentable values halt printing with a located error
@settings(max_examples=100)
@given(data=st.data())
def test_unrepresentable_value_halts_with_located_error(data):
    """A record list containing a value the target format cannot represent
    produces no output and an error naming the offending record index and
    field.

    **Validates: Requirements 3.3**
    """
    printer = PrettyPrinter()
    for fmt in _LOSSY_FORMATS:
        records = data.draw(record_lists(fmt=fmt, representable=False))

        result = printer.print(records, fmt)

        # No partial output is produced.
        assert result.text is None, fmt
        # The error locates the offending record and field within the input.
        assert result.error is not None, fmt
        assert result.error.fmt == fmt.value
        assert 0 <= result.error.record_index < len(records)
        offending = records[result.error.record_index]
        assert result.error.field_name in offending.fields
