"""Serialization of :class:`Record` lists back into a Supported_Format.

The Pretty_Printer is a standalone service used both for exporting analyzed
data and for round-trip verification (parse -> print -> parse). It operates
purely on the canonical ``Value`` model, so it never needs to understand a
format's native types beyond checking *representability*: whether each value
can be faithfully expressed in the requested target format.

Responsibilities (Requirement 3):

* Serialize a list of records, preserving the order supplied by the caller
  (Requirement 3.1).
* If any field value cannot be represented in the target format, halt, produce
  no output, and return an error identifying the offending record's index in
  the input list and the field name (Requirement 3.3).

This module implements all four Supported_Formats:

* **JSON** - a single top-level array, one object per record.
* **JSONL** - one JSON object per line.
* **CSV** - a header row followed by one row per record. CSV cells are scalar
  only, so a field whose value is a nested ``list``/``dict`` is *unrepresentable*
  and triggers the Requirement 3.3 error.
* **Parquet** - an Apache Arrow table written to the Parquet binary format. Arrow
  can represent every canonical ``Value`` (including nested lists/objects), so
  no value is unrepresentable in Parquet.

Parquet is a binary format, yet :class:`PrintResult` carries the serialized
output as ``str``. Parquet output is therefore encoded with ``latin-1`` (an
exact, reversible byte<->codepoint mapping), so the bytes can be recovered with
``text.encode("latin-1")`` without loss.

Empty record lists yield a valid empty representation per format and never an
error (Requirement 3.4): ``[]`` for JSON, an empty string for JSONL, an empty
string for CSV (no records means no header columns), and an empty Arrow table
for Parquet.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from analyzer.errors import UnrepresentableValueError
from analyzer.models import Record, SupportedFormat, Value


@dataclass
class PrintResult:
    """The outcome of a serialization request.

    Exactly one of ``text`` / ``error`` is populated. On success ``text`` holds
    the serialized representation and ``error`` is ``None``. On an
    unrepresentable value ``text`` is ``None`` (no partial output) and ``error``
    identifies the record index and field name that could not be represented.
    """

    text: str | None
    error: UnrepresentableValueError | None


def _json_representable(value: Value) -> bool:
    """Return True iff ``value`` can be represented in JSON / JSONL.

    Every canonical ``Value`` maps onto a JSON type, with one exception: JSON
    has no notion of non-finite floats, so ``NaN`` and the infinities are not
    representable. Containers are checked recursively.
    """
    if isinstance(value, bool):
        # bool is a JSON literal (true/false). Checked before float/int since
        # bool is a subclass of int.
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_representable(item) for item in value)
    if isinstance(value, dict):
        return all(_json_representable(item) for item in value.values())
    # str, int, None are always representable.
    return True


def _serialize_json(records: list[Record]) -> str:
    """Serialize records as a single JSON array, one element per record.

    Field insertion order is preserved within each record, and record order is
    preserved across the array.
    """
    payload = [dict(record.fields) for record in records]
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _serialize_jsonl(records: list[Record]) -> str:
    """Serialize records as JSONL: one JSON object per line, in input order."""
    lines = [
        json.dumps(dict(record.fields), ensure_ascii=False, allow_nan=False)
        for record in records
    ]
    return "\n".join(lines)


def _csv_representable(value: Value) -> bool:
    """Return True iff ``value`` can be represented as a single CSV cell.

    CSV cells are scalar: a record field whose value is a nested ``list`` or
    ``dict`` has no faithful single-cell representation and is therefore
    unrepresentable (Requirement 3.3). Every scalar canonical value
    (``str``/``int``/``float``/``bool``/``None``) maps onto a cell.
    """
    return not isinstance(value, (list, dict))


def _csv_field_names(records: list[Record]) -> list[str]:
    """Return the CSV header: every field name in first-seen order.

    Records parsed from a single CSV share one header, but a caller may supply
    records with differing field sets; the union (in first-seen order) yields a
    well-formed table where a record missing a column produces an empty cell.
    """
    field_names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for name in record.fields:
            if name not in seen:
                seen.add(name)
                field_names.append(name)
    return field_names


def _serialize_csv(records: list[Record]) -> str:
    """Serialize records as CSV: a header row followed by one row per record.

    The header is the union of field names across the records (first-seen
    order) and each row writes that record's scalar cell values in header
    order, preserving record order (Requirement 3.1). An empty record list has
    no columns and yields an empty string (Requirement 3.4).
    """
    if not records:
        return ""

    field_names = _csv_field_names(records)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=field_names,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for record in records:
        writer.writerow({name: record.fields.get(name) for name in field_names})
    return buffer.getvalue()


def _serialize_parquet(records: list[Record]) -> str:
    """Serialize records as a Parquet (Apache Arrow) document, in input order.

    The records are assembled into an Arrow table (one row per record) and
    written to the Parquet binary format. The resulting bytes are returned as a
    ``latin-1`` string so they fit :class:`PrintResult`'s ``text`` field
    losslessly. An empty record list produces an empty Arrow table, which is a
    valid Parquet document (Requirement 3.4).
    """
    table = pa.Table.from_pylist([dict(record.fields) for record in records])
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes().decode("latin-1")


# Dispatch tables keyed by target format. Each format contributes a
# representability predicate (used to locate the first unrepresentable value)
# and a serializer. New formats register here without changing the printing
# algorithm.
_REPRESENTABLE = {
    SupportedFormat.JSON: _json_representable,
    SupportedFormat.JSONL: _json_representable,
    SupportedFormat.CSV: _csv_representable,
    # Arrow represents every canonical Value (including nested containers and
    # non-finite floats), so no value is unrepresentable in Parquet.
    SupportedFormat.PARQUET: lambda value: True,
}

_SERIALIZERS = {
    SupportedFormat.JSON: _serialize_json,
    SupportedFormat.JSONL: _serialize_jsonl,
    SupportedFormat.CSV: _serialize_csv,
    SupportedFormat.PARQUET: _serialize_parquet,
}


class PrettyPrinter:
    """Serializes :class:`Record` lists back into a Supported_Format."""

    def print(self, records: list[Record], fmt: SupportedFormat) -> PrintResult:
        """Serialize ``records`` into ``fmt``.

        Performs a representability pass first: every field value of every
        record is checked against the target format. The first value (lowest
        record index, then field order) that cannot be represented halts
        serialization with no output and yields a located error (Requirement
        3.3). When all values are representable, the records are serialized in
        input order (Requirement 3.1).
        """
        if fmt not in _SERIALIZERS:
            raise NotImplementedError(
                f"Serialization to {fmt.value!r} is not implemented yet"
            )

        is_representable = _REPRESENTABLE[fmt]

        # Representability pass: locate the first offending value before
        # producing any output, so a failure yields no partial result.
        for index, record in enumerate(records):
            for field_name, value in record.fields.items():
                if not is_representable(value):
                    return PrintResult(
                        text=None,
                        error=UnrepresentableValueError(
                            record_index=index,
                            field_name=field_name,
                            fmt=fmt.value,
                        ),
                    )

        return PrintResult(text=_SERIALIZERS[fmt](records), error=None)
