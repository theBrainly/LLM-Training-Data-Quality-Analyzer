"""Property-based test for directory ingestion combination (task 4.2).

Drives the real :class:`~analyzer.ingestion.IngestionEngine` and
:class:`~analyzer.parsers.Parser` through :func:`~analyzer.combine.combine`
over temporary directories of generated supported-format files, asserting that
the combined :class:`Dataset` is exactly the concatenation of every supported
file's records in file-listing order then within-file parse order
(Requirements 1.1, 1.2).

The Ingestion_Engine enumerates a directory's files in name-sorted order, so we
write each generated file under a zero-padded ``file_NN`` stem whose sort order
matches generation order; the expected combined record sequence is therefore
the per-file record lists concatenated in generation order.

Hypothesis note: temp directories are created with :mod:`tempfile` inside the
test body rather than via the function-scoped ``tmp_path`` fixture, which is
unsafe to reuse across the many examples a single ``@given`` test drives.
"""

from __future__ import annotations

import json
import string
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from analyzer.combine import combine
from analyzer.ingestion import IngestionEngine
from analyzer.parsers import Parser

# Field names safe across JSON/JSONL.
_field_names = st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=6)

# Canonical scalar values that survive a JSON/JSONL round-trip unchanged.
_scalars = st.one_of(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=12),
    st.integers(),
    st.booleans(),
    st.none(),
    st.floats(allow_nan=False, allow_infinity=False),
)

# A single record's fields and a per-file list of such records.
_record_fields = st.dictionaries(_field_names, _scalars, max_size=4)
_file_records = st.lists(_record_fields, max_size=4)

# A file is a (format, records) pair; JSON and JSONL both preserve the canonical
# scalar values above, keeping the focus on combination order rather than
# format coercion (parse fidelity per format is Property 3's concern).
_files = st.lists(
    st.tuples(st.sampled_from(["json", "jsonl"]), _file_records),
    max_size=4,
)


def _write_file(directory: Path, index: int, fmt: str, fields_list: list[dict]) -> str:
    """Write one generated file and return its path.

    The stem ``file_NN`` is zero-padded so lexicographic (name-sorted) order,
    which the Ingestion_Engine uses to enumerate a directory, matches the
    generation order ``index``.
    """
    stem = f"file_{index:02d}"
    if fmt == "json":
        path = directory / f"{stem}.json"
        path.write_text(json.dumps(fields_list), encoding="utf-8")
    else:  # jsonl
        path = directory / f"{stem}.jsonl"
        body = "".join(json.dumps(fields) + "\n" for fields in fields_list)
        path.write_text(body, encoding="utf-8")
    return str(path)


# Feature: llm-training-data-quality-analyzer, Property 5: Directory ingestion combines all supported files in order
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(files=_files)
def test_directory_ingestion_combines_all_supported_files_in_order(files):
    """The combined Dataset concatenates every supported file's records in
    file-listing order then within-file parse order (Requirements 1.1, 1.2)."""
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)

        expected_fields: list[dict] = []
        expected_source_files: list[str] = []
        for index, (fmt, fields_list) in enumerate(files):
            path = _write_file(directory, index, fmt, fields_list)
            expected_fields.extend(fields_list)
            expected_source_files.append(path)

        result = IngestionEngine().ingest(str(directory))
        combined = combine(result, Parser())

        actual_fields = [record.fields for record in combined.dataset.records]

        # Record count equals the sum of per-file record counts (Requirement 1.2).
        assert len(actual_fields) == len(expected_fields)
        # Records appear in file-listing order then within-file order.
        assert actual_fields == expected_fields
        # source_files lists exactly the parsed files in name-sorted order.
        assert combined.dataset.source_files == sorted(expected_source_files)
        # No unsupported files were present, so nothing is skipped.
        assert combined.dataset.skipped_files == []
        # No malformed records were planted, so no parse issues arise.
        assert combined.issues == []
