"""Property-based tests for the Ingestion_Engine (tasks 2.3 and 2.4).

These cover two design Correctness Properties:

* Property 7 - the effective maximum file size is the configured value clamped
  into the inclusive range ``[1 MiB, 50 GiB]`` (Requirement 1.7).
* Property 6 - unsupported-extension files are skipped and recorded, never
  contributing records (Requirement 1.5).

Both run a minimum of 100 Hypothesis examples per the design's Testing Strategy.
Directory examples build a fresh temporary directory per example (rather than
using a function-scoped ``tmp_path`` fixture) so Hypothesis can drive many
inputs without tripping its fixture health checks.
"""

from __future__ import annotations

import os
import string
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from analyzer.ingestion import (
    DEFAULT_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    IngestionConfig,
    IngestionEngine,
    clamp_file_size,
    detect_format,
)


# --------------------------------------------------------------------------- #
# Property 7: file size clamping (task 2.3, Requirement 1.7)
# --------------------------------------------------------------------------- #

# Configured values span the categories called out by the property: unset
# (None), negative, zero, sub-1 MiB, in-range, and above-50 GiB.
_configured_sizes = st.one_of(
    st.none(),
    st.integers(min_value=-(10**18), max_value=10**18),
)


# Feature: llm-training-data-quality-analyzer, Property 7: Effective max file size is the clamped configured value
@settings(max_examples=200)
@given(configured=_configured_sizes)
def test_effective_max_file_size_is_clamped_configured_value(configured):
    effective = clamp_file_size(configured)

    if configured is None:
        # An unset value yields exactly the 5 GiB default.
        assert effective == DEFAULT_FILE_SIZE_BYTES
    else:
        # Any other value equals that value clamped into [1 MiB, 50 GiB].
        expected = min(max(configured, MIN_FILE_SIZE_BYTES), MAX_FILE_SIZE_BYTES)
        assert effective == expected

    # The effective value is always inside the inclusive bounds.
    assert MIN_FILE_SIZE_BYTES <= effective <= MAX_FILE_SIZE_BYTES

    # IngestionConfig stores the same effective (clamped) value.
    assert IngestionConfig(configured).max_file_size_bytes == effective


# --------------------------------------------------------------------------- #
# Property 6: unsupported-extension skipping (task 2.4, Requirement 1.5)
# --------------------------------------------------------------------------- #

_SUPPORTED_EXTS = (".json", ".jsonl", ".csv", ".parquet")
_UNSUPPORTED_EXTS = (".txt", ".md", ".png", ".dat", ".xml", ".yaml", ".log", ".bin", ".html")

_stems = st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=8)


@st.composite
def _mixed_directory_spec(draw):
    """A list of ``(filename, is_supported)`` for a directory.

    Unique stems guarantee unique filenames. At least one supported and one
    unsupported file are always present, so the directory both streams records
    and exercises the skip path.
    """
    stems = draw(st.lists(_stems, min_size=2, max_size=8, unique=True))

    supported_flags = [draw(st.booleans()) for _ in stems]
    # Force at least one of each classification.
    supported_flags[0] = True
    supported_flags[-1] = False

    files: list[tuple[str, bool]] = []
    for stem, is_supported in zip(stems, supported_flags):
        exts = _SUPPORTED_EXTS if is_supported else _UNSUPPORTED_EXTS
        ext = draw(st.sampled_from(exts))
        files.append((stem + ext, is_supported))
    return files


# Feature: llm-training-data-quality-analyzer, Property 6: Unsupported extensions are skipped and recorded
@settings(max_examples=100)
@given(spec=_mixed_directory_spec())
def test_unsupported_extensions_are_skipped_and_recorded(spec):
    with tempfile.TemporaryDirectory() as directory:
        supported_paths: list[str] = []
        unsupported_paths: list[str] = []
        for name, is_supported in spec:
            path = os.path.join(directory, name)
            with open(path, "wb") as handle:
                handle.write(b"[]")
            if is_supported:
                supported_paths.append(path)
            else:
                unsupported_paths.append(path)

        # Sanity-check the test's own classification against the engine's.
        assert all(detect_format(p) is not None for p in supported_paths)
        assert all(detect_format(p) is None for p in unsupported_paths)

        result = IngestionEngine().ingest(directory)

        # With at least one supported file present, there is no fail-fast error.
        assert result.error is None

        # Every unsupported-extension file is recorded in skipped_files...
        assert set(result.skipped_files) == set(unsupported_paths)

        # ...and contributes no records to the stream.
        unit_sources = {unit.source_file for unit in result.units}
        assert unit_sources.isdisjoint(unsupported_paths)
        assert unit_sources == set(supported_paths)
