"""Unit tests for Format_Validator schema-inference failure (Requirement 9.5).

When no schema is declared and inference is impossible - the dataset has zero
records, or the first record has no fields - the validator records exactly one
dataset-level ``SCHEMA_INFERENCE_FAILED`` issue and performs no per-record
validation. These example-based tests pin those failure paths; broad
property-based coverage of successful inference lives alongside in
``test_format_validator_properties.py``.
"""

from analyzer.detectors.format_validator import FormatValidator
from analyzer.models import (
    Dataset,
    IssueCategory,
    Record,
    RecordLocation,
)


def _rec(fields: dict, index: int) -> Record:
    return Record(
        fields=fields,
        location=RecordLocation(source_file="data", array_index=index),
    )


def _dataset(records: list[Record]) -> Dataset:
    return Dataset(records=records, source_files=["data"])


def _assert_single_inference_failure(issues) -> None:
    assert len(issues) == 1
    issue = issues[0]
    assert issue.category is IssueCategory.SCHEMA_INFERENCE_FAILED
    # Dataset-level issue: not tied to any single record.
    assert issue.location is None
    assert issue.detail


def test_empty_dataset_records_schema_inference_failed():
    issues = FormatValidator().validate(_dataset([]))
    _assert_single_inference_failure(issues)


def test_fieldless_first_record_records_schema_inference_failed():
    records = [_rec({}, 0)]
    issues = FormatValidator().validate(_dataset(records))
    _assert_single_inference_failure(issues)


def test_fieldless_first_record_skips_validation_of_later_records():
    # The first record is fieldless, so inference fails and no subsequent
    # record is validated - even though the later records have fields that
    # would otherwise be inspected.
    records = [
        _rec({}, 0),
        _rec({"text": "hello", "count": 1}, 1),
        _rec({"text": "world"}, 2),
    ]
    issues = FormatValidator().validate(_dataset(records))
    _assert_single_inference_failure(issues)


def test_inference_failure_only_applies_without_declared_schema():
    # Sanity guard: the failure path is specific to undeclared-schema inference;
    # passing schema=None on an empty dataset is the trigger, and the issue is
    # dataset-level rather than referencing a record location.
    issues = FormatValidator().validate(_dataset([]), schema=None)
    _assert_single_inference_failure(issues)
