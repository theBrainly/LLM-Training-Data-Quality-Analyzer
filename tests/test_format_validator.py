"""Unit tests for the Format_Validator (Requirements 9.1-9.6).

These example-based tests cover validation against a declared schema (missing
required fields and field type mismatches with typed issues), schema inference
from the first record and validation of subsequent records, and the
schema-inference-failed cases for empty/fieldless first records and empty
datasets. Property-based coverage lives in separate tasks.
"""

from analyzer.detectors.format_validator import (
    FormatValidator,
    value_field_type,
)
from analyzer.models import (
    Dataset,
    FieldSpec,
    FieldType,
    IssueCategory,
    Record,
    RecordLocation,
    Schema,
)


def _rec(fields: dict, index: int) -> Record:
    return Record(
        fields=fields,
        location=RecordLocation(source_file="data", array_index=index),
    )


def _dataset(records: list[Record]) -> Dataset:
    return Dataset(records=records, source_files=["data"])


class TestValueFieldTypeMapping:
    def test_maps_each_canonical_value_type(self):
        assert value_field_type("x") is FieldType.STRING
        assert value_field_type(3) is FieldType.INTEGER
        assert value_field_type(3.5) is FieldType.FLOAT
        assert value_field_type(None) is FieldType.NULL
        assert value_field_type([1, 2]) is FieldType.LIST
        assert value_field_type({"a": 1}) is FieldType.OBJECT

    def test_bool_maps_to_boolean_not_integer(self):
        # bool is a subclass of int; it must map to BOOLEAN.
        assert value_field_type(True) is FieldType.BOOLEAN
        assert value_field_type(False) is FieldType.BOOLEAN


class TestDeclaredSchemaMissingRequired:
    def test_absent_required_field_is_flagged(self):
        schema = Schema(
            fields=[
                FieldSpec("text", FieldType.STRING, required=True),
                FieldSpec("label", FieldType.STRING, required=True),
            ]
        )
        records = [_rec({"text": "hi"}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)

        missing = [
            i for i in issues if i.category is IssueCategory.MISSING_REQUIRED_FIELD
        ]
        assert len(missing) == 1
        assert missing[0].field_name == "label"
        assert missing[0].location == records[0].location

    def test_null_required_field_is_treated_as_missing(self):
        schema = Schema(
            fields=[FieldSpec("label", FieldType.STRING, required=True)]
        )
        records = [_rec({"label": None}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)

        missing = [
            i for i in issues if i.category is IssueCategory.MISSING_REQUIRED_FIELD
        ]
        assert len(missing) == 1
        assert missing[0].field_name == "label"
        # A null required field is missing, not a type mismatch.
        assert not any(
            i.category is IssueCategory.FIELD_TYPE_MISMATCH for i in issues
        )

    def test_optional_absent_field_is_not_flagged(self):
        schema = Schema(
            fields=[
                FieldSpec("text", FieldType.STRING, required=True),
                FieldSpec("note", FieldType.STRING, required=False),
            ]
        )
        records = [_rec({"text": "hi"}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)
        assert issues == []

    def test_flags_missing_field_in_every_offending_record(self):
        schema = Schema(
            fields=[FieldSpec("label", FieldType.STRING, required=True)]
        )
        records = [
            _rec({"label": "a"}, 0),
            _rec({}, 1),
            _rec({"label": None}, 2),
        ]
        issues = FormatValidator().validate(_dataset(records), schema)
        missing = [
            i for i in issues if i.category is IssueCategory.MISSING_REQUIRED_FIELD
        ]
        assert {i.location.array_index for i in missing} == {1, 2}


class TestDeclaredSchemaTypeMismatch:
    def test_type_mismatch_is_flagged_with_field_and_type(self):
        schema = Schema(
            fields=[FieldSpec("count", FieldType.INTEGER, required=True)]
        )
        records = [_rec({"count": "not a number"}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)

        mismatches = [
            i for i in issues if i.category is IssueCategory.FIELD_TYPE_MISMATCH
        ]
        assert len(mismatches) == 1
        assert mismatches[0].field_name == "count"
        assert mismatches[0].location == records[0].location

    def test_integer_does_not_satisfy_float_field(self):
        schema = Schema(
            fields=[FieldSpec("ratio", FieldType.FLOAT, required=True)]
        )
        records = [_rec({"ratio": 1}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)
        assert sum(
            i.category is IssueCategory.FIELD_TYPE_MISMATCH for i in issues
        ) == 1

    def test_bool_does_not_satisfy_integer_field(self):
        schema = Schema(
            fields=[FieldSpec("n", FieldType.INTEGER, required=True)]
        )
        records = [_rec({"n": True}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)
        assert sum(
            i.category is IssueCategory.FIELD_TYPE_MISMATCH for i in issues
        ) == 1

    def test_conformant_record_yields_no_issues(self):
        schema = Schema(
            fields=[
                FieldSpec("text", FieldType.STRING, required=True),
                FieldSpec("count", FieldType.INTEGER, required=True),
            ]
        )
        records = [_rec({"text": "hi", "count": 5}, 0)]
        issues = FormatValidator().validate(_dataset(records), schema)
        assert issues == []


class TestSchemaInference:
    def test_infers_schema_from_first_record_and_validates_rest(self):
        records = [
            _rec({"text": "hello", "count": 1}, 0),
            _rec({"text": "world", "count": 2}, 1),  # conformant
            _rec({"text": "bad", "count": "two"}, 2),  # count type mismatch
            _rec({"text": "missing"}, 3),  # count absent
        ]
        issues = FormatValidator().validate(_dataset(records))

        mismatches = [
            i for i in issues if i.category is IssueCategory.FIELD_TYPE_MISMATCH
        ]
        missing = [
            i for i in issues if i.category is IssueCategory.MISSING_REQUIRED_FIELD
        ]
        assert len(mismatches) == 1
        assert mismatches[0].location.array_index == 2
        assert mismatches[0].field_name == "count"
        assert len(missing) == 1
        assert missing[0].location.array_index == 3
        assert missing[0].field_name == "count"

    def test_first_record_itself_is_not_validated(self):
        # The first record always conforms to a schema inferred from it.
        records = [_rec({"a": "x"}, 0)]
        issues = FormatValidator().validate(_dataset(records))
        assert issues == []


class TestSchemaInferenceFailed:
    def test_empty_dataset_records_inference_failed(self):
        issues = FormatValidator().validate(_dataset([]))
        assert len(issues) == 1
        assert issues[0].category is IssueCategory.SCHEMA_INFERENCE_FAILED
        assert issues[0].location is None

    def test_fieldless_first_record_records_inference_failed(self):
        records = [_rec({}, 0), _rec({"a": 1}, 1)]
        issues = FormatValidator().validate(_dataset(records))
        assert len(issues) == 1
        assert issues[0].category is IssueCategory.SCHEMA_INFERENCE_FAILED
        assert issues[0].location is None
