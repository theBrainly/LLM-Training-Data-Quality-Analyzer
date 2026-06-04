"""Format consistency validation (Requirement 9).

The :class:`FormatValidator` checks that every Record in a
:class:`~analyzer.models.Dataset` conforms to a :class:`~analyzer.models.Schema`.
The schema is either *declared* by the caller or *inferred* from the first
Record of the Dataset when none is declared.

Two kinds of inconsistency are reported, each as a typed
:class:`~analyzer.models.QualityIssue` identifying the affected Record, the
nonconforming field, and the inconsistency type (Requirement 9.6):

* **Missing required field** - a field the schema declares as required that is
  *absent* from the Record or whose value is *null* (both treated as missing),
  reported as :class:`~analyzer.models.IssueCategory.MISSING_REQUIRED_FIELD`
  (Requirement 9.1).
* **Field type mismatch** - a field present with a non-null value whose
  canonical type differs from the type the schema declares, reported as
  :class:`~analyzer.models.IssueCategory.FIELD_TYPE_MISMATCH` (Requirement 9.2).

Schema handling follows the rules in Requirement 9:

* When no schema is declared and the first Record contains at least one field,
  the schema is inferred from the first Record's field names and value types,
  with every inferred field marked required (Requirement 9.3); every subsequent
  Record is then validated against that inferred schema (Requirement 9.4).
* When no schema is declared and the Dataset is empty or the first Record has
  no fields, a single dataset-level
  :class:`~analyzer.models.IssueCategory.SCHEMA_INFERENCE_FAILED` issue is
  recorded and no per-record validation is performed (Requirement 9.5).
"""

from __future__ import annotations

from analyzer.models import (
    Dataset,
    FieldSpec,
    FieldType,
    IssueCategory,
    QualityIssue,
    Record,
    Schema,
    Value,
)

__all__ = [
    "FormatValidator",
    "value_field_type",
]


def value_field_type(value: Value) -> FieldType:
    """Map a canonical :data:`~analyzer.models.Value` to its :class:`FieldType`.

    Booleans are checked before integers because ``bool`` is a subclass of
    ``int`` in Python; ``True``/``False`` map to :attr:`FieldType.BOOLEAN`, not
    :attr:`FieldType.INTEGER`. ``None`` maps to :attr:`FieldType.NULL`, lists to
    :attr:`FieldType.LIST`, and dicts to :attr:`FieldType.OBJECT`.
    """
    if value is None:
        return FieldType.NULL
    if isinstance(value, bool):
        return FieldType.BOOLEAN
    if isinstance(value, int):
        return FieldType.INTEGER
    if isinstance(value, float):
        return FieldType.FLOAT
    if isinstance(value, str):
        return FieldType.STRING
    if isinstance(value, list):
        return FieldType.LIST
    if isinstance(value, dict):
        return FieldType.OBJECT
    # Defensive: anything outside the canonical Value union is treated as an
    # object so validation never raises on unexpected input.
    return FieldType.OBJECT


class FormatValidator:
    """Validates records against a declared or first-record-inferred schema."""

    def validate(
        self, dataset: Dataset, schema: Schema | None = None
    ) -> list[QualityIssue]:
        """Return the Quality_Issues describing schema nonconformities.

        When ``schema`` is provided, every Record is validated against it. When
        ``schema`` is ``None`` the schema is inferred from the first Record (if
        it has at least one field) and the remaining Records are validated
        against the inferred schema; if inference is impossible a single
        ``SCHEMA_INFERENCE_FAILED`` issue is returned (Requirement 9.5).
        """
        if schema is not None:
            return self._validate_records(dataset.records, schema)

        # No declared schema: attempt to infer from the first record.
        records = dataset.records
        if not records or not records[0].fields:
            return [
                QualityIssue(
                    category=IssueCategory.SCHEMA_INFERENCE_FAILED,
                    location=None,
                    detail=(
                        "Schema inference could not be performed: the dataset is "
                        "empty"
                        if not records
                        else "Schema inference could not be performed: the first "
                        "record contains no fields"
                    ),
                )
            ]

        inferred = self._infer_schema(records[0])
        # The first record trivially conforms to a schema inferred from it;
        # validate every subsequent record against the inferred schema
        # (Requirement 9.4).
        return self._validate_records(records[1:], inferred)

    @staticmethod
    def _infer_schema(record: Record) -> Schema:
        """Infer a :class:`Schema` from ``record``'s field names and types.

        Every inferred field is marked required so subsequent records are
        expected to carry the same shape (Requirement 9.3).
        """
        fields = [
            FieldSpec(name=name, type=value_field_type(value), required=True)
            for name, value in record.fields.items()
        ]
        return Schema(fields=fields, inferred=True)

    @staticmethod
    def _validate_records(
        records: list[Record], schema: Schema
    ) -> list[QualityIssue]:
        """Validate each Record in ``records`` against ``schema``."""
        issues: list[QualityIssue] = []
        for record in records:
            for spec in schema.fields:
                present = spec.name in record.fields
                value = record.fields.get(spec.name)

                # Absent or null counts as missing for a required field
                # (Requirement 9.1).
                if not present or value is None:
                    if spec.required:
                        reason = "absent" if not present else "null"
                        issues.append(
                            QualityIssue(
                                category=IssueCategory.MISSING_REQUIRED_FIELD,
                                location=record.location,
                                field_name=spec.name,
                                detail=(
                                    f"Required field '{spec.name}' is {reason}"
                                ),
                            )
                        )
                    # Optional field that is absent/null is conformant; an
                    # optional null is not a type mismatch.
                    continue

                # Field present with a non-null value: compare types
                # (Requirement 9.2).
                actual = value_field_type(value)
                if actual is not spec.type:
                    issues.append(
                        QualityIssue(
                            category=IssueCategory.FIELD_TYPE_MISMATCH,
                            location=record.location,
                            field_name=spec.name,
                            detail=(
                                f"Field '{spec.name}' has type {actual.value}, "
                                f"expected {spec.type.value}"
                            ),
                        )
                    )
        return issues
