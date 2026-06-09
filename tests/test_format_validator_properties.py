"""Property-based tests for the Format_Validator (Requirement 9).

Each test validates exactly one Correctness Property from the design's
32-property numbering and references the requirement clauses it covers:

* Property 24 - missing-required-field detection (9.1, 9.6).
* Property 25 - field type-mismatch detection (9.2, 9.6).
* Property 26 - schema inference from the first record (9.3, 9.4).

The datasets are planted so the ground truth is known independently of the
implementation: each record's per-field state (present-and-valid, absent, null,
or wrong-type) is chosen by the generator, so the exact multiset of expected
``MISSING_REQUIRED_FIELD`` / ``FIELD_TYPE_MISMATCH`` issues (keyed by record
location and field name) can be reconstructed and compared against the
validator's output.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

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
from tests.strategies import field_names, field_values

_SETTINGS = settings(max_examples=100, deadline=None)

# Concrete (non-null) field types a declared/inferred schema field can carry.
# NULL is excluded for declared fields because a present ``None`` value is
# treated as *missing* (not a NULL-typed value), which would conflate the two
# inconsistency kinds the properties isolate.
_DECLARED_TYPES: tuple[FieldType, ...] = (
    FieldType.STRING,
    FieldType.INTEGER,
    FieldType.FLOAT,
    FieldType.BOOLEAN,
    FieldType.LIST,
    FieldType.OBJECT,
)

# Scalars used to fill generated lists/dicts; their nested contents do not
# affect the top-level FieldType (lists -> LIST, dicts -> OBJECT).
_SCALARS = st.one_of(
    st.text(max_size=8),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)


def value_of_type(ftype: FieldType) -> st.SearchStrategy:
    """A strategy whose every value maps to ``ftype`` via ``value_field_type``."""
    if ftype is FieldType.STRING:
        return st.text(max_size=10)
    if ftype is FieldType.INTEGER:
        # st.integers never yields bool, which would map to BOOLEAN.
        return st.integers()
    if ftype is FieldType.FLOAT:
        return st.floats(allow_nan=False, allow_infinity=False)
    if ftype is FieldType.BOOLEAN:
        return st.booleans()
    if ftype is FieldType.LIST:
        return st.lists(_SCALARS, max_size=4)
    if ftype is FieldType.OBJECT:
        return st.dictionaries(field_names(), _SCALARS, max_size=4)
    raise ValueError(f"unsupported field type: {ftype!r}")


def _rec(fields: dict, index: int) -> Record:
    return Record(
        fields=fields,
        location=RecordLocation(source_file="data", array_index=index),
    )


def _dataset(records: list[Record]) -> Dataset:
    return Dataset(records=records, source_files=["data"])


# --------------------------------------------------------------------------- #
# Property 24: Missing-required-field detection
# --------------------------------------------------------------------------- #

@st.composite
def _schema_and_records_for_missing(draw):
    """A declared schema plus records whose only defects are missing fields.

    Present fields always carry a value of the declared type, so the only
    issues the validator can emit are ``MISSING_REQUIRED_FIELD`` for required
    fields that are absent or null. Returns the schema, the records, and the
    ground-truth set of ``(array_index, field_name)`` expected to be flagged.
    """
    names = draw(st.lists(field_names(), min_size=1, max_size=5, unique=True))
    specs = [
        FieldSpec(
            name=name,
            type=draw(st.sampled_from(_DECLARED_TYPES)),
            required=draw(st.booleans()),
        )
        for name in names
    ]
    schema = Schema(fields=specs)

    n = draw(st.integers(min_value=0, max_value=5))
    records: list[Record] = []
    expected: list[tuple[int, str]] = []
    for idx in range(n):
        fields: dict = {}
        for spec in specs:
            state = draw(st.sampled_from(("present", "absent", "null")))
            if state == "present":
                fields[spec.name] = draw(value_of_type(spec.type))
            elif state == "null":
                fields[spec.name] = None
                if spec.required:
                    expected.append((idx, spec.name))
            else:  # absent
                if spec.required:
                    expected.append((idx, spec.name))
        records.append(_rec(fields, idx))
    return schema, records, expected


# Feature: llm-training-data-quality-analyzer, Property 24: Missing-required-field detection
@_SETTINGS
@given(_schema_and_records_for_missing())
def test_property_24_missing_required_field_detection(case):
    """Every required field that is absent or null is flagged, and nothing else.

    **Validates: Requirements 9.1, 9.6**
    """
    schema, records, expected = case
    issues = FormatValidator().validate(_dataset(records), schema)

    # Only missing-required-field issues can arise (present values conform).
    assert all(
        i.category is IssueCategory.MISSING_REQUIRED_FIELD for i in issues
    )
    # Each issue identifies the affected record, the field, and the type.
    for issue in issues:
        assert issue.field_name is not None
        assert issue.location is not None
        assert issue.detail

    got = sorted(
        (i.location.array_index, i.field_name) for i in issues
    )
    assert got == sorted(expected)


# --------------------------------------------------------------------------- #
# Property 25: Type-mismatch detection
# --------------------------------------------------------------------------- #

@st.composite
def _schema_and_records_for_mismatch(draw):
    """A declared schema plus records whose only defects are type mismatches.

    Every field is present with a non-null value (so no missing-field issues
    arise); each value is either of the declared type (conformant) or of a
    deliberately different type. Returns the schema, the records, and the
    ground-truth set of ``(array_index, field_name)`` expected to be flagged.
    """
    names = draw(st.lists(field_names(), min_size=1, max_size=5, unique=True))
    specs = [
        FieldSpec(
            name=name,
            type=draw(st.sampled_from(_DECLARED_TYPES)),
            required=True,
        )
        for name in names
    ]
    schema = Schema(fields=specs)

    n = draw(st.integers(min_value=0, max_value=5))
    records: list[Record] = []
    expected: list[tuple[int, str]] = []
    for idx in range(n):
        fields: dict = {}
        for spec in specs:
            if draw(st.booleans()):  # inject a mismatch
                actual = draw(
                    st.sampled_from(
                        [t for t in _DECLARED_TYPES if t is not spec.type]
                    )
                )
                fields[spec.name] = draw(value_of_type(actual))
                expected.append((idx, spec.name))
            else:  # conformant value of the declared type
                fields[spec.name] = draw(value_of_type(spec.type))
        records.append(_rec(fields, idx))
    return schema, records, expected


# Feature: llm-training-data-quality-analyzer, Property 25: Type-mismatch detection
@_SETTINGS
@given(_schema_and_records_for_mismatch())
def test_property_25_type_mismatch_detection(case):
    """Every present field whose type differs from the declared type is flagged.

    **Validates: Requirements 9.2, 9.6**
    """
    schema, records, expected = case
    issues = FormatValidator().validate(_dataset(records), schema)

    # Only type-mismatch issues can arise (every field is present and non-null).
    assert all(
        i.category is IssueCategory.FIELD_TYPE_MISMATCH for i in issues
    )
    for issue in issues:
        assert issue.field_name is not None
        assert issue.location is not None
        assert issue.detail

    got = sorted(
        (i.location.array_index, i.field_name) for i in issues
    )
    assert got == sorted(expected)


# --------------------------------------------------------------------------- #
# Property 26: Schema inference from the first record
# --------------------------------------------------------------------------- #

@st.composite
def _first_record_and_rest(draw):
    """A non-fieldless first record plus arbitrary subsequent records.

    The first record's field names and value types define the inferred schema
    (every field required). Subsequent records draw arbitrary canonical values
    for a subset of those fields (and may omit some), exercising both the
    missing and mismatch rules against the inferred schema.
    """
    names = draw(st.lists(field_names(), min_size=1, max_size=5, unique=True))
    first_fields = {
        name: draw(value_of_type(draw(st.sampled_from(_DECLARED_TYPES))))
        for name in names
    }
    first = _rec(first_fields, 0)

    n = draw(st.integers(min_value=0, max_value=5))
    rest: list[Record] = []
    for offset in range(n):
        rec_fields: dict = {}
        for name in names:
            if draw(st.booleans()):  # present with an arbitrary canonical value
                rec_fields[name] = draw(field_values())
        rest.append(_rec(rec_fields, offset + 1))
    return first, first_fields, rest


# Feature: llm-training-data-quality-analyzer, Property 26: Schema inference from the first record
@_SETTINGS
@given(_first_record_and_rest())
def test_property_26_schema_inference_from_first_record(case):
    """Inferred schema mirrors the first record; subsequent records are validated.

    **Validates: Requirements 9.3, 9.4**
    """
    first, first_fields, rest = case

    # (9.3) The inferred schema's field names and types equal the first record's.
    inferred = FormatValidator._infer_schema(first)
    assert inferred.inferred is True
    assert [f.name for f in inferred.fields] == list(first_fields.keys())
    inferred_types = {f.name: f.type for f in inferred.fields}
    assert inferred_types == {
        name: value_field_type(value) for name, value in first_fields.items()
    }
    assert all(f.required for f in inferred.fields)

    # (9.4) Every subsequent record is validated against the inferred schema
    # using the same missing/mismatch rules. Reconstruct the ground truth.
    expected: list[tuple[int, str, IssueCategory]] = []
    for rec in rest:
        for name in first_fields:
            if name not in rec.fields or rec.fields[name] is None:
                expected.append(
                    (rec.location.array_index, name,
                     IssueCategory.MISSING_REQUIRED_FIELD)
                )
            elif value_field_type(rec.fields[name]) is not inferred_types[name]:
                expected.append(
                    (rec.location.array_index, name,
                     IssueCategory.FIELD_TYPE_MISMATCH)
                )

    issues = FormatValidator().validate(_dataset([first, *rest]))
    got = sorted(
        (i.location.array_index, i.field_name, i.category) for i in issues
    )
    assert got == sorted(expected)
