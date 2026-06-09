"""Unit test for Report_Generator serialization failure (Requirement 10.7).

When serialization of a supported format fails, the Report_Generator must
produce no partial output and return an error naming the format. A non-finite
metric value (``NaN``) is used to force a JSON serialization failure under
``allow_nan=False``.

Imports come directly from ``analyzer.report`` to avoid coupling to the
package ``__init__`` exports.
"""

from analyzer.errors import SerializationError
from analyzer.models import (
    Dataset,
    Metrics,
    Record,
    RecordLocation,
)
from analyzer.report import OutputFormat, ReportGenerator


def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data", array_index=index)


def _dataset(n: int) -> Dataset:
    return Dataset(
        records=[Record(fields={"text": f"r{i}"}, location=_loc(i)) for i in range(n)],
        source_files=["data"],
    )


def _metrics(record_count: int) -> Metrics:
    return Metrics(
        record_count=record_count,
        mean_tokens=2,
        min_tokens=1,
        max_tokens=3,
        issue_record_proportion=0.0,
        quality_score=1.0,
    )


def test_serialization_failure_produces_no_partial_report_and_names_format():
    # A non-finite metric value has no JSON representation, so serialization
    # fails under allow_nan=False.
    report = ReportGenerator().build(_dataset(2), _metrics(2), [])
    report.metrics.quality_score = float("nan")

    result = ReportGenerator().serialize(report, OutputFormat.JSON)

    # No partial output is produced ...
    assert result.text is None
    # ... and the error names the failed output format (Requirement 10.7).
    assert isinstance(result.error, SerializationError)
    assert result.error.fmt == OutputFormat.JSON.value == "json"
