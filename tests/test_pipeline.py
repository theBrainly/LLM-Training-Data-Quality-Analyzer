"""Unit tests for the Analysis Pipeline (task 15.1).

These tests exercise :mod:`analyzer.pipeline`, which runs every detector over an
immutable :class:`Dataset`, accumulates findings into a single shared issue
collection, and feeds the dataset + issues to the Metrics_Engine. They verify
that:

* multiple detectors contribute issues to one shared collection,
* pre-existing parse issues are carried through into the result and metrics,
* metrics are computed from the dataset and accumulated issues, and
* the input records are not mutated by the run.
"""

from __future__ import annotations

import copy

from analyzer.detectors.quality import QualityConfig
from analyzer.detectors.toxicity import StubToxicityModel
from analyzer.models import (
    Dataset,
    IssueCategory,
    Metrics,
    QualityIssue,
    Record,
    RecordLocation,
)
from analyzer.pipeline import (
    AnalysisPipeline,
    AnalysisResult,
    PipelineConfig,
    analyze,
)


def _loc(index: int) -> RecordLocation:
    return RecordLocation(source_file="data.jsonl", line_number=index)


def _make_dataset() -> Dataset:
    """A small dataset deliberately exercising several detectors at once.

    - records 0 and 1 are byte-for-byte identical -> DUPLICATE
    - record 2 embeds an email -> PII
    - record 3 is a single token -> LOW_QUALITY_SHORT (default min tokens = 3)
    - record 4 is a clean, sufficiently long record
    """
    records = [
        Record(fields={"text": "the quick brown fox jumps"}, location=_loc(0)),
        Record(fields={"text": "the quick brown fox jumps"}, location=_loc(1)),
        Record(
            fields={"text": "please email me at jane.doe@example.com today"},
            location=_loc(2),
        ),
        Record(fields={"text": "hi"}, location=_loc(3)),
        Record(
            fields={"text": "a perfectly reasonable training example here"},
            location=_loc(4),
        ),
    ]
    return Dataset(records=records, source_files=["data.jsonl"])


def test_pipeline_accumulates_issues_from_multiple_detectors():
    dataset = _make_dataset()

    result = AnalysisPipeline().analyze(dataset)

    assert isinstance(result, AnalysisResult)
    categories = {issue.category for issue in result.issues}

    # Detectors of different concerns all contributed to one collection.
    assert IssueCategory.DUPLICATE in categories
    assert IssueCategory.PII in categories
    assert IssueCategory.LOW_QUALITY_SHORT in categories

    # The duplicate issue references both the duplicate and its original.
    dup = next(i for i in result.issues if i.category is IssueCategory.DUPLICATE)
    assert dup.location == _loc(1)
    assert dup.related_location == _loc(0)

    # The PII issue points at the record carrying the email.
    pii = next(i for i in result.issues if i.category is IssueCategory.PII)
    assert pii.location == _loc(2)
    assert pii.pii_category == "email"


def test_pipeline_computes_metrics_over_dataset_and_issues():
    dataset = _make_dataset()

    result = AnalysisPipeline().analyze(dataset)

    assert isinstance(result.metrics, Metrics)
    assert result.metrics.record_count == 5
    # At least the duplicate, PII, and short records carry issues.
    assert 0.0 < result.metrics.issue_record_proportion <= 1.0
    # quality_score == 1 - issue_record_proportion, bounded in [0, 1].
    assert result.metrics.quality_score == 1.0 - result.metrics.issue_record_proportion
    assert 0.0 <= result.metrics.quality_score <= 1.0


def test_pipeline_seeds_pre_existing_parse_issues():
    dataset = _make_dataset()
    parse_issue = QualityIssue(
        category=IssueCategory.PARSE_ERROR,
        location=None,
        detail="unparseable line 99",
    )

    result = AnalysisPipeline().analyze(dataset, parse_issues=[parse_issue])

    # The seeded parse issue survives into the combined collection.
    assert parse_issue in result.issues
    assert any(i.category is IssueCategory.PARSE_ERROR for i in result.issues)


def test_pipeline_does_not_mutate_input_records():
    dataset = _make_dataset()
    before = copy.deepcopy(dataset.records)

    AnalysisPipeline().analyze(dataset)

    # Same record objects, same field contents - nothing was modified.
    assert dataset.records is dataset.records
    for original, current in zip(before, dataset.records):
        assert current.fields == original.fields
        assert current.location == original.location
        assert current.metadata == original.metadata


def test_pipeline_returns_same_dataset_instance():
    dataset = _make_dataset()
    result = AnalysisPipeline().analyze(dataset)
    assert result.dataset is dataset


def test_pipeline_flags_toxicity_with_injected_model():
    # Inject a model that scores everything maximally toxic; with the default
    # threshold (0.8) every record should be flagged.
    toxic_model = StubToxicityModel(score=1.0)
    dataset = _make_dataset()

    result = analyze(dataset, toxicity_model=toxic_model)

    toxicity_issues = [
        i for i in result.issues if i.category is IssueCategory.TOXICITY
    ]
    assert len(toxicity_issues) == len(dataset.records)
    assert all(i.score == 1.0 for i in toxicity_issues)


def test_pipeline_respects_custom_quality_config():
    # With a high minimum-token threshold every short record is flagged.
    dataset = _make_dataset()
    cfg = PipelineConfig(quality_config=QualityConfig(min_token_threshold=100))

    result = AnalysisPipeline().analyze(dataset, config=cfg)

    short_issues = [
        i for i in result.issues if i.category is IssueCategory.LOW_QUALITY_SHORT
    ]
    assert len(short_issues) == len(dataset.records)


def test_pipeline_handles_empty_dataset():
    dataset = Dataset(records=[], source_files=[])
    result = AnalysisPipeline().analyze(dataset)

    # Empty dataset: detectors over records contribute nothing; the only issue
    # is the dataset-level schema-inference failure (no schema, no records).
    assert all(i.location is None for i in result.issues)
    assert any(
        i.category is IssueCategory.SCHEMA_INFERENCE_FAILED for i in result.issues
    )
    assert result.metrics is not None
    assert result.metrics.record_count == 0
    # Dataset-level issues reference no record, so the per-record proportion
    # (and thus the quality score) stays at its documented empty-dataset value.
    assert result.metrics.issue_record_proportion == 0.0
    assert result.metrics.quality_score == 0.0
