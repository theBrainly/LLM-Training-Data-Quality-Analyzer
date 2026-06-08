"""The Analysis Pipeline: run every detector over a Dataset and aggregate.

This module wires the independent quality detectors and the Metrics_Engine into
a single pass over an immutable :class:`~analyzer.models.Dataset` (the design's
"Analysis Pipeline" box). Following the design's Processing Flow:

* All detectors read the same immutable ``Dataset`` (or its individual
  ``Record`` objects) and append their findings to a single shared
  :class:`~analyzer.models.QualityIssue` collection. Detectors never mutate the
  input records (they each return copies or only read), so the ``Dataset`` that
  flows in is the same one that flows out.
* The collection is *seeded* with any pre-existing parse/ingestion issues
  (e.g. the issues produced by :mod:`analyzer.combine`) so that the final
  metrics and any downstream report see every issue discovered across the run.
* Once every detector has contributed, the ``Dataset`` and the accumulated
  issues are handed to the :class:`~analyzer.metrics.MetricsEngine` to compute
  the aggregate :class:`~analyzer.models.Metrics` (Requirement 4.3 consumes both
  the dataset and its issues).

Detector wiring:

* **Duplicate_Detector** runs once over the whole dataset (Requirement 5.1).
* **PII_Detector**, **Toxicity_Detector**, and **Quality_Detector** run once per
  record (Requirements 6.1, 7.1, 8.1).
* **Format_Validator** runs once over the whole dataset (Requirement 9.1).

Toxicity scoring is supplied via *dependency injection*: the pipeline takes a
:class:`~analyzer.detectors.toxicity.ToxicityModel` and defaults to the
deterministic :class:`~analyzer.detectors.toxicity.StubToxicityModel` so the
pipeline is runnable and testable without a real classifier. A production
classifier implementing the ``ToxicityModel`` protocol can be passed instead
without any other change to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analyzer.detectors.duplicate import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateDetector,
)
from analyzer.detectors.format_validator import FormatValidator
from analyzer.detectors.pii import PIIDetector
from analyzer.detectors.quality import QualityConfig, QualityDetector
from analyzer.detectors.toxicity import (
    DEFAULT_TOXICITY_THRESHOLD,
    StubToxicityModel,
    ToxicityDetector,
    ToxicityModel,
)
from analyzer.metrics import MetricsEngine
from analyzer.models import (
    Dataset,
    Metrics,
    QualityIssue,
    Schema,
)

__all__ = [
    "PipelineConfig",
    "AnalysisResult",
    "AnalysisPipeline",
    "analyze",
]


@dataclass
class PipelineConfig:
    """Configuration for a single analysis run.

    The threshold defaults match each detector's documented default
    (similarity ``0.9``, toxicity ``0.8``; the Quality_Detector defaults live in
    :class:`QualityConfig`). ``schema`` is the declared schema handed to the
    Format_Validator; when ``None`` the validator infers a schema from the first
    record (Requirement 9.3). Invalid threshold values are not rejected here -
    each detector validates its own threshold and falls back to its default,
    recording a ``CONFIG_ERROR`` issue (Requirements 5.5, 7.5, 8.5).
    """

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    toxicity_threshold: float = DEFAULT_TOXICITY_THRESHOLD
    quality_config: QualityConfig | None = None
    schema: Schema | None = None


@dataclass
class AnalysisResult:
    """The product of an analysis run.

    ``dataset`` is the immutable dataset that was analyzed (returned unchanged),
    ``issues`` is the single shared collection of every :class:`QualityIssue`
    discovered (seeded parse issues plus every detector's findings), and
    ``metrics`` is the :class:`Metrics` computed from the dataset and that issue
    collection.
    """

    dataset: Dataset
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: Metrics | None = None


class AnalysisPipeline:
    """Runs every detector over a Dataset and computes aggregate metrics.

    The toxicity model is injected (defaulting to
    :class:`StubToxicityModel`) so the pipeline can run without a real
    classifier; pass any :class:`ToxicityModel` implementation to plug in a
    production model.
    """

    def __init__(self, toxicity_model: ToxicityModel | None = None) -> None:
        self._duplicate_detector = DuplicateDetector()
        self._pii_detector = PIIDetector()
        self._toxicity_detector = ToxicityDetector(
            toxicity_model if toxicity_model is not None else StubToxicityModel()
        )
        self._quality_detector = QualityDetector()
        self._format_validator = FormatValidator()
        self._metrics_engine = MetricsEngine()

    def analyze(
        self,
        dataset: Dataset,
        config: PipelineConfig | None = None,
        parse_issues: list[QualityIssue] | None = None,
    ) -> AnalysisResult:
        """Analyze ``dataset`` and return the aggregated :class:`AnalysisResult`.

        All detectors run against the immutable ``dataset`` and append to a
        single shared issue collection, which is seeded with ``parse_issues``
        (pre-existing parse/ingestion findings) when supplied. The accumulated
        issues and the dataset are then fed to the Metrics_Engine to compute the
        aggregate metrics. The input records are never mutated; the same
        ``dataset`` is returned on the result.
        """
        cfg = config if config is not None else PipelineConfig()

        # Single shared collection, seeded with any pre-existing parse issues.
        issues: list[QualityIssue] = list(parse_issues) if parse_issues else []

        # Dataset-level: duplicate detection (Requirement 5.1).
        issues.extend(
            self._duplicate_detector.detect(dataset, cfg.similarity_threshold)
        )

        # Per-record: PII, toxicity, and quality detection
        # (Requirements 6.1, 7.1, 8.1).
        for record in dataset.records:
            issues.extend(self._pii_detector.detect(record))
            issues.extend(
                self._toxicity_detector.detect(record, cfg.toxicity_threshold)
            )
            issues.extend(
                self._quality_detector.detect(record, cfg.quality_config)
            )

        # Dataset-level: format/schema validation (Requirement 9.1).
        issues.extend(self._format_validator.validate(dataset, cfg.schema))

        # Feed the dataset + accumulated issues to the Metrics_Engine
        # (Requirement 4.3).
        metrics = self._metrics_engine.compute(dataset, issues)

        return AnalysisResult(dataset=dataset, issues=issues, metrics=metrics)


def analyze(
    dataset: Dataset,
    config: PipelineConfig | None = None,
    parse_issues: list[QualityIssue] | None = None,
    toxicity_model: ToxicityModel | None = None,
) -> AnalysisResult:
    """Convenience wrapper that builds an :class:`AnalysisPipeline` and runs it.

    ``toxicity_model`` is injected into the pipeline (defaulting to the
    deterministic stub); ``config`` and ``parse_issues`` are forwarded to
    :meth:`AnalysisPipeline.analyze`.
    """
    return AnalysisPipeline(toxicity_model=toxicity_model).analyze(
        dataset, config=config, parse_issues=parse_issues
    )
