"""Quality detectors for the LLM Training Data Quality Analyzer.

Each detector lives in its own module and reads the immutable ``Dataset`` (or a
single ``Record``) and emits :class:`~analyzer.models.QualityIssue` objects
without mutating its input. The package re-exports each detector's public API
so callers can import from ``analyzer.detectors`` directly.
"""

from analyzer.detectors.duplicate import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateDetector,
)
from analyzer.detectors.pii import (
    PII_CATEGORIES,
    PIIDetector,
    PIIMatch,
    find_pii_in_text,
    luhn_valid,
)

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DuplicateDetector",
    "PII_CATEGORIES",
    "PIIDetector",
    "PIIMatch",
    "find_pii_in_text",
    "luhn_valid",
]
