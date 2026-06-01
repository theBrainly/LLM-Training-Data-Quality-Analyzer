"""LLM Training Data Quality Analyzer.

A tool that ingests training datasets, parses them into a uniform in-memory
representation, runs a suite of quality detectors over the records, and emits
a structured report.
"""

from analyzer.errors import (
    AnalyzerError,
    ConfigError,
    FileSizeError,
    NoSupportedFilesError,
    PathNotFoundError,
    SerializationError,
    UnrepresentableValueError,
    UnsupportedFormatError,
)
from analyzer.ingestion import (
    DEFAULT_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    IngestionConfig,
    IngestionEngine,
    IngestionResult,
    clamp_file_size,
    detect_format,
)
from analyzer.models import (
    Dataset,
    FieldSpec,
    FieldType,
    IssueCategory,
    Metrics,
    QualityIssue,
    Record,
    RecordLocation,
    Report,
    Schema,
    Span,
    SupportedFormat,
    Value,
    fields_equivalent,
    records_equivalent,
    values_equal,
)
from analyzer.pretty_printer import PrettyPrinter, PrintResult
from analyzer.detectors import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateDetector,
)

__version__ = "0.1.0"

__all__ = [
    # Core data models
    "Dataset",
    "FieldSpec",
    "FieldType",
    "IssueCategory",
    "Metrics",
    "QualityIssue",
    "Record",
    "RecordLocation",
    "Report",
    "Schema",
    "Span",
    "SupportedFormat",
    "Value",
    "fields_equivalent",
    "records_equivalent",
    "values_equal",
    # Pretty printer
    "PrettyPrinter",
    "PrintResult",
    # Detectors
    "DuplicateDetector",
    "DEFAULT_SIMILARITY_THRESHOLD",
    # Typed errors
    "AnalyzerError",
    "ConfigError",
    "FileSizeError",
    "NoSupportedFilesError",
    "PathNotFoundError",
    "SerializationError",
    "UnrepresentableValueError",
    "UnsupportedFormatError",
    # Ingestion configuration and format detection
    "IngestionConfig",
    "IngestionEngine",
    "IngestionResult",
    "clamp_file_size",
    "detect_format",
    "MIN_FILE_SIZE_BYTES",
    "MAX_FILE_SIZE_BYTES",
    "DEFAULT_FILE_SIZE_BYTES",
]
