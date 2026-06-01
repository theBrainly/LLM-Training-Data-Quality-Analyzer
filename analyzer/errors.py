"""Typed errors for the LLM Training Data Quality Analyzer.

These errors model the *fail-fast* and *fail-safe* conditions described in the
design's Error Handling section. They are defined as exception subclasses so
they can either be raised internally or carried inside ``Result``-style return
values, while always exposing structured fields (path, format, parameter, etc.)
rather than relying on free-form message parsing.

Fail-fast conditions (an operation stops and returns/raises a typed error with
no partial output):

* :class:`PathNotFoundError`        - submitted path does not exist (Req 1.3)
* :class:`FileSizeError`            - file exceeds the configured limit (Req 1.4)
* :class:`NoSupportedFilesError`    - directory has no supported files (Req 1.6)
* :class:`UnrepresentableValueError`- value not representable on serialize (Req 3.3)
* :class:`SerializationError`       - report serialization failed (Req 10.7)
* :class:`UnsupportedFormatError`   - unsupported output format requested (Req 10.8)

Fail-safe configuration:

* :class:`ConfigError`              - invalid threshold rejected; default kept
                                      (Req 5.5, 7.5, 8.5)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AnalyzerError",
    "PathNotFoundError",
    "NoSupportedFilesError",
    "FileSizeError",
    "UnrepresentableValueError",
    "SerializationError",
    "UnsupportedFormatError",
    "ConfigError",
]


class AnalyzerError(Exception):
    """Base class for all typed errors raised by the Analyzer."""


class PathNotFoundError(AnalyzerError):
    """A submitted file or directory path does not exist (Requirement 1.3)."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Path does not exist: {path!r}")


class NoSupportedFilesError(AnalyzerError):
    """A submitted directory contains no Supported_Format files (Requirement 1.6)."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        super().__init__(
            f"No supported files found in directory: {directory!r}"
        )


class FileSizeError(AnalyzerError):
    """A submitted file exceeds the configured maximum size (Requirement 1.4)."""

    def __init__(self, path: str, limit_bytes: int, actual_bytes: int | None = None) -> None:
        self.path = path
        self.limit_bytes = limit_bytes
        self.actual_bytes = actual_bytes
        detail = (
            f"File {path!r} exceeds the configured size limit of "
            f"{limit_bytes} bytes"
        )
        if actual_bytes is not None:
            detail += f" (file is {actual_bytes} bytes)"
        super().__init__(detail)


class UnrepresentableValueError(AnalyzerError):
    """A field value cannot be represented in the target format (Requirement 3.3).

    Identifies the position of the offending record in the input list and the
    name of the field that could not be represented.
    """

    def __init__(self, record_index: int, field_name: str, fmt: str) -> None:
        self.record_index = record_index
        self.field_name = field_name
        self.fmt = fmt
        super().__init__(
            f"Value of field {field_name!r} in record at index {record_index} "
            f"cannot be represented in format {fmt!r}"
        )


class SerializationError(AnalyzerError):
    """Serialization of a report into the requested format failed (Requirement 10.7).

    No partial report is produced; the error identifies the failed output format.
    """

    def __init__(self, fmt: str, detail: str = "") -> None:
        self.fmt = fmt
        self.detail = detail
        message = f"Failed to serialize report to format {fmt!r}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class UnsupportedFormatError(AnalyzerError):
    """A requested output format is not JSON or Markdown (Requirement 10.8)."""

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        super().__init__(f"Unsupported output format: {fmt!r}")


class ConfigError(AnalyzerError):
    """An invalid threshold configuration was rejected (Requirements 5.5, 7.5, 8.5).

    The invalid value is recorded along with the documented default that is
    retained in its place, so callers can surface an error indication while
    continuing with the default.
    """

    def __init__(self, parameter: str, invalid_value: Any, retained_default: Any) -> None:
        self.parameter = parameter
        self.invalid_value = invalid_value
        self.retained_default = retained_default
        super().__init__(
            f"Invalid configuration for {parameter!r}: {invalid_value!r}; "
            f"retaining default {retained_default!r}"
        )
