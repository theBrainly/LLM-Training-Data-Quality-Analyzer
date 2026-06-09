"""Shared Hypothesis strategies and test stubs for the Analyzer test suite.

This module is *test support*, not a test module (it is not named ``test_*`` so
pytest does not collect it). It provides the generators and stubs that the
property-based tests across the suite draw from, per the design's Testing
Strategy / Generators section:

* :func:`field_values`        - a single canonical ``Value``, parameterizable by
  target :class:`~analyzer.models.SupportedFormat` and by whether the value must
  be *representable* in that format.
* :func:`records` / :func:`record_lists` - :class:`~analyzer.models.Record`
  objects and ordered lists of them, parameterizable by target format so they
  emit only representable values (for round-trip tests, Property 1/5) or
  deliberately *unrepresentable* values (for the negative serialization
  property, Property 2/6).
* :func:`format_sources`      - a :class:`~analyzer.models.SupportedFormat`.
* :func:`pii_text` / :func:`pii_free_text` - text with a known multiset of PII
  occurrences (with ground-truth spans) and PII-free filler, respectively
  (Properties 10/14/15).
* :func:`thresholds` / :func:`invalid_thresholds` - in-range and out-of-range
  threshold values (Properties 12/16/...).
* :class:`StubToxicityModel`  - a controllable deterministic toxicity model so
  threshold logic can be tested independently of any real classifier
  (Property 17/18 and Requirement 7.6 failure paths).

Choices made here are deliberately conservative so the values they emit
round-trip cleanly:

* JSON/JSONL/Parquet representable values are finite (no ``NaN``/``inf``) over
  the full canonical ``Value`` type, including nested lists/dicts.
* CSV representable values are text only, because CSV cells are textual and
  only strings reliably survive a parse -> print -> parse round-trip through a
  scalar-cell format; nested containers are the canonical *unrepresentable*
  case for CSV (per the design's Model Notes).
* Surrogate and control characters are excluded from generated text so that
  UTF-8 / CSV serialization does not spuriously fail.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

from hypothesis import strategies as st

from analyzer.models import (
    Record,
    RecordLocation,
    SupportedFormat,
    Value,
)

__all__ = [
    "ALL_FORMATS",
    "PII_CATEGORIES",
    "PIIOccurrence",
    "PIISample",
    "StubToxicityModel",
    "field_names",
    "field_values",
    "format_sources",
    "invalid_thresholds",
    "pii_free_text",
    "pii_instances",
    "pii_text",
    "record_lists",
    "record_locations",
    "records",
    "thresholds",
    "unrepresentable_values",
]


# --------------------------------------------------------------------------- #
# Format constants
# --------------------------------------------------------------------------- #

ALL_FORMATS: tuple[SupportedFormat, ...] = tuple(SupportedFormat)


def format_sources() -> st.SearchStrategy[SupportedFormat]:
    """A strategy over every :class:`SupportedFormat` (JSON/JSONL/CSV/Parquet)."""
    return st.sampled_from(ALL_FORMATS)


# --------------------------------------------------------------------------- #
# Text helpers (surrogate/control-char free)
# --------------------------------------------------------------------------- #

_FIELD_NAME_ALPHABET = string.ascii_letters + string.digits + "_"


def field_names(min_size: int = 1, max_size: int = 12) -> st.SearchStrategy[str]:
    """Non-empty field/key names safe for every format (incl. CSV headers)."""
    return st.text(alphabet=_FIELD_NAME_ALPHABET, min_size=min_size, max_size=max_size)


def _safe_text(max_size: int = 20) -> st.SearchStrategy[str]:
    """Text excluding surrogates so UTF-8 serialization never fails."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        max_size=max_size,
    )


def _csv_text(max_size: int = 20) -> st.SearchStrategy[str]:
    """Text safe for round-tripping through a CSV cell.

    Control characters (category ``Cc``: newlines, carriage returns, tabs, NUL)
    are excluded so that line-oriented CSV parsing does not split or alter a
    cell on the way back.
    """
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        max_size=max_size,
    )


# --------------------------------------------------------------------------- #
# Canonical value strategies
# --------------------------------------------------------------------------- #

def _json_scalars() -> st.SearchStrategy[Value]:
    return st.one_of(
        _safe_text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none(),
    )


def _json_values() -> st.SearchStrategy[Value]:
    """Finite canonical values representable in JSON/JSONL/Parquet.

    Recursively nests lists and dicts (string keys) over finite scalars; no
    ``NaN``/``inf`` so JSON serialization succeeds.
    """
    return st.recursive(
        _json_scalars(),
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(field_names(), children, max_size=4),
        ),
        max_leaves=8,
    )


def _nonfinite_floats() -> st.SearchStrategy[float]:
    return st.sampled_from([float("nan"), float("inf"), float("-inf")])


def unrepresentable_values(
    fmt: SupportedFormat | None = None,
) -> st.SearchStrategy[Value]:
    """A value that cannot be represented in ``fmt``.

    * For CSV the canonical unrepresentable values are nested containers
      (lists/dicts), since CSV cells are scalar text only.
    * For JSON/JSONL/Parquet (and the format-agnostic default) the
      unrepresentable values are the non-finite floats ``NaN``/``inf``/``-inf``,
      which have no JSON representation.
    """
    if fmt is SupportedFormat.CSV:
        scalar = st.one_of(
            _csv_text(),
            st.integers(),
            st.booleans(),
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False),
        )
        return st.one_of(
            st.lists(scalar, min_size=1, max_size=3),
            st.dictionaries(field_names(), scalar, min_size=1, max_size=3),
        )
    return _nonfinite_floats()


def field_values(
    fmt: SupportedFormat | None = None,
    representable: bool = True,
) -> st.SearchStrategy[Value]:
    """A single canonical :data:`~analyzer.models.Value`.

    When ``representable`` is True the value is guaranteed to round-trip through
    ``fmt`` (CSV -> text only; other formats -> finite nested canonical values).
    When ``representable`` is False the value is one that ``fmt`` cannot
    represent (see :func:`unrepresentable_values`).
    """
    if not representable:
        return unrepresentable_values(fmt)
    if fmt is SupportedFormat.CSV:
        return _csv_text()
    return _json_values()


# --------------------------------------------------------------------------- #
# Record strategies
# --------------------------------------------------------------------------- #

@st.composite
def record_locations(
    draw,
    fmt: SupportedFormat | None = None,
    source_file: str = "data",
) -> RecordLocation:
    """A :class:`RecordLocation` with format-appropriate coordinates."""
    if fmt is SupportedFormat.JSON:
        return RecordLocation(
            source_file=source_file, array_index=draw(st.integers(0, 100))
        )
    if fmt in (SupportedFormat.JSONL, SupportedFormat.CSV):
        return RecordLocation(
            source_file=source_file, line_number=draw(st.integers(1, 100))
        )
    if fmt is SupportedFormat.PARQUET:
        return RecordLocation(
            source_file=source_file,
            row_group=draw(st.integers(0, 5)),
            row_index=draw(st.integers(0, 100)),
        )
    return RecordLocation(source_file=source_file)


@st.composite
def records(
    draw,
    fmt: SupportedFormat | None = None,
    representable: bool = True,
    min_fields: int = 0,
    max_fields: int = 5,
    source_file: str = "data",
) -> Record:
    """A single :class:`Record` whose field values target ``fmt``.

    When ``representable`` is False, exactly one extra field is added carrying a
    value that ``fmt`` cannot represent, so the record alone is enough to
    trigger the located-serialization-error behaviour (Property 6).
    """
    names = draw(
        st.lists(
            field_names(), min_size=min_fields, max_size=max_fields, unique=True
        )
    )
    fields: dict[str, Value] = {
        name: draw(field_values(fmt=fmt, representable=True)) for name in names
    }
    if not representable:
        fields[draw(field_names())] = draw(unrepresentable_values(fmt))
    location = draw(record_locations(fmt=fmt, source_file=source_file))
    return Record(fields=fields, location=location)


@st.composite
def record_lists(
    draw,
    fmt: SupportedFormat | None = None,
    representable: bool = True,
    min_size: int = 0,
    max_size: int = 6,
    source_file: str = "data",
) -> list[Record]:
    """An ordered list of :class:`Record` objects targeting ``fmt``.

    When ``representable`` is True every value in every record is representable
    in ``fmt`` (suitable for round-trip tests). When ``representable`` is False
    the list contains at least one record, and exactly one of its records is
    augmented with an unrepresentable value (suitable for the negative
    serialization property).
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    recs: list[Record] = [
        draw(records(fmt=fmt, representable=True, source_file=source_file))
        for _ in range(n)
    ]
    if not representable:
        if not recs:
            recs.append(
                draw(records(fmt=fmt, representable=True, source_file=source_file))
            )
        idx = draw(st.integers(min_value=0, max_value=len(recs) - 1))
        target = recs[idx]
        new_fields = dict(target.fields)
        new_fields[draw(field_names())] = draw(unrepresentable_values(fmt))
        recs[idx] = Record(
            fields=new_fields,
            location=target.location,
            metadata=dict(target.metadata),
        )
    return recs


# --------------------------------------------------------------------------- #
# Threshold strategies
# --------------------------------------------------------------------------- #

def thresholds(
    min_value: float = 0.0, max_value: float = 1.0
) -> st.SearchStrategy[float]:
    """A valid threshold in the inclusive range ``[min_value, max_value]``.

    Defaults to ``[0.0, 1.0]``, the valid range shared by the duplicate
    similarity, toxicity, and gibberish thresholds.
    """
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def invalid_thresholds() -> st.SearchStrategy[float]:
    """A threshold value outside the valid ``[0.0, 1.0]`` range.

    Useful for the "invalid configuration is rejected and the default is
    retained" properties (Property 16 and friends).
    """
    return st.one_of(
        st.floats(
            min_value=-1.0e6,
            max_value=-1.0e-6,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.floats(
            min_value=1.0 + 1.0e-6,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


# --------------------------------------------------------------------------- #
# PII strategies
# --------------------------------------------------------------------------- #

# Category labels for the PII kinds the detector recognizes (Requirement 6.1).
PII_CATEGORIES: tuple[str, ...] = (
    "email",
    "phone",
    "physical_address",
    "government_id",
    "credit_card",
)

_ALPHA = "abcdefghijklmnopqrstuvwxyz"
_WORD = st.text(alphabet=_ALPHA, min_size=1, max_size=8)


@dataclass(frozen=True)
class PIIOccurrence:
    """Ground-truth record of a single PII instance embedded in text.

    ``text[start:end] == value`` always holds for the generated text, so tests
    can assert the detector reports the exact category and ``[start, end)`` span
    (Requirement 6.3).
    """

    category: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class PIISample:
    """A generated text plus the known multiset of PII occurrences within it."""

    text: str
    occurrences: tuple[PIIOccurrence, ...] = field(default_factory=tuple)


def _digits(n: int) -> st.SearchStrategy[str]:
    return st.text(alphabet="0123456789", min_size=n, max_size=n)


def _luhn_complete(partial: str) -> str:
    """Append a Luhn check digit so the full number validates (credit cards)."""
    total = 0
    for index, char in enumerate(reversed(partial)):
        digit = int(char)
        if index % 2 == 0:  # position that will be doubled once check digit added
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    check = (10 - (total % 10)) % 10
    return partial + str(check)


@st.composite
def _emails(draw) -> str:
    user = draw(st.text(alphabet=_ALPHA + "0123456789", min_size=1, max_size=10))
    domain = draw(st.text(alphabet=_ALPHA, min_size=1, max_size=10))
    tld = draw(st.sampled_from(["com", "org", "net", "io", "co"]))
    return f"{user}@{domain}.{tld}"


@st.composite
def _phones(draw) -> str:
    return f"{draw(_digits(3))}-{draw(_digits(3))}-{draw(_digits(4))}"


@st.composite
def _government_ids(draw) -> str:
    # SSN-style government identifier: ddd-dd-dddd.
    return f"{draw(_digits(3))}-{draw(_digits(2))}-{draw(_digits(4))}"


@st.composite
def _credit_cards(draw) -> str:
    partial = draw(_digits(15))
    return _luhn_complete(partial)


@st.composite
def _addresses(draw) -> str:
    number = draw(st.integers(min_value=1, max_value=9999))
    name = draw(st.text(alphabet=_ALPHA, min_size=3, max_size=10)).capitalize()
    suffix = draw(
        st.sampled_from(["Street", "Avenue", "Road", "Boulevard", "Lane", "Drive"])
    )
    return f"{number} {name} {suffix}"


@st.composite
def pii_instances(draw) -> tuple[str, str]:
    """A ``(category, value)`` pair for one concrete PII instance."""
    category = draw(st.sampled_from(PII_CATEGORIES))
    if category == "email":
        value = draw(_emails())
    elif category == "phone":
        value = draw(_phones())
    elif category == "government_id":
        value = draw(_government_ids())
    elif category == "credit_card":
        value = draw(_credit_cards())
    else:  # physical_address
        value = draw(_addresses())
    return category, value


@st.composite
def pii_free_text(draw, min_words: int = 1, max_words: int = 10) -> str:
    """Filler text guaranteed to contain no PII.

    Composed only of lowercase letter words separated by single spaces, so it
    contains no ``@``, no digit runs, and no street suffix patterns that the
    PII detectors look for (Requirement 6.2 / Property 14).
    """
    words = draw(st.lists(_WORD, min_size=min_words, max_size=max_words))
    return " ".join(words)


@st.composite
def _filler_segment(draw) -> str:
    """PII-free filler that begins and ends with a space.

    The surrounding spaces guarantee that embedded PII instances are delimited
    from filler and from one another, so their recorded spans stay exact.
    """
    words = draw(st.lists(_WORD, min_size=1, max_size=5))
    return " " + " ".join(words) + " "


@st.composite
def pii_text(draw, min_occurrences: int = 1, max_occurrences: int = 4) -> PIISample:
    """Text embedding a known multiset of PII instances into PII-free filler.

    Returns a :class:`PIISample` whose ``occurrences`` give the exact category
    and ``[start, end)`` span of every embedded instance, such that
    ``text[start:end] == value``. Repeated categories are allowed. This is the
    ground truth the PII detection properties (10/14/15) assert against.
    """
    count = draw(
        st.integers(min_value=min_occurrences, max_value=max_occurrences)
    )

    parts: list[str] = []
    occurrences: list[PIIOccurrence] = []
    cursor = 0

    def emit(segment: str) -> None:
        nonlocal cursor
        parts.append(segment)
        cursor += len(segment)

    emit(draw(_filler_segment()))  # leading filler
    for _ in range(count):
        category, value = draw(pii_instances())
        start = cursor
        emit(value)
        occurrences.append(
            PIIOccurrence(category=category, value=value, start=start, end=cursor)
        )
        emit(draw(_filler_segment()))  # separating / trailing filler

    return PIISample(text="".join(parts), occurrences=tuple(occurrences))


# --------------------------------------------------------------------------- #
# Toxicity stub
# --------------------------------------------------------------------------- #

class StubToxicityModel:
    """A deterministic, controllable stand-in for a real toxicity classifier.

    The stub decouples the Toxicity_Detector's threshold logic from any actual
    model (design's Testing Strategy). It can be configured to:

    * return a single fixed ``score`` for every input (the default), or
    * return per-input scores via a ``scores`` mapping keyed by the input, or
    * compute scores via a ``score_fn`` callable, or
    * simulate a scoring failure by setting ``fail=True`` (Requirement 7.6).

    Scores are clamped into ``[0.0, 1.0]`` by default so the stub honours the
    bound the detector relies on (Requirement 7.1); pass ``clamp=False`` to
    return raw values for tests that intentionally probe out-of-range handling.
    Every call's input is recorded in :attr:`calls` for inspection.
    """

    def __init__(
        self,
        *,
        score: float = 0.0,
        scores: dict | None = None,
        score_fn=None,
        fail: bool = False,
        clamp: bool = True,
    ) -> None:
        self._score = score
        self._scores = scores
        self._score_fn = score_fn
        self.fail = fail
        self._clamp = clamp
        self.calls: list = []

    def score(self, text):
        """Return a toxicity score for ``text`` (or raise when ``fail``)."""
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("toxicity scoring failed")

        if self._score_fn is not None:
            value = self._score_fn(text)
        elif self._scores is not None:
            value = self._scores[text]
        else:
            value = self._score

        if self._clamp and value is not None:
            value = max(0.0, min(1.0, float(value)))
        return value
