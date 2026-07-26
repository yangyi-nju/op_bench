from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


BOUNDARY_SUBCLASSES = ("B1", "B2", "B3", "B4", "B5")
PRECISION_SUBCLASSES = ("P1", "P2", "P3", "P4", "P5")
FAILURE_CONTRACTS = (
    "wrong-result",
    "exception",
    "crash-oob",
    "silent-acceptance",
)


@dataclass(frozen=True)
class BoundaryKeywordPack:
    pack_id: str
    subclass: str
    positive_phrases: tuple[str, ...]
    path_hints: tuple[str, ...]
    exclusion_phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_id = f"boundary-{self.subclass.lower()}-v1"
        if self.pack_id != expected_id:
            raise ValueError(f"pack_id must be {expected_id!r}")
        if self.subclass not in BOUNDARY_SUBCLASSES:
            raise ValueError(f"unsupported Boundary subclass {self.subclass!r}")
        for name, values in (
            ("positive_phrases", self.positive_phrases),
            ("path_hints", self.path_hints),
            ("exclusion_phrases", self.exclusion_phrases),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate phrases")
            if any(not phrase or phrase != phrase.lower() for phrase in values):
                raise ValueError(f"{name} must contain normalized lowercase phrases")


_COMMON_EXCLUSIONS = (
    "benchmark-only",
    "cleanup-only",
    "documentation-only",
    "performance benchmark",
    "performance-only",
    "pure refactor",
)


BOUNDARY_KEYWORD_PACKS = (
    BoundaryKeywordPack(
        pack_id="boundary-b1-v1",
        subclass="B1",
        positive_phrases=(
            "empty reduction",
            "empty tensor",
            "numel == 0",
            "zero batch",
            "zero size",
            "zero-size",
        ),
        path_hints=("reduce", "reduction", "tensor"),
        exclusion_phrases=(*_COMMON_EXCLUSIONS, "empty list"),
    ),
    BoundaryKeywordPack(
        pack_id="boundary-b2-v1",
        subclass="B2",
        positive_phrases=(
            "0-d",
            "degenerate shape",
            "rank 0",
            "scalar",
            "size one",
            "zero dimensional",
        ),
        path_hints=("broadcast", "shape", "squeeze"),
        exclusion_phrases=_COMMON_EXCLUSIONS,
    ),
    BoundaryKeywordPack(
        pack_id="boundary-b3-v1",
        subclass="B3",
        positive_phrases=(
            "index overflow",
            "int32",
            "large tensor",
            "numel overflow",
            "overflow",
            "size overflow",
            "stride overflow",
        ),
        path_hints=("index", "shape", "stride"),
        exclusion_phrases=_COMMON_EXCLUSIONS,
    ),
    BoundaryKeywordPack(
        pack_id="boundary-b4-v1",
        subclass="B4",
        positive_phrases=(
            "axis bounds",
            "groups",
            "invalid dim",
            "k == 0",
            "out of range",
            "parameter endpoint",
            "validation",
        ),
        path_hints=("padding", "parameter", "validation"),
        exclusion_phrases=_COMMON_EXCLUSIONS,
    ),
    BoundaryKeywordPack(
        pack_id="boundary-b5-v1",
        subclass="B5",
        positive_phrases=(
            "block limit",
            "cuda illegal memory",
            "grid limit",
            "large index",
            "launch bounds",
            "launch failure",
            "tail block",
        ),
        path_hints=("cuda", "kernel", "launch"),
        exclusion_phrases=_COMMON_EXCLUSIONS,
    ),
)

_PACKS_BY_ID = {pack.pack_id: pack for pack in BOUNDARY_KEYWORD_PACKS}


def keyword_pack(pack_id: str) -> BoundaryKeywordPack:
    try:
        return _PACKS_BY_ID[pack_id]
    except KeyError as exc:
        raise KeyError(f"unknown Boundary keyword pack {pack_id!r}") from exc


def match_keyword_packs(text: str, paths: Sequence[str]) -> tuple[str, ...]:
    """Return literal Boundary keyword matches in frozen registry order."""

    normalized_text = text.casefold()
    normalized_paths = " ".join(str(path).casefold() for path in paths)
    haystack = f"{normalized_text}\n{normalized_paths}"
    matches: list[str] = []
    for pack in BOUNDARY_KEYWORD_PACKS:
        if any(phrase in haystack for phrase in pack.exclusion_phrases):
            continue
        if any(phrase in haystack for phrase in pack.positive_phrases):
            matches.append(pack.pack_id)
    return tuple(matches)


def validate_problem_taxonomy(operator: Mapping[str, object]) -> tuple[str, ...]:
    """Validate optional v0.7 taxonomy fields without migrating old tasks."""

    fields = ("problem_dimension", "problem_subclass", "failure_contract")
    present = tuple(field in operator for field in fields)
    if not any(present):
        return ()
    dimension_present, subclass_present, failure_present = present
    if (
        dimension_present != subclass_present
        or (failure_present and not dimension_present)
    ):
        return ("operator taxonomy fields must be provided together",)

    errors: list[str] = []
    dimension = operator["problem_dimension"]
    subclass = operator["problem_subclass"]

    if dimension not in ("boundary", "precision"):
        errors.append(
            "operator.problem_dimension: expected 'boundary' or 'precision'"
        )
    elif dimension == "boundary" and subclass not in BOUNDARY_SUBCLASSES:
        errors.append("operator.problem_subclass: boundary requires B1..B5")
    elif dimension == "precision" and subclass not in PRECISION_SUBCLASSES:
        errors.append("operator.problem_subclass: precision requires P1..P5")

    failure_contract = operator.get("failure_contract")
    if failure_present and failure_contract not in FAILURE_CONTRACTS:
        errors.append(
            f"operator.failure_contract: unsupported value {failure_contract!r}"
        )
    return tuple(errors)
