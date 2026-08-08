"""Safe construction of Script-Fu expressions.

This module deliberately supports values, calls, and keyword arguments, but not
arbitrary source fragments. Raw Scheme is an implementation detail of the bridge,
never an MCP input type.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ProcedureNameError

_PROCEDURE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_!$%&*/:<=>?@^_|~+.-]*$")


@dataclass(frozen=True, slots=True)
class SchemeSymbol:
    """A validated Script-Fu symbol, usually a GIMP enum value."""

    name: str

    def __post_init__(self) -> None:
        if not _SYMBOL_RE.fullmatch(self.name):
            raise ValueError(f"Invalid Scheme symbol: {self.name!r}")


@dataclass(frozen=True, slots=True)
class SchemeNull:
    """The Script-Fu representation of a nullable GIMP object (ID -1)."""


def validate_procedure_name(name: str) -> str:
    if not isinstance(name, str) or not _PROCEDURE_RE.fullmatch(name):
        raise ProcedureNameError(
            "Procedure names must contain only ASCII letters, digits, '_' or '-'"
        )
    return name


def scheme_string(value: str) -> str:
    """Return a correctly escaped Scheme string literal."""

    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        + '"'
    )


def scheme_path(path: Path) -> str:
    return scheme_string(str(path))


def scheme_value(value: Any) -> str:
    """Serialize JSON-like values without allowing source-code injection."""

    if isinstance(value, SchemeSymbol):
        return value.name
    if isinstance(value, SchemeNull):
        return "-1"
    if value is None:
        return "#f"
    if value is True:
        return "#t"
    if value is False:
        return "#f"
    if isinstance(value, str):
        return scheme_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Scheme numbers must be finite")
        return format(value, ".17g")
    if isinstance(value, (list, tuple)):
        return "(" + " ".join(scheme_value(item) for item in value) + ")"
    if isinstance(value, Mapping):
        symbol = value.get("scheme_symbol")
        if isinstance(symbol, str) and len(value) == 1:
            return scheme_value(SchemeSymbol(symbol))
    raise TypeError(f"Unsupported structured Scheme value: {type(value).__name__}")


def scheme_call(
    procedure: str,
    args: list[Any] | tuple[Any, ...] = (),
    *,
    keywords: Mapping[str, Any] | None = None,
) -> str:
    """Build a positional, named, or mixed positional/named PDB invocation."""

    name = validate_procedure_name(procedure)
    parts = [scheme_value(value) for value in args]
    if keywords:
        for key, value in keywords.items():
            if not _SYMBOL_RE.fullmatch(key):
                raise ValueError(f"Invalid PDB argument name: {key!r}")
            parts.extend([f"#:{key}", scheme_value(value)])
    return "(" + " ".join([name, *parts]) + ")"


def with_v3(expression: str) -> str:
    """Evaluate an expression using GIMP 3's value dialect."""

    return f"(begin (script-fu-use-v3) {expression})"
