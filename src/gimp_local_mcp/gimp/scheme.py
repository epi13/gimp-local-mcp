"""A bounded reader for the simple Scheme values returned by Script-Fu."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..errors import SchemeParseError


@dataclass(frozen=True, slots=True)
class SchemeSymbolValue:
    name: str


@dataclass(frozen=True, slots=True)
class SchemeVector:
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SchemeOpaque:
    representation: str


_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$")


class SchemeReader:
    def __init__(self, text: str, *, max_depth: int = 100, max_items: int = 10_000) -> None:
        self.text = text.strip()
        self.index = 0
        self.max_depth = max_depth
        self.items = 0
        self.max_items = max_items

    def parse(self) -> Any:
        if not self.text:
            return None
        value = self._value(0)
        self._skip_space()
        if self.index != len(self.text):
            raise SchemeParseError(f"Trailing Scheme data at offset {self.index}")
        return value

    def _skip_space(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _value(self, depth: int) -> Any:
        if depth > self.max_depth:
            raise SchemeParseError("Scheme response nesting exceeds the safety limit")
        self._skip_space()
        if self.index >= len(self.text):
            raise SchemeParseError("Unexpected end of Scheme response")
        char = self.text[self.index]
        if char == "(":
            return self._list(depth + 1)
        if char == "#" and self.text[self.index : self.index + 2] == "#(":
            self.index += 2
            return SchemeVector(tuple(self._sequence(")", depth + 1)))
        if char == '"':
            return self._string()
        if char == "'":
            self.index += 1
            return [SchemeSymbolValue("quote"), self._value(depth + 1)]
        token = self._token()
        if token == "#t":
            return True
        if token == "#f":
            return False
        if token.startswith("#<") and token.endswith(">"):
            return SchemeOpaque(token)
        if token == "()":
            return []
        if _NUMBER_RE.fullmatch(token):
            try:
                return float(token) if any(c in token for c in ".eE") else int(token)
            except ValueError as exc:
                raise SchemeParseError(f"Invalid numeric token {token!r}") from exc
        return SchemeSymbolValue(token)

    def _list(self, depth: int) -> list[Any]:
        self.index += 1
        return self._sequence(")", depth)

    def _sequence(self, closing: str, depth: int) -> list[Any]:
        values: list[Any] = []
        while True:
            self._skip_space()
            if self.index >= len(self.text):
                raise SchemeParseError("Unclosed Scheme list or vector")
            if self.text[self.index] == closing:
                self.index += 1
                return values
            self.items += 1
            if self.items > self.max_items:
                raise SchemeParseError("Scheme response item count exceeds the safety limit")
            values.append(self._value(depth))

    def _token(self) -> str:
        start = self.index
        while (
            self.index < len(self.text)
            and not self.text[self.index].isspace()
            and self.text[self.index] not in "()'"
        ):
            self.index += 1
        if start == self.index:
            raise SchemeParseError(f"Unexpected character {self.text[self.index]!r}")
        return self.text[start : self.index]

    def _string(self) -> str:
        self.index += 1
        output: list[str] = []
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if char == '"':
                return "".join(output)
            if char != "\\":
                output.append(char)
                continue
            if self.index >= len(self.text):
                raise SchemeParseError("Unclosed escape in Scheme string")
            escaped = self.text[self.index]
            self.index += 1
            output.append(
                {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(escaped, escaped)
            )
        raise SchemeParseError("Unclosed Scheme string")


def parse_scheme(text: str) -> Any:
    return SchemeReader(text).parse()


def unwrap(value: Any) -> Any:
    """Convert parser wrapper values to JSON-compatible values where possible."""

    if isinstance(value, SchemeVector):
        return [unwrap(item) for item in value.values]
    if isinstance(value, SchemeSymbolValue):
        return value.name
    if isinstance(value, SchemeOpaque):
        return value.representation
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    return value
