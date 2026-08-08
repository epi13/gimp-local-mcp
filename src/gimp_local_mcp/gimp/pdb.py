"""Runtime PDB catalog, metadata adapters, and structured invoker.

Script-Fu exposes procedure existence, counts, and documentation reliably.  It
does not expose the full ``GimpProcedure``/``GParamSpec`` object model that
libgimp bindings can inspect.  The adapter boundary below keeps that limitation
explicit while allowing a future trusted adapter to provide richer metadata
without changing invocation or MCP-facing models.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..errors import ProcedureNameError
from .scheme import parse_scheme, unwrap
from .serializer import scheme_call, validate_procedure_name, with_v3
from .transport import ScriptFuClient

MetadataStatus = Literal["available", "partial", "unavailable", "malformed"]
ParameterDirection = Literal["argument", "auxiliary", "return"]

_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_MAX_METADATA_PARAMETERS = 256
_MAX_METADATA_TEXT = 2048
_MAX_METADATA_VALUES = 256
_MAX_METADATA_DEPTH = 8


@dataclass(frozen=True, slots=True)
class PdbParameter:
    """A bounded, immutable representation of reported PDB parameter data."""

    name: str | None
    position: int
    direction: ParameterDirection
    parameter_type: str | None = None
    default: Any = None
    default_available: bool = False
    choices: tuple[Any, ...] = ()
    nullable: bool | None = None
    required: bool | None = None
    description: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "direction": self.direction,
            "type": self.parameter_type,
            "default": self.default,
            "default_available": self.default_available,
            "choices": list(self.choices),
            "nullable": self.nullable,
            "required": self.required,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class PdbIntrospectionResult:
    """The result of one bounded argument/return metadata probe."""

    status: MetadataStatus
    source: str
    arguments: tuple[PdbParameter, ...] = ()
    returns: tuple[PdbParameter, ...] = ()
    notes: tuple[str, ...] = ()
    argument_names_complete: bool = False
    requiredness_complete: bool = False

    @classmethod
    def unavailable(cls, source: str, note: str) -> PdbIntrospectionResult:
        return cls(status="unavailable", source=source, notes=(note,))

    @classmethod
    def malformed(cls, source: str, note: str) -> PdbIntrospectionResult:
        return cls(status="malformed", source=source, notes=(note,))

    @property
    def argument_metadata_available(self) -> bool:
        """True only when the adapter reported complete useful metadata."""

        return self.status == "available"

    @property
    def named_argument_names_available(self) -> bool:
        """Whether every reported argument has a trustworthy Scheme keyword name."""

        return (
            self.argument_names_complete
            and self.status in {"available", "partial"}
            and all(parameter.name is not None for parameter in self.arguments)
        )

    @property
    def required_arguments_available(self) -> bool:
        return (
            self.requiredness_complete
            and self.named_argument_names_available
            and all(parameter.required is not None for parameter in self.arguments)
        )


class PdbIntrospectionAdapter(Protocol):
    """Source of optional, structured PDB argument metadata."""

    def inspect(
        self,
        procedure_name: str,
        *,
        argument_count: int | None,
        return_value_count: int | None,
    ) -> PdbIntrospectionResult: ...


class ScriptFuPdbIntrospection:
    """Conservative default adapter for the Script-Fu TCP bridge.

    The standard Script-Fu bridge has no stable operation for returning the
    ``GParamSpec`` objects behind a ``GimpProcedure``.  This adapter therefore
    reports the limitation instead of probing undocumented Scheme functions or
    guessing signatures.  A future bridge-specific source can be injected via
    ``StructuredPdbIntrospection``.
    """

    source = "Script-Fu TCP"

    def inspect(
        self,
        procedure_name: str,
        *,
        argument_count: int | None,
        return_value_count: int | None,
    ) -> PdbIntrospectionResult:
        del procedure_name, argument_count, return_value_count
        return PdbIntrospectionResult.unavailable(
            self.source,
            "Script-Fu reports procedure counts and documentation, but not stable "
            "GimpProcedure/GParamSpec argument metadata through this bridge",
        )


class StructuredPdbIntrospection:
    """Adapt a trusted structured metadata source without accepting Scheme text."""

    def __init__(self, provider: Callable[[str], Any], *, source: str) -> None:
        self.provider = provider
        self.source = source

    def inspect(
        self,
        procedure_name: str,
        *,
        argument_count: int | None,
        return_value_count: int | None,
    ) -> PdbIntrospectionResult:
        del argument_count, return_value_count
        try:
            payload = self.provider(procedure_name)
        except Exception as exc:  # adapter failures must not erase count/doc metadata
            return PdbIntrospectionResult.unavailable(
                self.source, f"metadata source unavailable: {type(exc).__name__}"
            )
        return parse_pdb_metadata(payload, source=self.source)


def _metadata_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata field {field!r} must be a non-empty string")
    if len(value) > _MAX_METADATA_TEXT:
        raise ValueError(f"metadata field {field!r} exceeds the safety limit")
    return value


def _metadata_value(value: Any, *, depth: int = 0) -> Any:
    """Validate values that may be returned as defaults or enum choices."""

    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("metadata value nesting exceeds the safety limit")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > _MAX_METADATA_TEXT:
            raise ValueError("metadata string exceeds the safety limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_METADATA_VALUES:
            raise ValueError("metadata sequence exceeds the safety limit")
        return [_metadata_value(item, depth=depth + 1) for item in value]
    raise ValueError(f"unsupported metadata value: {type(value).__name__}")


def _parse_parameter(
    value: Any,
    *,
    direction: ParameterDirection,
    position: int,
    complete: bool,
) -> PdbParameter:
    if not isinstance(value, Mapping):
        raise ValueError("each metadata parameter must be an object")

    name_value = value.get("name")
    name: str | None
    if name_value is None:
        name = None
    elif isinstance(name_value, str) and _PARAMETER_NAME_RE.fullmatch(name_value):
        name = name_value
    else:
        raise ValueError("metadata parameter names must be valid Scheme identifiers")

    reported_position = value.get("position", position)
    if not isinstance(reported_position, int) or isinstance(reported_position, bool):
        raise ValueError("metadata parameter position must be a non-negative integer")
    if reported_position < 0:
        raise ValueError("metadata parameter position must be non-negative")

    parameter_type = value.get("type")
    if parameter_type is not None:
        parameter_type = _metadata_text(parameter_type, "type")
    if complete and (name is None or parameter_type is None):
        raise ValueError("complete metadata requires parameter names and types")

    raw_direction = value.get("direction", direction)
    if raw_direction != direction:
        raise ValueError(f"metadata parameter direction must be {direction!r}")

    default_available = "default" in value
    default = _metadata_value(value["default"]) if default_available else None

    choices_value = value.get("choices", ())
    if not isinstance(choices_value, (list, tuple)):
        raise ValueError("metadata choices must be a sequence")
    if len(choices_value) > _MAX_METADATA_VALUES:
        raise ValueError("metadata choices exceed the safety limit")
    choices = tuple(_metadata_value(item) for item in choices_value)

    nullable = value.get("nullable")
    if nullable is not None and not isinstance(nullable, bool):
        raise ValueError("metadata nullable must be boolean or null")
    required = value.get("required")
    if required is not None and not isinstance(required, bool):
        raise ValueError("metadata required must be boolean or null")
    description = value.get("description")
    if description is not None:
        description = _metadata_text(description, "description")

    return PdbParameter(
        name=name,
        position=reported_position,
        direction=direction,
        parameter_type=parameter_type,
        default=default,
        default_available=default_available,
        choices=choices,
        nullable=nullable,
        required=required,
        description=description,
    )


def _parse_parameter_list(
    value: Any,
    *,
    direction: ParameterDirection,
    complete: bool,
) -> tuple[PdbParameter, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"metadata {direction} collection must be a sequence")
    if len(value) > _MAX_METADATA_PARAMETERS:
        raise ValueError("metadata parameter count exceeds the safety limit")
    return tuple(
        _parse_parameter(item, direction=direction, position=index, complete=complete)
        for index, item in enumerate(value)
    )


def parse_pdb_metadata(payload: Any, *, source: str) -> PdbIntrospectionResult:
    """Parse one explicit structured metadata payload.

    The parser accepts only an object with an explicit status and ``arguments``
    / ``returns`` arrays.  It does not infer types, defaults, enum choices, or
    optionality from procedure names or documentation.
    """

    try:
        if not isinstance(payload, Mapping):
            raise ValueError("metadata response must be an object")
        status = payload.get("status")
        if status not in {"available", "partial", "unavailable", "malformed"}:
            raise ValueError("metadata response has an invalid status")

        notes_value = payload.get("notes", ())
        if not isinstance(notes_value, (list, tuple)):
            raise ValueError("metadata notes must be a sequence")
        notes = tuple(_metadata_text(note, "note") for note in notes_value[:16])
        if status in {"unavailable", "malformed"}:
            return PdbIntrospectionResult(status=status, source=source, notes=notes)

        names_complete = payload.get("names_complete", status == "available")
        if not isinstance(names_complete, bool):
            raise ValueError("metadata names_complete must be boolean")
        requiredness_complete = payload.get("requiredness_complete", False)
        if not isinstance(requiredness_complete, bool):
            raise ValueError("metadata requiredness_complete must be boolean")

        arguments = _parse_parameter_list(
            payload.get("arguments", ()), direction="argument", complete=status == "available"
        )
        returns = _parse_parameter_list(
            payload.get("returns", ()), direction="return", complete=status == "available"
        )
        if names_complete and any(parameter.name is None for parameter in arguments):
            raise ValueError("metadata names_complete requires names for every argument")
        if requiredness_complete and any(parameter.required is None for parameter in arguments):
            raise ValueError(
                "metadata requiredness_complete requires requiredness for every argument"
            )
        return PdbIntrospectionResult(
            status=status,
            source=source,
            arguments=arguments,
            returns=returns,
            notes=notes,
            argument_names_complete=names_complete,
            requiredness_complete=requiredness_complete,
        )
    except (TypeError, ValueError) as exc:
        return PdbIntrospectionResult.malformed(source, str(exc))


@dataclass(frozen=True, slots=True)
class PdbProcedure:
    name: str
    procedure_type: str | int | None
    argument_count: int | None
    return_value_count: int | None
    blurb: str | None
    help: str | None
    help_id: str | None
    argument_metadata_available: bool = False
    argument_metadata_status: MetadataStatus = "unavailable"
    argument_metadata_source: str | None = None
    arguments: tuple[PdbParameter, ...] = ()
    returns: tuple[PdbParameter, ...] = ()
    metadata_notes: tuple[str, ...] = ()
    argument_names_complete: bool = False
    requiredness_complete: bool = False

    @property
    def named_argument_names_available(self) -> bool:
        return (
            self.argument_names_complete
            and self.argument_metadata_status
            in {
                "available",
                "partial",
            }
            and all(parameter.name is not None for parameter in self.arguments)
        )

    @property
    def required_arguments_available(self) -> bool:
        return (
            self.requiredness_complete
            and self.named_argument_names_available
            and all(parameter.required is not None for parameter in self.arguments)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "procedure_type": self.procedure_type,
            "argument_count": self.argument_count,
            "return_value_count": self.return_value_count,
            "blurb": self.blurb,
            "help": self.help,
            "help_id": self.help_id,
            "argument_metadata_available": self.argument_metadata_available,
            "argument_metadata_status": self.argument_metadata_status,
            "argument_metadata_source": self.argument_metadata_source,
            "argument_names_complete": self.argument_names_complete,
            "requiredness_complete": self.requiredness_complete,
            "arguments": [parameter.as_dict() for parameter in self.arguments],
            "returns": [parameter.as_dict() for parameter in self.returns],
            "metadata_notes": list(self.metadata_notes),
        }


def _as_list(value: Any) -> list[Any]:
    value = unwrap(value)
    return value if isinstance(value, list) else []


class PdbCatalog:
    def __init__(
        self,
        client: ScriptFuClient,
        introspection: PdbIntrospectionAdapter | None = None,
    ) -> None:
        self.client = client
        self.introspection = introspection or ScriptFuPdbIntrospection()

    def exists(self, name: str) -> bool:
        validate_procedure_name(name)
        result = unwrap(
            parse_scheme(self.client.execute(with_v3(scheme_call("gimp-pdb-proc-exists", [name]))))
        )
        return bool(result)

    def search(self, query: str, *, limit: int = 50) -> list[str]:
        if not query.strip():
            raise ValueError("PDB search query cannot be empty")
        if not 1 <= limit <= 500:
            raise ValueError("PDB search limit must be between 1 and 500")
        pattern = ".*" + re.escape(query.strip()) + ".*"
        result = self.client.execute(
            with_v3(
                scheme_call(
                    "gimp-pdb-query",
                    [pattern, ".*", ".*", ".*", ".*", ".*", ".*"],
                )
            )
        )
        names = _as_list(parse_scheme(result))
        if len(names) >= 2 and isinstance(names[1], list):
            names = names[1]
        return [str(name) for name in names[:limit]]

    def describe(self, name: str) -> PdbProcedure:
        validate_procedure_name(name)
        if not self.exists(name):
            raise ProcedureNameError(f"GIMP PDB procedure does not exist: {name}")
        info = _as_list(
            parse_scheme(
                self.client.execute(with_v3(scheme_call("gimp-pdb-get-proc-info", [name])))
            )
        )
        documentation = _as_list(
            parse_scheme(
                self.client.execute(with_v3(scheme_call("gimp-pdb-get-proc-documentation", [name])))
            )
        )
        procedure_type = info[0] if len(info) > 0 else None
        argument_count = info[1] if len(info) > 1 and isinstance(info[1], int) else None
        return_value_count = info[2] if len(info) > 2 and isinstance(info[2], int) else None
        try:
            metadata = self.introspection.inspect(
                name,
                argument_count=argument_count,
                return_value_count=return_value_count,
            )
        except Exception as exc:  # malformed optional adapters fall back safely
            metadata = PdbIntrospectionResult.malformed(
                "adapter", f"metadata adapter failed: {type(exc).__name__}"
            )
        return PdbProcedure(
            name=name,
            procedure_type=procedure_type,
            argument_count=argument_count,
            return_value_count=return_value_count,
            blurb=documentation[0]
            if len(documentation) > 0 and isinstance(documentation[0], str)
            else None,
            help=documentation[1]
            if len(documentation) > 1 and isinstance(documentation[1], str)
            else None,
            help_id=documentation[2]
            if len(documentation) > 2 and isinstance(documentation[2], str)
            else None,
            argument_metadata_available=metadata.argument_metadata_available,
            argument_metadata_status=metadata.status,
            argument_metadata_source=metadata.source,
            arguments=metadata.arguments,
            returns=metadata.returns,
            metadata_notes=metadata.notes,
            argument_names_complete=metadata.argument_names_complete,
            requiredness_complete=metadata.requiredness_complete,
        )


class PdbInvoker:
    def __init__(self, client: ScriptFuClient, catalog: PdbCatalog | None = None) -> None:
        self.client = client
        self.catalog = catalog or PdbCatalog(client)

    @staticmethod
    def _validate_keywords(procedure: PdbProcedure, keywords: Mapping[str, Any]) -> None:
        if not procedure.named_argument_names_available:
            return
        known_names = {parameter.name for parameter in procedure.arguments}
        unknown = sorted(set(keywords) - known_names)
        if unknown:
            known = ", ".join(sorted(name for name in known_names if name is not None))
            raise ValueError(
                f"Unknown named arguments for {procedure.name}: {unknown}; known: {known}"
            )
        if procedure.required_arguments_available:
            required = {
                parameter.name for parameter in procedure.arguments if parameter.required is True
            }
            missing = sorted(name for name in required if name not in keywords)
            if missing:
                raise ValueError(
                    f"Missing required named arguments for {procedure.name}: {missing}"
                )

    def invoke(
        self,
        name: str,
        args: list[Any] | tuple[Any, ...] = (),
        *,
        keywords: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        procedure = self.catalog.describe(name)
        has_keywords = keywords is not None
        if args and has_keywords:
            raise ValueError("PDB invocation cannot mix positional and named arguments")
        if procedure.argument_count is not None and len(args) > procedure.argument_count:
            raise ValueError(
                f"{name} accepts at most {procedure.argument_count} positional arguments; "
                f"received {len(args)}"
            )
        if keywords is not None:
            self._validate_keywords(procedure, keywords)
        expression = with_v3(scheme_call(name, args, keywords=keywords))
        result = unwrap(parse_scheme(self.client.execute(expression)))
        return {
            "procedure": name,
            "result": result,
            "metadata": procedure.as_dict(),
            "validation": {
                "argument_count_checked": procedure.argument_count is not None and not has_keywords,
                "named_argument_names_checked": (
                    procedure.named_argument_names_available if has_keywords else None
                ),
                "required_arguments_checked": (
                    procedure.required_arguments_available if has_keywords else None
                ),
                "metadata_status": procedure.argument_metadata_status,
                "note": (
                    "Named argument validation is unavailable because the active PDB metadata "
                    f"source reported {procedure.argument_metadata_status} metadata."
                    if has_keywords and not procedure.named_argument_names_available
                    else None
                ),
            },
        }
