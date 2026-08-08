"""Runtime PDB catalog and structured invoker."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ..errors import ProcedureNameError
from .scheme import parse_scheme, unwrap
from .serializer import scheme_call, validate_procedure_name, with_v3
from .transport import ScriptFuClient


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(value: Any) -> list[Any]:
    value = unwrap(value)
    return value if isinstance(value, list) else []


class PdbCatalog:
    def __init__(self, client: ScriptFuClient) -> None:
        self.client = client

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
        return PdbProcedure(
            name=name,
            procedure_type=info[0] if len(info) > 0 else None,
            argument_count=info[1] if len(info) > 1 and isinstance(info[1], int) else None,
            return_value_count=info[2] if len(info) > 2 and isinstance(info[2], int) else None,
            blurb=documentation[0]
            if len(documentation) > 0 and isinstance(documentation[0], str)
            else None,
            help=documentation[1]
            if len(documentation) > 1 and isinstance(documentation[1], str)
            else None,
            help_id=documentation[2]
            if len(documentation) > 2 and isinstance(documentation[2], str)
            else None,
        )


class PdbInvoker:
    def __init__(self, client: ScriptFuClient, catalog: PdbCatalog | None = None) -> None:
        self.client = client
        self.catalog = catalog or PdbCatalog(client)

    def invoke(
        self,
        name: str,
        args: list[Any] | tuple[Any, ...] = (),
        *,
        keywords: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        procedure = self.catalog.describe(name)
        if procedure.argument_count is not None and len(args) > procedure.argument_count:
            raise ValueError(
                f"{name} accepts at most {procedure.argument_count} positional arguments; "
                f"received {len(args)}"
            )
        expression = with_v3(scheme_call(name, args, keywords=keywords))
        result = unwrap(parse_scheme(self.client.execute(expression)))
        return {
            "procedure": name,
            "result": result,
            "metadata": procedure.as_dict(),
            "validation": {
                "argument_count_checked": procedure.argument_count is not None and not keywords,
                "named_argument_names_checked": False if keywords else None,
                "note": (
                    "GIMP returned procedure counts and documentation. Script-Fu does not expose "
                    "a stable JSON representation of GParamSpec argument names in this bridge yet."
                    if keywords
                    else None
                ),
            },
        }
