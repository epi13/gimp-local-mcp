from __future__ import annotations

import pytest

from gimp_local_mcp.errors import ProcedureNameError
from gimp_local_mcp.gimp.pdb import PdbCatalog, PdbInvoker


class FakeClient:
    def __init__(self) -> None:
        self.expressions: list[str] = []

    def execute(self, expression: str) -> str:
        self.expressions.append(expression)
        if "gimp-pdb-proc-exists" in expression:
            return "#t"
        if "gimp-pdb-get-proc-info" in expression:
            return "(0 2 1)"
        if "gimp-pdb-get-proc-documentation" in expression:
            return '("A test" "Detailed" "help-id")'
        if "gimp-pdb-query" in expression:
            return '("gimp-image-new" "gimp-image-scale")'
        return "42"


def test_catalog_discovers_live_metadata_without_faking_arguments() -> None:
    client = FakeClient()
    procedure = PdbCatalog(client).describe("gimp-image-new")
    assert procedure.argument_count == 2
    assert procedure.return_value_count == 1
    assert procedure.argument_metadata_available is False


def test_invoker_serializes_structured_arguments_and_validates_count() -> None:
    client = FakeClient()
    invoker = PdbInvoker(client)
    result = invoker.invoke("gimp-image-new", [100, 200])
    assert result["result"] == 42
    assert "(gimp-image-new 100 200)" in client.expressions[-1]
    with pytest.raises(ValueError, match="at most 2"):
        invoker.invoke("gimp-image-new", [1, 2, 3])


def test_catalog_rejects_invalid_names_before_transport() -> None:
    client = FakeClient()
    with pytest.raises(ProcedureNameError):
        PdbCatalog(client).exists('(display "bad")')
    assert client.expressions == []
