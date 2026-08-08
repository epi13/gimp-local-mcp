from __future__ import annotations

import pytest

from gimp_local_mcp.errors import ProcedureNameError
from gimp_local_mcp.gimp.pdb import (
    PdbCatalog,
    PdbInvoker,
    StructuredPdbIntrospection,
    parse_pdb_metadata,
)


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
    assert procedure.argument_metadata_status == "unavailable"
    assert procedure.arguments == ()


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


def test_structured_metadata_adapter_exposes_typed_parameters() -> None:
    client = FakeClient()
    payload = {
        "status": "available",
        "requiredness_complete": True,
        "arguments": [
            {
                "name": "width",
                "position": 0,
                "type": "integer",
                "default": 100,
                "required": True,
                "description": "Image width",
            },
            {
                "name": "height",
                "position": 1,
                "type": "integer",
                "required": True,
            },
        ],
        "returns": [{"name": "image", "position": 0, "type": "GimpImage"}],
        "notes": ["reported by a structured metadata adapter"],
    }
    catalog = PdbCatalog(
        client,
        StructuredPdbIntrospection(lambda _name: payload, source="fixture"),
    )

    procedure = catalog.describe("gimp-image-new")

    assert procedure.argument_metadata_available is True
    assert procedure.argument_metadata_status == "available"
    assert procedure.arguments[0].name == "width"
    assert procedure.arguments[0].parameter_type == "integer"
    assert procedure.arguments[0].default == 100
    assert procedure.returns[0].direction == "return"

    result = PdbInvoker(client, catalog).invoke(
        "gimp-image-new", keywords={"width": 100, "height": 200}
    )
    assert result["validation"]["named_argument_names_checked"] is True
    assert result["validation"]["required_arguments_checked"] is True
    assert "#:width 100" in client.expressions[-1]

    with pytest.raises(ValueError, match="Unknown named arguments"):
        PdbInvoker(client, catalog).invoke("gimp-image-new", keywords={"width": 100, "depth": 8})
    with pytest.raises(ValueError, match="Missing required named arguments"):
        PdbInvoker(client, catalog).invoke("gimp-image-new", keywords={"width": 100})


def test_partial_metadata_validates_names_but_not_speculative_types() -> None:
    client = FakeClient()
    catalog = PdbCatalog(
        client,
        StructuredPdbIntrospection(
            lambda _name: {
                "status": "partial",
                "names_complete": True,
                "arguments": [{"name": "width"}, {"name": "height"}],
                "returns": [],
                "notes": ["types unavailable"],
            },
            source="partial-fixture",
        ),
    )

    procedure = catalog.describe("gimp-image-new")
    assert procedure.argument_metadata_available is False
    assert procedure.argument_metadata_status == "partial"
    assert procedure.named_argument_names_available is True

    result = PdbInvoker(client, catalog).invoke(
        "gimp-image-new", keywords={"width": 100, "height": 200}
    )
    assert result["validation"]["named_argument_names_checked"] is True
    assert result["validation"]["required_arguments_checked"] is False


def test_malformed_metadata_is_reported_without_failing_basic_description() -> None:
    result = parse_pdb_metadata(
        {
            "status": "available",
            "arguments": [{"name": "width"}],
            "returns": [],
        },
        source="malformed-fixture",
    )

    assert result.status == "malformed"
    assert result.arguments == ()
    assert result.notes


def test_partial_metadata_without_complete_names_keeps_named_validation_unknown() -> None:
    client = FakeClient()
    catalog = PdbCatalog(
        client,
        StructuredPdbIntrospection(
            lambda _name: {
                "status": "partial",
                "arguments": [{"name": "width"}],
                "returns": [],
            },
            source="incomplete-fixture",
        ),
    )

    procedure = catalog.describe("gimp-image-new")
    assert procedure.named_argument_names_available is False
    result = PdbInvoker(client, catalog).invoke("gimp-image-new", keywords={"future_arg": 1})
    assert result["validation"]["named_argument_names_checked"] is False
    assert result["validation"]["metadata_status"] == "partial"


def test_metadata_limits_keep_pathological_payloads_unavailable() -> None:
    result = parse_pdb_metadata(
        {
            "status": "available",
            "arguments": [{"name": f"arg{index}", "type": "integer"} for index in range(257)],
            "returns": [],
        },
        source="oversized-fixture",
    )

    assert result.status == "malformed"
    assert "safety limit" in result.notes[0]


def test_metadata_parser_rejects_nested_scheme_source() -> None:
    result = parse_pdb_metadata(
        {
            "status": "available",
            "arguments": [
                {
                    "name": "script",
                    "type": "string",
                    "default": {"scheme": '(display "bad")'},
                }
            ],
            "returns": [],
        },
        source="security-fixture",
    )

    assert result.status == "malformed"
    assert "unsupported metadata value" in result.notes[0]
