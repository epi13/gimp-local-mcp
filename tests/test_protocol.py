from __future__ import annotations

import pytest

from gimp_local_mcp.errors import ProcedureNameError, ProtocolError
from gimp_local_mcp.gimp.protocol import decode_request, decode_response_header, encode_request
from gimp_local_mcp.gimp.scheme import SchemeOpaque, SchemeVector, parse_scheme, unwrap
from gimp_local_mcp.gimp.serializer import (
    SchemeNull,
    scheme_call,
    scheme_path,
    scheme_string,
    scheme_value,
    validate_procedure_name,
)


def test_request_framing_is_utf8_and_big_endian() -> None:
    frame = encode_request('(gimp-message "café")')
    assert frame[:1] == b"G"
    assert int.from_bytes(frame[1:3], "big") == len(frame[3:])
    assert decode_request(frame) == '(gimp-message "café")'


def test_protocol_rejects_bad_magic_and_length() -> None:
    with pytest.raises(ProtocolError):
        decode_request(b"X\x00\x00")
    with pytest.raises(ProtocolError):
        decode_request(b"G\x00\x04abc")
    with pytest.raises(ProtocolError):
        decode_response_header(b"X\x00\x00\x00")
    with pytest.raises(ProtocolError):
        decode_response_header(b"G\x02\x00\x00")


def test_scheme_string_and_structured_call_escape_untrusted_text() -> None:
    assert scheme_string('a"b\\c\n') == '"a\\"b\\\\c\\n"'
    assert scheme_path(__import__("pathlib").Path('/tmp/a"b')) == '"/tmp/a\\"b"'
    assert scheme_value(SchemeNull()) == "-1"
    assert (
        scheme_call("gimp-item-set-name", [12, 'a") (display "oops")'])
        == '(gimp-item-set-name 12 "a\\") (display \\"oops\\")")'
    )
    assert (
        scheme_call("plug-in-test", keywords={"value": {"scheme_symbol": "RGB"}})
        == "(plug-in-test #:value RGB)"
    )


def test_procedure_names_are_restricted() -> None:
    assert validate_procedure_name("gimp-image-new") == "gimp-image-new"
    with pytest.raises(ProcedureNameError):
        validate_procedure_name('(begin (display "bad"))')


def test_scheme_reader_supports_gimp_v3_values() -> None:
    parsed = parse_scheme('(12 #t "hello\\nworld" #(3 -1) #<GimpImage>)')
    assert unwrap(parsed[:3]) == [12, True, "hello\nworld"]
    assert isinstance(parsed[3], SchemeVector)
    assert unwrap(parsed[3]) == [3, -1]
    assert isinstance(parsed[4], SchemeOpaque)
