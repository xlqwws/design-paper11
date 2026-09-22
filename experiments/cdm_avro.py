"""Small streaming Avro OCF reader for DARPA TC CDM archives.

E5 CADETS files are gzip-wrapped Avro object container files.  Reading one
whole decompressed shard can require multiple GB, so this reader decodes one
OCF block at a time and has no dependency on fastavro.
"""

from __future__ import annotations

import gzip
import json
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Tuple


PRIMITIVES = {
    "null", "boolean", "int", "long", "float", "double", "bytes", "string",
}
PROJECTED_FIELDS = {
    "Event": {
        "uuid", "type", "subject", "predicateObject", "predicateObject2",
        "timestampNanos", "predicateObjectPath", "properties",
    },
    "Subject": {"uuid", "cmdLine"},
    "FileObject": {"uuid", "type"},
    "NetFlowObject": {"uuid", "localAddress", "localPort", "remoteAddress", "remotePort"},
}
EVENT_ONLY_FIELDS = {
    "uuid", "type", "subject", "predicateObject", "timestampNanos",
}


def read_long_buffer(buf: bytes, index: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte = buf[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return (value >> 1) ^ -(value & 1), index


def read_long_stream(stream: BinaryIO) -> int:
    shift = 0
    value = 0
    while True:
        raw = stream.read(1)
        if not raw:
            raise EOFError
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return (value >> 1) ^ -(value & 1)


def read_bytes_buffer(buf: bytes, index: int) -> Tuple[bytes, int]:
    size, index = read_long_buffer(buf, index)
    return buf[index:index + size], index + size


def read_string_buffer(buf: bytes, index: int) -> Tuple[str, int]:
    raw, index = read_bytes_buffer(buf, index)
    return raw.decode("utf-8", errors="ignore"), index


def read_bytes_stream(stream: BinaryIO) -> bytes:
    size = read_long_stream(stream)
    value = stream.read(size)
    if len(value) != size:
        raise EOFError
    return value


def read_string_stream(stream: BinaryIO) -> str:
    return read_bytes_stream(stream).decode("utf-8", errors="ignore")


def schema_fullname(schema: Dict[str, object], namespace: str = "") -> str:
    name = str(schema.get("name", ""))
    selected_namespace = str(schema.get("namespace") or namespace or "")
    return f"{selected_namespace}.{name}" if selected_namespace and "." not in name else name


def collect_names(schema: object, names: Dict[str, object], namespace: str = "") -> None:
    if isinstance(schema, list):
        for item in schema:
            collect_names(item, names, namespace)
        return
    if not isinstance(schema, dict):
        return
    schema_type = schema.get("type")
    if isinstance(schema_type, (dict, list)):
        collect_names(schema_type, names, namespace)
        return
    if schema_type in {"record", "enum", "fixed"} and "name" in schema:
        full_name = schema_fullname(schema, namespace)
        names[str(schema["name"])] = schema
        names[full_name] = schema
        namespace = full_name.rsplit(".", 1)[0] if "." in full_name else namespace
    if schema_type == "record":
        for field in schema.get("fields", []):
            collect_names(field.get("type"), names, namespace)
    elif schema_type == "array":
        collect_names(schema.get("items"), names, namespace)
    elif schema_type == "map":
        collect_names(schema.get("values"), names, namespace)


def resolve(schema: object, names: Dict[str, object]) -> object:
    if isinstance(schema, str) and schema not in PRIMITIVES:
        return names.get(schema, schema)
    if isinstance(schema, dict):
        schema_type = schema.get("type")
        if isinstance(schema_type, str) and schema_type in names:
            return names[schema_type]
    return schema


def decode(schema: object, buf: bytes, index: int, names: Dict[str, object]):
    schema = resolve(schema, names)
    if isinstance(schema, list):
        branch, index = read_long_buffer(buf, index)
        return decode(schema[int(branch)], buf, index, names)
    if isinstance(schema, str):
        if schema == "null":
            return None, index
        if schema == "boolean":
            return bool(buf[index]), index + 1
        if schema in {"int", "long"}:
            return read_long_buffer(buf, index)
        if schema == "float":
            return struct.unpack("<f", buf[index:index + 4])[0], index + 4
        if schema == "double":
            return struct.unpack("<d", buf[index:index + 8])[0], index + 8
        if schema == "bytes":
            return read_bytes_buffer(buf, index)
        if schema == "string":
            return read_string_buffer(buf, index)
        raise ValueError(f"Unsupported Avro reference: {schema}")
    if not isinstance(schema, dict):
        raise ValueError(f"Unsupported Avro schema: {schema!r}")
    schema_type = resolve(schema.get("type"), names)
    if isinstance(schema_type, (list, dict)):
        return decode(schema_type, buf, index, names)
    if schema_type == "record":
        output = {"__record_name": str(schema.get("name", ""))}
        for field in schema.get("fields", []):
            output[field["name"]], index = decode(field["type"], buf, index, names)
        return output, index
    if schema_type == "enum":
        symbol_index, index = read_long_buffer(buf, index)
        symbols = schema.get("symbols", [])
        value = symbols[int(symbol_index)] if 0 <= int(symbol_index) < len(symbols) else str(symbol_index)
        return str(value), index
    if schema_type == "fixed":
        size = int(schema["size"])
        return buf[index:index + size], index + size
    if schema_type == "array":
        output = []
        while True:
            count, index = read_long_buffer(buf, index)
            if count == 0:
                break
            if count < 0:
                count = -count
                _, index = read_long_buffer(buf, index)
            for _ in range(count):
                value, index = decode(schema["items"], buf, index, names)
                output.append(value)
        return output, index
    if schema_type == "map":
        output = {}
        while True:
            count, index = read_long_buffer(buf, index)
            if count == 0:
                break
            if count < 0:
                count = -count
                _, index = read_long_buffer(buf, index)
            for _ in range(count):
                key, index = read_string_buffer(buf, index)
                output[key], index = decode(schema["values"], buf, index, names)
        return output, index
    return decode(str(schema_type), buf, index, names)


def skip(schema: object, buf: bytes, index: int, names: Dict[str, object]) -> int:
    schema = resolve(schema, names)
    if isinstance(schema, list):
        branch, index = read_long_buffer(buf, index)
        return skip(schema[int(branch)], buf, index, names)
    if isinstance(schema, str):
        if schema == "null":
            return index
        if schema == "boolean":
            return index + 1
        if schema in {"int", "long"}:
            _, index = read_long_buffer(buf, index)
            return index
        if schema == "float":
            return index + 4
        if schema == "double":
            return index + 8
        if schema in {"bytes", "string"}:
            size, index = read_long_buffer(buf, index)
            return index + size
        raise ValueError(f"Unsupported Avro reference while skipping: {schema}")
    if not isinstance(schema, dict):
        raise ValueError(f"Unsupported Avro schema while skipping: {schema!r}")
    schema_type = resolve(schema.get("type"), names)
    if isinstance(schema_type, (list, dict)):
        return skip(schema_type, buf, index, names)
    if schema_type == "record":
        for field in schema.get("fields", []):
            index = skip(field["type"], buf, index, names)
        return index
    if schema_type == "enum":
        _, index = read_long_buffer(buf, index)
        return index
    if schema_type == "fixed":
        return index + int(schema["size"])
    if schema_type == "array":
        while True:
            count, index = read_long_buffer(buf, index)
            if count == 0:
                return index
            if count < 0:
                count = -count
                block_size, index = read_long_buffer(buf, index)
                index += block_size
                continue
            for _ in range(count):
                index = skip(schema["items"], buf, index, names)
    if schema_type == "map":
        while True:
            count, index = read_long_buffer(buf, index)
            if count == 0:
                return index
            if count < 0:
                count = -count
                block_size, index = read_long_buffer(buf, index)
                index += block_size
                continue
            for _ in range(count):
                index = skip("string", buf, index, names)
                index = skip(schema["values"], buf, index, names)
    return skip(str(schema_type), buf, index, names)


def decode_projected(schema: object, buf: bytes, index: int, names: Dict[str, object]):
    schema = resolve(schema, names)
    if isinstance(schema, list):
        branch, index = read_long_buffer(buf, index)
        return decode_projected(schema[int(branch)], buf, index, names)
    if not isinstance(schema, dict):
        return decode(schema, buf, index, names)
    schema_type = resolve(schema.get("type"), names)
    if isinstance(schema_type, (list, dict)):
        return decode_projected(schema_type, buf, index, names)
    if schema_type != "record":
        return decode(schema, buf, index, names)

    record_name = str(schema.get("name", ""))
    selected = PROJECTED_FIELDS.get(record_name)
    is_envelope = "datum" in {str(field.get("name")) for field in schema.get("fields", [])}
    output = {"__record_name": record_name}
    for field in schema.get("fields", []):
        field_name = str(field["name"])
        should_decode = field_name == "datum" if is_envelope else (
            field_name == "uuid" if selected is None else field_name in selected
        )
        if should_decode:
            output[field_name], index = decode_projected(field["type"], buf, index, names)
        else:
            index = skip(field["type"], buf, index, names)
    return output, index


def decode_event_envelope(schema: object, buf: bytes, index: int, names: Dict[str, object]):
    """Decode only Event payloads from a top-level CDM envelope.

    E5 contains tens of millions of entity and marker records.  The only-type
    temporal builder does not consume them, so constructing projected Python
    dictionaries for those branches is needless overhead.
    """
    schema = resolve(schema, names)
    if not isinstance(schema, dict) or schema.get("type") != "record":
        raise ValueError("Expected a top-level CDM envelope record")
    event = None
    for field in schema.get("fields", []):
        field_schema = resolve(field["type"], names)
        if field.get("name") != "datum" or not isinstance(field_schema, list):
            index = skip(field_schema, buf, index, names)
            continue
        branch, index = read_long_buffer(buf, index)
        branch_schema = resolve(field_schema[int(branch)], names)
        branch_name = str(branch_schema.get("name", "")) if isinstance(branch_schema, dict) else ""
        if branch_name != "Event":
            index = skip(branch_schema, buf, index, names)
            continue
        event = {"__record_name": "Event"}
        for event_field in branch_schema.get("fields", []):
            name = str(event_field["name"])
            if name in EVENT_ONLY_FIELDS:
                event[name], index = decode(event_field["type"], buf, index, names)
            else:
                index = skip(event_field["type"], buf, index, names)
    return event, index


def read_metadata(stream: BinaryIO) -> Dict[str, bytes]:
    metadata = {}
    while True:
        count = read_long_stream(stream)
        if count == 0:
            break
        if count < 0:
            count = -count
            read_long_stream(stream)
        for _ in range(count):
            key = read_string_stream(stream)
            value = read_bytes_stream(stream)
            metadata[key] = value
    return metadata


def iter_ocf_blocks(path: Path):
    """Yield `(schema, names, record_count, decoded_block)` tuples."""
    with gzip.open(path, "rb") as stream:
        if stream.read(4) != b"Obj\x01":
            raise ValueError(f"Not an Avro object container: {path}")
        metadata = read_metadata(stream)
        schema = json.loads(metadata["avro.schema"].decode("utf-8"))
        names = {}
        collect_names(schema, names, str(schema.get("namespace", "")))
        expected_sync = stream.read(16)
        codec = metadata.get("avro.codec", b"null").decode("utf-8", errors="ignore")
        while True:
            try:
                count = read_long_stream(stream)
                block_size = read_long_stream(stream)
            except EOFError:
                return
            block = stream.read(block_size)
            if len(block) != block_size:
                raise EOFError(f"Truncated Avro block in {path}")
            if codec == "deflate":
                block = zlib.decompress(block, -15)
            elif codec != "null":
                raise ValueError(f"Unsupported Avro codec `{codec}` in {path}")
            yield schema, names, int(count), block
            if stream.read(16) != expected_sync:
                raise ValueError(f"Avro sync marker mismatch in {path}")


def decode_projected_block(schema, names, count: int, block: bytes):
    index = 0
    for _ in range(count):
        record, index = decode_projected(schema, block, index, names)
        if isinstance(record, dict):
            yield record


def iter_ocf_records(path: Path, max_records: int = 0) -> Iterable[Dict[str, object]]:
    emitted = 0
    with gzip.open(path, "rb") as stream:
        if stream.read(4) != b"Obj\x01":
            raise ValueError(f"Not an Avro object container: {path}")
        metadata = read_metadata(stream)
        schema = json.loads(metadata["avro.schema"].decode("utf-8"))
        names = {}
        collect_names(schema, names, str(schema.get("namespace", "")))
        expected_sync = stream.read(16)
        codec = metadata.get("avro.codec", b"null").decode("utf-8", errors="ignore")
        while True:
            try:
                count = read_long_stream(stream)
                block_size = read_long_stream(stream)
            except EOFError:
                return
            block = stream.read(block_size)
            if len(block) != block_size:
                raise EOFError(f"Truncated Avro block in {path}")
            if codec == "deflate":
                block = zlib.decompress(block, -15)
            elif codec != "null":
                raise ValueError(f"Unsupported Avro codec `{codec}` in {path}")
            index = 0
            for _ in range(count):
                record, index = decode(schema, block, index, names)
                if isinstance(record, dict):
                    yield record
                    emitted += 1
                    if max_records and emitted >= max_records:
                        return
            actual_sync = stream.read(16)
            if actual_sync != expected_sync:
                raise ValueError(f"Avro sync marker mismatch in {path}")


def iter_cdm_records(path: Path, max_records: int = 0) -> Iterable[Dict[str, object]]:
    """Yield projected CDM records while preserving exact OCF boundaries."""
    emitted = 0
    with gzip.open(path, "rb") as stream:
        if stream.read(4) != b"Obj\x01":
            raise ValueError(f"Not an Avro object container: {path}")
        metadata = read_metadata(stream)
        schema = json.loads(metadata["avro.schema"].decode("utf-8"))
        names = {}
        collect_names(schema, names, str(schema.get("namespace", "")))
        expected_sync = stream.read(16)
        codec = metadata.get("avro.codec", b"null").decode("utf-8", errors="ignore")
        while True:
            try:
                count = read_long_stream(stream)
                block_size = read_long_stream(stream)
            except EOFError:
                return
            block = stream.read(block_size)
            if len(block) != block_size:
                raise EOFError(f"Truncated Avro block in {path}")
            if codec == "deflate":
                block = zlib.decompress(block, -15)
            elif codec != "null":
                raise ValueError(f"Unsupported Avro codec `{codec}` in {path}")
            index = 0
            for _ in range(count):
                record, index = decode_projected(schema, block, index, names)
                if isinstance(record, dict):
                    yield record
                    emitted += 1
                    if max_records and emitted >= max_records:
                        return
            actual_sync = stream.read(16)
            if actual_sync != expected_sync:
                raise ValueError(f"Avro sync marker mismatch in {path}")


def iter_cdm_events(path: Path, max_records: int = 0) -> Iterable[Dict[str, object]]:
    """Yield only minimal Event records from a CDM OCF stream."""
    scanned = 0
    with gzip.open(path, "rb") as stream:
        if stream.read(4) != b"Obj\x01":
            raise ValueError(f"Not an Avro object container: {path}")
        metadata = read_metadata(stream)
        schema = json.loads(metadata["avro.schema"].decode("utf-8"))
        names = {}
        collect_names(schema, names, str(schema.get("namespace", "")))
        expected_sync = stream.read(16)
        codec = metadata.get("avro.codec", b"null").decode("utf-8", errors="ignore")
        while True:
            try:
                count = read_long_stream(stream)
                block_size = read_long_stream(stream)
            except EOFError:
                return
            block = stream.read(block_size)
            if len(block) != block_size:
                raise EOFError(f"Truncated Avro block in {path}")
            if codec == "deflate":
                block = zlib.decompress(block, -15)
            elif codec != "null":
                raise ValueError(f"Unsupported Avro codec `{codec}` in {path}")
            index = 0
            for _ in range(count):
                event, index = decode_event_envelope(schema, block, index, names)
                scanned += 1
                if event is not None:
                    yield event
                if max_records and scanned >= max_records:
                    return
            if stream.read(16) != expected_sync:
                raise ValueError(f"Avro sync marker mismatch in {path}")
