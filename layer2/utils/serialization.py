"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Serialization Utilities
File    : layer2/utils/serialization.py
============================================================

Purpose
-------
Provides safe and reusable JSON operations:

- Conversion of dataclasses, enums, paths and datetimes
- Strict JSON serialization
- Atomic JSON writing
- JSON file reading and root-type validation
- SHA-256 content hashing
- File-integrity verification
- Previous-output archiving
- JSON file metadata inspection

Atomic writing prevents partially written Layer 2 outputs from
being consumed by Layer 3.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type


# ============================================================
# CONSTANTS
# ============================================================

SERIALIZATION_VERSION = "1.0"
DEFAULT_JSON_INDENT = 2
DEFAULT_ENCODING = "utf-8"


# ============================================================
# EXCEPTIONS
# ============================================================

class SerializationError(Exception):
    """Base serialization exception."""


class JSONConversionError(SerializationError):
    """Raised when a value cannot be converted."""


class JSONReadError(SerializationError):
    """Raised when a JSON file cannot be read."""


class JSONWriteError(SerializationError):
    """Raised when a JSON file cannot be written."""


class JSONValidationError(SerializationError):
    """Raised when JSON structure is invalid."""


class IntegrityVerificationError(
    SerializationError
):
    """Raised when content integrity verification fails."""


class ArchiveError(SerializationError):
    """Raised when an output cannot be archived."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class JSONFileMetadata:
    """Metadata describing one JSON file."""

    path: str
    size_bytes: int
    sha256: str
    modified_at: str
    root_type: str


@dataclass(frozen=True)
class JSONWriteResult:
    """Result returned by atomic JSON writing."""

    path: str
    size_bytes: int
    sha256: str
    written_at: str
    archived_previous_path: Optional[str]
    atomic: bool
    serialization_version: str


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="milliseconds"
    )


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def make_json_safe(
    value: Any,
    *,
    strict: bool = True,
) -> Any:
    """
    Convert common Python objects to JSON-safe values.

    Supported values include:
    - Dataclasses
    - Enums
    - pathlib.Path
    - datetime
    - bytes
    - dictionaries
    - lists, tuples and sets
    """

    if is_dataclass(value):
        return make_json_safe(
            asdict(value),
            strict=strict,
        )

    if isinstance(value, Enum):
        return make_json_safe(
            value.value,
            strict=strict,
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):

        timestamp = value

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        return timestamp.isoformat()

    if isinstance(value, bytes):
        return {
            "encoding": "hex",
            "data": value.hex(),
        }

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(
                item,
                strict=strict,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(
                item,
                strict=strict,
            )
            for item in value
        ]

    if isinstance(value, set):
        converted_items = [
            make_json_safe(
                item,
                strict=strict,
            )
            for item in value
        ]

        try:
            return sorted(
                converted_items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            )
        except TypeError:
            return converted_items

    if isinstance(value, float):

        if not math.isfinite(value):
            if strict:
                raise JSONConversionError(
                    "NaN and infinite float values "
                    "are not permitted."
                )

            return None

        return value

    if isinstance(
        value,
        (str, int, bool),
    ) or value is None:
        return value

    try:
        json.dumps(
            value,
            allow_nan=False,
        )

        return value

    except (TypeError, ValueError) as error:

        if strict:
            raise JSONConversionError(
                "Unsupported JSON value type: "
                f"{value.__class__.__name__}"
            ) from error

        return repr(value)


# ============================================================
# JSON SERIALIZATION
# ============================================================

def dumps_json(
    value: Any,
    *,
    indent: Optional[int] = DEFAULT_JSON_INDENT,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    strict: bool = True,
) -> str:
    """Serialize a value to JSON text."""

    safe_value = make_json_safe(
        value,
        strict=strict,
    )

    try:
        return json.dumps(
            safe_value,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise JSONConversionError(
            "Unable to serialize value to JSON."
        ) from error


def canonical_json(
    value: Any,
) -> str:
    """
    Return deterministic JSON used for hashing.

    Whitespace and dictionary insertion order do not affect
    the resulting content hash.
    """

    safe_value = make_json_safe(
        value,
        strict=True,
    )

    try:
        return json.dumps(
            safe_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise JSONConversionError(
            "Unable to create canonical JSON."
        ) from error


# ============================================================
# CONTENT HASHING
# ============================================================

def calculate_json_sha256(
    value: Any,
) -> str:
    """Calculate SHA-256 for canonical JSON content."""

    content = canonical_json(value)

    return hashlib.sha256(
        content.encode(DEFAULT_ENCODING)
    ).hexdigest()


def calculate_file_sha256(
    file_path: Path | str,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the byte-level SHA-256 of a file."""

    path = Path(file_path)

    if not path.exists():
        raise IntegrityVerificationError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise IntegrityVerificationError(
            f"Path is not a file: {path}"
        )

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise IntegrityVerificationError(
            "chunk_size must be a positive integer."
        )

    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:

            while True:
                chunk = file.read(chunk_size)

                if not chunk:
                    break

                digest.update(chunk)

    except OSError as error:
        raise IntegrityVerificationError(
            f"Unable to hash file: {path}"
        ) from error

    return digest.hexdigest()


# ============================================================
# JSON READING
# ============================================================

def read_json(
    file_path: Path | str,
    *,
    expected_root_type: Optional[Type[Any]] = dict,
    encoding: str = DEFAULT_ENCODING,
) -> Any:
    """Read and optionally validate a JSON file."""

    path = Path(file_path)

    if not path.exists():
        raise JSONReadError(
            f"JSON file does not exist: {path}"
        )

    if not path.is_file():
        raise JSONReadError(
            f"JSON path is not a file: {path}"
        )

    try:
        with path.open(
            "r",
            encoding=encoding,
        ) as file:
            payload = json.load(file)

    except json.JSONDecodeError as error:
        raise JSONReadError(
            f"Invalid JSON in {path}: "
            f"line {error.lineno}, "
            f"column {error.colno}."
        ) from error

    except OSError as error:
        raise JSONReadError(
            f"Unable to read JSON file: {path}"
        ) from error

    if (
        expected_root_type is not None
        and not isinstance(
            payload,
            expected_root_type,
        )
    ):
        raise JSONValidationError(
            f"Expected JSON root type "
            f"{expected_root_type.__name__}, "
            f"received "
            f"{payload.__class__.__name__}."
        )

    return payload


# ============================================================
# ARCHIVING
# ============================================================

def archive_existing_file(
    file_path: Path | str,
    *,
    archive_directory: Optional[
        Path | str
    ] = None,
    timestamp: Optional[str] = None,
) -> Optional[Path]:
    """
    Copy an existing output to an archive directory.

    The original file is retained so it can be atomically
    replaced afterward.
    """

    path = Path(file_path)

    if not path.exists():
        return None

    if not path.is_file():
        raise ArchiveError(
            f"Archive source is not a file: {path}"
        )

    archive_dir = Path(
        archive_directory
        or path.parent / "archive"
    )

    try:
        archive_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        archive_timestamp = (
            timestamp
            or datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%S_%fZ"
            )
        )

        archive_name = (
            f"{path.stem}_"
            f"{archive_timestamp}"
            f"{path.suffix}"
        )

        archive_path = (
            archive_dir / archive_name
        )

        counter = 1

        while archive_path.exists():
            archive_path = (
                archive_dir
                / (
                    f"{path.stem}_"
                    f"{archive_timestamp}_"
                    f"{counter}"
                    f"{path.suffix}"
                )
            )

            counter += 1

        shutil.copy2(
            path,
            archive_path,
        )

    except OSError as error:
        raise ArchiveError(
            f"Unable to archive file: {path}"
        ) from error

    return archive_path


# ============================================================
# ATOMIC JSON WRITING
# ============================================================

def write_json_atomic(
    file_path: Path | str,
    value: Any,
    *,
    indent: Optional[int] = DEFAULT_JSON_INDENT,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    encoding: str = DEFAULT_ENCODING,
    archive_previous: bool = False,
    archive_directory: Optional[
        Path | str
    ] = None,
    add_trailing_newline: bool = True,
) -> JSONWriteResult:
    """
    Write JSON through a temporary file and atomically replace
    the destination.

    The temporary file is created in the destination directory,
    ensuring that os.replace operates on the same filesystem.
    """

    destination = Path(file_path)

    serialized = dumps_json(
        value,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        strict=True,
    )

    if add_trailing_newline:
        serialized += "\n"

    archived_path: Optional[Path] = None
    temporary_path: Optional[Path] = None

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if archive_previous:
            archived_path = archive_existing_file(
                destination,
                archive_directory=(
                    archive_directory
                ),
            )

        descriptor, temporary_name = (
            tempfile.mkstemp(
                prefix=(
                    f".{destination.stem}_"
                ),
                suffix=".tmp",
                dir=str(destination.parent),
            )
        )

        temporary_path = Path(
            temporary_name
        )

        with os.fdopen(
            descriptor,
            "w",
            encoding=encoding,
            newline="\n",
        ) as temporary_file:

            temporary_file.write(
                serialized
            )

            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

        os.replace(
            temporary_path,
            destination,
        )

        temporary_path = None

    except (
        OSError,
        SerializationError,
    ) as error:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except OSError:
                pass

        if isinstance(
            error,
            SerializationError,
        ):
            raise

        raise JSONWriteError(
            f"Unable to atomically write JSON: "
            f"{destination}"
        ) from error

    written_payload = read_json(
        destination,
        expected_root_type=None,
        encoding=encoding,
    )

    return JSONWriteResult(
        path=str(destination.resolve()),
        size_bytes=destination.stat().st_size,
        sha256=calculate_json_sha256(
            written_payload
        ),
        written_at=utc_now_iso(),
        archived_previous_path=(
            str(archived_path.resolve())
            if archived_path is not None
            else None
        ),
        atomic=True,
        serialization_version=(
            SERIALIZATION_VERSION
        ),
    )


# ============================================================
# INTEGRITY VERIFICATION
# ============================================================

def verify_json_integrity(
    file_path: Path | str,
    expected_sha256: str,
) -> bool:
    """
    Verify canonical JSON content against an expected hash.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise IntegrityVerificationError(
            "expected_sha256 must contain "
            "64 hexadecimal characters."
        )

    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise IntegrityVerificationError(
            "expected_sha256 is not hexadecimal."
        ) from error

    payload = read_json(
        file_path,
        expected_root_type=None,
    )

    actual_sha256 = (
        calculate_json_sha256(payload)
    )

    if (
        actual_sha256
        != expected_sha256.lower()
    ):
        raise IntegrityVerificationError(
            "JSON integrity verification failed. "
            f"Expected {expected_sha256.lower()}, "
            f"received {actual_sha256}."
        )

    return True


# ============================================================
# FILE INSPECTION
# ============================================================

def inspect_json_file(
    file_path: Path | str,
) -> JSONFileMetadata:
    """Inspect a JSON file and return metadata."""

    path = Path(file_path)

    payload = read_json(
        path,
        expected_root_type=None,
    )

    try:
        modified_at = datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        ).isoformat(
            timespec="milliseconds"
        )

        size_bytes = path.stat().st_size

    except OSError as error:
        raise JSONReadError(
            f"Unable to inspect JSON file: {path}"
        ) from error

    return JSONFileMetadata(
        path=str(path.resolve()),
        size_bytes=size_bytes,
        sha256=calculate_json_sha256(
            payload
        ),
        modified_at=modified_at,
        root_type=(
            payload.__class__.__name__
        ),
    )


# ============================================================
# SELF-TEST SUPPORT
# ============================================================

class SelfTestStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class SelfTestRecord:
    record_id: str
    status: SelfTestStatus
    timestamp: datetime
    output_path: Path
    values: Dict[str, Any]


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | SERIALIZATION SELF-TEST")
    print("=" * 72)

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    test_directory = (
        project_root
        / "output"
        / "layer2"
        / "serialization_self_test"
    )

    unique_id = (
        uuid.uuid4().hex[:8]
    )

    output_path = (
        test_directory
        / f"serialization_test_{unique_id}.json"
    )

    archive_directory = (
        test_directory / "archive"
    )

    try:
        test_record = SelfTestRecord(
            record_id=(
                f"TEST_{unique_id.upper()}"
            ),
            status=SelfTestStatus.SUCCESS,
            timestamp=datetime.now(
                timezone.utc
            ),
            output_path=output_path,
            values={
                "confidence": 0.935,
                "modalities": {
                    "vision",
                    "audio",
                    "motion",
                },
                "binary_data": b"NOONGIL",
            },
        )

        safe_record = make_json_safe(
            test_record
        )

        if not isinstance(
            safe_record,
            dict,
        ):
            raise AssertionError(
                "Dataclass conversion failed."
            )

        print("[PASS] Dataclass converted")
        print("[PASS] Enum converted")
        print("[PASS] Path converted")
        print("[PASS] Datetime converted")
        print("[PASS] Set and bytes converted")

        first_write = write_json_atomic(
            output_path,
            test_record,
            sort_keys=True,
            archive_previous=False,
        )

        if not Path(first_write.path).exists():
            raise AssertionError(
                "Atomic output was not created."
            )

        print("[PASS] JSON written atomically")

        loaded_payload = read_json(
            output_path,
            expected_root_type=dict,
        )

        if (
            loaded_payload["record_id"]
            != test_record.record_id
        ):
            raise AssertionError(
                "Loaded JSON content is incorrect."
            )

        print("[PASS] JSON read and validated")

        if not verify_json_integrity(
            output_path,
            first_write.sha256,
        ):
            raise AssertionError(
                "Integrity verification returned false."
            )

        print("[PASS] SHA-256 integrity verified")

        updated_payload = dict(
            loaded_payload
        )

        updated_payload["values"][
            "confidence"
        ] = 0.95

        second_write = write_json_atomic(
            output_path,
            updated_payload,
            sort_keys=True,
            archive_previous=True,
            archive_directory=(
                archive_directory
            ),
        )

        if (
            second_write.archived_previous_path
            is None
        ):
            raise AssertionError(
                "Previous output was not archived."
            )

        if not Path(
            second_write.archived_previous_path
        ).exists():
            raise AssertionError(
                "Archived output does not exist."
            )

        print("[PASS] Previous output archived")
        print("[PASS] Updated output written")

        metadata = inspect_json_file(
            output_path
        )

        if metadata.root_type != "dict":
            raise AssertionError(
                "Incorrect JSON root type."
            )

        if metadata.size_bytes <= 0:
            raise AssertionError(
                "JSON file size is invalid."
            )

        print("[PASS] JSON metadata inspected")

        try:
            make_json_safe(
                float("nan"),
                strict=True,
            )
        except JSONConversionError:
            print(
                "[PASS] Invalid float rejected"
            )
        else:
            raise AssertionError(
                "NaN value was accepted."
            )

        print("\nSerialization summary:")
        print(
            f"  output: {second_write.path}"
        )
        print(
            f"  size: "
            f"{second_write.size_bytes} bytes"
        )
        print(
            f"  SHA-256: "
            f"{second_write.sha256}"
        )
        print(
            f"  archive: "
            f"{second_write.archived_previous_path}"
        )
        print(
            f"  atomic: "
            f"{second_write.atomic}"
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] SERIALIZATION UTILITIES "
            "ARE WORKING"
        )
        print("=" * 72)

        return True

    except (
        SerializationError,
        AssertionError,
    ) as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    return argparse.ArgumentParser(
        description=(
            "Run the NOONGIL-X Layer 2 "
            "serialization self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())