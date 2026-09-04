"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Unified Layer 2 Output Schema
File    : layer2/schemas/layer2_output.py
============================================================

Purpose
-------
Defines the final structured perception output produced by
Layer 2 and consumed by the frozen Layer 3 pipeline.

Required Layer 3 compatibility fields:
- timestamp
- scene.type
- objects
- sounds
- speech_transcript
- user_activity.state

Additional perception fields:
- recognized_text
- depth
- obstacles
- perception_confidence
- fusion
- metadata

Architectural Boundary
----------------------
This schema does not construct:
- Entities
- Events
- Context graphs
- Episodic memories
- Semantic memories
- Final hazard decisions

Those responsibilities belong to Layers 3 and 4.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import json
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ============================================================
# CONSTANTS
# ============================================================

LAYER2_OUTPUT_SCHEMA_VERSION = "1.0"
LAYER2_OUTPUT_TYPE = "noongil_layer2_perception_output"

ALLOWED_OUTPUT_STATUSES = {
    "complete",
    "partial",
    "degraded",
    "failed",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class Layer2OutputError(Exception):
    """Base exception for Layer 2 output operations."""


class Layer2OutputValidationError(
    Layer2OutputError
):
    """Raised when Layer 2 output is invalid."""


class Layer2OutputSerializationError(
    Layer2OutputError
):
    """Raised when output serialization fails."""


class Layer2OutputWriteError(
    Layer2OutputError
):
    """Raised when output cannot be written."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="milliseconds"
    )


def generate_output_id() -> str:
    """Generate a unique Layer 2 output ID."""

    return (
        "L2_OUTPUT_"
        f"{uuid.uuid4().hex[:16].upper()}"
    )


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise Layer2OutputValidationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def validate_iso_timestamp(
    value: Any,
    field_name: str,
) -> str:
    """Validate an ISO-8601 timestamp."""

    timestamp = require_non_empty_string(
        value,
        field_name,
    )

    normalized = timestamp.replace(
        "Z",
        "+00:00",
    )

    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise Layer2OutputValidationError(
            f"{field_name} is not a valid "
            f"ISO-8601 timestamp: {timestamp!r}"
        ) from error

    return timestamp


def validate_confidence(
    value: Any,
    field_name: str,
) -> Optional[float]:
    """Validate a confidence score."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise Layer2OutputValidationError(
            f"{field_name} must be numeric."
        )

    confidence = float(value)

    if not math.isfinite(confidence):
        raise Layer2OutputValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= confidence <= 1.0:
        raise Layer2OutputValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return confidence


def validate_non_negative_number(
    value: Any,
    field_name: str,
) -> Optional[float]:
    """Validate an optional non-negative number."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise Layer2OutputValidationError(
            f"{field_name} must be numeric."
        )

    number = float(value)

    if not math.isfinite(number):
        raise Layer2OutputValidationError(
            f"{field_name} must be finite."
        )

    if number < 0.0:
        raise Layer2OutputValidationError(
            f"{field_name} cannot be negative."
        )

    return number


def require_dictionary(
    value: Any,
    field_name: str,
) -> Dict[str, Any]:
    """Validate and return a dictionary."""

    if not isinstance(value, Mapping):
        raise Layer2OutputValidationError(
            f"{field_name} must be a JSON object."
        )

    return dict(value)


def require_list(
    value: Any,
    field_name: str,
) -> List[Any]:
    """Validate and return a list."""

    if not isinstance(value, list):
        raise Layer2OutputValidationError(
            f"{field_name} must be a list."
        )

    return list(value)


def make_json_safe(
    value: Any,
) -> Any:
    """Convert supported Python values to JSON-safe data."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


# ============================================================
# LAYER 2 OUTPUT SCHEMA
# ============================================================

@dataclass
class Layer2Output:
    """
    Final unified perception output sent to Layer 3.
    """

    timestamp: str
    scene: Dict[str, Any]
    objects: List[Dict[str, Any]]
    sounds: List[Dict[str, Any]]
    speech_transcript: List[str]
    user_activity: Dict[str, Any]

    location: Dict[str, Any] = field(
        default_factory= dict
        )

    recognized_text: List[Dict[str, Any]] = field(
        default_factory=list
    )

    depth: Dict[str, Any] = field(
        default_factory=dict
    )

    obstacles: List[Dict[str, Any]] = field(
        default_factory=list
    )

    perception_confidence: Dict[str, Any] = field(
        default_factory=dict
    )

    fusion: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    output_id: str = field(
        default_factory=generate_output_id
    )

    output_type: str = LAYER2_OUTPUT_TYPE
    schema_version: str = (
        LAYER2_OUTPUT_SCHEMA_VERSION
    )

    status: str = "complete"

    source_packet_id: Optional[str] = None
    processing_time_ms: float = 0.0

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.validate()

    @property
    def ready_for_layer3(self) -> bool:
        """Return whether Layer 3 can process the output."""

        return (
            self.status
            in {
                "complete",
                "partial",
                "degraded",
            }
            and bool(self.scene.get("type"))
        )

    @property
    def scene_type(self) -> str:
        """Return the detected scene type."""

        return str(
            self.scene.get("type", "")
        )

    @property
    def overall_confidence(
        self,
    ) -> Optional[float]:
        """Return the overall perception confidence."""

        value = self.perception_confidence.get(
            "overall"
        )

        if value is None:
            value = self.perception_confidence.get(
                "overall_confidence"
            )

        if value is None:
            return None

        return float(value)

    def validate(self) -> None:
        """Validate the complete Layer 2 output."""

        self.output_id = require_non_empty_string(
            self.output_id,
            "output_id",
        )

        self.output_type = (
            require_non_empty_string(
                self.output_type,
                "output_type",
            )
        )

        self.schema_version = (
            require_non_empty_string(
                self.schema_version,
                "schema_version",
            )
        )

        self.timestamp = validate_iso_timestamp(
            self.timestamp,
            "timestamp",
        )

        self.status = require_non_empty_string(
            self.status,
            "status",
        ).lower()

        if self.status not in ALLOWED_OUTPUT_STATUSES:
            raise Layer2OutputValidationError(
                f"Unsupported output status: "
                f"{self.status!r}"
            )

        self.scene = require_dictionary(
            self.scene,
            "scene",
        )

        scene_type = require_non_empty_string(
            self.scene.get("type"),
            "scene.type",
        )

        self.scene["type"] = scene_type

        scene_confidence = self.scene.get(
            "confidence"
        )

        if scene_confidence is not None:
            self.scene["confidence"] = (
                validate_confidence(
                    scene_confidence,
                    "scene.confidence",
                )
            )

        self.objects = require_list(
            self.objects,
            "objects",
        )

        self.sounds = require_list(
            self.sounds,
            "sounds",
        )

        self.speech_transcript = require_list(
            self.speech_transcript,
            "speech_transcript",
        )

        self.user_activity = (
            require_dictionary(
                self.user_activity,
                "user_activity",
            )
        )
        self.location = require_dictionary(
            self.location,
            "location",
        )

        self.recognized_text = require_list(
            self.recognized_text,
            "recognized_text",
        )

        self.depth = require_dictionary(
            self.depth,
            "depth",
        )

        self.obstacles = require_list(
            self.obstacles,
            "obstacles",
        )

        self.perception_confidence = (
            require_dictionary(
                self.perception_confidence,
                "perception_confidence",
            )
        )

        self.fusion = require_dictionary(
            self.fusion,
            "fusion",
        )

        self.metadata = require_dictionary(
            self.metadata,
            "metadata",
        )

        self._validate_objects()
        self._validate_sounds()
        self._validate_transcript()
        self._validate_user_activity()
        self._validate_location()
        self._validate_recognized_text()
        self._validate_depth()
        self._validate_obstacles()
        self._validate_perception_confidence()
        self._validate_fusion()
        self._validate_metadata()


        self.processing_time_ms = (
            validate_non_negative_number(
                self.processing_time_ms,
                "processing_time_ms",
            )
            or 0.0
        )

        if self.source_packet_id is not None:
            self.source_packet_id = (
                require_non_empty_string(
                    self.source_packet_id,
                    "source_packet_id",
                )
            )

        if not isinstance(self.warnings, list):
            raise Layer2OutputValidationError(
                "warnings must be a list."
            )

        if not isinstance(self.errors, list):
            raise Layer2OutputValidationError(
                "errors must be a list."
            )

        self.warnings = [
            str(item)
            for item in self.warnings
            if str(item).strip()
        ]

        self.errors = [
            str(item)
            for item in self.errors
            if str(item).strip()
        ]

        if self.status == "failed" and not self.errors:
            raise Layer2OutputValidationError(
                "A failed output must contain an error."
            )

    def _validate_objects(self) -> None:

        validated_objects = []

        for index, item in enumerate(self.objects):

            obj = require_dictionary(
                item,
                f"objects[{index}]",
            )

            obj["label"] = (
                require_non_empty_string(
                    obj.get("label"),
                    f"objects[{index}].label",
                )
            )

            if obj.get("confidence") is not None:
                obj["confidence"] = (
                    validate_confidence(
                        obj["confidence"],
                        (
                            f"objects[{index}]"
                            ".confidence"
                        ),
                    )
                )

            if obj.get("distance_m") is not None:
                obj["distance_m"] = (
                    validate_non_negative_number(
                        obj["distance_m"],
                        (
                            f"objects[{index}]"
                            ".distance_m"
                        ),
                    )
                )

            bounding_box = obj.get(
                "bounding_box"
            )

            if bounding_box is not None:
                if (
                    not isinstance(
                        bounding_box,
                        list,
                    )
                    or len(bounding_box) != 4
                ):
                    raise Layer2OutputValidationError(
                        f"objects[{index}].bounding_box "
                        "must contain four values."
                    )

                for coordinate in bounding_box:
                    if isinstance(
                        coordinate,
                        bool,
                    ) or not isinstance(
                        coordinate,
                        (int, float),
                    ):
                        raise Layer2OutputValidationError(
                            f"objects[{index}]"
                            ".bounding_box must be numeric."
                        )

            validated_objects.append(obj)

        self.objects = validated_objects

    def _validate_sounds(self) -> None:

        validated_sounds = []

        for index, item in enumerate(self.sounds):

            sound = require_dictionary(
                item,
                f"sounds[{index}]",
            )

            sound["label"] = (
                require_non_empty_string(
                    sound.get("label"),
                    f"sounds[{index}].label",
                )
            )

            if sound.get("confidence") is not None:
                sound["confidence"] = (
                    validate_confidence(
                        sound["confidence"],
                        (
                            f"sounds[{index}]"
                            ".confidence"
                        ),
                    )
                )

            validated_sounds.append(sound)

        self.sounds = validated_sounds

    def _validate_transcript(self) -> None:

        validated_transcript = []

        for index, item in enumerate(
            self.speech_transcript
        ):
            transcript = require_non_empty_string(
                item,
                f"speech_transcript[{index}]",
            )

            validated_transcript.append(
                transcript
            )

        self.speech_transcript = (
            validated_transcript
        )

    def _validate_user_activity(self) -> None:

        state = self.user_activity.get("state")

        if state is None:
            self.user_activity["state"] = "unknown"
        else:
            self.user_activity["state"] = (
                require_non_empty_string(
                    state,
                    "user_activity.state",
                )
            )

        if (
            self.user_activity.get("confidence")
            is not None
        ):
            self.user_activity["confidence"] = (
                validate_confidence(
                    self.user_activity[
                        "confidence"
                    ],
                    "user_activity.confidence",
                )
            )

    def _validate_location(self) -> None:
        """Validate optional GPS location."""

        latitude = self.location.get(
            "latitude"
        )

        longitude = self.location.get(
            "longitude"
        )

        if latitude is not None:

            if (
                isinstance(latitude, bool)
                or not isinstance(
                    latitude,
                    (int, float),
                )
            ):
                raise Layer2OutputValidationError(
                    "location.latitude must "
                    "be numeric."
                )

            latitude = float(
                latitude
            )

            if (
                not math.isfinite(latitude)
                or not -90.0
                <= latitude
                <= 90.0
            ):
                raise Layer2OutputValidationError(
                    "location.latitude must be "
                    "between -90 and 90."
                )

            self.location[
                "latitude"
            ] = latitude

        if longitude is not None:

            if (
                isinstance(longitude, bool)
                or not isinstance(
                    longitude,
                    (int, float),
                )
            ):
                raise Layer2OutputValidationError(
                    "location.longitude must "
                    "be numeric."
                )

            longitude = float(
                longitude
            )

            if (
                not math.isfinite(longitude)
                or not -180.0
                <= longitude
                <= 180.0
            ):
                raise Layer2OutputValidationError(
                    "location.longitude must be "
                    "between -180 and 180."
                )

            self.location[
                "longitude"
            ] = longitude

        accuracy_m = self.location.get(
            "accuracy_m"
        )

        if accuracy_m is not None:
            self.location["accuracy_m"] = (
                validate_non_negative_number(
                    accuracy_m,
                    "location.accuracy_m",
                )
            )
         
    def _validate_recognized_text(self) -> None:

        validated_text = []

        for index, item in enumerate(
            self.recognized_text
        ):

            if isinstance(item, str):
                text_item = {
                    "text": require_non_empty_string(
                        item,
                        (
                            f"recognized_text[{index}]"
                        ),
                    )
                }
            else:
                text_item = require_dictionary(
                    item,
                    f"recognized_text[{index}]",
                )

                text_item["text"] = (
                    require_non_empty_string(
                        text_item.get("text"),
                        (
                            f"recognized_text[{index}]"
                            ".text"
                        ),
                    )
                )

                if (
                    text_item.get("confidence")
                    is not None
                ):
                    text_item["confidence"] = (
                        validate_confidence(
                            text_item["confidence"],
                            (
                                f"recognized_text[{index}]"
                                ".confidence"
                            ),
                        )
                    )

            validated_text.append(text_item)

        self.recognized_text = validated_text

    def _validate_depth(self) -> None:

        for field_name in (
            "nearest_obstacle_m",
            "minimum_depth_m",
            "maximum_depth_m",
        ):
            if self.depth.get(field_name) is not None:
                self.depth[field_name] = (
                    validate_non_negative_number(
                        self.depth[field_name],
                        f"depth.{field_name}",
                    )
                )

    def _validate_obstacles(self) -> None:

        validated_obstacles = []

        for index, item in enumerate(
            self.obstacles
        ):
            obstacle = require_dictionary(
                item,
                f"obstacles[{index}]",
            )

            obstacle["label"] = (
                require_non_empty_string(
                    obstacle.get(
                        "label",
                        "obstacle",
                    ),
                    (
                        f"obstacles[{index}]"
                        ".label"
                    ),
                )
            )

            if (
                obstacle.get("confidence")
                is not None
            ):
                obstacle["confidence"] = (
                    validate_confidence(
                        obstacle["confidence"],
                        (
                            f"obstacles[{index}]"
                            ".confidence"
                        ),
                    )
                )

            if obstacle.get("distance_m") is not None:
                obstacle["distance_m"] = (
                    validate_non_negative_number(
                        obstacle["distance_m"],
                        (
                            f"obstacles[{index}]"
                            ".distance_m"
                        ),
                    )
                )

            validated_obstacles.append(obstacle)

        self.obstacles = validated_obstacles

    def _validate_perception_confidence(
        self,
    ) -> None:

        for key, value in list(
            self.perception_confidence.items()
        ):
            if value is None:
                continue

            if isinstance(value, (int, float)):
                self.perception_confidence[key] = (
                    validate_confidence(
                        value,
                        (
                            "perception_confidence."
                            f"{key}"
                        ),
                    )
                )

    def _validate_fusion(self) -> None:

        status = self.fusion.get("status")

        if status is not None:
            self.fusion["status"] = (
                require_non_empty_string(
                    status,
                    "fusion.status",
                )
            )

        available_modalities = (
            self.fusion.get(
                "available_modalities"
            )
        )

        if available_modalities is not None:
            if not isinstance(
                available_modalities,
                list,
            ):
                raise Layer2OutputValidationError(
                    "fusion.available_modalities "
                    "must be a list."
                )

            self.fusion[
                "available_modalities"
            ] = [
                require_non_empty_string(
                    modality,
                    (
                        "fusion.available_modalities"
                    ),
                )
                for modality
                in available_modalities
            ]

    def _validate_metadata(self) -> None:

        self.metadata.setdefault(
            "source_layer",
            "layer2",
        )

        self.metadata.setdefault(
            "destination_layer",
            "layer3",
        )

        self.metadata.setdefault(
            "ready_for_layer3",
            self.ready_for_layer3,
        )

        if (
            self.metadata["source_layer"]
            != "layer2"
        ):
            raise Layer2OutputValidationError(
                "metadata.source_layer must be "
                "'layer2'."
            )

        if (
            self.metadata["destination_layer"]
            != "layer3"
        ):
            raise Layer2OutputValidationError(
                "metadata.destination_layer must "
                "be 'layer3'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to a JSON-safe dictionary."""

        self.validate()

        return make_json_safe(
            asdict(self)
        )

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        """Convert output to formatted JSON."""

        try:
            return json.dumps(
                self.to_dict(),
                indent=indent,
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as error:
            raise Layer2OutputSerializationError(
                "Unable to serialize Layer 2 output."
            ) from error

    def write_json(
        self,
        file_path: Path | str,
    ) -> Path:
        """Write output to a JSON file."""

        output_path = Path(file_path)

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_text(
                self.to_json() + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise Layer2OutputWriteError(
                f"Unable to write Layer 2 output: "
                f"{output_path}"
            ) from error

        return output_path

    def summary(self) -> Dict[str, Any]:
        """Return a compact output summary."""

        return {
            "output_id": self.output_id,
            "source_packet_id": (
                self.source_packet_id
            ),
            "status": self.status,
            "ready_for_layer3": (
                self.ready_for_layer3
            ),
            "scene_type": self.scene_type,
            "object_count": len(self.objects),
            "sound_count": len(self.sounds),
            "transcript_count": len(
                self.speech_transcript
            ),
            "recognized_text_count": len(
                self.recognized_text
            ),
            "obstacle_count": len(
                self.obstacles
            ),
            "user_activity": (
                self.user_activity.get("state")
            ),
            "location_available": bool(
                self.location),
            "location": dict(
                self.location),
            "overall_confidence": (
                self.overall_confidence
            ),
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Layer2Output":
        """Create Layer2Output from a dictionary."""

        if not isinstance(payload, Mapping):
            raise Layer2OutputValidationError(
                "Layer 2 output payload must be "
                "a dictionary."
            )

        metadata = dict(
            payload.get("metadata", {})
        )

        metadata.setdefault(
            "source_layer",
            "layer2",
        )

        metadata.setdefault(
            "destination_layer",
            "layer3",
        )

        return cls(
            timestamp=payload.get(
                "timestamp",
                utc_now_iso(),
            ),
            scene=dict(
                payload.get("scene", {})
            ),
            objects=list(
                payload.get("objects", [])
            ),
            sounds=list(
                payload.get("sounds", [])
            ),
            speech_transcript=list(
                payload.get(
                    "speech_transcript",
                    [],
                )
            ),
            user_activity=dict(
                payload.get(
                    "user_activity",
                    {"state": "unknown"},
                )
            ),
            location=dict(
                payload.get("location", {})
            ),
            recognized_text=list(
                payload.get(
                    "recognized_text",
                    [],
                )
            ),
            depth=dict(
                payload.get("depth", {})
            ),
            obstacles=list(
                payload.get("obstacles", [])
            ),
            perception_confidence=dict(
                payload.get(
                    "perception_confidence",
                    {},
                )
            ),
            fusion=dict(
                payload.get("fusion", {})
            ),
            metadata=metadata,
            output_id=payload.get(
                "output_id",
                generate_output_id(),
            ),
            output_type=payload.get(
                "output_type",
                LAYER2_OUTPUT_TYPE,
            ),
            schema_version=payload.get(
                "schema_version",
                LAYER2_OUTPUT_SCHEMA_VERSION,
            ),
            status=payload.get(
                "status",
                "complete",
            ),
            source_packet_id=payload.get(
                "source_packet_id"
            ),
            processing_time_ms=payload.get(
                "processing_time_ms",
                0.0,
            ),
            warnings=list(
                payload.get("warnings", [])
            ),
            errors=list(
                payload.get("errors", [])
            ),
        )


# ============================================================
# JSON LOADING
# ============================================================

def load_layer2_output(
    file_path: Path | str,
) -> Layer2Output:
    """Load and validate a Layer 2 output file."""

    path = Path(file_path)

    if not path.exists():
        raise Layer2OutputError(
            f"Layer 2 output does not exist: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise Layer2OutputSerializationError(
            f"Invalid JSON in {path}: "
            f"line {error.lineno}, "
            f"column {error.colno}."
        ) from error
    except OSError as error:
        raise Layer2OutputError(
            f"Unable to read {path}"
        ) from error

    return Layer2Output.from_dict(payload)


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "park_walking",
) -> bool:
    """Test using an expected Layer 2 scenario."""

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    expected_output_path = (
        project_root
        / "layer1_output_test_scenarios"
        / scenario_name
        / "expected_layer2_output.json"
    )

    test_output_path = (
        project_root
        / "output"
        / "layer2"
        / "schema_self_test"
        / "layer2_output.json"
    )

    print("=" * 72)
    print("NOONGIL-X | LAYER 2 OUTPUT SCHEMA SELF-TEST")
    print("=" * 72)
    print(f"Scenario : {scenario_name}")
    print(f"Input    : {expected_output_path}")

    try:
        output = load_layer2_output(
            expected_output_path
        )

        print("\n[PASS] Expected output loaded")
        print("[PASS] Required Layer 3 fields validated")
        print("[PASS] Scene structure validated")
        print("[PASS] Objects validated")
        print("[PASS] Sounds validated")
        print("[PASS] Speech transcript validated")
        print("[PASS] User activity validated")

        written_path = output.write_json(
            test_output_path
        )

        print(
            f"[PASS] Output serialized: "
            f"{written_path}"
        )

        restored = load_layer2_output(
            written_path
        )

        if (
            restored.scene_type
            != output.scene_type
        ):
            raise AssertionError(
                "Scene type changed after serialization."
            )

        if (
            len(restored.objects)
            != len(output.objects)
        ):
            raise AssertionError(
                "Object count changed after serialization."
            )

        if not restored.ready_for_layer3:
            raise AssertionError(
                "Output is not ready for Layer 3."
            )

        print("[PASS] Serialized output restored")
        print("[PASS] Layer 3 compatibility confirmed")

        print("\nOutput summary:")

        for key, value in output.summary().items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 2 OUTPUT SCHEMA IS WORKING")
        print("=" * 72)

        return True

    except (
        Layer2OutputError,
        AssertionError,
    ) as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Validate the unified Layer 2 output "
            "schema."
        )
    )

    parser.add_argument(
        "--scenario",
        default="park_walking",
        help=(
            "Scenario used by the self-test. "
            "Default: park_walking"
        ),
    )

    return parser


def main() -> int:

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    return (
        0
        if run_self_test(arguments.scenario)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())