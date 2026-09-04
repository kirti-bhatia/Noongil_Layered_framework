"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : AI Model Configuration
File    : layer2/config/model_config.py
============================================================

Purpose
-------
Defines model specifications for:

- Scene classification
- Object detection
- Object tracking
- Visual activity recognition
- OCR
- Speech recognition
- Sound-event detection
- Depth estimation
- Multimodal embeddings

Runtime thresholds remain in config/settings.py.
Actual model loading will be handled by utils/model_loader.py.

Important
---------
Running this file does not download or load models.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import json
import os

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


# ============================================================
# CONSTANTS
# ============================================================

MODEL_CONFIG_VERSION = "1.0"

SUPPORTED_MODEL_TASKS = {
    "scene_classification",
    "object_detection",
    "object_tracking",
    "activity_recognition",
    "ocr",
    "speech_recognition",
    "sound_event_detection",
    "depth_estimation",
    "multimodal_embedding",
}

SUPPORTED_DEVICES = {
    "auto",
    "cpu",
    "cuda",
    "mps",
}

SUPPORTED_PRECISIONS = {
    "auto",
    "float32",
    "float16",
    "bfloat16",
    "int8",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelConfigError(Exception):
    """Base exception for model configuration."""


class ModelConfigValidationError(
    ModelConfigError
):
    """Raised when model configuration is invalid."""


class ModelConfigSerializationError(
    ModelConfigError
):
    """Raised when model configuration cannot be serialized."""


# ============================================================
# ENUMERATIONS
# ============================================================

class ModelBackend(str, Enum):
    """Supported model-loading backends."""

    ULTRALYTICS = "ultralytics"
    HUGGINGFACE = "huggingface"
    PADDLEOCR = "paddleocr"
    OPENAI_WHISPER = "openai_whisper"
    OPENCV = "opencv"
    INTERNAL = "internal"


# ============================================================
# VALIDATION HELPERS
# ============================================================

def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ModelConfigValidationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def validate_optional_string(
    value: Any,
    field_name: str,
) -> Optional[str]:
    """Validate an optional non-empty string."""

    if value is None:
        return None

    return require_non_empty_string(
        value,
        field_name,
    )


def validate_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """Validate a positive integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ModelConfigValidationError(
            f"{field_name} must be a positive integer."
        )

    return value


# ============================================================
# INDIVIDUAL MODEL SPECIFICATION
# ============================================================

@dataclass
class ModelSpec:
    """
    Configuration for one Layer 2 AI model.

    Parameters
    ----------
    task:
        Layer 2 task performed by the model.

    model_id:
        Model identifier, local filename or repository ID.

    backend:
        Library used to load the model.

    enabled:
        Whether this model may be used by the pipeline.

    local_path:
        Optional explicit local model path.

    allow_download:
        Whether the model loader may download missing files.
    """

    task: str
    model_id: str
    backend: ModelBackend

    enabled: bool = True
    required: bool = False

    local_path: Optional[str] = None
    revision: Optional[str] = None

    device: str = "auto"
    precision: str = "auto"

    allow_download: bool = True
    trust_remote_code: bool = False

    batch_size: int = 1

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    description: str = ""

    def __post_init__(self) -> None:

        if isinstance(self.backend, str):
            try:
                self.backend = ModelBackend(
                    self.backend.lower()
                )
            except ValueError as error:
                raise ModelConfigValidationError(
                    f"Unsupported model backend: "
                    f"{self.backend!r}"
                ) from error

        self.validate()

    def validate(self) -> None:
        """Validate one model specification."""

        self.task = require_non_empty_string(
            self.task,
            "model.task",
        ).lower()

        if self.task not in SUPPORTED_MODEL_TASKS:
            raise ModelConfigValidationError(
                f"Unsupported model task: "
                f"{self.task!r}"
            )

        self.model_id = (
            require_non_empty_string(
                self.model_id,
                f"{self.task}.model_id",
            )
        )

        if not isinstance(
            self.backend,
            ModelBackend,
        ):
            raise ModelConfigValidationError(
                f"{self.task}.backend must be "
                "a ModelBackend."
            )

        if not isinstance(self.enabled, bool):
            raise ModelConfigValidationError(
                f"{self.task}.enabled must be boolean."
            )

        if not isinstance(self.required, bool):
            raise ModelConfigValidationError(
                f"{self.task}.required must be boolean."
            )

        if not isinstance(
            self.allow_download,
            bool,
        ):
            raise ModelConfigValidationError(
                f"{self.task}.allow_download "
                "must be boolean."
            )

        if not isinstance(
            self.trust_remote_code,
            bool,
        ):
            raise ModelConfigValidationError(
                f"{self.task}.trust_remote_code "
                "must be boolean."
            )

        self.local_path = validate_optional_string(
            self.local_path,
            f"{self.task}.local_path",
        )

        self.revision = validate_optional_string(
            self.revision,
            f"{self.task}.revision",
        )

        self.device = require_non_empty_string(
            self.device,
            f"{self.task}.device",
        ).lower()

        if self.device not in SUPPORTED_DEVICES:
            raise ModelConfigValidationError(
                f"Unsupported device for {self.task}: "
                f"{self.device!r}"
            )

        self.precision = require_non_empty_string(
            self.precision,
            f"{self.task}.precision",
        ).lower()

        if self.precision not in SUPPORTED_PRECISIONS:
            raise ModelConfigValidationError(
                f"Unsupported precision for "
                f"{self.task}: {self.precision!r}"
            )

        self.batch_size = validate_positive_integer(
            self.batch_size,
            f"{self.task}.batch_size",
        )

        if not isinstance(self.parameters, dict):
            raise ModelConfigValidationError(
                f"{self.task}.parameters must be "
                "a dictionary."
            )

        if not isinstance(self.description, str):
            raise ModelConfigValidationError(
                f"{self.task}.description must be "
                "a string."
            )

    @property
    def source(self) -> str:
        """Return the effective model source."""

        return self.local_path or self.model_id

    @property
    def uses_local_path(self) -> bool:
        """Return whether a local path is configured."""

        return self.local_path is not None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe model specification."""

        self.validate()

        payload = asdict(self)
        payload["backend"] = self.backend.value

        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelSpec":
        """Create a model specification from a dictionary."""

        if not isinstance(payload, Mapping):
            raise ModelConfigValidationError(
                "Model specification must be "
                "a dictionary."
            )

        return cls(
            task=payload.get("task", ""),
            model_id=payload.get(
                "model_id",
                "",
            ),
            backend=payload.get(
                "backend",
                ModelBackend.INTERNAL,
            ),
            enabled=payload.get(
                "enabled",
                True,
            ),
            required=payload.get(
                "required",
                False,
            ),
            local_path=payload.get(
                "local_path"
            ),
            revision=payload.get(
                "revision"
            ),
            device=payload.get(
                "device",
                "auto",
            ),
            precision=payload.get(
                "precision",
                "auto",
            ),
            allow_download=payload.get(
                "allow_download",
                True,
            ),
            trust_remote_code=payload.get(
                "trust_remote_code",
                False,
            ),
            batch_size=payload.get(
                "batch_size",
                1,
            ),
            parameters=dict(
                payload.get(
                    "parameters",
                    {},
                )
            ),
            description=payload.get(
                "description",
                "",
            ),
        )


# ============================================================
# COMPLETE MODEL CONFIGURATION
# ============================================================

@dataclass
class Layer2ModelConfig:
    """Complete Layer 2 model registry."""

    scene_classification: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="scene_classification",
            model_id=(
                "openai/clip-vit-base-patch32"
            ),
            backend=ModelBackend.HUGGINGFACE,
            required=True,
            parameters={
                "candidate_labels": [
                    "park",
                    "classroom",
                    "shopping mall",
                    "cafe",
                    "home",
                    "street",
                    "road",
                    "hospital",
                    "office",
                    "unknown environment",
                ]
            },
            description=(
                "Zero-shot visual scene classification."
            ),
        )
    )

    object_detection: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="object_detection",
            model_id="yolov8n.pt",
            backend=ModelBackend.ULTRALYTICS,
            required=True,
            parameters={
                "image_size": 640,
                "agnostic_nms": False,
            },
            description=(
                "Lightweight real-time object detection."
            ),
        )
    )

    object_tracking: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="object_tracking",
            model_id="bytetrack.yaml",
            backend=ModelBackend.ULTRALYTICS,
            required=False,
            parameters={
                "persist_tracks": True,
            },
            description=(
                "Multi-frame object identity tracking."
            ),
        )
    )

    activity_recognition: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="activity_recognition",
            model_id=(
                "MCG-NJU/"
                "videomae-base-finetuned-kinetics"
            ),
            backend=ModelBackend.HUGGINGFACE,
            required=False,
            parameters={
                "number_of_frames": 16,
                "visual_fallback_enabled": True,
            },
            description=(
                "Visual activity recognition from "
                "short frame sequences."
            ),
        )
    )

    ocr: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="ocr",
            model_id="paddleocr-en",
            backend=ModelBackend.PADDLEOCR,
            required=False,
            parameters={
                "language": "en",
                # "use_angle_classifier": True,
                "use_angle_cls": True,
                "show_log": False,
            },
            description=(
                "Environmental text detection "
                "and recognition."
            ),
        )
    )

    speech_recognition: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="speech_recognition",
            model_id="small",
            backend=ModelBackend.OPENAI_WHISPER,
            required=False,
            parameters={
                "language": "en",
                "task": "transcribe",
                "temperature": 0.0,
            },
            description=(
                "Speech-to-text transcription."
            ),
        )
    )

    sound_event_detection: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="sound_event_detection",
            model_id=(
                "MIT/"
                "ast-finetuned-audioset-10-10-0.4593"
            ),
            backend=ModelBackend.HUGGINGFACE,
            required=False,
            parameters={
                "maximum_labels": 5,
            },
            description=(
                "Environmental sound-event "
                "classification."
            ),
        )
    )

    depth_estimation: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="depth_estimation",
            model_id=(
                "depth-anything/"
                "Depth-Anything-V2-Small-hf"
            ),
            backend=ModelBackend.HUGGINGFACE,
            required=False,
            parameters={
                "output_type": "relative_depth",
                "normalize_output": True,
            },
            description=(
                "Monocular relative-depth estimation."
            ),
        )
    )

    multimodal_embedding: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            task="multimodal_embedding",
            model_id=(
                "openai/clip-vit-base-patch32"
            ),
            backend=ModelBackend.HUGGINGFACE,
            required=False,
            parameters={
                "normalize_embeddings": True,
            },
            description=(
                "Shared image-text embedding support."
            ),
        )
    )

    cache_directory: str = (
        "models/layer2/cache"
    )

    offline_mode: bool = False
    config_version: str = (
        MODEL_CONFIG_VERSION
    )

    def __post_init__(self) -> None:
        self.validate()

    def model_registry(
        self,
    ) -> Dict[str, ModelSpec]:
        """Return all configured models by task."""

        return {
            "scene_classification": (
                self.scene_classification
            ),
            "object_detection": (
                self.object_detection
            ),
            "object_tracking": (
                self.object_tracking
            ),
            "activity_recognition": (
                self.activity_recognition
            ),
            "ocr": self.ocr,
            "speech_recognition": (
                self.speech_recognition
            ),
            "sound_event_detection": (
                self.sound_event_detection
            ),
            "depth_estimation": (
                self.depth_estimation
            ),
            "multimodal_embedding": (
                self.multimodal_embedding
            ),
        }

    def validate(self) -> None:
        """Validate the complete model registry."""

        self.config_version = (
            require_non_empty_string(
                self.config_version,
                "config_version",
            )
        )

        self.cache_directory = (
            require_non_empty_string(
                self.cache_directory,
                "cache_directory",
            )
        )

        if not isinstance(self.offline_mode, bool):
            raise ModelConfigValidationError(
                "offline_mode must be boolean."
            )

        registry = self.model_registry()

        if set(registry) != SUPPORTED_MODEL_TASKS:
            raise ModelConfigValidationError(
                "Model registry tasks do not match "
                "the supported Layer 2 tasks."
            )

        for task, model in registry.items():

            if not isinstance(model, ModelSpec):
                raise ModelConfigValidationError(
                    f"{task} must be a ModelSpec."
                )

            model.validate()

            if model.task != task:
                raise ModelConfigValidationError(
                    f"Registry key {task!r} does not "
                    f"match model task {model.task!r}."
                )

            if (
                self.offline_mode
                and model.enabled
                and not model.uses_local_path
            ):
                model.allow_download = False

    def enabled_models(
        self,
    ) -> Dict[str, ModelSpec]:
        """Return all enabled models."""

        return {
            task: model
            for task, model
            in self.model_registry().items()
            if model.enabled
        }

    def required_models(
        self,
    ) -> Dict[str, ModelSpec]:
        """Return all required models."""

        return {
            task: model
            for task, model
            in self.model_registry().items()
            if model.enabled and model.required
        }

    def get(
        self,
        task: str,
    ) -> ModelSpec:
        """Return the model for a task."""

        normalized_task = (
            require_non_empty_string(
                task,
                "task",
            ).lower()
        )

        registry = self.model_registry()

        if normalized_task not in registry:
            raise ModelConfigValidationError(
                f"No model configured for task: "
                f"{normalized_task!r}"
            )

        return registry[normalized_task]

    def resolve_cache_directory(
        self,
        project_root: Path | str,
    ) -> Path:
        """Resolve the model cache directory."""

        cache_path = Path(
            self.cache_directory
        )

        if cache_path.is_absolute():
            return cache_path.resolve()

        return (
            Path(project_root)
            / cache_path
        ).resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete configuration."""

        self.validate()

        return {
            "config_version": (
                self.config_version
            ),
            "cache_directory": (
                self.cache_directory
            ),
            "offline_mode": (
                self.offline_mode
            ),
            "models": {
                task: model.to_dict()
                for task, model
                in self.model_registry().items()
            },
        }

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        """Serialize model configuration."""

        try:
            return json.dumps(
                self.to_dict(),
                indent=indent,
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as error:
            raise ModelConfigSerializationError(
                "Unable to serialize model "
                "configuration."
            ) from error

    def write_json(
        self,
        file_path: Path | str,
    ) -> Path:
        """Write model configuration to JSON."""

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
            raise ModelConfigSerializationError(
                f"Unable to write model configuration: "
                f"{output_path}"
            ) from error

        return output_path

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Layer2ModelConfig":
        """Create configuration from a dictionary."""

        if not isinstance(payload, Mapping):
            raise ModelConfigValidationError(
                "Model configuration must be "
                "a dictionary."
            )

        model_payloads = payload.get(
            "models",
            {},
        )

        if not isinstance(
            model_payloads,
            Mapping,
        ):
            raise ModelConfigValidationError(
                "models must be a dictionary."
            )

        defaults = cls()

        model_specs: Dict[str, ModelSpec] = {}

        for task, default_model in (
            defaults.model_registry().items()
        ):
            configured_model = (
                model_payloads.get(task)
            )

            if configured_model is None:
                model_specs[task] = default_model
            else:
                model_specs[task] = (
                    ModelSpec.from_dict(
                        configured_model
                    )
                )

        return cls(
            scene_classification=(
                model_specs[
                    "scene_classification"
                ]
            ),
            object_detection=(
                model_specs["object_detection"]
            ),
            object_tracking=(
                model_specs["object_tracking"]
            ),
            activity_recognition=(
                model_specs[
                    "activity_recognition"
                ]
            ),
            ocr=model_specs["ocr"],
            speech_recognition=(
                model_specs[
                    "speech_recognition"
                ]
            ),
            sound_event_detection=(
                model_specs[
                    "sound_event_detection"
                ]
            ),
            depth_estimation=(
                model_specs["depth_estimation"]
            ),
            multimodal_embedding=(
                model_specs[
                    "multimodal_embedding"
                ]
            ),
            cache_directory=payload.get(
                "cache_directory",
                "models/layer2/cache",
            ),
            offline_mode=payload.get(
                "offline_mode",
                False,
            ),
            config_version=payload.get(
                "config_version",
                MODEL_CONFIG_VERSION,
            ),
        )


# ============================================================
# CONFIGURATION FACTORIES
# ============================================================

def create_default_model_config(
) -> Layer2ModelConfig:
    """Create the default model configuration."""

    offline_value = os.getenv(
        "NOONGIL_LAYER2_OFFLINE",
        "false",
    ).strip().lower()

    offline_mode = offline_value in {
        "1",
        "true",
        "yes",
        "on",
    }

    cache_directory = os.getenv(
        "NOONGIL_LAYER2_MODEL_CACHE",
        "models/layer2/cache",
    )

    config = Layer2ModelConfig(
        cache_directory=cache_directory,
        offline_mode=offline_mode,
    )

    config.validate()
    return config


def create_test_model_config(
) -> Layer2ModelConfig:
    """
    Create a configuration for unit tests.

    Models stay configured but downloads are disabled.
    """

    config = Layer2ModelConfig(
        cache_directory=(
            "models/layer2/test_cache"
        ),
        offline_mode=True,
    )

    for model in (
        config.model_registry().values()
    ):
        model.allow_download = False

    config.validate()
    return config


def load_model_config(
    file_path: Path | str,
) -> Layer2ModelConfig:
    """Load model configuration from JSON."""

    path = Path(file_path)

    if not path.exists():
        raise ModelConfigError(
            f"Model configuration does not exist: "
            f"{path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ModelConfigSerializationError(
            f"Invalid JSON in {path}: "
            f"line {error.lineno}, "
            f"column {error.colno}."
        ) from error
    except OSError as error:
        raise ModelConfigSerializationError(
            f"Unable to read model configuration: "
            f"{path}"
        ) from error

    return Layer2ModelConfig.from_dict(
        payload
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | MODEL CONFIGURATION SELF-TEST")
    print("=" * 72)
    print(
        "This test validates configuration only. "
        "No models will be downloaded."
    )

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    test_output_path = (
        project_root
        / "output"
        / "layer2"
        / "config_self_test"
        / "model_config.json"
    )

    try:
        config = create_default_model_config()

        print("\n[PASS] Default model configuration created")
        print("[PASS] Model registry validated")

        registry = config.model_registry()

        if len(registry) != 9:
            raise AssertionError(
                "Expected nine model tasks."
            )

        print("[PASS] Nine model tasks registered")

        required = config.required_models()

        if not required:
            raise AssertionError(
                "No required models were configured."
            )

        print("[PASS] Required models identified")

        test_config = create_test_model_config()

        if not test_config.offline_mode:
            raise AssertionError(
                "Test model configuration must "
                "use offline mode."
            )

        for model in (
            test_config.model_registry().values()
        ):
            if model.allow_download:
                raise AssertionError(
                    "Test configuration unexpectedly "
                    "allows model downloads."
                )

        print("[PASS] Offline test configuration created")
        print("[PASS] Test downloads disabled")

        written_path = config.write_json(
            test_output_path
        )

        print(
            f"[PASS] Configuration serialized: "
            f"{written_path}"
        )

        restored = load_model_config(
            written_path
        )

        if restored.to_dict() != config.to_dict():
            raise AssertionError(
                "Restored model configuration does "
                "not match the original."
            )

        print("[PASS] Configuration restored from JSON")

        scene_model = restored.get(
            "scene_classification"
        )

        if scene_model.backend != (
            ModelBackend.HUGGINGFACE
        ):
            raise AssertionError(
                "Scene backend was not preserved."
            )

        print("[PASS] Task lookup verified")

        cache_path = (
            restored.resolve_cache_directory(
                project_root
            )
        )

        print("\nModel summary:")

        for task, model in (
            restored.model_registry().items()
        ):
            print(
                f"  {task}: "
                f"{model.model_id} "
                f"[{model.backend.value}]"
            )

        print(
            f"\n  cache directory: {cache_path}"
        )
        print(
            f"  offline mode: "
            f"{restored.offline_mode}"
        )
        print(
            f"  enabled models: "
            f"{len(restored.enabled_models())}"
        )
        print(
            f"  required models: "
            f"{len(restored.required_models())}"
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] MODEL CONFIGURATION IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        ModelConfigError,
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
            "Run the NOONGIL-X Layer 2 model "
            "configuration self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())