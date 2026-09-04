"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Lazy AI Model Loader
File    : layer2/utils/model_loader.py
============================================================

Purpose
-------
Provides centralized model management:

- Dependency inspection
- CPU, CUDA and MPS device resolution
- Lazy model loading
- In-memory model caching
- Duplicate-load prevention
- Offline-mode enforcement
- Model unloading
- Model-status reporting
- Backend-specific loading

Supported backends:
- Ultralytics
- Hugging Face Transformers
- PaddleOCR
- OpenAI Whisper
- OpenCV
- Internal/custom models

The self-test uses an internal mock model. It does not download
or load any external AI model.

Compatibility
-------------
Python 3.10+
============================================================
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import math
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from layer2.config.model_config import (
    Layer2ModelConfig,
    ModelBackend,
    ModelSpec,
    create_default_model_config,
    create_test_model_config,
)

from layer2.utils.exceptions import (
    DependencyMissingError,
    ModelLoadingError,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    get_logger,
    log_event,
    log_exception,
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_LOADER_VERSION = "1.0"

BACKEND_DEPENDENCIES = {
    ModelBackend.ULTRALYTICS: (
        "ultralytics",
    ),
    ModelBackend.HUGGINGFACE: (
        "transformers",
    ),
    ModelBackend.PADDLEOCR: (
        "paddleocr",
        "paddle",
    ),
    ModelBackend.OPENAI_WHISPER: (
        "whisper",
    ),
    ModelBackend.OPENCV: (
        "cv2",
    ),
    ModelBackend.INTERNAL: (),
}

HUGGINGFACE_PIPELINE_TASKS = {
    "scene_classification": (
        "zero-shot-image-classification"
    ),
    "activity_recognition": (
        "video-classification"
    ),
    "sound_event_detection": (
        "audio-classification"
    ),
    "depth_estimation": (
        "depth-estimation"
    ),
}


# ============================================================
# ENUMERATIONS
# ============================================================

class ModelState(str, Enum):
    """Runtime state of one model."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"
    UNLOADED = "unloaded"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ModelBundle:
    """
    Model and optional processor/tokenizer combination.
    """

    model: Any
    processor: Any = None
    tokenizer: Any = None
    backend: Optional[str] = None
    task: Optional[str] = None
    device: str = "cpu"
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ModelLoadRecord:
    """Runtime information for one configured model."""

    task: str
    model_id: str
    backend: str
    state: ModelState

    device: str
    precision: str

    load_attempts: int = 0
    load_time_ms: Optional[float] = None
    loaded_at: Optional[str] = None

    cache_hits: int = 0
    error: Optional[str] = None

    required: bool = False
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable record."""

        return {
            "task": self.task,
            "model_id": self.model_id,
            "backend": self.backend,
            "state": self.state.value,
            "device": self.device,
            "precision": self.precision,
            "load_attempts": self.load_attempts,
            "load_time_ms": self.load_time_ms,
            "loaded_at": self.loaded_at,
            "cache_hits": self.cache_hits,
            "error": self.error,
            "required": self.required,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class DependencyStatus:
    """Installation state of one dependency."""

    module_name: str
    installed: bool
    version: Optional[str]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "installed": self.installed,
            "version": self.version,
            "error": self.error,
        }


@dataclass
class ModelLoaderStatistics:
    """Aggregate model-loader statistics."""

    total_load_requests: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    cache_hits: int = 0
    total_unloads: int = 0

    cumulative_load_time_ms: float = 0.0

    @property
    def average_load_time_ms(self) -> float:

        if self.successful_loads == 0:
            return 0.0

        return round(
            self.cumulative_load_time_ms
            / self.successful_loads,
            3,
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "total_load_requests": (
                self.total_load_requests
            ),
            "successful_loads": (
                self.successful_loads
            ),
            "failed_loads": self.failed_loads,
            "cache_hits": self.cache_hits,
            "total_unloads": self.total_unloads,
            "cumulative_load_time_ms": round(
                self.cumulative_load_time_ms,
                3,
            ),
            "average_load_time_ms": (
                self.average_load_time_ms
            ),
        }


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:

    from datetime import datetime, timezone

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="milliseconds"
    )


def module_is_installed(
    module_name: str,
) -> bool:
    """Check whether a Python module is installed."""

    try:
        return (
            importlib.util.find_spec(
                module_name
            )
            is not None
        )
    except (
        ImportError,
        ModuleNotFoundError,
        ValueError,
    ):
        return False


def inspect_dependency(
    module_name: str,
) -> DependencyStatus:
    """Inspect one dependency without requiring it."""

    if not module_is_installed(module_name):
        return DependencyStatus(
            module_name=module_name,
            installed=False,
            version=None,
            error=None,
        )

    version = None
    error_message = None

    try:
        module = importlib.import_module(
            module_name
        )

        version_value = getattr(
            module,
            "__version__",
            None,
        )

        if version_value is not None:
            version = str(version_value)

    except Exception as error:
        error_message = (
            f"{error.__class__.__name__}: "
            f"{error}"
        )

    return DependencyStatus(
        module_name=module_name,
        installed=True,
        version=version,
        error=error_message,
    )


# ============================================================
# DEVICE RESOLUTION
# ============================================================

def resolve_device(
    requested_device: str = "auto",
) -> str:
    """Resolve the best available execution device."""

    normalized = (
        requested_device.strip().lower()
    )

    if normalized not in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }:
        raise ModelLoadingError(
            f"Unsupported device: "
            f"{requested_device!r}",
            module="model_loader",
            details={
                "requested_device": (
                    requested_device
                )
            },
        )

    if normalized == "cpu":
        return "cpu"

    if not module_is_installed("torch"):

        if normalized == "auto":
            return "cpu"

        raise DependencyMissingError(
            "PyTorch is required to use "
            f"device {normalized!r}.",
            module="model_loader",
            details={
                "dependency": "torch",
                "requested_device": normalized,
            },
        )

    try:
        torch = importlib.import_module(
            "torch"
        )

        cuda_available = bool(
            torch.cuda.is_available()
        )

        mps_backend = getattr(
            torch.backends,
            "mps",
            None,
        )

        mps_available = bool(
            mps_backend is not None
            and mps_backend.is_available()
        )

    except Exception as error:

        if normalized == "auto":
            return "cpu"

        raise ModelLoadingError(
            "Unable to inspect PyTorch devices.",
            module="model_loader",
            details={
                "requested_device": normalized
            },
            cause=error,
        ) from error

    if normalized == "cuda":

        if not cuda_available:
            raise ModelLoadingError(
                "CUDA was requested but is "
                "not available.",
                module="model_loader",
                details={
                    "requested_device": "cuda"
                },
            )

        return "cuda"

    if normalized == "mps":

        if not mps_available:
            raise ModelLoadingError(
                "MPS was requested but is "
                "not available.",
                module="model_loader",
                details={
                    "requested_device": "mps"
                },
            )

        return "mps"

    if cuda_available:
        return "cuda"

    if mps_available:
        return "mps"

    return "cpu"


# ============================================================
# MODEL LOADER
# ============================================================

BackendLoader = Callable[
    [ModelSpec, str, Path, bool],
    Any,
]


class ModelLoader:
    """
    Lazily load and cache Layer 2 models.
    """

    def __init__(
        self,
        model_config: Optional[
            Layer2ModelConfig
        ] = None,
        *,
        project_root: Optional[
            Path | str
        ] = None,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.model_config = (
            model_config
            or create_default_model_config()
        )

        self.model_config.validate()

        default_project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.project_root = Path(
            project_root
            or default_project_root
        ).resolve()

        self.cache_directory = (
            self.model_config
            .resolve_cache_directory(
                self.project_root
            )
        )

        self.logger = (
            logger
            or get_logger(
                "model_loader"
            )
        )

        self._models: Dict[str, Any] = {}
        self._records: Dict[
            str,
            ModelLoadRecord
        ] = {}

        self._lock = threading.RLock()

        self.statistics = (
            ModelLoaderStatistics()
        )

        self._backend_loaders: Dict[
            ModelBackend,
            BackendLoader,
        ] = {
            ModelBackend.ULTRALYTICS: (
                self._load_ultralytics
            ),
            ModelBackend.HUGGINGFACE: (
                self._load_huggingface
            ),
            ModelBackend.PADDLEOCR: (
                self._load_paddleocr
            ),
            ModelBackend.OPENAI_WHISPER: (
                self._load_whisper
            ),
            ModelBackend.OPENCV: (
                self._load_opencv
            ),
            ModelBackend.INTERNAL: (
                self._load_internal
            ),
        }

        self._initialize_records()

    def _initialize_records(self) -> None:

        for task, spec in (
            self.model_config
            .model_registry()
            .items()
        ):
            state = (
                ModelState.NOT_LOADED
                if spec.enabled
                else ModelState.DISABLED
            )

            self._records[task] = (
                ModelLoadRecord(
                    task=task,
                    model_id=spec.model_id,
                    backend=spec.backend.value,
                    state=state,
                    device=spec.device,
                    precision=spec.precision,
                    required=spec.required,
                    enabled=spec.enabled,
                )
            )

    def register_backend_loader(
        self,
        backend: ModelBackend,
        loader: BackendLoader,
    ) -> None:
        """Register or replace a backend loader."""

        if not isinstance(
            backend,
            ModelBackend,
        ):
            raise ModelLoadingError(
                "backend must be a ModelBackend.",
                module="model_loader",
            )

        if not callable(loader):
            raise ModelLoadingError(
                "Backend loader must be callable.",
                module="model_loader",
            )

        with self._lock:
            self._backend_loaders[
                backend
            ] = loader

    def dependency_report(
        self,
    ) -> Dict[str, DependencyStatus]:
        """Inspect dependencies used by enabled models."""

        dependency_names = set()

        for spec in (
            self.model_config
            .enabled_models()
            .values()
        ):
            dependency_names.update(
                BACKEND_DEPENDENCIES.get(
                    spec.backend,
                    (),
                )
            )

        dependency_names.add("torch")

        return {
            dependency: inspect_dependency(
                dependency
            )
            for dependency in sorted(
                dependency_names
            )
        }

    def check_backend_dependencies(
        self,
        backend: ModelBackend,
    ) -> None:
        """Ensure backend dependencies are installed."""

        required_dependencies = (
            BACKEND_DEPENDENCIES.get(
                backend,
                (),
            )
        )

        missing = [
            dependency
            for dependency
            in required_dependencies
            if not module_is_installed(
                dependency
            )
        ]

        if missing:
            raise DependencyMissingError(
                "Missing dependencies for "
                f"{backend.value}: "
                f"{', '.join(missing)}",
                module="model_loader",
                details={
                    "backend": backend.value,
                    "missing_dependencies": (
                        missing
                    ),
                },
            )

    def is_loaded(
        self,
        task: str,
    ) -> bool:
        """Return whether a model is cached."""

        with self._lock:
            return task in self._models

    def get_record(
        self,
        task: str,
    ) -> ModelLoadRecord:
        """Return one model load record."""

        if task not in self._records:
            raise ModelLoadingError(
                f"Unknown model task: {task!r}",
                module="model_loader",
            )

        return self._records[task]

    def load(
        self,
        task: str,
        *,
        force_reload: bool = False,
    ) -> Any:
        """Load or return the cached model."""

        normalized_task = (
            task.strip().lower()
            if isinstance(task, str)
            else ""
        )

        if not normalized_task:
            raise ModelLoadingError(
                "task must be a non-empty string.",
                module="model_loader",
            )

        with self._lock:

            self.statistics.total_load_requests += 1

            spec = self.model_config.get(
                normalized_task
            )

            record = self.get_record(
                normalized_task
            )

            if not spec.enabled:
                record.state = ModelState.DISABLED

                raise ModelLoadingError(
                    f"Model task "
                    f"{normalized_task!r} is disabled.",
                    module="model_loader",
                    recoverable=True,
                    details={
                        "task": normalized_task
                    },
                )

            if (
                normalized_task in self._models
                and not force_reload
            ):
                record.cache_hits += 1
                self.statistics.cache_hits += 1

                log_event(
                    self.logger,
                    event="model_cache_hit",
                    message=(
                        f"Using cached model for "
                        f"{normalized_task}."
                    ),
                    details={
                        "task": normalized_task,
                        "model_id": spec.model_id,
                    },
                )

                return self._models[
                    normalized_task
                ]

            if (
                force_reload
                and normalized_task
                in self._models
            ):
                self.unload(
                    normalized_task
                )

            record.state = ModelState.LOADING
            record.load_attempts += 1
            record.error = None

            started_at = time.perf_counter()

            log_event(
                self.logger,
                event="model_loading_started",
                message=(
                    f"Loading model for "
                    f"{normalized_task}."
                ),
                details={
                    "task": normalized_task,
                    "model_id": spec.model_id,
                    "backend": (
                        spec.backend.value
                    ),
                },
            )

            try:
                self.check_backend_dependencies(
                    spec.backend
                )

                device = resolve_device(
                    spec.device
                )

                loader_function = (
                    self._backend_loaders.get(
                        spec.backend
                    )
                )

                if loader_function is None:
                    raise ModelLoadingError(
                        "No loader is registered for "
                        f"{spec.backend.value}.",
                        module="model_loader",
                    )

                loaded_model = loader_function(
                    spec,
                    device,
                    self.cache_directory,
                    self.model_config.offline_mode,
                )

                if loaded_model is None:
                    raise ModelLoadingError(
                        f"Model loader returned None "
                        f"for {normalized_task}.",
                        module="model_loader",
                    )

                load_time_ms = round(
                    (
                        time.perf_counter()
                        - started_at
                    )
                    * 1000.0,
                    3,
                )

                if not math.isfinite(load_time_ms):
                    load_time_ms = 0.0

                self._models[
                    normalized_task
                ] = loaded_model

                record.state = ModelState.LOADED
                record.device = device
                record.load_time_ms = (
                    load_time_ms
                )
                record.loaded_at = utc_now_iso()
                record.error = None

                self.statistics.successful_loads += 1
                self.statistics\
                    .cumulative_load_time_ms += (
                        load_time_ms
                    )

                log_event(
                    self.logger,
                    event="model_loading_completed",
                    message=(
                        f"Model for {normalized_task} "
                        "loaded."
                    ),
                    details={
                        "task": normalized_task,
                        "model_id": spec.model_id,
                        "device": device,
                        "load_time_ms": (
                            load_time_ms
                        ),
                    },
                )

                return loaded_model

            except Exception as error:
                record.state = ModelState.FAILED
                record.error = (
                    f"{error.__class__.__name__}: "
                    f"{error}"
                )

                self.statistics.failed_loads += 1

                log_exception(
                    self.logger,
                    error,
                    event="model_loading_failed",
                    message=(
                        f"Failed loading model for "
                        f"{normalized_task}."
                    ),
                    details={
                        "task": normalized_task,
                        "model_id": spec.model_id,
                        "backend": (
                            spec.backend.value
                        ),
                        "required": spec.required,
                    },
                )

                if isinstance(
                    error,
                    (
                        ModelLoadingError,
                        DependencyMissingError,
                    ),
                ):
                    raise

                raise ModelLoadingError(
                    f"Unable to load model for "
                    f"{normalized_task}.",
                    module="model_loader",
                    recoverable=not spec.required,
                    details={
                        "task": normalized_task,
                        "model_id": spec.model_id,
                        "backend": (
                            spec.backend.value
                        ),
                    },
                    cause=error,
                ) from error

    def get(
        self,
        task: str,
        *,
        load_if_missing: bool = True,
    ) -> Any:
        """Get a loaded model."""

        with self._lock:

            if task in self._models:
                return self._models[task]

        if load_if_missing:
            return self.load(task)

        raise ModelLoadingError(
            f"Model for {task!r} is not loaded.",
            module="model_loader",
            recoverable=True,
            details={"task": task},
        )

    def unload(
        self,
        task: str,
    ) -> bool:
        """Unload one cached model."""

        normalized_task = task.strip().lower()

        with self._lock:

            if normalized_task not in self._models:
                return False

            model = self._models.pop(
                normalized_task
            )

            del model
            gc.collect()

            if module_is_installed("torch"):
                try:
                    torch = importlib.import_module(
                        "torch"
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    pass

            record = self.get_record(
                normalized_task
            )

            record.state = ModelState.UNLOADED
            record.loaded_at = None

            self.statistics.total_unloads += 1

            log_event(
                self.logger,
                event="model_unloaded",
                message=(
                    f"Model for {normalized_task} "
                    "unloaded."
                ),
                details={
                    "task": normalized_task
                },
            )

            return True

    def unload_all(self) -> int:
        """Unload all cached models."""

        with self._lock:
            tasks = list(
                self._models.keys()
            )

        unloaded_count = 0

        for task in tasks:
            if self.unload(task):
                unloaded_count += 1

        return unloaded_count

    def status(self) -> Dict[str, Any]:
        """Return complete loader status."""

        return {
            "loader_version": (
                MODEL_LOADER_VERSION
            ),
            "offline_mode": (
                self.model_config.offline_mode
            ),
            "cache_directory": str(
                self.cache_directory
            ),
            "loaded_tasks": sorted(
                self._models.keys()
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
            "models": {
                task: record.to_dict()
                for task, record
                in self._records.items()
            },
        }

    # ========================================================
    # BACKEND LOADERS
    # ========================================================

    def _load_ultralytics(
        self,
        spec: ModelSpec,
        device: str,
        cache_directory: Path,
        offline_mode: bool,
    ) -> Any:

        ultralytics = importlib.import_module(
            "ultralytics"
        )

        source = spec.source

        if (
            offline_mode
            and not spec.uses_local_path
            and not Path(source).exists()
        ):
            raise ModelLoadingError(
                "Offline mode requires an existing "
                "local Ultralytics model file.",
                module="model_loader",
                details={
                    "task": spec.task,
                    "source": source,
                },
            )

        model = ultralytics.YOLO(
            source
        )

        return ModelBundle(
            model=model,
            backend=spec.backend.value,
            task=spec.task,
            device=device,
            metadata={
                "model_id": spec.model_id,
                "parameters": spec.parameters,
            },
        )

    def _load_huggingface(
        self,
        spec: ModelSpec,
        device: str,
        cache_directory: Path,
        offline_mode: bool,
    ) -> Any:

        transformers = importlib.import_module(
            "transformers"
        )

        cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        device_argument: Any = -1

        if device == "cuda":
            device_argument = 0
        elif device == "mps":
            device_argument = "mps"

        model_kwargs: Dict[str, Any] = {}

        if offline_mode:
            model_kwargs[
                "local_files_only"
            ] = True

        if spec.revision:
            model_kwargs[
                "revision"
            ] = spec.revision

        if spec.task in (
            HUGGINGFACE_PIPELINE_TASKS
        ):
            pipeline_task = (
                HUGGINGFACE_PIPELINE_TASKS[
                    spec.task
                ]
            )

            pipeline = transformers.pipeline(
                task=pipeline_task,
                model=spec.source,
                device=device_argument,
                model_kwargs=model_kwargs,
            )

            return ModelBundle(
                model=pipeline,
                backend=spec.backend.value,
                task=spec.task,
                device=device,
                metadata={
                    "model_id": spec.model_id,
                    "pipeline_task": (
                        pipeline_task
                    ),
                    "parameters": (
                        spec.parameters
                    ),
                },
            )

        if (
            spec.task
            == "multimodal_embedding"
        ):
            clip_model_class = getattr(
                transformers,
                "CLIPModel",
            )

            clip_processor_class = getattr(
                transformers,
                "CLIPProcessor",
            )

            model = (
                clip_model_class
                .from_pretrained(
                    spec.source,
                    cache_dir=str(
                        cache_directory
                    ),
                    local_files_only=(
                        offline_mode
                    ),
                    trust_remote_code=(
                        spec.trust_remote_code
                    ),
                    revision=spec.revision,
                )
            )

            processor = (
                clip_processor_class
                .from_pretrained(
                    spec.source,
                    cache_dir=str(
                        cache_directory
                    ),
                    local_files_only=(
                        offline_mode
                    ),
                    trust_remote_code=(
                        spec.trust_remote_code
                    ),
                    revision=spec.revision,
                )
            )

            if hasattr(model, "to"):
                model = model.to(device)

            if hasattr(model, "eval"):
                model.eval()

            return ModelBundle(
                model=model,
                processor=processor,
                backend=spec.backend.value,
                task=spec.task,
                device=device,
                metadata={
                    "model_id": spec.model_id,
                    "parameters": (
                        spec.parameters
                    ),
                },
            )

        raise ModelLoadingError(
            "No Hugging Face loading strategy "
            f"exists for {spec.task!r}.",
            module="model_loader",
            details={
                "task": spec.task,
                "model_id": spec.model_id,
            },
        )

    def _load_paddleocr(
        self,
        spec: ModelSpec,
        device: str,
        cache_directory: Path,
        offline_mode: bool,
    ) -> Any:

        paddleocr_module = (
            importlib.import_module(
                "paddleocr"
            )
        )

        paddle_ocr_class = getattr(
            paddleocr_module,
            "PaddleOCR",
        )

        parameters = dict(
            spec.parameters
        )

        language = parameters.pop(
            "language",
            parameters.pop(
                "lang",
                "en",
            ),
        )

        use_gpu = device == "cuda"

        parameters.setdefault(
            "lang",
            language,
        )

        parameters.setdefault(
            "use_gpu",
            use_gpu,
        )

        model = paddle_ocr_class(
            **parameters
        )

        return ModelBundle(
            model=model,
            backend=spec.backend.value,
            task=spec.task,
            device=device,
            metadata={
                "model_id": spec.model_id,
                "parameters": spec.parameters,
                "offline_mode": offline_mode,
            },
        )

    def _load_whisper(
        self,
        spec: ModelSpec,
        device: str,
        cache_directory: Path,
        offline_mode: bool,
    ) -> Any:

        whisper = importlib.import_module(
            "whisper"
        )

        cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        source = spec.source

        if (
            offline_mode
            and spec.uses_local_path
        ):
            source = str(
                Path(spec.local_path).resolve()
            )

        model = whisper.load_model(
            source,
            device=device,
            download_root=str(
                cache_directory
            ),
        )

        return ModelBundle(
            model=model,
            backend=spec.backend.value,
            task=spec.task,
            device=device,
            metadata={
                "model_id": spec.model_id,
                "parameters": spec.parameters,
            },
        )

    def _load_opencv(
        self,
        spec: ModelSpec,
        device: str,
        cache_directory: Path,
        offline_mode: bool,
    ) -> Any:

        cv2 = importlib.import_module(
            "cv2"
        )

        return ModelBundle(
            model=cv2,
            backend=spec.backend.value,
            task=spec.task,
            device=device,
            metadata={
                "model_id": spec.model_id,
                "parameters": spec.parameters,
                "cache_directory": str(
                    cache_directory
                ),
                "offline_mode": offline_mode,
            },
        )

    def _load_internal(
        self,
        spec: ModelSpec,
        device: str,
        cache_directory: Path,
        offline_mode: bool,
    ) -> Any:
        """
        Create an internal placeholder model.

        Custom internal backends may replace this loader through
        register_backend_loader().
        """

        return ModelBundle(
            model={
                "model_id": spec.model_id,
                "task": spec.task,
                "ready": True,
            },
            backend=spec.backend.value,
            task=spec.task,
            device=device,
            metadata={
                "parameters": spec.parameters,
                "cache_directory": str(
                    cache_directory
                ),
                "offline_mode": offline_mode,
            },
        )


# ============================================================
# SHARED LOADER
# ============================================================

_shared_loader: Optional[
    ModelLoader
] = None

_shared_loader_lock = threading.Lock()


def get_shared_model_loader(
    model_config: Optional[
        Layer2ModelConfig
    ] = None,
    *,
    project_root: Optional[
        Path | str
    ] = None,
) -> ModelLoader:
    """Return the shared Layer 2 model loader."""

    global _shared_loader

    with _shared_loader_lock:

        if _shared_loader is None:
            _shared_loader = ModelLoader(
                model_config=model_config,
                project_root=project_root,
            )

        return _shared_loader


def reset_shared_model_loader() -> None:
    """Unload and remove the shared loader."""

    global _shared_loader

    with _shared_loader_lock:

        if _shared_loader is not None:
            _shared_loader.unload_all()

        _shared_loader = None


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | MODEL LOADER SELF-TEST")
    print("=" * 72)
    print(
        "This test uses an internal mock model. "
        "No external model will be downloaded."
    )

    try:
        config = create_test_model_config()

        config.scene_classification = (
            ModelSpec(
                task="scene_classification",
                model_id=(
                    "internal-test-scene-model"
                ),
                backend=ModelBackend.INTERNAL,
                enabled=True,
                required=True,
                device="cpu",
                precision="float32",
                allow_download=False,
                parameters={
                    "test_mode": True
                },
                description=(
                    "Internal model-loader test."
                ),
            )
        )

        config.validate()

        loader = ModelLoader(
            model_config=config
        )

        print("\n[PASS] Model loader created")

        dependency_report = (
            loader.dependency_report()
        )

        if not isinstance(
            dependency_report,
            dict,
        ):
            raise AssertionError(
                "Dependency report is invalid."
            )

        print("[PASS] Dependencies inspected")

        resolved_device = resolve_device(
            "cpu"
        )

        if resolved_device != "cpu":
            raise AssertionError(
                "CPU device was not resolved."
            )

        print("[PASS] CPU device resolved")

        first_model = loader.load(
            "scene_classification"
        )

        if not isinstance(
            first_model,
            ModelBundle,
        ):
            raise AssertionError(
                "Internal model did not return "
                "a ModelBundle."
            )

        print("[PASS] Internal model loaded")

        if not loader.is_loaded(
            "scene_classification"
        ):
            raise AssertionError(
                "Loaded model was not cached."
            )

        print("[PASS] Loaded model cached")

        second_model = loader.load(
            "scene_classification"
        )

        if first_model is not second_model:
            raise AssertionError(
                "Cached model object was not reused."
            )

        print("[PASS] Model cache reused")

        record = loader.get_record(
            "scene_classification"
        )

        if record.state != ModelState.LOADED:
            raise AssertionError(
                "Model state is not loaded."
            )

        if record.cache_hits != 1:
            raise AssertionError(
                "Model cache hit was not recorded."
            )

        print("[PASS] Model status recorded")

        status = loader.status()

        if (
            "scene_classification"
            not in status["loaded_tasks"]
        ):
            raise AssertionError(
                "Loader status omitted the "
                "loaded model."
            )

        print("[PASS] Loader status generated")

        unloaded = loader.unload(
            "scene_classification"
        )

        if not unloaded:
            raise AssertionError(
                "Model was not unloaded."
            )

        if loader.is_loaded(
            "scene_classification"
        ):
            raise AssertionError(
                "Model remained cached after unload."
            )

        print("[PASS] Model unloaded")

        print("\nModel-loader summary:")
        print(
            f"  device: {resolved_device}"
        )
        print(
            f"  state: {record.state.value}"
        )
        print(
            f"  load attempts: "
            f"{record.load_attempts}"
        )
        print(
            f"  cache hits: {record.cache_hits}"
        )
        print(
            f"  load time: "
            f"{record.load_time_ms} ms"
        )
        print(
            f"  statistics: "
            f"{loader.statistics.to_dict()}"
        )

        print("\nDependency availability:")

        for dependency, dependency_status in (
            dependency_report.items()
        ):
            print(
                f"  {dependency}: "
                f"{dependency_status.installed}"
            )

        print("\n" + "=" * 72)
        print(
            "[PASSED] MODEL LOADER IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        ModelLoadingError,
        DependencyMissingError,
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
            "model-loader self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())