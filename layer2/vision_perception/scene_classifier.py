
"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Scene Classifier
File    : layer2/vision_perception/scene_classifier.py
============================================================

Purpose
-------
Classifies the environmental scene in a processed camera frame.

Supported scene classes:
- park
- classroom
- shopping_mall
- cafe
- home
- street
- road
- hospital
- office
- unknown

The module uses the configured zero-shot image-classification
model. It does not use scenario filenames or expected outputs
to make predictions.

Dependencies
------------
PyTorch
Torchvision
Transformers
Pillow
NumPy

Compatibility
-------------
Python 3.10+
============================================================
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from layer2.config.model_config import (
    Layer2ModelConfig,
    create_default_model_config,
)

from layer2.config.settings import (
    Layer2Settings,
    create_default_settings,
)

from layer2.input_reception.layer1_packet_adapter import (
    AdaptedLayer1Input,
    Layer1PacketAdapter,
)

from layer2.schemas.module_result import (
    ModuleResult,
)

from layer2.utils.exceptions import (
    DependencyMissingError,
    ModelLoadingError,
    SceneClassificationError,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    ModuleTimer,
    get_logger,
    log_event,
    log_exception,
)

from layer2.utils.model_loader import (
    ModelBundle,
    ModelLoader,
)

from layer2.vision_perception.vision_processor import (
    VisionProcessingOutput,
    VisionProcessor,
)


# ============================================================
# CONSTANTS
# ============================================================

SCENE_CLASSIFIER_VERSION = "1.0"

DEFAULT_SCENE_LABELS = [
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

SCENE_LABEL_MAPPING = {
    "park": "park",
    "classroom": "classroom",
    "shopping mall": "shopping_mall",
    "shopping_mall": "shopping_mall",
    "mall": "shopping_mall",
    "cafe": "cafe",
    "coffee shop": "cafe",
    "home": "home",
    "living room": "home",
    "street": "street",
    "city street": "street",
    "road": "road",
    "hospital": "hospital",
    "office": "office",
    "unknown environment": "unknown",
    "unknown": "unknown",
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class SceneCandidate:
    """One candidate scene prediction."""

    label: str
    normalized_label: str
    confidence: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "label": self.label,
            "normalized_label": (
                self.normalized_label
            ),
            "confidence": round(
                self.confidence,
                6,
            ),
            "rank": self.rank,
        }


@dataclass
class SceneClassificationOutput:
    """Complete scene-classification output."""

    result: ModuleResult
    scene_type: str
    confidence: Optional[float]
    candidates: List[SceneCandidate]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    def to_dict(self) -> Dict[str, Any]:

        return {
            "scene_type": self.scene_type,
            "confidence": self.confidence,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "result": self.result.to_dict(),
        }


# ============================================================
# SCENE CLASSIFIER
# ============================================================

class SceneClassifier:
    """Zero-shot environmental scene classifier."""

    def __init__(
        self,
        settings: Optional[
            Layer2Settings
        ] = None,
        model_config: Optional[
            Layer2ModelConfig
        ] = None,
        model_loader: Optional[
            ModelLoader
        ] = None,
        *,
        project_root: Optional[
            Path | str
        ] = None,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.settings = (
            settings
            or create_default_settings()
        )

        self.settings.validate()

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

        self.logger = (
            logger
            or get_logger(
                "scene_classifier"
            )
        )

        self.model_loader = (
            model_loader
            or ModelLoader(
                model_config=(
                    self.model_config
                ),
                project_root=(
                    self.project_root
                ),
                logger=self.logger,
            )
        )

    def classify(
        self,
        packet: AdaptedLayer1Input,
        vision_output: VisionProcessingOutput,
    ) -> SceneClassificationOutput:
        """Classify one processed camera frame."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise SceneClassificationError(
                "packet must be an "
                "AdaptedLayer1Input.",
                module="scene_classifier",
            )

        if not isinstance(
            vision_output,
            VisionProcessingOutput,
        ):
            raise SceneClassificationError(
                "vision_output must be a "
                "VisionProcessingOutput.",
                module="scene_classifier",
            )

        if not (
            self.settings.modules
            .scene_classification
        ):
            result = ModuleResult.skipped(
                module_name="scene_classifier",
                modality="vision",
                reason=(
                    "Scene classification is "
                    "disabled in settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return SceneClassificationOutput(
                result=result,
                scene_type="unknown",
                confidence=None,
                candidates=[],
            )

        if (
            not vision_output.succeeded
            or vision_output.image_rgb
            is None
        ):
            result = ModuleResult.failure(
                module_name="scene_classifier",
                modality="vision",
                error=(
                    "A successful processed vision "
                    "frame is required."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return SceneClassificationOutput(
                result=result,
                scene_type="unknown",
                confidence=None,
                candidates=[],
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        log_event(
            packet_logger,
            event=(
                "scene_classification_started"
            ),
            message=(
                "Scene classification started."
            ),
        )

        with ModuleTimer(
            "scene_classifier",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                pipeline = (
                    self._get_pipeline()
                )

                image = self._get_input_image(
                    vision_output
                )

                candidate_labels = (
                    self._candidate_labels()
                )

                raw_predictions = pipeline(
                    image,
                    candidate_labels=(
                        candidate_labels
                    ),
                )

                candidates = (
                    self._normalize_predictions(
                        raw_predictions
                    )
                )

                if not candidates:
                    raise SceneClassificationError(
                        "The scene model returned "
                        "no predictions.",
                        module="scene_classifier",
                    )

                top_candidate = candidates[0]

                threshold = (
                    self.settings.vision
                    .scene_confidence_threshold
                )

                warnings = []

                if (
                    top_candidate.confidence
                    < threshold
                ):
                    scene_type = "unknown"

                    warnings.append(
                        "Scene confidence is below "
                        "the configured threshold."
                    )

                    status_factory = (
                        ModuleResult.partial
                    )
                else:
                    scene_type = (
                        top_candidate
                        .normalized_label
                    )

                    status_factory = (
                        ModuleResult.success
                    )

                data = {
                    "scene": {
                        "type": scene_type,
                        "confidence": round(
                            top_candidate.confidence,
                            6,
                        ),
                        "raw_label": (
                            top_candidate.label
                        ),
                    },
                    "candidates": [
                        candidate.to_dict()
                        for candidate
                        in candidates
                    ],
                    "candidate_labels": (
                        candidate_labels
                    ),
                    "threshold": threshold,
                    "frame_quality_confidence": (
                        vision_output
                        .result.confidence
                    ),
                    "model_id": (
                        self.model_config
                        .scene_classification
                        .model_id
                    ),
                    "classifier_version": (
                        SCENE_CLASSIFIER_VERSION
                    ),
                }

                result = status_factory(
                    module_name=(
                        "scene_classifier"
                    ),
                    modality="vision",
                    data=data,
                    confidence=(
                        top_candidate.confidence
                    ),
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        packet.packet_id
                    ),
                    warnings=warnings,
                    metadata={
                        "scenario": (
                            packet.scenario
                        ),
                        "model_backend": (
                            self.model_config
                            .scene_classification
                            .backend.value
                        ),
                    },
                )

                log_event(
                    packet_logger,
                    event=(
                        "scene_classification_completed"
                    ),
                    message=(
                        "Scene classification "
                        "completed."
                    ),
                    details={
                        "scene_type": scene_type,
                        "confidence": (
                            top_candidate.confidence
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return SceneClassificationOutput(
                    result=result,
                    scene_type=scene_type,
                    confidence=(
                        top_candidate.confidence
                    ),
                    candidates=candidates,
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "scene_classification_failed"
                    ),
                    message=(
                        "Scene classification failed."
                    ),
                    details={
                        "processing_time_ms": (
                            timer.elapsed_ms
                        )
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "scene_classifier"
                    ),
                    modality="vision",
                    error=(
                        f"{error.__class__.__name__}: "
                        f"{error}"
                    ),
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        packet.packet_id
                    ),
                    metadata={
                        "model_id": (
                            self.model_config
                            .scene_classification
                            .model_id
                        )
                    },
                )

                return SceneClassificationOutput(
                    result=result,
                    scene_type="unknown",
                    confidence=None,
                    candidates=[],
                )

    def _get_pipeline(self) -> Any:
        """Load and return the scene pipeline."""

        loaded = self.model_loader.load(
            "scene_classification"
        )

        if isinstance(loaded, ModelBundle):
            pipeline = loaded.model
        else:
            pipeline = loaded

        if not callable(pipeline):
            raise ModelLoadingError(
                "Loaded scene-classification "
                "object is not callable.",
                module="scene_classifier",
            )

        return pipeline

    def _get_input_image(
        self,
        vision_output: VisionProcessingOutput,
    ) -> Any:
        """
        Return a Pillow image for CLIP.

        The original orientation-corrected image is preferred
        because scene classification does not require letterbox
        padding.
        """

        if (
            vision_output.original_image
            is not None
        ):
            return (
                vision_output.original_image
            )

        try:
            from PIL import Image

            return Image.fromarray(
                vision_output.image_rgb
            )

        except Exception as error:
            raise SceneClassificationError(
                "Unable to create scene-model "
                "input image.",
                module="scene_classifier",
                cause=error,
            ) from error

    def _candidate_labels(self) -> List[str]:
        """Return configured candidate scene labels."""

        parameters = (
            self.model_config
            .scene_classification
            .parameters
        )

        configured_labels = (
            parameters.get(
                "candidate_labels"
            )
        )

        if not isinstance(
            configured_labels,
            list,
        ):
            return list(
                DEFAULT_SCENE_LABELS
            )

        labels = [
            str(label).strip()
            for label in configured_labels
            if str(label).strip()
        ]

        return (
            labels
            if labels
            else list(
                DEFAULT_SCENE_LABELS
            )
        )

    def _normalize_predictions(
        self,
        raw_predictions: Any,
    ) -> List[SceneCandidate]:
        """Validate and normalize model predictions."""

        if not isinstance(
            raw_predictions,
            list,
        ):
            raise SceneClassificationError(
                "Scene predictions must be a list.",
                module="scene_classifier",
                details={
                    "received_type": (
                        raw_predictions
                        .__class__.__name__
                    )
                },
            )

        normalized_predictions = []

        for item in raw_predictions:

            if not isinstance(item, dict):
                continue

            label = item.get("label")
            score = item.get("score")

            if (
                not isinstance(label, str)
                or not label.strip()
            ):
                continue

            if (
                not isinstance(
                    score,
                    (int, float),
                )
                or isinstance(score, bool)
            ):
                continue

            confidence = float(score)

            if (
                not math.isfinite(confidence)
                or not 0.0
                <= confidence
                <= 1.0
            ):
                continue

            normalized_label = (
                self._normalize_label(
                    label
                )
            )

            normalized_predictions.append(
                (
                    label.strip(),
                    normalized_label,
                    confidence,
                )
            )

        normalized_predictions.sort(
            key=lambda value: value[2],
            reverse=True,
        )

        return [
            SceneCandidate(
                label=label,
                normalized_label=(
                    normalized_label
                ),
                confidence=confidence,
                rank=index,
            )
            for index, (
                label,
                normalized_label,
                confidence,
            )
            in enumerate(
                normalized_predictions,
                start=1,
            )
        ]

    def _normalize_label(
        self,
        label: str,
    ) -> str:
        """Convert a model label to Layer 3 format."""

        normalized = (
            label.strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

        normalized = " ".join(
            normalized.split()
        )

        if normalized in SCENE_LABEL_MAPPING:
            return SCENE_LABEL_MAPPING[
                normalized
            ]

        return normalized.replace(
            " ",
            "_",
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def classify_scene(
    packet: AdaptedLayer1Input,
    vision_output: VisionProcessingOutput,
    *,
    settings: Optional[
        Layer2Settings
    ] = None,
    model_config: Optional[
        Layer2ModelConfig
    ] = None,
    model_loader: Optional[
        ModelLoader
    ] = None,
) -> SceneClassificationOutput:

    classifier = SceneClassifier(
        settings=settings,
        model_config=model_config,
        model_loader=model_loader,
    )

    return classifier.classify(
        packet,
        vision_output,
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "park_walking",
    *,
    test_all: bool = False,
) -> bool:

    print("=" * 72)
    print("NOONGIL-X | SCENE CLASSIFIER SELF-TEST")
    print("=" * 72)
    print(
        "The first run may download the configured "
        "CLIP model."
    )

    loader: Optional[ModelLoader] = None

    try:
        adapter = Layer1PacketAdapter(
            require_media=True
        )

        settings = (
            create_default_settings()
        )

        model_config = (
            create_default_model_config()
        )

        loader = ModelLoader(
            model_config=model_config
        )

        vision_processor = (
            VisionProcessor(
                settings=settings
            )
        )

        classifier = SceneClassifier(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        if test_all:
            scenarios = (
                adapter.discover_scenarios()
            )
        else:
            scenarios = [scenario_name]

        if not scenarios:
            raise AssertionError(
                "No scenarios selected."
            )

        print(
            f"[PASS] Testing "
            f"{len(scenarios)} scenario(s)"
        )

        successful_predictions = 0

        for scenario in scenarios:

            packet = adapter.load_scenario(
                scenario
            )

            vision_output = (
                vision_processor.process(
                    packet
                )
            )

            if not vision_output.succeeded:
                raise AssertionError(
                    f"Vision processing failed "
                    f"for {scenario}."
                )

            output = classifier.classify(
                packet,
                vision_output,
            )

            if not output.succeeded:
                raise AssertionError(
                    f"Scene classification failed "
                    f"for {scenario}: "
                    f"{output.result.errors}"
                )

            if not output.candidates:
                raise AssertionError(
                    f"No scene candidates returned "
                    f"for {scenario}."
                )

            if output.confidence is None:
                raise AssertionError(
                    f"Scene confidence missing "
                    f"for {scenario}."
                )

            if not 0.0 <= output.confidence <= 1.0:
                raise AssertionError(
                    f"Invalid confidence for "
                    f"{scenario}."
                )

            successful_predictions += 1

            expected_file = (
                adapter.scenario_directory
                / scenario
                / "expected_layer2_output.json"
            )

            expected_scene = None

            if expected_file.exists():
                import json

                expected_payload = json.loads(
                    expected_file.read_text(
                        encoding="utf-8"
                    )
                )

                expected_scene = (
                    expected_payload
                    .get("scene", {})
                    .get("type")
                )

            matched = (
                output.scene_type
                == expected_scene
            )

            print(
                f"[PASS] {scenario}: "
                f"predicted={output.scene_type}, "
                f"confidence="
                f"{output.confidence:.6f}, "
                f"expected={expected_scene}, "
                f"match={matched}"
            )

        if (
            successful_predictions
            != len(scenarios)
        ):
            raise AssertionError(
                "Not every scenario produced "
                "a scene prediction."
            )

        print("[PASS] Model loaded lazily")
        print("[PASS] Candidate labels evaluated")
        print("[PASS] Confidence scores validated")
        print("[PASS] Layer 3 scene format generated")
        print("[PASS] ModuleResult generated")

        print("\n" + "=" * 72)
        print(
            "[PASSED] SCENE CLASSIFIER IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        SceneClassificationError,
        AssertionError,
    ) as error:

        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False

    finally:

        if loader is not None:
            loader.unload_all()


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the NOONGIL-X Layer 2 "
            "scene-classifier self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="park_walking",
        help=(
            "Scenario tested by default. "
            "Default: park_walking"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all eight scenarios.",
    )

    return parser


def main() -> int:

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    return (
        0
        if run_self_test(
            arguments.scenario,
            test_all=arguments.all,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())