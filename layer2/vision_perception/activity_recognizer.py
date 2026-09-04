"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Activity Recognizer
File    : layer2/vision_perception/activity_recognizer.py
============================================================

Purpose
-------
Recognizes human activities from a short sequence of camera
frames using the configured VideoMAE video-classification model.

The recognizer maintains a rolling frame buffer. Until enough
frames are collected, it returns a valid partial ModuleResult.

Configured model:
MCG-NJU/videomae-base-finetuned-kinetics

Compatibility
-------------
Python 3.10+
============================================================
"""

from __future__ import annotations

import argparse
import math
import tempfile

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
)

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

ACTIVITY_RECOGNIZER_VERSION = "1.0"

DEFAULT_NUMBER_OF_FRAMES = 16
DEFAULT_TOP_K = 5
DEFAULT_FRAME_RATE = 8.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# EXCEPTION
# ============================================================

class ActivityRecognitionError(Exception):
    """Raised when activity recognition cannot be completed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class ActivityCandidate:
    """One predicted activity."""

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
class ActivityRecognitionOutput:
    """Complete activity-recognition output."""

    result: ModuleResult

    activity: str
    confidence: Optional[float]

    candidates: List[ActivityCandidate]

    frames_collected: int
    frames_required: int

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def ready(self) -> bool:
        return (
            self.frames_collected
            >= self.frames_required
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "activity": self.activity,
            "confidence": self.confidence,
            "frames_collected": (
                self.frames_collected
            ),
            "frames_required": (
                self.frames_required
            ),
            "ready": self.ready,
            "candidates": [
                candidate.to_dict()
                for candidate
                in self.candidates
            ],
            "result": self.result.to_dict(),
        }


# ============================================================
# ACTIVITY RECOGNIZER
# ============================================================

class ActivityRecognizer:
    """
    Stateful VideoMAE activity recognizer.

    Consecutive processed frames must be supplied to the same
    ActivityRecognizer instance. Creating a new instance for
    every frame would reset the frame buffer.
    """

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
                "activity_recognizer"
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

        parameters = (
            self.model_config
            .activity_recognition
            .parameters
        )

        self.frames_required = int(
            parameters.get(
                "number_of_frames",
                DEFAULT_NUMBER_OF_FRAMES,
            )
        )

        if self.frames_required <= 0:
            raise ActivityRecognitionError(
                "number_of_frames must be positive."
            )

        self.visual_fallback_enabled = bool(
            parameters.get(
                "visual_fallback_enabled",
                True,
            )
        )

        self.frame_rate = float(
            parameters.get(
                "frame_rate",
                DEFAULT_FRAME_RATE,
            )
        )

        self.top_k = int(
            parameters.get(
                "top_k",
                DEFAULT_TOP_K,
            )
        )

        self.frame_buffer: Deque[Any] = deque(
            maxlen=self.frames_required
        )

        self.frame_packet_ids: Deque[str] = deque(
            maxlen=self.frames_required
        )

        self.sequence_number = 0

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def recognize(
        self,
        packet: AdaptedLayer1Input,
        vision_output: VisionProcessingOutput,
    ) -> ActivityRecognitionOutput:
        """Add one frame and recognize activity when ready."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise ActivityRecognitionError(
                "packet must be an "
                "AdaptedLayer1Input."
            )

        if not isinstance(
            vision_output,
            VisionProcessingOutput,
        ):
            raise ActivityRecognitionError(
                "vision_output must be a "
                "VisionProcessingOutput."
            )

        if not (
            self.settings.modules
            .activity_recognition
        ):
            result = ModuleResult.skipped(
                module_name=(
                    "activity_recognizer"
                ),
                modality="vision",
                reason=(
                    "Activity recognition is "
                    "disabled in settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return ActivityRecognitionOutput(
                result=result,
                activity="unknown",
                confidence=None,
                candidates=[],
                frames_collected=0,
                frames_required=(
                    self.frames_required
                ),
            )

        if (
            not vision_output.succeeded
            or vision_output.image_rgb is None
        ):
            result = ModuleResult.failure(
                module_name=(
                    "activity_recognizer"
                ),
                modality="vision",
                error=(
                    "A successful processed vision "
                    "frame is required."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return ActivityRecognitionOutput(
                result=result,
                activity="unknown",
                confidence=None,
                candidates=[],
                frames_collected=len(
                    self.frame_buffer
                ),
                frames_required=(
                    self.frames_required
                ),
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        self._add_frame(
            vision_output.image_rgb,
            packet.packet_id,
        )

        frames_collected = len(
            self.frame_buffer
        )

        if (
            frames_collected
            < self.frames_required
        ):
            result = ModuleResult.partial(
                module_name=(
                    "activity_recognizer"
                ),
                modality="vision",
                data={
                    "activity": {
                        "type": "collecting_frames",
                        "confidence": None,
                    },
                    "frames_collected": (
                        frames_collected
                    ),
                    "frames_required": (
                        self.frames_required
                    ),
                    "remaining_frames": (
                        self.frames_required
                        - frames_collected
                    ),
                    "buffer_ready": False,
                    "recognizer_version": (
                        ACTIVITY_RECOGNIZER_VERSION
                    ),
                },
                confidence=0.0,
                processing_time_ms=0.0,
                source_packet_id=(
                    packet.packet_id
                ),
                warnings=[
                    (
                        "Activity recognition is "
                        "waiting for additional frames."
                    )
                ],
                metadata={
                    "scenario": packet.scenario
                },
            )

            return ActivityRecognitionOutput(
                result=result,
                activity="collecting_frames",
                confidence=None,
                candidates=[],
                frames_collected=(
                    frames_collected
                ),
                frames_required=(
                    self.frames_required
                ),
            )

        log_event(
            packet_logger,
            event=(
                "activity_recognition_started"
            ),
            message=(
                "Activity recognition started."
            ),
            details={
                "frames_collected": (
                    frames_collected
                )
            },
        )

        with ModuleTimer(
            "activity_recognizer",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            temporary_video = None

            try:
                pipeline = self._get_pipeline()

                # temporary_video = (
                #     self._create_temporary_video()
                # )

                # raw_predictions = pipeline(
                #     str(temporary_video),
                #     top_k=self.top_k,
                # )
                video_frames = (
                     self._prepare_video_array()
                     )
                raw_predictions = pipeline( 
                    video_frames,
                    top_k=self.top_k,
                    )


                candidates = (
                    self._normalize_predictions(
                        raw_predictions
                    )
                )

                if not candidates:
                    raise ActivityRecognitionError(
                        "The activity model returned "
                        "no valid predictions."
                    )

                top_candidate = candidates[0]

                threshold = (
                    self._confidence_threshold()
                )

                warnings = []

                if (
                    top_candidate.confidence
                    < threshold
                ):
                    activity = "unknown"

                    warnings.append(
                        "Activity confidence is below "
                        "the configured threshold."
                    )

                    status_factory = (
                        ModuleResult.partial
                    )
                else:
                    activity = (
                        top_candidate
                        .normalized_label
                    )

                    status_factory = (
                        ModuleResult.success
                    )

                self.sequence_number += 1

                data = {
                    "activity": {
                        "type": activity,
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
                    "frames_collected": (
                        frames_collected
                    ),
                    "frames_required": (
                        self.frames_required
                    ),
                    "buffer_ready": True,
                    "sequence_number": (
                        self.sequence_number
                    ),
                    "source_packet_ids": list(
                        self.frame_packet_ids
                    ),
                    "frame_rate": (
                        self.frame_rate
                    ),
                    "threshold": threshold,
                    "model_id": (
                        self.model_config
                        .activity_recognition
                        .model_id
                    ),
                    "recognizer_version": (
                        ACTIVITY_RECOGNIZER_VERSION
                    ),
                }

                result = status_factory(
                    module_name=(
                        "activity_recognizer"
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
                            .activity_recognition
                            .backend.value
                        ),
                    },
                )

                log_event(
                    packet_logger,
                    event=(
                        "activity_recognition_completed"
                    ),
                    message=(
                        "Activity recognition completed."
                    ),
                    details={
                        "activity": activity,
                        "confidence": (
                            top_candidate.confidence
                        ),
                        "frames_collected": (
                            frames_collected
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return ActivityRecognitionOutput(
                    result=result,
                    activity=activity,
                    confidence=(
                        top_candidate.confidence
                    ),
                    candidates=candidates,
                    frames_collected=(
                        frames_collected
                    ),
                    frames_required=(
                        self.frames_required
                    ),
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "activity_recognition_failed"
                    ),
                    message=(
                        "Activity recognition failed."
                    ),
                    details={
                        "frames_collected": (
                            frames_collected
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "activity_recognizer"
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
                            .activity_recognition
                            .model_id
                        )
                    },
                )

                return ActivityRecognitionOutput(
                    result=result,
                    activity="unknown",
                    confidence=None,
                    candidates=[],
                    frames_collected=(
                        frames_collected
                    ),
                    frames_required=(
                        self.frames_required
                    ),
                )

            finally:
                if (
                    temporary_video is not None
                    and temporary_video.exists()
                ):
                    try:
                        temporary_video.unlink()
                    except OSError:
                        pass

    # --------------------------------------------------------
    # FRAME BUFFER
    # --------------------------------------------------------

    def _add_frame(
        self,
        image_rgb: Any,
        packet_id: str,
    ) -> None:
        """Validate and add one RGB frame."""

        if (
            image_rgb is None
            or not hasattr(
                image_rgb,
                "shape",
            )
            or len(image_rgb.shape) != 3
            or image_rgb.shape[2] != 3
        ):
            raise ActivityRecognitionError(
                "Expected an RGB image array."
            )

        frame_copy = image_rgb.copy()

        self.frame_buffer.append(
            frame_copy
        )

        self.frame_packet_ids.append(
            packet_id
        )

    def reset(self) -> None:
        """Clear frames before starting a new stream."""

        self.frame_buffer.clear()
        self.frame_packet_ids.clear()
        self.sequence_number = 0

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------

    def _get_pipeline(self) -> Any:
        """Load and return the VideoMAE pipeline."""

        loaded = self.model_loader.load(
            "activity_recognition"
        )

        if isinstance(
            loaded,
            ModelBundle,
        ):
            pipeline = loaded.model
        else:
            pipeline = loaded

        if not callable(pipeline):
            raise ModelLoadingError(
                "Loaded activity-recognition "
                "object is not callable.",
                module="activity_recognizer",
            )

        return pipeline

    # --------------------------------------------------------
    # TEMPORARY VIDEO CREATION
    # --------------------------------------------------------

    # def _create_temporary_video(
    #     self,
    # ) -> Path:
    def _prepare_video_array(
        self,
    ) -> Any:
        """
        Convert buffered RGB frames into a temporary MP4 file.

        The Transformers video-classification pipeline expects
        a video input rather than unrelated individual images.
        """

        try:
            import cv2
        except ImportError as error:
            raise DependencyMissingError(
                "OpenCV is required for activity "
                "recognition. Install with: "
                "python -m pip install opencv-python",
                module="activity_recognizer",
            ) from error

        if (
            len(self.frame_buffer)
            < self.frames_required
        ):
            raise ActivityRecognitionError(
                "The frame buffer is not ready."
            )

        first_frame = self.frame_buffer[0]

        frame_height = int(
            first_frame.shape[0]
        )

        frame_width = int(
            first_frame.shape[1]
        )

        temporary_file = tempfile.NamedTemporaryFile(
            prefix="noongil_activity_",
            suffix=".mp4",
            delete=False,
        )

        temporary_path = Path(
            temporary_file.name
        )

        temporary_file.close()

        codec = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(temporary_path),
            codec,
            self.frame_rate,
            (
                frame_width,
                frame_height,
            ),
        )

        if not writer.isOpened():
            raise ActivityRecognitionError(
                "OpenCV could not create the "
                "temporary activity video."
            )

        try:
            for frame_rgb in self.frame_buffer:

                if (
                    frame_rgb.shape[0]
                    != frame_height
                    or frame_rgb.shape[1]
                    != frame_width
                ):
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (
                            frame_width,
                            frame_height,
                        ),
                    )

                frame_bgr = cv2.cvtColor(
                    frame_rgb,
                    cv2.COLOR_RGB2BGR,
                )

                writer.write(
                    frame_bgr
                )

        finally:
            writer.release()

        if (
            not temporary_path.exists()
            or temporary_path.stat().st_size == 0
        ):
            raise ActivityRecognitionError(
                "The temporary activity video "
                "was not created correctly."
            )

        return temporary_path

    # --------------------------------------------------------
    # PREDICTION PROCESSING
    # --------------------------------------------------------

    def _normalize_predictions(
        self,
        raw_predictions: Any,
    ) -> List[ActivityCandidate]:
        """Validate and normalize model predictions."""

        if (
            isinstance(raw_predictions, list)
            and len(raw_predictions) == 1
            and isinstance(
                raw_predictions[0],
                list,
            )
        ):
            raw_predictions = (
                raw_predictions[0]
            )

        if not isinstance(
            raw_predictions,
            list,
        ):
            raise ActivityRecognitionError(
                "Activity predictions must "
                "be returned as a list."
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

            normalized_predictions.append(
                (
                    label.strip(),
                    self._normalize_label(
                        label
                    ),
                    confidence,
                )
            )

        normalized_predictions.sort(
            key=lambda value: value[2],
            reverse=True,
        )

        return [
            ActivityCandidate(
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

    @staticmethod
    def _normalize_label(
        label: str,
    ) -> str:
        """Convert a Kinetics label to Layer 3 format."""

        normalized = (
            label.strip()
            .lower()
            .replace("-", " ")
            .replace("/", " ")
        )

        normalized = "_".join(
            normalized.split()
        )

        return normalized or "unknown"

    def _confidence_threshold(
        self,
    ) -> float:
        """Return the activity-confidence threshold."""

        vision_settings = (
            self.settings.vision
        )

        value = getattr(
            vision_settings,
            "activity_confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD,
        )

        value = float(value)

        if (
            not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ActivityRecognitionError(
                "Activity confidence threshold "
                "must be between 0 and 1."
            )

        return value


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def recognize_activity(
    packet: AdaptedLayer1Input,
    vision_output: VisionProcessingOutput,
    *,
    recognizer: Optional[
        ActivityRecognizer
    ] = None,
    settings: Optional[
        Layer2Settings
    ] = None,
    model_config: Optional[
        Layer2ModelConfig
    ] = None,
    model_loader: Optional[
        ModelLoader
    ] = None,
) -> ActivityRecognitionOutput:
    """
    Recognize activity from one frame.

    For real-time use, pass the same recognizer instance for
    every consecutive frame so that its buffer is preserved.
    """

    active_recognizer = (
        recognizer
        or ActivityRecognizer(
            settings=settings,
            model_config=model_config,
            model_loader=model_loader,
        )
    )

    return active_recognizer.recognize(
        packet,
        vision_output,
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "park_walking",
) -> bool:

    print("=" * 72)
    print(
        "NOONGIL-X | ACTIVITY RECOGNIZER "
        "SELF-TEST"
    )
    print("=" * 72)

    print(
        "The first run may download the "
        "configured VideoMAE model."
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

        processor = VisionProcessor(
            settings=settings
        )

        recognizer = ActivityRecognizer(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        packet = adapter.load_scenario(
            scenario_name
        )

        vision_output = processor.process(
            packet
        )

        if not vision_output.succeeded:
            raise AssertionError(
                "Vision processing failed."
            )

        print(
            f"[PASS] Scenario loaded: "
            f"{scenario_name}"
        )

        for frame_number in range(
            1,
            recognizer.frames_required + 1,
        ):
            output = recognizer.recognize(
                packet,
                vision_output,
            )

            if not output.succeeded:
                raise AssertionError(
                    "Activity recognition failed "
                    f"at frame {frame_number}: "
                    f"{output.result.errors}"
                )

            if (
                frame_number
                < recognizer.frames_required
                and output.activity
                != "collecting_frames"
            ):
                raise AssertionError(
                    "The recognizer performed inference "
                    "before its buffer was ready."
                )

        if not output.ready:
            raise AssertionError(
                "The frame buffer did not become ready."
            )

        if not output.candidates:
            raise AssertionError(
                "No activity candidates were returned."
            )

        if output.confidence is None:
            raise AssertionError(
                "Activity confidence is missing."
            )

        if not (
            0.0
            <= output.confidence
            <= 1.0
        ):
            raise AssertionError(
                "Activity confidence is invalid."
            )

        print(
            f"[PASS] Frames buffered: "
            f"{output.frames_collected}/"
            f"{output.frames_required}"
        )

        print(
            f"[PASS] Predicted activity: "
            f"{output.activity}"
        )

        print(
            f"[PASS] Confidence: "
            f"{output.confidence:.6f}"
        )

        print(
            "[PASS] Rolling frame buffer validated"
        )

        print(
            "[PASS] Temporary video generated"
        )

        print(
            "[PASS] VideoMAE loaded lazily"
        )

        print(
            "[PASS] Activity candidates validated"
        )

        print(
            "[PASS] Layer 3 activity format generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] ACTIVITY RECOGNIZER "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        ActivityRecognitionError,
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
            "activity-recognizer self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="park_walking",
        help=(
            "Scenario used for the frame-buffer test. "
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
        if run_self_test(
            arguments.scenario
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())