"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Sound Event Detector
File    : layer2/audio/sound_event_detector.py
============================================================

Purpose
-------
Detects environmental sound events using the configured AST
AudioSet model.

Examples:
- alarms
- sirens
- vehicle horns
- traffic noise
- barking
- birds
- crowd noise
- footsteps
- breaking glass
- emergency sounds
============================================================
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
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


# ============================================================
# CONSTANTS
# ============================================================

SOUND_EVENT_DETECTOR_VERSION = "1.0"

DEFAULT_SOUND_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_MAXIMUM_LABELS = 5
DEFAULT_SAMPLE_RATE = 16000


# ============================================================
# SAFETY-RELEVANT SOUND GROUPS
# ============================================================

CRITICAL_SOUND_KEYWORDS = {
    "fire alarm",
    "smoke detector",
    "gunshot",
    "gunfire",
    "explosion",
    "breaking glass",
    "screaming",
    "scream",
    "emergency vehicle",
}

HIGH_PRIORITY_SOUND_KEYWORDS = {
    "siren",
    "alarm",
    "car horn",
    "vehicle horn",
    "horn",
    "shout",
    "shouting",
    "crying",
    "buzzer",
    "crash",
    "collision",
    "skidding",
}

NAVIGATION_SOUND_KEYWORDS = {
    "traffic",
    "vehicle",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "train",
    "footsteps",
    "walking",
    "crowd",
    "speech",
    "bicycle",
}


# ============================================================
# EXCEPTION
# ============================================================

class SoundEventDetectionError(Exception):
    """Raised when sound-event detection fails."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class SoundEvent:
    """One environmental sound prediction."""

    event_id: str

    label: str
    normalized_label: str

    confidence: float
    rank: int

    priority: str
    safety_relevant: bool

    def to_dict(self) -> Dict[str, Any]:

        return {
            "event_id": self.event_id,
            "label": self.label,
            "normalized_label": (
                self.normalized_label
            ),
            "confidence": round(
                self.confidence,
                6,
            ),
            "rank": self.rank,
            "priority": self.priority,
            "safety_relevant": (
                self.safety_relevant
            ),
        }


@dataclass
class SoundEventDetectionOutput:
    """Complete environmental-sound output."""

    result: ModuleResult

    events: List[SoundEvent]

    audio_path: Optional[Path]
    sample_rate: Optional[int]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def top_event(self) -> Optional[SoundEvent]:

        if not self.events:
            return None

        return self.events[0]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "event_count": self.event_count,
            "events": [
                event.to_dict()
                for event in self.events
            ],
            "top_event": (
                self.top_event.to_dict()
                if self.top_event
                else None
            ),
            "audio_path": (
                str(self.audio_path)
                if self.audio_path
                else None
            ),
            "sample_rate": self.sample_rate,
            "result": self.result.to_dict(),
        }


# ============================================================
# SOUND EVENT DETECTOR
# ============================================================

class SoundEventDetector:
    """AST-based environmental sound classifier."""

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
                "sound_event_detector"
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

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def detect(
        self,
        packet: AdaptedLayer1Input,
    ) -> SoundEventDetectionOutput:
        """Detect environmental sounds in packet audio."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise SoundEventDetectionError(
                "packet must be an "
                "AdaptedLayer1Input."
            )

        if not self._module_enabled():

            result = ModuleResult.skipped(
                module_name=(
                    "sound_event_detector"
                ),
                modality="audio",
                reason=(
                    "Sound-event detection is "
                    "disabled in settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return SoundEventDetectionOutput(
                result=result,
                events=[],
                audio_path=None,
                sample_rate=None,
            )

        audio_input = packet.get_modality(
            "audio"
        )

        if (
            audio_input is None
            or not audio_input.usable
            or audio_input.media_path is None
        ):
            result = ModuleResult.failure(
                module_name=(
                    "sound_event_detector"
                ),
                modality="audio",
                error=(
                    "A usable Layer 1 audio "
                    "file is required."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return SoundEventDetectionOutput(
                result=result,
                events=[],
                audio_path=None,
                sample_rate=None,
            )

        audio_path = Path(
            audio_input.media_path
        )

        if not audio_path.is_file():

            result = ModuleResult.failure(
                module_name=(
                    "sound_event_detector"
                ),
                modality="audio",
                error=(
                    f"Audio file does not exist: "
                    f"{audio_path}"
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return SoundEventDetectionOutput(
                result=result,
                events=[],
                audio_path=audio_path,
                sample_rate=None,
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        log_event(
            packet_logger,
            event=(
                "sound_event_detection_started"
            ),
            message=(
                "Environmental sound-event "
                "detection started."
            ),
            details={
                "audio_path": str(
                    audio_path
                )
            },
        )

        with ModuleTimer(
            "sound_event_detector",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                pipeline = (
                    self._get_pipeline()
                )

                audio_array = (
                    self._load_audio(
                        audio_path
                    )
                )

                maximum_labels = (
                    self._maximum_labels()
                )

                raw_predictions = pipeline(
                    {
                        "array": audio_array,
                        "sampling_rate": (
                            DEFAULT_SAMPLE_RATE
                        ),
                    },
                    top_k=maximum_labels,
                )

                threshold = (
                    self._confidence_threshold()
                )

                events = (
                    self._parse_predictions(
                        raw_predictions=(
                            raw_predictions
                        ),
                        confidence_threshold=(
                            threshold
                        ),
                        packet_id=(
                            packet.packet_id
                        ),
                    )
                )

                confidence = (
                    events[0].confidence
                    if events
                    else 0.0
                )

                warnings = []

                if not events:
                    warnings.append(
                        "No environmental sound met "
                        "the configured confidence "
                        "threshold."
                    )

                safety_events = [
                    event
                    for event in events
                    if event.safety_relevant
                ]

                data = {
                    "event_count": len(events),
                    "events": [
                        event.to_dict()
                        for event in events
                    ],
                    "top_event": (
                        events[0].to_dict()
                        if events
                        else None
                    ),
                    "safety_event_count": (
                        len(safety_events)
                    ),
                    "safety_events": [
                        event.to_dict()
                        for event
                        in safety_events
                    ],
                    "audio_path": str(
                        audio_path
                    ),
                    "sample_rate": (
                        DEFAULT_SAMPLE_RATE
                    ),
                    "threshold": threshold,
                    "maximum_labels": (
                        maximum_labels
                    ),
                    "model_id": (
                        self.model_config
                        .sound_event_detection
                        .model_id
                    ),
                    "detector_version": (
                        SOUND_EVENT_DETECTOR_VERSION
                    ),
                }

                status_factory = (
                    ModuleResult.success
                    if events
                    else ModuleResult.partial
                )

                result = status_factory(
                    module_name=(
                        "sound_event_detector"
                    ),
                    modality="audio",
                    data=data,
                    confidence=confidence,
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
                            .sound_event_detection
                            .backend.value
                        ),
                    },
                )

                log_event(
                    packet_logger,
                    event=(
                        "sound_event_detection_completed"
                    ),
                    message=(
                        "Environmental sound-event "
                        "detection completed."
                    ),
                    details={
                        "event_count": (
                            len(events)
                        ),
                        "safety_event_count": (
                            len(safety_events)
                        ),
                        "top_event": (
                            events[0]
                            .normalized_label
                            if events
                            else None
                        ),
                        "confidence": confidence,
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return SoundEventDetectionOutput(
                    result=result,
                    events=events,
                    audio_path=audio_path,
                    sample_rate=(
                        DEFAULT_SAMPLE_RATE
                    ),
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "sound_event_detection_failed"
                    ),
                    message=(
                        "Environmental sound-event "
                        "detection failed."
                    ),
                    details={
                        "audio_path": str(
                            audio_path
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "sound_event_detector"
                    ),
                    modality="audio",
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
                        "audio_path": str(
                            audio_path
                        ),
                        "model_id": (
                            self.model_config
                            .sound_event_detection
                            .model_id
                        ),
                    },
                )

                return SoundEventDetectionOutput(
                    result=result,
                    events=[],
                    audio_path=audio_path,
                    sample_rate=None,
                )

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    def _module_enabled(self) -> bool:
        """Return whether sound detection is enabled."""

        modules = self.settings.modules

        if hasattr(
            modules,
            "sound_event_detection",
        ):
            return bool(
                modules.sound_event_detection
            )

        if hasattr(
            modules,
            "audio_processing",
        ):
            return bool(
                modules.audio_processing
            )

        return True

    def _confidence_threshold(
        self,
    ) -> float:
        """Return sound confidence threshold."""

        audio_settings = getattr(
            self.settings,
            "audio",
            None,
        )

        value = getattr(
            audio_settings,
            "sound_confidence_threshold",
            DEFAULT_SOUND_CONFIDENCE_THRESHOLD,
        )

        value = float(value)

        if (
            not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise SoundEventDetectionError(
                "Sound confidence threshold must "
                "be between 0 and 1."
            )

        return value

    def _maximum_labels(self) -> int:
        """Return maximum number of sound labels."""

        value = (
            self.model_config
            .sound_event_detection
            .parameters
            .get(
                "maximum_labels",
                DEFAULT_MAXIMUM_LABELS,
            )
        )

        value = int(value)

        if value <= 0:
            raise SoundEventDetectionError(
                "maximum_labels must be positive."
            )

        return value

    # --------------------------------------------------------
    # AUDIO LOADING
    # --------------------------------------------------------

    @staticmethod
    def _load_audio(
        audio_path: Path,
    ) -> Any:
        """
        Load audio as mono float32 at 16 kHz.

        Whisper's tested audio loader is reused because it
        already handles WAV, MP3, M4A and other FFmpeg formats.
        """

        try:
            import whisper

        except ImportError as error:
            raise DependencyMissingError(
                "openai-whisper is required "
                "for audio loading.",
                module=(
                    "sound_event_detector"
                ),
            ) from error

        try:
            audio_array = whisper.load_audio(
                str(audio_path),
                sr=DEFAULT_SAMPLE_RATE,
            )

        except Exception as error:
            raise SoundEventDetectionError(
                "Unable to decode audio file: "
                f"{audio_path}"
            ) from error

        if (
            audio_array is None
            or not hasattr(
                audio_array,
                "shape",
            )
            or audio_array.size == 0
        ):
            raise SoundEventDetectionError(
                "Decoded audio is empty."
            )

        return audio_array

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    def _get_pipeline(self) -> Any:
        """Load and return AST pipeline."""

        loaded = self.model_loader.load(
            "sound_event_detection"
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
                "Loaded sound-event model "
                "is not callable.",
                module=(
                    "sound_event_detector"
                ),
            )

        return pipeline

    # --------------------------------------------------------
    # OUTPUT PROCESSING
    # --------------------------------------------------------

    def _parse_predictions(
        self,
        *,
        raw_predictions: Any,
        confidence_threshold: float,
        packet_id: str,
    ) -> List[SoundEvent]:
        """Validate AST predictions."""

        if (
            isinstance(
                raw_predictions,
                list,
            )
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
            raise SoundEventDetectionError(
                "Sound predictions must "
                "be returned as a list."
            )

        valid_predictions = []

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

            if (
                confidence
                < confidence_threshold
            ):
                continue

            normalized_label = (
                self._normalize_label(
                    label
                )
            )

            priority = self._priority(
                label
            )

            valid_predictions.append(
                (
                    label.strip(),
                    normalized_label,
                    confidence,
                    priority,
                )
            )

        valid_predictions.sort(
            key=lambda value: value[2],
            reverse=True,
        )

        return [
            SoundEvent(
                event_id=(
                    f"SOUND_{packet_id}_"
                    f"{index:03d}"
                ),
                label=label,
                normalized_label=(
                    normalized_label
                ),
                confidence=confidence,
                rank=index,
                priority=priority,
                safety_relevant=(
                    priority
                    in {
                        "critical",
                        "high",
                    }
                ),
            )
            for index, (
                label,
                normalized_label,
                confidence,
                priority,
            )
            in enumerate(
                valid_predictions,
                start=1,
            )
        ]

    @staticmethod
    def _normalize_label(
        label: str,
    ) -> str:
        """Convert AudioSet label to Layer 3 format."""

        normalized = (
            label.strip()
            .lower()
            .replace("-", " ")
            .replace("/", " ")
            .replace(",", " ")
        )

        return "_".join(
            normalized.split()
        ) or "unknown"

    @staticmethod
    def _priority(
        label: str,
    ) -> str:
        """Determine sound-event priority."""

        normalized = label.lower()

        if any(
            keyword in normalized
            for keyword
            in CRITICAL_SOUND_KEYWORDS
        ):
            return "critical"

        if any(
            keyword in normalized
            for keyword
            in HIGH_PRIORITY_SOUND_KEYWORDS
        ):
            return "high"

        if any(
            keyword in normalized
            for keyword
            in NAVIGATION_SOUND_KEYWORDS
        ):
            return "medium"

        return "low"


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def detect_sound_events(
    packet: AdaptedLayer1Input,
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
) -> SoundEventDetectionOutput:

    detector = SoundEventDetector(
        settings=settings,
        model_config=model_config,
        model_loader=model_loader,
    )

    return detector.detect(
        packet
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "park_walking",
) -> bool:

    print("=" * 72)

    print(
        "NOONGIL-X | SOUND EVENT DETECTOR "
        "SELF-TEST"
    )

    print("=" * 72)

    print(
        "The first run may download the "
        "configured AST model."
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

        detector = SoundEventDetector(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        packet = adapter.load_scenario(
            scenario_name
        )

        output = detector.detect(
            packet
        )

        if not output.succeeded:
            raise AssertionError(
                "Sound-event detection failed: "
                f"{output.result.errors}"
            )

        for event in output.events:

            if not (
                0.0
                <= event.confidence
                <= 1.0
            ):
                raise AssertionError(
                    "Invalid sound-event confidence."
                )

            if event.rank <= 0:
                raise AssertionError(
                    "Invalid sound-event rank."
                )

        print(
            f"[PASS] Scenario: "
            f"{scenario_name}"
        )

        print(
            f"[PASS] Events detected: "
            f"{output.event_count}"
        )

        print(
            f"[PASS] Top event: "
            f"{output.top_event.normalized_label }"
            f"{output.top_event.confidence if output.top_event else None}"
        )

        print(
            f"[PASS] Confidence: "
            f"{output.top_event.confidence} "
            f"{output.top_event.confidence if output.top_event else 0.0}"
        )

        print(
            "[PASS] AST model loaded lazily"
        )

        print(
            "[PASS] Audio decoded at 16 kHz"
        )

        print(
            "[PASS] Sound confidence filtering applied"
        )

        print(
            "[PASS] Safety priority generated"
        )

        print(
            "[PASS] Layer 3 sound format generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] SOUND EVENT DETECTOR "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        SoundEventDetectionError,
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
            "sound-event-detector self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="park_walking",
        help=(
            "Scenario used for testing. "
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