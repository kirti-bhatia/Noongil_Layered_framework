"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Speech Recognizer
File    : layer2/audio/speech_recognizer.py
============================================================

Purpose
-------
Transcribes speech from Layer 1 audio using OpenAI Whisper.

Responsibilities:
- validate Layer 1 audio input
- load Whisper lazily
- transcribe speech
- detect language
- extract timestamped speech segments
- estimate transcription confidence
- handle silence safely
- generate a standardized ModuleResult
============================================================
"""

from __future__ import annotations

import argparse
import math
import shutil

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

SPEECH_RECOGNIZER_VERSION = "1.0"

DEFAULT_NO_SPEECH_THRESHOLD = 0.60
DEFAULT_MINIMUM_TEXT_LENGTH = 1


# ============================================================
# EXCEPTION
# ============================================================

class SpeechRecognitionError(Exception):
    """Raised when speech recognition cannot be completed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class SpeechSegment:
    """One timestamped Whisper segment."""

    segment_id: int

    start_time: float
    end_time: float

    text: str

    confidence: float
    no_speech_probability: float

    def to_dict(self) -> Dict[str, Any]:

        return {
            "segment_id": (
                self.segment_id
            ),
            "start_time": round(
                self.start_time,
                3,
            ),
            "end_time": round(
                self.end_time,
                3,
            ),
            "duration": round(
                max(
                    0.0,
                    self.end_time
                    - self.start_time,
                ),
                3,
            ),
            "text": self.text,
            "confidence": round(
                self.confidence,
                6,
            ),
            "no_speech_probability": (
                round(
                    self.no_speech_probability,
                    6,
                )
            ),
        }


@dataclass
class SpeechRecognitionOutput:
    """Complete Whisper transcription output."""

    result: ModuleResult

    transcript: str
    language: Optional[str]

    confidence: Optional[float]

    segments: List[SpeechSegment]

    audio_path: Optional[Path]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def speech_detected(self) -> bool:
        return bool(
            self.transcript.strip()
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "transcript": self.transcript,
            "language": self.language,
            "confidence": self.confidence,
            "speech_detected": (
                self.speech_detected
            ),
            "segment_count": (
                self.segment_count
            ),
            "segments": [
                segment.to_dict()
                for segment in self.segments
            ],
            "audio_path": (
                str(self.audio_path)
                if self.audio_path
                else None
            ),
            "result": self.result.to_dict(),
        }


# ============================================================
# SPEECH RECOGNIZER
# ============================================================

class SpeechRecognizer:
    """OpenAI Whisper speech recognizer."""

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
                "speech_recognizer"
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

    def transcribe(
        self,
        packet: AdaptedLayer1Input,
    ) -> SpeechRecognitionOutput:
        """Transcribe the packet's audio input."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise SpeechRecognitionError(
                "packet must be an "
                "AdaptedLayer1Input."
            )                                        #detecting audio input from layer 1

        if not self._module_enabled():

            result = ModuleResult.skipped(
                module_name=(
                    "speech_recognizer"
                ),
                modality="audio",
                reason=(
                    "Speech recognition is "
                    "disabled in settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return SpeechRecognitionOutput(
                result=result,
                transcript="",
                language=None,
                confidence=None,
                segments=[],
                audio_path=None,
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
                    "speech_recognizer"
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

            return SpeechRecognitionOutput(
                result=result,
                transcript="",
                language=None,
                confidence=None,
                segments=[],
                audio_path=None,
            )

        audio_path = Path(
            audio_input.media_path
        )

        if not audio_path.is_file():

            result = ModuleResult.failure(
                module_name=(
                    "speech_recognizer"
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

            return SpeechRecognitionOutput(
                result=result,
                transcript="",
                language=None,
                confidence=None,
                segments=[],
                audio_path=audio_path,
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        log_event(
            packet_logger,
            event=(
                "speech_recognition_started"
            ),
            message=(
                "Speech recognition started."
            ),
            details={
                "audio_path": str(
                    audio_path
                )
            },
        )

        with ModuleTimer(
            "speech_recognizer",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                self._require_ffmpeg()

                model = self._get_model()

                raw_result = (
                    self._run_whisper(
                        model,
                        audio_path,
                    )
                )

                transcript = (
                    self._clean_transcript(
                        raw_result.get(
                            "text",
                            "",
                        )
                    )
                )

                language = (
                    raw_result.get(
                        "language"
                    )
                )

                if language is not None:
                    language = str(
                        language
                    ).strip().lower()

                segments = (
                    self._parse_segments(
                        raw_result.get(
                            "segments",
                            [],
                        )
                    )
                )

                confidence = (
                    self._overall_confidence(
                        segments
                    )
                )

                speech_detected = (
                    len(transcript)
                    >= DEFAULT_MINIMUM_TEXT_LENGTH
                )

                warnings = []

                if not speech_detected:
                    warnings.append(
                        "No intelligible speech "
                        "was detected."
                    )

                data = {
                    "transcript": transcript,
                    "language": language,
                    "speech_detected": (
                        speech_detected
                    ),
                    "segment_count": (
                        len(segments)
                    ),
                    "segments": [
                        segment.to_dict()
                        for segment
                        in segments
                    ],
                    "audio_path": str(
                        audio_path
                    ),
                    "model_id": (
                        self.model_config
                        .speech_recognition
                        .model_id
                    ),
                    "recognizer_version": (
                        SPEECH_RECOGNIZER_VERSION
                    ),
                }

                status_factory = (
                    ModuleResult.success
                    if speech_detected
                    else ModuleResult.partial
                )

                result = status_factory(
                    module_name=(
                        "speech_recognizer"
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
                            .speech_recognition
                            .backend.value
                        ),
                    },
                )

                log_event(
                    packet_logger,
                    event=(
                        "speech_recognition_completed"
                    ),
                    message=(
                        "Speech recognition completed."
                    ),
                    details={
                        "speech_detected": (
                            speech_detected
                        ),
                        "segment_count": (
                            len(segments)
                        ),
                        "language": language,
                        "confidence": confidence,
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return SpeechRecognitionOutput(
                    result=result,
                    transcript=transcript,
                    language=language,
                    confidence=confidence,
                    segments=segments,
                    audio_path=audio_path,
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "speech_recognition_failed"
                    ),
                    message=(
                        "Speech recognition failed."
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
                        "speech_recognizer"
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
                            .speech_recognition
                            .model_id
                        ),
                    },
                )

                return SpeechRecognitionOutput(
                    result=result,
                    transcript="",
                    language=None,
                    confidence=None,
                    segments=[],
                    audio_path=audio_path,
                )

    # --------------------------------------------------------
    # SETTINGS AND DEPENDENCIES
    # --------------------------------------------------------

    def _module_enabled(self) -> bool:
        """Return whether speech recognition is enabled."""

        modules = self.settings.modules

        if hasattr(
            modules,
            "speech_recognition",
        ):
            return bool(
                modules.speech_recognition
            )

        if hasattr(
            modules,
            "audio_processing",
        ):
            return bool(
                modules.audio_processing
            )

        return True

    @staticmethod
    def _require_ffmpeg() -> None:
        """Ensure Whisper can decode the audio file."""

        if shutil.which("ffmpeg") is None:
            raise DependencyMissingError(
                "FFmpeg is required by Whisper but "
                "was not found in PATH. Install it "
                "and restart PowerShell.",
                module="speech_recognizer",
            )

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    def _get_model(self) -> Any:
        """Load and return Whisper."""

        loaded = self.model_loader.load(
            "speech_recognition"
        )

        if isinstance(
            loaded,
            ModelBundle,
        ):
            model = loaded.model
        else:
            model = loaded

        if not callable(
            getattr(
                model,
                "transcribe",
                None,
            )
        ):
            raise ModelLoadingError(
                "Loaded Whisper model has no "
                "callable transcribe() method.",
                module="speech_recognizer",
            )

        return model

    def _run_whisper(
        self,
        model: Any,
        audio_path: Path,
    ) -> Dict[str, Any]:
        """Run configured Whisper transcription."""

        parameters = dict(
            self.model_config
            .speech_recognition
            .parameters
        )

        language = parameters.get(
            "language"
        )

        task = str(
            parameters.get(
                "task",
                "transcribe",
            )
        )

        temperature = float(
            parameters.get(
                "temperature",
                0.0,
            )
        )

        options: Dict[str, Any] = {
            "task": task,
            "temperature": temperature,
            "verbose": False,
            "fp16": False,
        }

        if language:
            options["language"] = str(
                language
            )

        raw_result = model.transcribe(
            str(audio_path),
            **options,
        )

        if not isinstance(
            raw_result,
            dict,
        ):
            raise SpeechRecognitionError(
                "Whisper output must be a dictionary."
            )

        return raw_result

    # --------------------------------------------------------
    # OUTPUT PROCESSING
    # --------------------------------------------------------

    @staticmethod
    def _clean_transcript(
        transcript: Any,
    ) -> str:
        """Normalize Whisper transcript text."""

        if not isinstance(
            transcript,
            str,
        ):
            return ""

        return " ".join(
            transcript.strip().split()
        )

    def _parse_segments(
        self,
        raw_segments: Any,
    ) -> List[SpeechSegment]:
        """Parse Whisper timestamped segments."""

        if not isinstance(
            raw_segments,
            list,
        ):
            return []

        segments = []

        for index, raw_segment in enumerate(
            raw_segments,
            start=1,
        ):

            if not isinstance(
                raw_segment,
                dict,
            ):
                continue

            text = self._clean_transcript(
                raw_segment.get(
                    "text",
                    "",
                )
            )

            if not text:
                continue

            start_time = self._safe_float(
                raw_segment.get(
                    "start",
                    0.0,
                ),
                default=0.0,
            )

            end_time = self._safe_float(
                raw_segment.get(
                    "end",
                    start_time,
                ),
                default=start_time,
            )

            start_time = max(
                0.0,
                start_time,
            )

            end_time = max(
                start_time,
                end_time,
            )

            average_log_probability = (
                self._safe_float(
                    raw_segment.get(
                        "avg_logprob",
                        -10.0,
                    ),
                    default=-10.0,
                )
            )

            confidence = math.exp(
                min(
                    0.0,
                    average_log_probability,
                )
            )

            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )

            no_speech_probability = (
                self._safe_float(
                    raw_segment.get(
                        "no_speech_prob",
                        0.0,
                    ),
                    default=0.0,
                )
            )

            no_speech_probability = max(
                0.0,
                min(
                    1.0,
                    no_speech_probability,
                ),
            )

            if (
                no_speech_probability
                >= DEFAULT_NO_SPEECH_THRESHOLD
            ):
                continue

            segments.append(
                SpeechSegment(
                    segment_id=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    confidence=confidence,
                    no_speech_probability=(
                        no_speech_probability
                    ),
                )
            )

        return segments

    @staticmethod
    def _overall_confidence(
        segments: List[
            SpeechSegment
        ],
    ) -> float:
        """Calculate duration-weighted confidence."""

        if not segments:
            return 0.0

        weighted_sum = 0.0
        total_duration = 0.0

        for segment in segments:

            duration = max(
                0.01,
                segment.end_time
                - segment.start_time,
            )

            weighted_sum += (
                segment.confidence
                * duration
            )

            total_duration += duration

        if total_duration <= 0.0:
            return 0.0

        confidence = (
            weighted_sum
            / total_duration
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            6,
        )

    @staticmethod
    def _safe_float(
        value: Any,
        *,
        default: float,
    ) -> float:
        """Safely convert a value to finite float."""

        try:
            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return default

        if not math.isfinite(result):
            return default

        return result


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def recognize_speech(
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
) -> SpeechRecognitionOutput:

    recognizer = SpeechRecognizer(
        settings=settings,
        model_config=model_config,
        model_loader=model_loader,
    )

    return recognizer.transcribe(
        packet
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "classroom",
) -> bool:

    print("=" * 72)

    print(
        "NOONGIL-X | SPEECH RECOGNIZER "
        "SELF-TEST"
    )

    print("=" * 72)

    print(
        "The first run may download the "
        "configured Whisper model."
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

        recognizer = SpeechRecognizer(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        packet = adapter.load_scenario(
            scenario_name
        )

        output = recognizer.transcribe(
            packet
        )

        if not output.succeeded:
            raise AssertionError(
                "Speech recognition failed: "
                f"{output.result.errors}"
            )

        if (
            output.confidence is not None
            and not (
                0.0
                <= output.confidence
                <= 1.0
            )
        ):
            raise AssertionError(
                "Speech confidence is invalid."
            )

        for segment in output.segments:

            if (
                segment.start_time < 0.0
                or segment.end_time
                < segment.start_time
            ):
                raise AssertionError(
                    "Invalid segment timestamps."
                )

            if not (
                0.0
                <= segment.confidence
                <= 1.0
            ):
                raise AssertionError(
                    "Invalid segment confidence."
                )

        print(
            f"[PASS] Scenario: "
            f"{scenario_name}"
        )

        print(
            f"[PASS] Speech detected: "
            f"{output.speech_detected}"
        )

        print(
            f"[PASS] Language: "
            f"{output.language}"
        )

        print(
            f"[PASS] Segments: "
            f"{output.segment_count}"
        )

        print(
            f"[PASS] Confidence: "
            f"{output.confidence}"
        )

        print(
            f"[PASS] Transcript: "
            f"{output.transcript!r}"
        )

        print(
            "[PASS] Whisper loaded lazily"
        )

        print(
            "[PASS] Segment timestamps validated"
        )

        print(
            "[PASS] Speech confidence validated"
        )

        print(
            "[PASS] Layer 3 transcript format generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] SPEECH RECOGNIZER "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        SpeechRecognitionError,
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
            "speech-recognizer self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="classroom",
        help=(
            "Scenario used for testing. "
            "Default: classroom"
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