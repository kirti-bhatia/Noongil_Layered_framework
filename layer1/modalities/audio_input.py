"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Audio Input Processor
File    : layer1/modalities/audio_input.py
============================================================

Purpose
-------
Consumes normalized audio packets from MultimodalReceiver and
produces validated Layer 1 AudioData objects.

Responsibilities
----------------
1. Validate audio packet structure
2. Decode supported base64 audio payloads
3. Extract audio metadata
4. Perform signal-level preprocessing metadata
5. Estimate amplitude, clipping, silence, packet integrity,
   and a simple signal-to-noise quality score
6. Save raw/preprocessed WAV files when configured
7. Build AudioData for the final Multimodal Sensor Packet
8. Log processing, quality, and errors
9. Provide diagnostics and a standalone self-test

Architectural Boundary
----------------------
This module does NOT perform:
- speech transcription;
- command interpretation;
- emotion recognition;
- sound-event classification;
- semantic audio understanding;
- reasoning;
- LLM processing.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import audioop
import base64
import io
import json
import math
import statistics
import struct
import time
import wave

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from layer1.acquisition.multimodal_receiver import (
    MultimodalReceiver,
    ReceivedSensorPacket,
)
from layer1.config.paths import (
    PREPROCESSED_AUDIO_DIR,
    RAW_AUDIO_DIR,
    ensure_directory,
)
from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.schemas.sensor_packet import (
    AudioData,
    ModalityMetadata,
    ModalityStatus,
)
from layer1.utils.logger import (
    PipelineTimer,
    get_logger,
    log_exception,
    log_sensor_event,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class AudioInputError(Exception):
    """Base exception for audio input processing."""


class AudioPacketValidationError(AudioInputError):
    """Raised when an audio packet is invalid."""


class AudioDecodeError(AudioInputError):
    """Raised when encoded audio cannot be decoded."""


class AudioProcessingError(AudioInputError):
    """Raised when signal-level audio processing fails."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AudioProcessingResult:
    """
    Result returned after processing one audio packet.
    """

    success: bool
    audio_data: Optional[AudioData] = None
    packet_id: Optional[str] = None
    chunk_id: Optional[str] = None
    raw_audio_path: Optional[str] = None
    preprocessed_audio_path: Optional[str] = None
    processing_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudioProcessorStatistics:
    """
    Runtime statistics for AudioInputProcessor.
    """

    total_received: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_metadata_only: int = 0
    total_audio_decoded: int = 0
    total_saved_raw: int = 0
    total_saved_preprocessed: int = 0
    cumulative_processing_seconds: float = 0.0
    last_packet_id: Optional[str] = None
    last_chunk_id: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return (
            self.cumulative_processing_seconds
            / self.total_processed
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["average_processing_seconds"] = (
            self.average_processing_seconds
        )
        return payload


# ============================================================
# HELPERS
# ============================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def require_positive_int(
    value: Any,
    field_name: str,
) -> int:
    if isinstance(value, bool):
        raise AudioPacketValidationError(
            f"{field_name} must be a positive integer."
        )

    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise AudioPacketValidationError(
            f"{field_name} must be a positive integer."
        ) from error

    if parsed <= 0:
        raise AudioPacketValidationError(
            f"{field_name} must be a positive integer."
        )

    return parsed


def require_non_negative_float(
    value: Any,
    field_name: str,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AudioPacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed) or parsed < 0:
        raise AudioPacketValidationError(
            f"{field_name} must be finite and non-negative."
        )

    return parsed


def require_probability(
    value: Any,
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AudioPacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed):
        raise AudioPacketValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= parsed <= 1.0:
        raise AudioPacketValidationError(
            f"{field_name} must be between 0.0 and 1.0."
        )

    return parsed


def safe_filename_component(value: str) -> str:
    normalized = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in value
    )
    return normalized.strip("_") or "audio"


# ============================================================
# AUDIO INPUT PROCESSOR
# ============================================================

class AudioInputProcessor:
    """
    Convert receiver audio packets into validated AudioData.

    Two processing paths are supported:

    1. Metadata-only processing
       Used by the current phone simulator.

    2. Audio-backed processing
       Used later when the phone sends actual base64 PCM or WAV
       audio data.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger("modalities.audio_input")
        self.statistics = AudioProcessorStatistics()

        ensure_directory(RAW_AUDIO_DIR)
        ensure_directory(PREPROCESSED_AUDIO_DIR)

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process_packet(
        self,
        packet: ReceivedSensorPacket,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> AudioProcessingResult:
        """
        Process one normalized audio packet.
        """

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        self.statistics.total_received += 1
        started = time.perf_counter()

        try:
            with PipelineTimer(
                "audio_input.process_packet",
                logger=self.logger,
                metadata={
                    "packet_id": packet.packet_id,
                    "device_id": packet.device_id,
                },
            ):
                self._validate_packet(packet)

                warnings: List[str] = []
                decoded = self._decode_audio_if_present(
                    packet.payload
                )

                if decoded is None:
                    self.statistics.total_metadata_only += 1
                    audio_data = self._build_from_metadata(
                        packet,
                        warnings=warnings,
                    )
                    raw_path = None
                    preprocessed_path = None
                else:
                    self.statistics.total_audio_decoded += 1
                    (
                        audio_data,
                        raw_path,
                        preprocessed_path,
                    ) = self._process_audio_backed_packet(
                        packet,
                        decoded,
                        warnings=warnings,
                    )

                audio_data.validate()

                elapsed = time.perf_counter() - started

                self.statistics.total_processed += 1
                self.statistics.cumulative_processing_seconds += (
                    elapsed
                )
                self.statistics.last_packet_id = packet.packet_id
                self.statistics.last_chunk_id = (
                    audio_data.chunk_id
                )
                self.statistics.last_error = None

                log_sensor_event(
                    modality="audio",
                    event="Audio packet processed",
                    device_id=packet.device_id,
                    sensor_type=packet.sensor_type,
                    packet_id=packet.packet_id,
                    sequence_number=packet.sequence_number,
                    details={
                        "chunk_id": audio_data.chunk_id,
                        "sample_rate_hz": (
                            audio_data.sample_rate_hz
                        ),
                        "duration_ms": audio_data.duration_ms,
                        "signal_to_noise_score": (
                            audio_data.signal_to_noise_score
                        ),
                        "clipping_ratio": (
                            audio_data.clipping_ratio
                        ),
                        "silence_ratio": (
                            audio_data.silence_ratio
                        ),
                        "processing_seconds": round(
                            elapsed,
                            6,
                        ),
                        "metadata_only": decoded is None,
                    },
                )

                return AudioProcessingResult(
                    success=True,
                    audio_data=audio_data,
                    packet_id=packet.packet_id,
                    chunk_id=audio_data.chunk_id,
                    raw_audio_path=raw_path,
                    preprocessed_audio_path=(
                        preprocessed_path
                    ),
                    processing_seconds=elapsed,
                    warnings=warnings,
                )

        except Exception as error:
            elapsed = time.perf_counter() - started

            self.statistics.total_failed += 1
            self.statistics.last_packet_id = getattr(
                packet,
                "packet_id",
                None,
            )
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Audio packet processing failed",
                error=error,
                details={
                    "packet_id": getattr(
                        packet,
                        "packet_id",
                        None,
                    ),
                    "device_id": getattr(
                        packet,
                        "device_id",
                        None,
                    ),
                },
            )

            if should_raise:
                raise

            return AudioProcessingResult(
                success=False,
                packet_id=getattr(
                    packet,
                    "packet_id",
                    None,
                ),
                processing_seconds=elapsed,
                error=f"{type(error).__name__}: {error}",
            )

    def process_receiver_queue(
        self,
        receiver: MultimodalReceiver,
        *,
        maximum_items: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
    ) -> List[AudioProcessingResult]:
        """
        Drain and process audio packets from MultimodalReceiver.
        """

        packets = receiver.drain(
            "audio",
            maximum_items=maximum_items,
        )

        return [
            self.process_packet(
                packet,
                raise_on_error=raise_on_error,
            )
            for packet in packets
        ]

    def process_latest_from_receiver(
        self,
        receiver: MultimodalReceiver,
        *,
        remove: bool = True,
        raise_on_error: Optional[bool] = None,
    ) -> Optional[AudioProcessingResult]:
        """
        Process the most recent audio packet from a receiver.
        """

        packet = receiver.get_latest(
            "audio",
            remove=remove,
        )

        if packet is None:
            return None

        return self.process_packet(
            packet,
            raise_on_error=raise_on_error,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_packet(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        if not isinstance(packet, ReceivedSensorPacket):
            raise AudioPacketValidationError(
                "packet must be ReceivedSensorPacket."
            )

        packet.validate()

        if packet.modality != "audio":
            raise AudioPacketValidationError(
                "AudioInputProcessor accepts only "
                "modality='audio'."
            )

        if not isinstance(packet.payload, dict):
            raise AudioPacketValidationError(
                "Audio packet payload must be a dictionary."
            )

    # ========================================================
    # DECODING
    # ========================================================

    def _decode_audio_if_present(
        self,
        payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Decode actual audio data when present.

        Supported keys:
        - encoded_audio
        - audio_base64
        - pcm_base64
        - wav_base64
        - audio_bytes

        Returns
        -------
        dict or None
            {
                "raw_bytes": ...,
                "pcm_bytes": ...,
                "sample_rate_hz": ...,
                "channels": ...,
                "sample_width_bits": ...,
                "encoding": ...
            }
        """

        encoded_value = (
            payload.get("encoded_audio")
            or payload.get("audio_base64")
            or payload.get("pcm_base64")
            or payload.get("wav_base64")
            or payload.get("audio_bytes")
        )

        if encoded_value is None:
            return None

        if isinstance(encoded_value, Mapping):
            encoded_value = encoded_value.get("data")

        if not isinstance(encoded_value, str):
            raise AudioDecodeError(
                "Encoded audio must be a base64 string."
            )

        try:
            raw_bytes = base64.b64decode(
                encoded_value,
                validate=True,
            )
        except Exception as error:
            raise AudioDecodeError(
                "Encoded audio is not valid base64."
            ) from error

        encoding = str(
            payload.get(
                "encoding",
                self.settings.audio.encoding.value,
            )
        ).lower()

        if encoding == "wav":
            return self._decode_wav_bytes(raw_bytes)

        sample_rate_hz = require_positive_int(
            payload.get(
                "sample_rate_hz",
                self.settings.audio.sample_rate_hz,
            ),
            "sample_rate_hz",
        )

        channels = require_positive_int(
            payload.get(
                "channels",
                self.settings.audio.channels,
            ),
            "channels",
        )

        sample_width_bits = require_positive_int(
            payload.get(
                "sample_width_bits",
                self.settings.audio.sample_width_bits,
            ),
            "sample_width_bits",
        )

        if sample_width_bits not in {8, 16, 24, 32}:
            raise AudioDecodeError(
                "sample_width_bits must be one of "
                "{8, 16, 24, 32}."
            )

        return {
            "raw_bytes": raw_bytes,
            "pcm_bytes": raw_bytes,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "sample_width_bits": sample_width_bits,
            "encoding": encoding,
        }

    def _decode_wav_bytes(
        self,
        raw_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Decode WAV bytes using the standard wave module.
        """

        try:
            with wave.open(
                io.BytesIO(raw_bytes),
                "rb",
            ) as reader:
                channels = reader.getnchannels()
                sample_width_bytes = reader.getsampwidth()
                sample_rate_hz = reader.getframerate()
                frame_count = reader.getnframes()
                pcm_bytes = reader.readframes(frame_count)
        except Exception as error:
            raise AudioDecodeError(
                "Unable to decode WAV audio."
            ) from error

        return {
            "raw_bytes": raw_bytes,
            "pcm_bytes": pcm_bytes,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "sample_width_bits": (
                sample_width_bytes * 8
            ),
            "encoding": "wav",
        }

    # ========================================================
    # METADATA-ONLY PROCESSING
    # ========================================================

    def _build_from_metadata(
        self,
        packet: ReceivedSensorPacket,
        *,
        warnings: List[str],
    ) -> AudioData:
        payload = packet.payload

        chunk_id = str(
            payload.get("chunk_id")
            or packet.packet_id
        )

        sample_rate_hz = require_positive_int(
            payload.get(
                "sample_rate_hz",
                self.settings.audio.sample_rate_hz,
            ),
            "sample_rate_hz",
        )

        channels = require_positive_int(
            payload.get(
                "channels",
                self.settings.audio.channels,
            ),
            "channels",
        )

        sample_width_bits = require_positive_int(
            payload.get(
                "sample_width_bits",
                self.settings.audio.sample_width_bits,
            ),
            "sample_width_bits",
        )

        if sample_width_bits not in {8, 16, 24, 32}:
            raise AudioPacketValidationError(
                "sample_width_bits must be one of "
                "{8, 16, 24, 32}."
            )

        duration_ms = require_non_negative_float(
            payload.get(
                "duration_ms",
                self.settings.audio.chunk_duration_ms,
            ),
            "duration_ms",
        )

        amplitude = require_probability(
            payload.get("amplitude_score"),
            "amplitude_score",
        )
        snr = require_probability(
            payload.get("signal_to_noise_score"),
            "signal_to_noise_score",
        )
        clipping = require_probability(
            payload.get("clipping_ratio"),
            "clipping_ratio",
        )
        silence = require_probability(
            payload.get("silence_ratio"),
            "silence_ratio",
        )
        integrity = require_probability(
            payload.get("packet_integrity_score"),
            "packet_integrity_score",
        )

        if amplitude is None:
            amplitude = 0.50
            warnings.append(
                "amplitude_score_missing_default_used"
            )

        if snr is None:
            snr = 0.50
            warnings.append(
                "signal_to_noise_score_missing_default_used"
            )

        if clipping is None:
            clipping = 0.0
            warnings.append(
                "clipping_ratio_missing_default_used"
            )

        if silence is None:
            silence = 0.0
            warnings.append(
                "silence_ratio_missing_default_used"
            )

        if integrity is None:
            integrity = 1.0
            warnings.append(
                "packet_integrity_score_missing_default_used"
            )

        limitations: List[str] = []

        if snr < (
            self.settings.audio.minimum_signal_to_noise_score
        ):
            limitations.append("low_signal_to_noise")

        if clipping > (
            self.settings.audio.maximum_clipping_ratio
        ):
            limitations.append("high_clipping")

        if silence >= 0.95:
            limitations.append("mostly_silent")

        if payload.get("degraded") is True:
            limitations.append("simulated_degraded_quality")

        preprocessing_steps = [
            "packet_validation",
            "metadata_extraction",
        ]

        if self.settings.audio.convert_to_mono:
            preprocessing_steps.append(
                "mono_conversion_requested_metadata_only"
            )

        if self.settings.audio.normalize_amplitude:
            preprocessing_steps.append(
                "amplitude_normalization_requested_metadata_only"
            )

        if self.settings.audio.generate_mel_spectrogram:
            preprocessing_steps.append(
                "mel_spectrogram_deferred"
            )

        if self.settings.audio.generate_mfcc:
            preprocessing_steps.append(
                "mfcc_deferred"
            )

        metadata = ModalityMetadata(
            modality="audio",
            status=ModalityStatus.OBSERVED,
            source_timestamp=packet.source_timestamp,
            arrival_timestamp=packet.arrival_timestamp,
            sequence_number=packet.sequence_number,
            sampling_rate_hz=packet.sampling_rate_hz,
            latency_ms=packet.latency_ms,
            source_device_id=packet.device_id,
            data_reference=str(
                payload.get(
                    "audio_reference",
                    packet.packet_id,
                )
            ),
            preprocessing_steps=preprocessing_steps,
            limitations=limitations,
            metadata={
                "sensor_type": packet.sensor_type,
                "payload_encoding": (
                    packet.payload_encoding.value
                ),
                "simulated": bool(
                    packet.metadata.get("simulated", False)
                ),
                "scenario": packet.metadata.get("scenario"),
                "metadata_only": True,
            },
        )

        return AudioData(
            metadata=metadata,
            chunk_id=chunk_id,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bits=sample_width_bits,
            duration_ms=duration_ms,
            encoding=str(
                payload.get(
                    "encoding",
                    self.settings.audio.encoding.value,
                )
            ),
            amplitude_score=amplitude,
            signal_to_noise_score=snr,
            clipping_ratio=clipping,
            silence_ratio=silence,
            packet_integrity_score=integrity,
            audio_path=(
                str(payload.get("audio_path"))
                if payload.get("audio_path")
                else None
            ),
            encoded_audio=None,
            feature_type=(
                str(payload.get("feature_type"))
                if payload.get("feature_type")
                else None
            ),
            feature_reference=(
                str(payload.get("feature_reference"))
                if payload.get("feature_reference")
                else None
            ),
        )

    # ========================================================
    # AUDIO-BACKED PROCESSING
    # ========================================================

    def _process_audio_backed_packet(
        self,
        packet: ReceivedSensorPacket,
        decoded: Mapping[str, Any],
        *,
        warnings: List[str],
    ) -> Tuple[AudioData, Optional[str], Optional[str]]:
        payload = packet.payload

        chunk_id = str(
            payload.get("chunk_id")
            or packet.packet_id
        )

        pcm_bytes = bytes(decoded["pcm_bytes"])
        raw_bytes = bytes(decoded["raw_bytes"])

        sample_rate_hz = int(
            decoded["sample_rate_hz"]
        )
        channels = int(decoded["channels"])
        sample_width_bits = int(
            decoded["sample_width_bits"]
        )
        sample_width_bytes = sample_width_bits // 8
        encoding = str(decoded["encoding"])

        if sample_width_bytes <= 0:
            raise AudioProcessingError(
                "Invalid sample width."
            )

        preprocessing_steps = [
            "packet_validation",
            "base64_decode",
        ]

        raw_path: Optional[str] = None
        preprocessed_path: Optional[str] = None

        if self.settings.audio.save_raw_audio:
            raw_path = self._save_audio_bytes(
                raw_bytes=raw_bytes,
                pcm_bytes=pcm_bytes,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                sample_width_bytes=sample_width_bytes,
                chunk_id=chunk_id,
                suffix="raw",
                source_encoding=encoding,
                directory=RAW_AUDIO_DIR,
            )
            self.statistics.total_saved_raw += 1

        working_pcm = pcm_bytes
        working_channels = channels

        if (
            self.settings.audio.convert_to_mono
            and channels > 1
        ):
            working_pcm = self._convert_to_mono(
                working_pcm,
                sample_width_bytes=sample_width_bytes,
                channels=channels,
            )
            working_channels = 1
            preprocessing_steps.append("convert_to_mono")

        if self.settings.audio.normalize_amplitude:
            working_pcm = self._normalize_pcm(
                working_pcm,
                sample_width_bytes=sample_width_bytes,
            )
            preprocessing_steps.append(
                "amplitude_normalization"
            )

        if self.settings.audio.basic_noise_reduction:
            warnings.append(
                "basic_noise_reduction_requested_but_not_"
                "applied_to_avoid_signal_distortion"
            )

        amplitude_score = self._calculate_amplitude_score(
            working_pcm,
            sample_width_bytes=sample_width_bytes,
        )
        clipping_ratio = self._calculate_clipping_ratio(
            working_pcm,
            sample_width_bytes=sample_width_bytes,
        )
        silence_ratio = self._calculate_silence_ratio(
            working_pcm,
            sample_width_bytes=sample_width_bytes,
        )
        signal_to_noise_score = (
            self._estimate_signal_to_noise_score(
                working_pcm,
                sample_width_bytes=sample_width_bytes,
            )
        )
        integrity_score = self._calculate_integrity_score(
            working_pcm,
            sample_width_bytes=sample_width_bytes,
            channels=working_channels,
        )

        frame_count = (
            len(working_pcm)
            / max(
                sample_width_bytes * working_channels,
                1,
            )
        )

        duration_ms = (
            frame_count
            / max(sample_rate_hz, 1)
            * 1000.0
        )

        limitations: List[str] = []

        if signal_to_noise_score < (
            self.settings.audio.minimum_signal_to_noise_score
        ):
            limitations.append("low_signal_to_noise")

        if clipping_ratio > (
            self.settings.audio.maximum_clipping_ratio
        ):
            limitations.append("high_clipping")

        if silence_ratio >= 0.95:
            limitations.append("mostly_silent")

        if self.settings.audio.save_preprocessed_audio:
            preprocessed_path = self._save_wav(
                pcm_bytes=working_pcm,
                sample_rate_hz=sample_rate_hz,
                channels=working_channels,
                sample_width_bytes=sample_width_bytes,
                chunk_id=chunk_id,
                suffix="preprocessed",
                directory=PREPROCESSED_AUDIO_DIR,
            )
            self.statistics.total_saved_preprocessed += 1

        metadata = ModalityMetadata(
            modality="audio",
            status=ModalityStatus.OBSERVED,
            source_timestamp=packet.source_timestamp,
            arrival_timestamp=packet.arrival_timestamp,
            sequence_number=packet.sequence_number,
            sampling_rate_hz=packet.sampling_rate_hz,
            latency_ms=packet.latency_ms,
            source_device_id=packet.device_id,
            data_reference=(
                preprocessed_path
                or raw_path
                or str(
                    payload.get(
                        "audio_reference",
                        packet.packet_id,
                    )
                )
            ),
            preprocessing_steps=preprocessing_steps,
            limitations=limitations,
            metadata={
                "sensor_type": packet.sensor_type,
                "payload_encoding": (
                    packet.payload_encoding.value
                ),
                "simulated": bool(
                    packet.metadata.get("simulated", False)
                ),
                "scenario": packet.metadata.get("scenario"),
                "metadata_only": False,
                "source_encoding": encoding,
                "original_channels": channels,
            },
        )

        audio_data = AudioData(
            metadata=metadata,
            chunk_id=chunk_id,
            sample_rate_hz=sample_rate_hz,
            channels=working_channels,
            sample_width_bits=sample_width_bits,
            duration_ms=round(duration_ms, 3),
            encoding="wav",
            amplitude_score=amplitude_score,
            signal_to_noise_score=signal_to_noise_score,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
            packet_integrity_score=integrity_score,
            audio_path=preprocessed_path or raw_path,
            encoded_audio=None,
            feature_type=None,
            feature_reference=None,
        )

        return audio_data, raw_path, preprocessed_path

    # ========================================================
    # SIGNAL PROCESSING HELPERS
    # ========================================================

    def _convert_to_mono(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
        channels: int,
    ) -> bytes:
        """
        Convert interleaved multi-channel PCM to mono.

        Stereo uses audioop.tomono. More than two channels are
        averaged manually.
        """

        if channels == 1:
            return pcm_bytes

        if channels == 2:
            return audioop.tomono(
                pcm_bytes,
                sample_width_bytes,
                0.5,
                0.5,
            )

        samples = self._decode_samples(
            pcm_bytes,
            sample_width_bytes=sample_width_bytes,
        )

        if len(samples) % channels != 0:
            raise AudioProcessingError(
                "PCM sample count is not divisible by channels."
            )

        mono_samples: List[int] = []

        for index in range(0, len(samples), channels):
            frame = samples[index:index + channels]
            mono_samples.append(
                int(round(statistics.fmean(frame)))
            )

        return self._encode_samples(
            mono_samples,
            sample_width_bytes=sample_width_bytes,
        )

    def _normalize_pcm(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
    ) -> bytes:
        if not pcm_bytes:
            return pcm_bytes

        maximum = audioop.max(
            pcm_bytes,
            sample_width_bytes,
        )

        if maximum <= 0:
            return pcm_bytes

        full_scale = (
            2 ** (sample_width_bytes * 8 - 1)
        ) - 1

        target = full_scale * 0.90
        factor = target / maximum

        if factor <= 1.0:
            return pcm_bytes

        return audioop.mul(
            pcm_bytes,
            sample_width_bytes,
            factor,
        )

    def _calculate_amplitude_score(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
    ) -> float:
        if not pcm_bytes:
            return 0.0

        rms = audioop.rms(
            pcm_bytes,
            sample_width_bytes,
        )

        full_scale = (
            2 ** (sample_width_bytes * 8 - 1)
        ) - 1

        return round(
            clamp(rms / max(full_scale, 1), 0.0, 1.0),
            6,
        )

    def _calculate_clipping_ratio(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
    ) -> float:
        samples = self._decode_samples(
            pcm_bytes,
            sample_width_bytes=sample_width_bytes,
        )

        if not samples:
            return 0.0

        full_scale = (
            2 ** (sample_width_bytes * 8 - 1)
        ) - 1

        threshold = full_scale * 0.98

        clipped = sum(
            1
            for sample in samples
            if abs(sample) >= threshold
        )

        return round(
            clamp(clipped / len(samples), 0.0, 1.0),
            6,
        )

    def _calculate_silence_ratio(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
    ) -> float:
        samples = self._decode_samples(
            pcm_bytes,
            sample_width_bytes=sample_width_bytes,
        )

        if not samples:
            return 1.0

        full_scale = (
            2 ** (sample_width_bytes * 8 - 1)
        ) - 1

        threshold = (
            full_scale
            * self.settings.audio.silence_threshold
        )

        silent = sum(
            1
            for sample in samples
            if abs(sample) <= threshold
        )

        return round(
            clamp(silent / len(samples), 0.0, 1.0),
            6,
        )

    def _estimate_signal_to_noise_score(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
    ) -> float:
        """
        Estimate signal quality from robust amplitude statistics.

        This is not a physical SNR measurement. It is a bounded
        signal-quality indicator suitable for Layer 1 confidence.
        """

        samples = self._decode_samples(
            pcm_bytes,
            sample_width_bytes=sample_width_bytes,
        )

        if len(samples) < 4:
            return 0.0

        absolute_values = sorted(
            abs(sample)
            for sample in samples
        )

        signal_start = int(len(absolute_values) * 0.75)
        noise_end = max(
            1,
            int(len(absolute_values) * 0.25),
        )

        signal_values = absolute_values[signal_start:]
        noise_values = absolute_values[:noise_end]

        signal_level = (
            statistics.fmean(signal_values)
            if signal_values
            else 0.0
        )

        noise_level = (
            statistics.fmean(noise_values)
            if noise_values
            else 0.0
        )

        if signal_level <= 0:
            return 0.0

        ratio = signal_level / max(noise_level, 1.0)

        score = math.log10(1.0 + ratio) / 2.0

        return round(
            clamp(score, 0.0, 1.0),
            6,
        )

    def _calculate_integrity_score(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
        channels: int,
    ) -> float:
        frame_size = sample_width_bytes * channels

        if frame_size <= 0:
            return 0.0

        if not pcm_bytes:
            return 0.0

        remainder = len(pcm_bytes) % frame_size

        if remainder == 0:
            return 1.0

        return round(
            clamp(
                1.0 - (remainder / frame_size),
                0.0,
                1.0,
            ),
            6,
        )

    def _decode_samples(
        self,
        pcm_bytes: bytes,
        *,
        sample_width_bytes: int,
    ) -> List[int]:
        if sample_width_bytes == 1:
            return [
                value - 128
                for value in pcm_bytes
            ]

        if sample_width_bytes == 2:
            count = len(pcm_bytes) // 2
            return list(
                struct.unpack(
                    f"<{count}h",
                    pcm_bytes[:count * 2],
                )
            )

        if sample_width_bytes == 4:
            count = len(pcm_bytes) // 4
            return list(
                struct.unpack(
                    f"<{count}i",
                    pcm_bytes[:count * 4],
                )
            )

        if sample_width_bytes == 3:
            samples: List[int] = []

            for index in range(
                0,
                len(pcm_bytes) - 2,
                3,
            ):
                value = int.from_bytes(
                    pcm_bytes[index:index + 3],
                    byteorder="little",
                    signed=False,
                )

                if value & 0x800000:
                    value -= 1 << 24

                samples.append(value)

            return samples

        raise AudioProcessingError(
            f"Unsupported sample width: "
            f"{sample_width_bytes} bytes."
        )

    def _encode_samples(
        self,
        samples: Sequence[int],
        *,
        sample_width_bytes: int,
    ) -> bytes:
        if sample_width_bytes == 1:
            return bytes(
                clamp(sample + 128, 0, 255)
                for sample in samples
            )

        if sample_width_bytes == 2:
            bounded = [
                int(clamp(sample, -32768, 32767))
                for sample in samples
            ]
            return struct.pack(
                f"<{len(bounded)}h",
                *bounded,
            )

        if sample_width_bytes == 4:
            minimum = -(2 ** 31)
            maximum = (2 ** 31) - 1
            bounded = [
                int(clamp(sample, minimum, maximum))
                for sample in samples
            ]
            return struct.pack(
                f"<{len(bounded)}i",
                *bounded,
            )

        if sample_width_bytes == 3:
            output = bytearray()

            for sample in samples:
                bounded = int(
                    clamp(
                        sample,
                        -(2 ** 23),
                        (2 ** 23) - 1,
                    )
                )

                if bounded < 0:
                    bounded += 1 << 24

                output.extend(
                    bounded.to_bytes(
                        3,
                        byteorder="little",
                        signed=False,
                    )
                )

            return bytes(output)

        raise AudioProcessingError(
            f"Unsupported sample width: "
            f"{sample_width_bytes} bytes."
        )

    # ========================================================
    # FILE OUTPUT
    # ========================================================

    def _save_audio_bytes(
        self,
        *,
        raw_bytes: bytes,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        channels: int,
        sample_width_bytes: int,
        chunk_id: str,
        suffix: str,
        source_encoding: str,
        directory: Path,
    ) -> str:
        if source_encoding == "wav":
            ensure_directory(directory)

            filename = (
                f"{safe_filename_component(chunk_id)}_"
                f"{safe_filename_component(suffix)}.wav"
            )

            output_path = directory / filename
            output_path.write_bytes(raw_bytes)

            return str(output_path.resolve())

        return self._save_wav(
            pcm_bytes=pcm_bytes,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            chunk_id=chunk_id,
            suffix=suffix,
            directory=directory,
        )

    def _save_wav(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        channels: int,
        sample_width_bytes: int,
        chunk_id: str,
        suffix: str,
        directory: Path,
    ) -> str:
        ensure_directory(directory)

        filename = (
            f"{safe_filename_component(chunk_id)}_"
            f"{safe_filename_component(suffix)}.wav"
        )

        output_path = directory / filename

        try:
            with wave.open(
                str(output_path),
                "wb",
            ) as writer:
                writer.setnchannels(channels)
                writer.setsampwidth(sample_width_bytes)
                writer.setframerate(sample_rate_hz)
                writer.writeframes(pcm_bytes)
        except Exception as error:
            raise AudioProcessingError(
                f"Unable to save WAV file: {output_path}"
            ) from error

        return str(output_path.resolve())

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "audio_enabled": self.settings.audio.enabled,
            "raw_output_dir": str(RAW_AUDIO_DIR),
            "preprocessed_output_dir": str(
                PREPROCESSED_AUDIO_DIR
            ),
            "statistics": self.statistics.to_dict(),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_audio_input_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 AUDIO INPUT TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating test settings...")

        settings = create_test_settings()
        settings.audio.save_raw_audio = False
        settings.audio.save_preprocessed_audio = False

        processor = AudioInputProcessor(settings)

        print("[SUCCESS] Audio processor initialized.")

        print("[2/6] Creating receiver and simulator...")

        from layer1.acquisition.phone_sensor_simulator import (
            PhoneSensorSimulator,
            PhoneSimulatorConfig,
            SimulationScenario,
        )

        receiver = MultimodalReceiver(settings)
        receiver.start()

        simulator = PhoneSensorSimulator(
            PhoneSimulatorConfig(
                scenario=SimulationScenario.NAVIGATION,
                random_seed=42,
            )
        )

        packets = simulator.generate_cycle()

        receipts = receiver.receive_batch(
            packets,
            raise_on_error=True,
        )

        if not all(receipt.accepted for receipt in receipts):
            raise AssertionError(
                "Simulator packets were not accepted."
            )

        print("[SUCCESS] Simulator packets routed.")

        print("[3/6] Processing latest audio packet...")

        result = processor.process_latest_from_receiver(
            receiver,
            remove=True,
            raise_on_error=True,
        )

        if result is None:
            raise AssertionError(
                "No audio packet was available."
            )

        if not result.success:
            raise AssertionError(
                f"Audio processing failed: {result.error}"
            )

        if result.audio_data is None:
            raise AssertionError(
                "AudioData was not produced."
            )

        print("[SUCCESS] Audio packet processed.")

        print("[4/6] Validating AudioData...")

        audio = result.audio_data
        audio.validate()

        if audio.metadata.modality != "audio":
            raise AssertionError(
                "AudioData modality is incorrect."
            )

        if audio.sample_rate_hz != 16000:
            raise AssertionError(
                "Unexpected sample rate."
            )

        if audio.chunk_id != "AUDIO_000001":
            raise AssertionError(
                "Unexpected chunk ID."
            )

        print("[SUCCESS] AudioData is valid.")

        print("[5/6] Testing invalid modality rejection...")

        vision_packet = receiver.get_latest(
            "vision",
            remove=False,
        )

        if vision_packet is None:
            raise AssertionError(
                "Vision packet missing from receiver."
            )

        invalid_result = processor.process_packet(
            vision_packet,
            raise_on_error=False,
        )

        if invalid_result.success:
            raise AssertionError(
                "Non-audio packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid modality was rejected.")

        print("[6/6] Checking diagnostics...")

        health = processor.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Audio processor health check failed."
            )

        if health["statistics"]["total_processed"] != 1:
            raise AssertionError(
                "Processed count is incorrect."
            )

        if health["statistics"]["total_failed"] != 1:
            raise AssertionError(
                "Failed count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nAudioData:")
        print(
            json.dumps(
                result.audio_data.metadata.metadata
                | {
                    "chunk_id": result.audio_data.chunk_id,
                    "sample_rate_hz": (
                        result.audio_data.sample_rate_hz
                    ),
                    "channels": result.audio_data.channels,
                    "duration_ms": (
                        result.audio_data.duration_ms
                    ),
                    "amplitude_score": (
                        result.audio_data.amplitude_score
                    ),
                    "signal_to_noise_score": (
                        result.audio_data.signal_to_noise_score
                    ),
                    "clipping_ratio": (
                        result.audio_data.clipping_ratio
                    ),
                    "silence_ratio": (
                        result.audio_data.silence_ratio
                    ),
                    "packet_integrity_score": (
                        result.audio_data.packet_integrity_score
                    ),
                    "limitations": (
                        result.audio_data.metadata.limitations
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nProcessor health:")
        print(
            json.dumps(
                health,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 1 AUDIO INPUT IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 AUDIO INPUT TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    if not run_audio_input_self_test():
        raise SystemExit(1)