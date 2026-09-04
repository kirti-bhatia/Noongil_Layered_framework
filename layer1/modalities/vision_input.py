"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Vision Input Processor
File    : layer1/modalities/vision_input.py
============================================================

Purpose
-------
Consumes normalized vision packets from MultimodalReceiver and
produces validated Layer 1 VisionData objects.

Responsibilities
----------------
1. Validate camera packet structure
2. Decode supported image references or base64 image payloads
3. Extract frame metadata
4. Apply optional resize/normalization metadata
5. Estimate brightness, sharpness, contrast, and frame integrity
6. Save raw/preprocessed frames when configured
7. Build VisionData for the final Multimodal Sensor Packet
8. Log processing, quality, and errors
9. Provide diagnostics and a standalone self-test

Architectural Boundary
----------------------
This module does NOT perform:
- object detection;
- OCR;
- scene classification;
- face recognition;
- activity recognition;
- hazard detection;
- reasoning;
- LLM processing.

Compatibility
-------------
Python 3.10+
Pillow is optional. The module works without Pillow when only
metadata-based simulated packets are processed.
============================================================
"""

from __future__ import annotations

import base64
import io
import json
import math
import statistics
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from layer1.acquisition.multimodal_receiver import (
    MultimodalReceiver,
    ReceivedSensorPacket,
)
from layer1.config.paths import (
    PREPROCESSED_VISION_DIR,
    RAW_VISION_DIR,
    ensure_directory,
)
from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.schemas.sensor_packet import (
    ModalityMetadata,
    ModalityStatus,
    VisionData,
)
from layer1.utils.logger import (
    PipelineTimer,
    get_logger,
    log_exception,
    log_sensor_event,
)

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError:
    Image = None
    ImageFilter = None
    ImageStat = None


# ============================================================
# EXCEPTIONS
# ============================================================

class VisionInputError(Exception):
    """Base exception for vision input processing."""


class VisionPacketValidationError(VisionInputError):
    """Raised when a received vision packet is invalid."""


class VisionDecodeError(VisionInputError):
    """Raised when image content cannot be decoded."""


class VisionProcessingError(VisionInputError):
    """Raised when frame processing fails."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VisionProcessingResult:
    """
    Result returned after processing one vision packet.
    """

    success: bool
    vision_data: Optional[VisionData] = None
    packet_id: Optional[str] = None
    frame_id: Optional[str] = None
    raw_frame_path: Optional[str] = None
    preprocessed_frame_path: Optional[str] = None
    processing_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisionProcessorStatistics:
    """
    Runtime statistics for VisionInputProcessor.
    """

    total_received: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_metadata_only: int = 0
    total_image_decoded: int = 0
    total_saved_raw: int = 0
    total_saved_preprocessed: int = 0
    cumulative_processing_seconds: float = 0.0
    last_packet_id: Optional[str] = None
    last_frame_id: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.cumulative_processing_seconds / self.total_processed

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


def require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise VisionPacketValidationError(
            f"{field_name} must be a positive integer."
        )

    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise VisionPacketValidationError(
            f"{field_name} must be a positive integer."
        ) from error

    if parsed <= 0:
        raise VisionPacketValidationError(
            f"{field_name} must be a positive integer."
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
        raise VisionPacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed):
        raise VisionPacketValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= parsed <= 1.0:
        raise VisionPacketValidationError(
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
    return normalized.strip("_") or "frame"


# ============================================================
# VISION INPUT PROCESSOR
# ============================================================

class VisionInputProcessor:
    """
    Convert receiver vision packets into validated VisionData.

    The processor supports two operation paths:

    1. Metadata-only processing
       Used by the current phone simulator.

    2. Image-backed processing
       Used later when the real smartphone sends base64 or binary
       image content. Pillow is used when available.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger("modalities.vision_input")
        self.statistics = VisionProcessorStatistics()

        ensure_directory(RAW_VISION_DIR)
        ensure_directory(PREPROCESSED_VISION_DIR)

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process_packet(
        self,
        packet: ReceivedSensorPacket,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> VisionProcessingResult:
        """
        Process one normalized receiver packet.

        Parameters
        ----------
        packet:
            ReceivedSensorPacket routed to the vision queue.

        raise_on_error:
            Override runtime fail-fast behavior.
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
                "vision_input.process_packet",
                logger=self.logger,
                metadata={
                    "packet_id": packet.packet_id,
                    "device_id": packet.device_id,
                },
            ):
                self._validate_packet(packet)

                payload = packet.payload
                warnings: List[str] = []

                image = self._decode_image_if_present(payload)

                if image is None:
                    self.statistics.total_metadata_only += 1
                    vision_data = self._build_from_metadata(
                        packet,
                        warnings=warnings,
                    )
                    raw_path = None
                    preprocessed_path = None
                else:
                    self.statistics.total_image_decoded += 1
                    (
                        vision_data,
                        raw_path,
                        preprocessed_path,
                    ) = self._process_image_backed_packet(
                        packet,
                        image,
                        warnings=warnings,
                    )

                vision_data.validate()

                elapsed = time.perf_counter() - started

                self.statistics.total_processed += 1
                self.statistics.cumulative_processing_seconds += elapsed
                self.statistics.last_packet_id = packet.packet_id
                self.statistics.last_frame_id = vision_data.frame_id
                self.statistics.last_error = None

                log_sensor_event(
                    modality="vision",
                    event="Vision packet processed",
                    device_id=packet.device_id,
                    sensor_type=packet.sensor_type,
                    packet_id=packet.packet_id,
                    sequence_number=packet.sequence_number,
                    details={
                        "frame_id": vision_data.frame_id,
                        "width": vision_data.width,
                        "height": vision_data.height,
                        "brightness_score": (
                            vision_data.brightness_score
                        ),
                        "sharpness_score": (
                            vision_data.sharpness_score
                        ),
                        "processing_seconds": round(
                            elapsed,
                            6,
                        ),
                        "metadata_only": image is None,
                    },
                )

                return VisionProcessingResult(
                    success=True,
                    vision_data=vision_data,
                    packet_id=packet.packet_id,
                    frame_id=vision_data.frame_id,
                    raw_frame_path=raw_path,
                    preprocessed_frame_path=preprocessed_path,
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
                "Vision packet processing failed",
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

            return VisionProcessingResult(
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
    ) -> List[VisionProcessingResult]:
        """
        Drain and process vision packets from MultimodalReceiver.
        """

        packets = receiver.drain(
            "vision",
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
    ) -> Optional[VisionProcessingResult]:
        """
        Process the most recent vision packet from a receiver.
        """

        packet = receiver.get_latest(
            "vision",
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
            raise VisionPacketValidationError(
                "packet must be ReceivedSensorPacket."
            )

        packet.validate()

        if packet.modality != "vision":
            raise VisionPacketValidationError(
                "VisionInputProcessor accepts only "
                "modality='vision'."
            )

        if not isinstance(packet.payload, dict):
            raise VisionPacketValidationError(
                "Vision packet payload must be a dictionary."
            )

    # ========================================================
    # IMAGE DECODING
    # ========================================================

    def _decode_image_if_present(
        self,
        payload: Mapping[str, Any],
    ) -> Any:
        """
        Decode image data when actual image content is present.

        Supported payload keys:
        - encoded_frame
        - frame_base64
        - image_base64
        - image_bytes encoded as base64 string
        """

        encoded_value = (
            payload.get("encoded_frame")
            or payload.get("frame_base64")
            or payload.get("image_base64")
            or payload.get("image_bytes")
        )

        if encoded_value is None:
            return None

        if Image is None:
            raise VisionDecodeError(
                "Image content was provided, but Pillow is not "
                "installed. Install it using: pip install pillow"
            )

        if isinstance(encoded_value, Mapping):
            encoded_value = encoded_value.get("data")

        if not isinstance(encoded_value, str):
            raise VisionDecodeError(
                "Encoded frame must be a base64 string."
            )

        try:
            image_bytes = base64.b64decode(
                encoded_value,
                validate=True,
            )
        except Exception as error:
            raise VisionDecodeError(
                "Encoded frame is not valid base64."
            ) from error

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            return image
        except Exception as error:
            raise VisionDecodeError(
                "Unable to decode image bytes."
            ) from error

    # ========================================================
    # METADATA-ONLY PROCESSING
    # ========================================================

    def _build_from_metadata(
        self,
        packet: ReceivedSensorPacket,
        *,
        warnings: List[str],
    ) -> VisionData:
        payload = packet.payload

        frame_id = (
            payload.get("frame_id")
            or packet.metadata.get("frame_id")
            or packet.packet_id
        )

        width = require_positive_int(
            payload.get(
                "width",
                self.settings.vision.target_width,
            ),
            "width",
        )

        height = require_positive_int(
            payload.get(
                "height",
                self.settings.vision.target_height,
            ),
            "height",
        )

        channels = require_positive_int(
            payload.get(
                "channels",
                self.settings.vision.target_channels,
            ),
            "channels",
        )

        brightness = require_probability(
            payload.get("brightness_score"),
            "brightness_score",
        )
        sharpness = require_probability(
            payload.get("sharpness_score"),
            "sharpness_score",
        )
        contrast = require_probability(
            payload.get("contrast_score"),
            "contrast_score",
        )
        integrity = require_probability(
            payload.get("frame_integrity_score"),
            "frame_integrity_score",
        )

        if brightness is None:
            brightness = 0.50
            warnings.append(
                "brightness_score_missing_default_used"
            )

        if sharpness is None:
            sharpness = 0.50
            warnings.append(
                "sharpness_score_missing_default_used"
            )

        if contrast is None:
            contrast = 0.50
            warnings.append(
                "contrast_score_missing_default_used"
            )

        if integrity is None:
            integrity = 1.0
            warnings.append(
                "frame_integrity_score_missing_default_used"
            )

        limitations: List[str] = []

        if brightness < self.settings.vision.low_brightness_threshold:
            limitations.append("low_brightness")

        if sharpness < (
            self.settings.vision.minimum_sharpness_threshold
        ):
            limitations.append("low_sharpness")

        if integrity < (
            self.settings.vision.minimum_frame_integrity
        ):
            limitations.append("low_frame_integrity")

        if payload.get("degraded") is True:
            limitations.append("simulated_degraded_quality")

        preprocessing_steps = [
            "packet_validation",
            "metadata_extraction",
        ]

        if self.settings.vision.resize_frames:
            preprocessing_steps.append(
                "resize_requested_metadata_only"
            )

        if self.settings.vision.normalize_pixels:
            preprocessing_steps.append(
                "normalization_requested_metadata_only"
            )

        metadata = ModalityMetadata(
            modality="vision",
            status=ModalityStatus.OBSERVED,
            source_timestamp=packet.source_timestamp,
            arrival_timestamp=packet.arrival_timestamp,
            sequence_number=packet.sequence_number,
            sampling_rate_hz=packet.sampling_rate_hz,
            latency_ms=packet.latency_ms,
            source_device_id=packet.device_id,
            data_reference=(
                str(
                    payload.get(
                        "frame_reference",
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
                "metadata_only": True,
            },
        )

        return VisionData(
            metadata=metadata,
            frame_id=str(frame_id),
            width=width,
            height=height,
            channels=channels,
            encoding=str(
                payload.get(
                    "encoding",
                    self.settings.vision.encoding.value,
                )
            ),
            color_space=str(
                payload.get("color_space", "RGB")
            ),
            frame_rate_fps=(
                float(packet.sampling_rate_hz)
                if packet.sampling_rate_hz is not None
                else self.settings.vision.default_fps
            ),
            brightness_score=brightness,
            sharpness_score=sharpness,
            contrast_score=contrast,
            frame_integrity_score=integrity,
            frame_path=(
                str(payload.get("frame_path"))
                if payload.get("frame_path")
                else None
            ),
            encoded_frame=None,
        )

    # ========================================================
    # IMAGE-BACKED PROCESSING
    # ========================================================

    def _process_image_backed_packet(
        self,
        packet: ReceivedSensorPacket,
        image: Any,
        *,
        warnings: List[str],
    ) -> Tuple[VisionData, Optional[str], Optional[str]]:
        if Image is None:
            raise VisionProcessingError(
                "Pillow is required for image-backed processing."
            )

        payload = packet.payload
        frame_id = str(
            payload.get("frame_id")
            or packet.packet_id
        )

        original = image.copy()
        working = image.copy()

        raw_path: Optional[str] = None
        preprocessed_path: Optional[str] = None

        if self.settings.vision.save_raw_frames:
            raw_path = self._save_image(
                original,
                directory=RAW_VISION_DIR,
                frame_id=frame_id,
                suffix="raw",
            )
            self.statistics.total_saved_raw += 1

        preprocessing_steps = [
            "packet_validation",
            "base64_decode",
        ]

        if working.mode != "RGB":
            working = working.convert("RGB")
            preprocessing_steps.append("convert_to_rgb")

        if self.settings.vision.resize_frames:
            working = self._resize_image(working)
            preprocessing_steps.append("resize")

        if self.settings.vision.denoise_frames:
            if ImageFilter is None:
                warnings.append(
                    "denoise_requested_but_pillow_filter_unavailable"
                )
            else:
                working = working.filter(
                    ImageFilter.MedianFilter(size=3)
                )
                preprocessing_steps.append("median_denoise")

        if self.settings.vision.normalize_pixels:
            preprocessing_steps.append(
                "pixel_normalization_deferred_to_layer2_tensor_stage"
            )

        brightness = self._calculate_brightness(working)
        sharpness = self._calculate_sharpness(working)
        contrast = self._calculate_contrast(working)
        integrity = self._calculate_integrity(
            image=working,
            payload=payload,
        )

        limitations: List[str] = []

        if brightness < self.settings.vision.low_brightness_threshold:
            limitations.append("low_brightness")

        if sharpness < (
            self.settings.vision.minimum_sharpness_threshold
        ):
            limitations.append("low_sharpness")

        if integrity < (
            self.settings.vision.minimum_frame_integrity
        ):
            limitations.append("low_frame_integrity")

        if self.settings.vision.save_preprocessed_frames:
            preprocessed_path = self._save_image(
                working,
                directory=PREPROCESSED_VISION_DIR,
                frame_id=frame_id,
                suffix="preprocessed",
            )
            self.statistics.total_saved_preprocessed += 1

        width, height = working.size
        channels = len(working.getbands())

        metadata = ModalityMetadata(
            modality="vision",
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
                        "frame_reference",
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
                "original_size": list(original.size),
            },
        )

        vision_data = VisionData(
            metadata=metadata,
            frame_id=frame_id,
            width=width,
            height=height,
            channels=channels,
            encoding=self.settings.vision.encoding.value,
            color_space="RGB",
            frame_rate_fps=(
                float(packet.sampling_rate_hz)
                if packet.sampling_rate_hz is not None
                else self.settings.vision.default_fps
            ),
            brightness_score=brightness,
            sharpness_score=sharpness,
            contrast_score=contrast,
            frame_integrity_score=integrity,
            frame_path=preprocessed_path or raw_path,
            encoded_frame=None,
        )

        return vision_data, raw_path, preprocessed_path

    def _resize_image(self, image: Any) -> Any:
        target_width = self.settings.vision.target_width
        target_height = self.settings.vision.target_height

        if self.settings.vision.preserve_aspect_ratio:
            copy = image.copy()
            copy.thumbnail(
                (target_width, target_height)
            )

            canvas = Image.new(
                "RGB",
                (target_width, target_height),
                color=(0, 0, 0),
            )

            left = (target_width - copy.width) // 2
            top = (target_height - copy.height) // 2

            canvas.paste(copy, (left, top))
            return canvas

        return image.resize(
            (target_width, target_height)
        )

    def _calculate_brightness(self, image: Any) -> float:
        grayscale = image.convert("L")

        if ImageStat is not None:
            mean = ImageStat.Stat(grayscale).mean[0]
        else:
            values = list(grayscale.getdata())
            mean = statistics.fmean(values) if values else 0.0

        return round(clamp(mean / 255.0, 0.0, 1.0), 6)

    def _calculate_contrast(self, image: Any) -> float:
        grayscale = image.convert("L")

        if ImageStat is not None:
            standard_deviation = ImageStat.Stat(
                grayscale
            ).stddev[0]
        else:
            values = list(grayscale.getdata())
            standard_deviation = (
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0
            )

        return round(
            clamp(standard_deviation / 128.0, 0.0, 1.0),
            6,
        )

    def _calculate_sharpness(self, image: Any) -> float:
        grayscale = image.convert("L")

        if ImageFilter is None:
            return 0.50

        edges = grayscale.filter(
            ImageFilter.FIND_EDGES
        )

        if ImageStat is not None:
            edge_mean = ImageStat.Stat(edges).mean[0]
        else:
            values = list(edges.getdata())
            edge_mean = (
                statistics.fmean(values)
                if values
                else 0.0
            )

        return round(
            clamp(edge_mean / 64.0, 0.0, 1.0),
            6,
        )

    def _calculate_integrity(
        self,
        *,
        image: Any,
        payload: Mapping[str, Any],
    ) -> float:
        score = require_probability(
            payload.get("frame_integrity_score"),
            "frame_integrity_score",
        )

        if score is not None:
            return score

        width, height = image.size

        if width <= 0 or height <= 0:
            return 0.0

        return 1.0

    def _save_image(
        self,
        image: Any,
        *,
        directory: Path,
        frame_id: str,
        suffix: str,
    ) -> str:
        ensure_directory(directory)

        extension = (
            "jpg"
            if self.settings.vision.encoding.value == "jpeg"
            else self.settings.vision.encoding.value
        )

        filename = (
            f"{safe_filename_component(frame_id)}_"
            f"{safe_filename_component(suffix)}."
            f"{extension}"
        )

        output_path = directory / filename

        save_format = (
            "JPEG"
            if extension in {"jpg", "jpeg"}
            else extension.upper()
        )

        save_kwargs: Dict[str, Any] = {}

        if save_format == "JPEG":
            save_kwargs["quality"] = (
                self.settings.vision.jpeg_quality
            )

        image.save(
            output_path,
            format=save_format,
            **save_kwargs,
        )

        return str(output_path.resolve())

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "pillow_available": Image is not None,
            "vision_enabled": self.settings.vision.enabled,
            "raw_output_dir": str(RAW_VISION_DIR),
            "preprocessed_output_dir": str(
                PREPROCESSED_VISION_DIR
            ),
            "statistics": self.statistics.to_dict(),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_vision_input_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 VISION INPUT TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating test settings...")

        settings = create_test_settings()
        settings.vision.save_raw_frames = False
        settings.vision.save_preprocessed_frames = False

        processor = VisionInputProcessor(settings)

        print("[SUCCESS] Vision processor initialized.")

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

        print("[3/6] Processing latest vision packet...")

        result = processor.process_latest_from_receiver(
            receiver,
            remove=True,
            raise_on_error=True,
        )

        if result is None:
            raise AssertionError(
                "No vision packet was available."
            )

        if not result.success:
            raise AssertionError(
                f"Vision processing failed: {result.error}"
            )

        if result.vision_data is None:
            raise AssertionError(
                "VisionData was not produced."
            )

        print("[SUCCESS] Vision packet processed.")

        print("[4/6] Validating VisionData...")

        vision = result.vision_data
        vision.validate()

        if vision.metadata.modality != "vision":
            raise AssertionError(
                "VisionData modality is incorrect."
            )

        if vision.width != 640 or vision.height != 480:
            raise AssertionError(
                "Vision dimensions are incorrect."
            )

        if vision.frame_id != "FRAME_000001":
            raise AssertionError(
                "Unexpected frame ID."
            )

        print("[SUCCESS] VisionData is valid.")

        print("[5/6] Testing invalid modality rejection...")

        audio_packet = receiver.get_latest(
            "audio",
            remove=False,
        )

        if audio_packet is None:
            raise AssertionError(
                "Audio packet missing from receiver."
            )

        invalid_result = processor.process_packet(
            audio_packet,
            raise_on_error=False,
        )

        if invalid_result.success:
            raise AssertionError(
                "Non-vision packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid modality was rejected.")

        print("[6/6] Checking diagnostics...")

        health = processor.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Vision processor health check failed."
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

        print("\nVisionData:")
        print(
            json.dumps(
                result.vision_data.metadata.metadata
                | {
                    "frame_id": result.vision_data.frame_id,
                    "width": result.vision_data.width,
                    "height": result.vision_data.height,
                    "brightness_score": (
                        result.vision_data.brightness_score
                    ),
                    "sharpness_score": (
                        result.vision_data.sharpness_score
                    ),
                    "contrast_score": (
                        result.vision_data.contrast_score
                    ),
                    "frame_integrity_score": (
                        result.vision_data.frame_integrity_score
                    ),
                    "limitations": (
                        result.vision_data.metadata.limitations
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
        print("[PASSED] LAYER 1 VISION INPUT IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 VISION INPUT TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    if not run_vision_input_self_test():
        raise SystemExit(1)