"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Phone Sensor Simulator
File    : layer1/acquisition/phone_sensor_simulator.py
============================================================

Purpose
-------
Simulates smartphone and connected-device sensor packets for
NOONGIL Layer 1.

The simulator can produce:

- RGB camera metadata packets
- Microphone/audio metadata packets
- GPS and compass packets
- Accelerometer packets
- Gyroscope packets
- Magnetometer packets
- Touch/button interaction packets
- Earphone/wearable status packets
- Device-state packets
- Optional environmental-context packets

The generated packets are compatible with:

    layer1.acquisition.multimodal_receiver.MultimodalReceiver

Architectural Boundary
----------------------
This simulator produces synthetic sensor-level data only.

It does NOT perform:
- object detection;
- OCR;
- speech recognition;
- scene understanding;
- activity recognition;
- hazard reasoning;
- intent reasoning;
- LLM processing.

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
import random
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


# ============================================================
# ENUMERATIONS
# ============================================================

class SimulationScenario(str, Enum):
    IDLE = "idle"
    HOME = "home"
    CLASSROOM = "classroom"
    PARK = "park"
    SHOPPING_MALL = "shopping_mall"
    NAVIGATION = "navigation"
    READING = "reading"
    EMERGENCY = "emergency"
    DEGRADED_NETWORK = "degraded_network"


class SimulatorStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PhoneSimulatorConfig:
    """
    Runtime configuration for the simulated smartphone.
    """

    device_id: str = "PHONE_001"
    device_name: str = "NOONGIL Simulated Android Phone"
    operating_system: str = "Android"
    application_version: str = "1.0.0"

    scenario: SimulationScenario = SimulationScenario.NAVIGATION
    random_seed: int = 42

    camera_fps: float = 10.0
    audio_chunk_rate_hz: float = 1.0
    gps_rate_hz: float = 1.0
    imu_rate_hz: float = 25.0
    wearable_rate_hz: float = 0.5
    device_status_rate_hz: float = 0.2
    environment_rate_hz: float = 0.1

    camera_width: int = 640
    camera_height: int = 480
    camera_channels: int = 3
    camera_encoding: str = "jpeg"

    audio_sample_rate_hz: int = 16000
    audio_channels: int = 1
    audio_sample_width_bits: int = 16
    audio_chunk_duration_ms: float = 1000.0

    base_latitude: float = 31.6340
    base_longitude: float = 74.8720
    base_altitude_meters: float = 234.0

    earphone_name: str = "realme Buds T200"
    earphone_connected: bool = True

    battery_level: float = 0.85
    is_charging: bool = False
    battery_drain_per_cycle: float = 0.0005

    network_strength: float = 0.90
    network_latency_ms: float = 20.0

    introduce_random_latency: bool = True
    minimum_latency_ms: float = 5.0
    maximum_latency_ms: float = 80.0

    introduce_missing_packets: bool = False
    missing_packet_probability: float = 0.02

    introduce_degraded_quality: bool = False
    degraded_quality_probability: float = 0.10

    include_environment: bool = False

    def validate(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id cannot be empty.")

        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("Camera dimensions must be positive.")

        if self.camera_channels <= 0:
            raise ValueError("camera_channels must be positive.")

        for name, value in (
            ("camera_fps", self.camera_fps),
            ("audio_chunk_rate_hz", self.audio_chunk_rate_hz),
            ("gps_rate_hz", self.gps_rate_hz),
            ("imu_rate_hz", self.imu_rate_hz),
            ("wearable_rate_hz", self.wearable_rate_hz),
            ("device_status_rate_hz", self.device_status_rate_hz),
            ("environment_rate_hz", self.environment_rate_hz),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        for name, value in (
            ("battery_level", self.battery_level),
            ("network_strength", self.network_strength),
            ("missing_packet_probability", self.missing_packet_probability),
            ("degraded_quality_probability", self.degraded_quality_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.minimum_latency_ms < 0:
            raise ValueError("minimum_latency_ms cannot be negative.")

        if self.maximum_latency_ms < self.minimum_latency_ms:
            raise ValueError(
                "maximum_latency_ms cannot be less than minimum_latency_ms."
            )

        if not -90.0 <= self.base_latitude <= 90.0:
            raise ValueError("base_latitude is invalid.")

        if not -180.0 <= self.base_longitude <= 180.0:
            raise ValueError("base_longitude is invalid.")


# ============================================================
# SIMULATOR STATE
# ============================================================

@dataclass
class SimulatorState:
    """
    Mutable state maintained across simulation cycles.
    """

    status: SimulatorStatus = SimulatorStatus.CREATED
    cycle_number: int = 0

    camera_sequence: int = 0
    audio_sequence: int = 0
    gps_sequence: int = 0
    accelerometer_sequence: int = 0
    gyroscope_sequence: int = 0
    magnetometer_sequence: int = 0
    interaction_sequence: int = 0
    wearable_sequence: int = 0
    device_sequence: int = 0
    environment_sequence: int = 0

    latitude: float = 31.6340
    longitude: float = 74.8720
    altitude_meters: float = 234.0
    heading_degrees: float = 0.0
    speed_meters_per_second: float = 0.0

    battery_level: float = 0.85
    network_strength: float = 0.90
    network_latency_ms: float = 20.0

    last_interaction_action: Optional[str] = None
    emergency_active: bool = False

    generated_packets: int = 0
    dropped_packets: int = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def generate_packet_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    token = uuid.uuid4().hex[:8].upper()
    return f"{prefix}_{timestamp}_{token}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalized_random(
    rng: random.Random,
    center: float,
    spread: float,
) -> float:
    return clamp(rng.gauss(center, spread), 0.0, 1.0)


# ============================================================
# MAIN SIMULATOR
# ============================================================

class PhoneSensorSimulator:
    """
    Generates realistic synthetic smartphone sensor packets.

    Typical use
    -----------
    simulator = PhoneSensorSimulator()

    packets = simulator.generate_cycle()

    for packet in packets:
        receiver.receive(packet)
    """

    def __init__(
        self,
        config: Optional[PhoneSimulatorConfig] = None,
    ) -> None:
        self.config = config or PhoneSimulatorConfig()
        self.config.validate()

        self.rng = random.Random(self.config.random_seed)

        self.state = SimulatorState(
            latitude=self.config.base_latitude,
            longitude=self.config.base_longitude,
            altitude_meters=self.config.base_altitude_meters,
            battery_level=self.config.battery_level,
            network_strength=self.config.network_strength,
            network_latency_ms=self.config.network_latency_ms,
        )

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> None:
        self.state.status = SimulatorStatus.RUNNING

    def stop(self) -> None:
        self.state.status = SimulatorStatus.STOPPED

    def reset(self) -> None:
        self.rng = random.Random(self.config.random_seed)

        self.state = SimulatorState(
            latitude=self.config.base_latitude,
            longitude=self.config.base_longitude,
            altitude_meters=self.config.base_altitude_meters,
            battery_level=self.config.battery_level,
            network_strength=self.config.network_strength,
            network_latency_ms=self.config.network_latency_ms,
        )

    def set_scenario(
        self,
        scenario: SimulationScenario | str,
    ) -> None:
        if isinstance(scenario, str):
            scenario = SimulationScenario(scenario)

        self.config.scenario = scenario

    # ========================================================
    # COMMON PACKET BUILDER
    # ========================================================

    def _build_packet(
        self,
        *,
        modality: str,
        sensor_type: str,
        sequence_number: int,
        sampling_rate_hz: Optional[float],
        payload: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = utc_now_iso()

        packet_metadata = {
            "scenario": self.config.scenario.value,
            "cycle_number": self.state.cycle_number,
            "simulated": True,
            "network_strength": round(self.state.network_strength, 4),
            "network_latency_ms": round(self.state.network_latency_ms, 3),
        }

        if metadata:
            packet_metadata.update(dict(metadata))

        packet = {
            "packet_id": generate_packet_id(
                f"SIM_{sensor_type.upper()}"
            ),
            "schema_version": "1.0",
            "device_id": self.config.device_id,
            "modality": modality,
            "sensor_type": sensor_type,
            "timestamp": timestamp,
            "sequence_number": sequence_number,
            "payload_encoding": "json",
            "payload": dict(payload),
            "metadata": packet_metadata,
        }

        if sampling_rate_hz is not None:
            packet["sampling_rate_hz"] = sampling_rate_hz

        return packet

    def _should_drop_packet(self) -> bool:
        if not self.config.introduce_missing_packets:
            return False

        return (
            self.rng.random()
            < self.config.missing_packet_probability
        )

    def _quality_is_degraded(self) -> bool:
        if not self.config.introduce_degraded_quality:
            return False

        return (
            self.rng.random()
            < self.config.degraded_quality_probability
        )

    def _simulate_network_state(self) -> None:
        if self.config.scenario == SimulationScenario.DEGRADED_NETWORK:
            strength_center = 0.30
            latency_center = 350.0
        elif self.config.scenario == SimulationScenario.EMERGENCY:
            strength_center = 0.75
            latency_center = 45.0
        else:
            strength_center = self.config.network_strength
            latency_center = self.config.network_latency_ms

        self.state.network_strength = normalized_random(
            self.rng,
            strength_center,
            0.04,
        )

        if self.config.introduce_random_latency:
            random_latency = self.rng.uniform(
                self.config.minimum_latency_ms,
                self.config.maximum_latency_ms,
            )
            self.state.network_latency_ms = (
                0.5 * latency_center + 0.5 * random_latency
            )
        else:
            self.state.network_latency_ms = latency_center

    def _update_battery(self) -> None:
        if self.config.is_charging:
            self.state.battery_level = clamp(
                self.state.battery_level + 0.001,
                0.0,
                1.0,
            )
        else:
            self.state.battery_level = clamp(
                self.state.battery_level
                - self.config.battery_drain_per_cycle,
                0.0,
                1.0,
            )

    # ========================================================
    # SCENARIO BEHAVIOUR
    # ========================================================

    def _scenario_motion_profile(self) -> Dict[str, float]:
        scenario = self.config.scenario

        if scenario == SimulationScenario.IDLE:
            return {
                "motion_intensity": 0.05,
                "speed": 0.0,
                "accel_spread": 0.04,
                "gyro_spread": 0.01,
            }

        if scenario in {
            SimulationScenario.HOME,
            SimulationScenario.CLASSROOM,
            SimulationScenario.READING,
        }:
            return {
                "motion_intensity": 0.15,
                "speed": 0.1,
                "accel_spread": 0.08,
                "gyro_spread": 0.03,
            }

        if scenario in {
            SimulationScenario.PARK,
            SimulationScenario.SHOPPING_MALL,
            SimulationScenario.NAVIGATION,
        }:
            return {
                "motion_intensity": 0.45,
                "speed": 1.1,
                "accel_spread": 0.35,
                "gyro_spread": 0.12,
            }

        if scenario == SimulationScenario.EMERGENCY:
            return {
                "motion_intensity": 0.90,
                "speed": 2.5,
                "accel_spread": 1.5,
                "gyro_spread": 0.8,
            }

        return {
            "motion_intensity": 0.30,
            "speed": 0.8,
            "accel_spread": 0.25,
            "gyro_spread": 0.10,
        }

    def _scenario_interaction_action(self) -> Optional[str]:
        scenario = self.config.scenario

        mapping = {
            SimulationScenario.NAVIGATION: "navigation_mode_requested",
            SimulationScenario.READING: "reading_mode_requested",
            SimulationScenario.EMERGENCY: "emergency_mode_requested",
            SimulationScenario.PARK: "awareness_mode_requested",
            SimulationScenario.SHOPPING_MALL: "navigation_mode_requested",
            SimulationScenario.CLASSROOM: "awareness_mode_requested",
            SimulationScenario.HOME: "awareness_mode_requested",
        }

        return mapping.get(scenario)

    # ========================================================
    # SENSOR GENERATORS
    # ========================================================

    def generate_camera_packet(self) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.camera_sequence += 1

        degraded = self._quality_is_degraded()

        if self.config.scenario == SimulationScenario.READING:
            brightness_center = 0.80
            sharpness_center = 0.90
        elif self.config.scenario == SimulationScenario.EMERGENCY:
            brightness_center = 0.55
            sharpness_center = 0.60
        else:
            brightness_center = 0.72
            sharpness_center = 0.78

        if degraded:
            brightness_center *= 0.35
            sharpness_center *= 0.45

        payload = {
            "frame_id": f"FRAME_{self.state.camera_sequence:06d}",
            "width": self.config.camera_width,
            "height": self.config.camera_height,
            "channels": self.config.camera_channels,
            "encoding": self.config.camera_encoding,
            "color_space": "RGB",
            "frame_reference": (
                f"memory://simulated/frame/"
                f"{self.state.camera_sequence:06d}"
            ),
            "brightness_score": round(
                normalized_random(
                    self.rng,
                    brightness_center,
                    0.05,
                ),
                4,
            ),
            "sharpness_score": round(
                normalized_random(
                    self.rng,
                    sharpness_center,
                    0.05,
                ),
                4,
            ),
            "contrast_score": round(
                normalized_random(
                    self.rng,
                    0.75 if not degraded else 0.35,
                    0.05,
                ),
                4,
            ),
            "frame_integrity_score": round(
                normalized_random(
                    self.rng,
                    0.98 if not degraded else 0.70,
                    0.02,
                ),
                4,
            ),
            "degraded": degraded,
        }

        return self._build_packet(
            modality="camera",
            sensor_type="rgb_camera",
            sequence_number=self.state.camera_sequence,
            sampling_rate_hz=self.config.camera_fps,
            payload=payload,
            metadata={
                "orientation": "portrait",
                "camera_position": "rear",
            },
        )

    def generate_audio_packet(self) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.audio_sequence += 1

        degraded = self._quality_is_degraded()

        if self.config.scenario == SimulationScenario.EMERGENCY:
            snr_center = 0.55
            amplitude_center = 0.85
        elif self.config.scenario == SimulationScenario.CLASSROOM:
            snr_center = 0.72
            amplitude_center = 0.55
        elif self.config.scenario == SimulationScenario.SHOPPING_MALL:
            snr_center = 0.48
            amplitude_center = 0.70
        else:
            snr_center = 0.70
            amplitude_center = 0.60

        if degraded:
            snr_center *= 0.45

        payload = {
            "chunk_id": f"AUDIO_{self.state.audio_sequence:06d}",
            "sample_rate_hz": self.config.audio_sample_rate_hz,
            "channels": self.config.audio_channels,
            "sample_width_bits": self.config.audio_sample_width_bits,
            "duration_ms": self.config.audio_chunk_duration_ms,
            "encoding": "pcm_s16le",
            "audio_reference": (
                f"memory://simulated/audio/"
                f"{self.state.audio_sequence:06d}"
            ),
            "amplitude_score": round(
                normalized_random(
                    self.rng,
                    amplitude_center,
                    0.08,
                ),
                4,
            ),
            "signal_to_noise_score": round(
                normalized_random(
                    self.rng,
                    snr_center,
                    0.06,
                ),
                4,
            ),
            "clipping_ratio": round(
                normalized_random(
                    self.rng,
                    0.02 if not degraded else 0.20,
                    0.02,
                ),
                4,
            ),
            "silence_ratio": round(
                normalized_random(
                    self.rng,
                    0.10,
                    0.04,
                ),
                4,
            ),
            "packet_integrity_score": round(
                normalized_random(
                    self.rng,
                    0.98 if not degraded else 0.75,
                    0.02,
                ),
                4,
            ),
            "degraded": degraded,
        }

        return self._build_packet(
            modality="microphone",
            sensor_type="phone_microphone",
            sequence_number=self.state.audio_sequence,
            sampling_rate_hz=self.config.audio_sample_rate_hz,
            payload=payload,
        )

    def generate_gps_packet(self) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.gps_sequence += 1

        profile = self._scenario_motion_profile()
        speed = profile["speed"]

        heading_change = self.rng.gauss(0.0, 4.0)
        self.state.heading_degrees = (
            self.state.heading_degrees + heading_change
        ) % 360.0

        distance_meters = speed / max(self.config.gps_rate_hz, 1e-6)

        heading_radians = math.radians(self.state.heading_degrees)

        delta_lat = (
            distance_meters * math.cos(heading_radians)
        ) / 111_111.0

        longitude_scale = max(
            math.cos(math.radians(self.state.latitude)),
            0.01,
        )

        delta_lon = (
            distance_meters * math.sin(heading_radians)
        ) / (111_111.0 * longitude_scale)

        self.state.latitude += delta_lat
        self.state.longitude += delta_lon
        self.state.speed_meters_per_second = speed

        degraded = self._quality_is_degraded()

        horizontal_accuracy = (
            self.rng.uniform(5.0, 12.0)
            if not degraded
            else self.rng.uniform(35.0, 90.0)
        )

        payload = {
            "latitude": round(self.state.latitude, 7),
            "longitude": round(self.state.longitude, 7),
            "altitude_meters": round(
                self.state.altitude_meters
                + self.rng.gauss(0.0, 0.8),
                3,
            ),
            "horizontal_accuracy_meters": round(
                horizontal_accuracy,
                3,
            ),
            "heading_degrees": round(
                self.state.heading_degrees,
                3,
            ),
            "heading_accuracy_degrees": round(
                5.0 if not degraded else 25.0,
                3,
            ),
            "speed_meters_per_second": round(speed, 3),
            "provider": "simulated_gps",
            "degraded": degraded,
        }

        return self._build_packet(
            modality="gps",
            sensor_type="gps",
            sequence_number=self.state.gps_sequence,
            sampling_rate_hz=self.config.gps_rate_hz,
            payload=payload,
        )

    def generate_accelerometer_packet(
        self,
    ) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.accelerometer_sequence += 1

        profile = self._scenario_motion_profile()
        spread = profile["accel_spread"]

        payload = {
            "x": round(self.rng.gauss(0.0, spread), 5),
            "y": round(self.rng.gauss(9.81, spread), 5),
            "z": round(self.rng.gauss(0.0, spread), 5),
            "unit": "m/s^2",
            "motion_intensity_hint": round(
                profile["motion_intensity"],
                4,
            ),
        }

        if self.config.scenario == SimulationScenario.EMERGENCY:
            payload["x"] = round(
                self.rng.gauss(4.0, 2.0),
                5,
            )
            payload["z"] = round(
                self.rng.gauss(6.0, 2.5),
                5,
            )

        return self._build_packet(
            modality="accelerometer",
            sensor_type="accelerometer",
            sequence_number=self.state.accelerometer_sequence,
            sampling_rate_hz=self.config.imu_rate_hz,
            payload=payload,
        )

    def generate_gyroscope_packet(
        self,
    ) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.gyroscope_sequence += 1

        spread = self._scenario_motion_profile()["gyro_spread"]

        payload = {
            "x": round(self.rng.gauss(0.0, spread), 5),
            "y": round(self.rng.gauss(0.0, spread), 5),
            "z": round(self.rng.gauss(0.0, spread), 5),
            "unit": "rad/s",
        }

        return self._build_packet(
            modality="gyroscope",
            sensor_type="gyroscope",
            sequence_number=self.state.gyroscope_sequence,
            sampling_rate_hz=self.config.imu_rate_hz,
            payload=payload,
        )

    def generate_magnetometer_packet(
        self,
    ) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.magnetometer_sequence += 1

        heading = math.radians(self.state.heading_degrees)

        field_strength = 45.0

        payload = {
            "x": round(
                field_strength * math.cos(heading)
                + self.rng.gauss(0.0, 1.0),
                5,
            ),
            "y": round(
                field_strength * math.sin(heading)
                + self.rng.gauss(0.0, 1.0),
                5,
            ),
            "z": round(
                self.rng.gauss(5.0, 1.0),
                5,
            ),
            "unit": "microtesla",
            "heading_degrees_hint": round(
                self.state.heading_degrees,
                3,
            ),
        }

        return self._build_packet(
            modality="magnetometer",
            sensor_type="magnetometer",
            sequence_number=self.state.magnetometer_sequence,
            sampling_rate_hz=self.config.imu_rate_hz,
            payload=payload,
        )

    def generate_interaction_packet(
        self,
        *,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        action = self._scenario_interaction_action()

        if action is None:
            return None

        if (
            not force
            and self.state.last_interaction_action == action
        ):
            return None

        self.state.interaction_sequence += 1
        self.state.last_interaction_action = action

        emergency = (
            self.config.scenario == SimulationScenario.EMERGENCY
        )

        self.state.emergency_active = emergency

        payload = {
            "interaction_id": (
                f"INTERACTION_"
                f"{self.state.interaction_sequence:06d}"
            ),
            "interaction_type": (
                "emergency_trigger"
                if emergency
                else "button"
            ),
            "action": action,
            "value": None,
            "emergency_flag": emergency,
        }

        return self._build_packet(
            modality=(
                "emergency"
                if emergency
                else "button"
            ),
            sensor_type=(
                "emergency_trigger"
                if emergency
                else "touchscreen_button"
            ),
            sequence_number=self.state.interaction_sequence,
            sampling_rate_hz=None,
            payload=payload,
        )

    def generate_wearable_packet(
        self,
    ) -> Optional[Dict[str, Any]]:
        if self._should_drop_packet():
            self.state.dropped_packets += 1
            return None

        self.state.wearable_sequence += 1

        payload = {
            "device_id": "EARPHONE_001",
            "device_name": self.config.earphone_name,
            "device_type": "wireless_earphones",
            "connected": self.config.earphone_connected,
            "connection_type": "bluetooth",
            "battery_level": round(
                clamp(
                    self.state.battery_level - 0.08,
                    0.0,
                    1.0,
                ),
                4,
            ),
            "capabilities": [
                "audio_output",
                "microphone_input",
            ],
            "microphone_available": (
                self.config.earphone_connected
            ),
            "audio_output_available": (
                self.config.earphone_connected
            ),
            "haptic_output_available": False,
        }

        return self._build_packet(
            modality="earphone",
            sensor_type="wireless_earphones",
            sequence_number=self.state.wearable_sequence,
            sampling_rate_hz=self.config.wearable_rate_hz,
            payload=payload,
        )

    def generate_device_status_packet(
        self,
    ) -> Dict[str, Any]:
        self.state.device_sequence += 1

        payload = {
            "device_id": self.config.device_id,
            "device_name": self.config.device_name,
            "device_type": "android_smartphone",
            "operating_system": self.config.operating_system,
            "application_version": (
                self.config.application_version
            ),
            "battery_level": round(
                self.state.battery_level,
                4,
            ),
            "is_charging": self.config.is_charging,
            "network_type": "wifi",
            "network_strength": round(
                self.state.network_strength,
                4,
            ),
            "network_latency_ms": round(
                self.state.network_latency_ms,
                3,
            ),
            "available_sensors": [
                "camera",
                "microphone",
                "gps",
                "accelerometer",
                "gyroscope",
                "magnetometer",
                "touchscreen",
                "bluetooth",
            ],
        }

        return self._build_packet(
            modality="wearable",
            sensor_type="device_status",
            sequence_number=self.state.device_sequence,
            sampling_rate_hz=self.config.device_status_rate_hz,
            payload=payload,
        )

    def generate_environment_packet(
        self,
    ) -> Optional[Dict[str, Any]]:
        if not self.config.include_environment:
            return None

        self.state.environment_sequence += 1

        weather_by_scenario = {
            SimulationScenario.PARK: {
                "condition": "partly_cloudy",
                "temperature_c": 28.0,
            },
            SimulationScenario.EMERGENCY: {
                "condition": "unknown",
                "temperature_c": 27.0,
            },
            SimulationScenario.NAVIGATION: {
                "condition": "clear",
                "temperature_c": 30.0,
            },
        }

        weather = weather_by_scenario.get(
            self.config.scenario,
            {
                "condition": "clear",
                "temperature_c": 26.0,
            },
        )

        payload = {
            "weather": weather,
            "map_context": {
                "latitude": round(self.state.latitude, 7),
                "longitude": round(self.state.longitude, 7),
                "region": "simulated_region",
            },
            "traffic_context": {
                "status": (
                    "heavy"
                    if self.config.scenario
                    == SimulationScenario.SHOPPING_MALL
                    else "normal"
                )
            },
            "provider_names": [
                "simulated_weather",
                "simulated_maps",
            ],
            "retrieved_at": utc_now_iso(),
            "cache_age_seconds": 0.0,
        }

        return self._build_packet(
            modality="environment",
            sensor_type="environment_api",
            sequence_number=self.state.environment_sequence,
            sampling_rate_hz=self.config.environment_rate_hz,
            payload=payload,
        )

    # ========================================================
    # CYCLE GENERATION
    # ========================================================

    def generate_cycle(
        self,
        *,
        include_interaction: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate one complete multimodal acquisition cycle.
        """

        if self.state.status != SimulatorStatus.RUNNING:
            self.start()

        self.state.cycle_number += 1

        self._simulate_network_state()
        self._update_battery()

        candidate_packets = [
            self.generate_camera_packet(),
            self.generate_audio_packet(),
            self.generate_gps_packet(),
            self.generate_accelerometer_packet(),
            self.generate_gyroscope_packet(),
            self.generate_magnetometer_packet(),
            (
                self.generate_interaction_packet()
                if include_interaction
                else None
            ),
            self.generate_wearable_packet(),
            self.generate_device_status_packet(),
            self.generate_environment_packet(),
        ]

        packets = [
            packet
            for packet in candidate_packets
            if packet is not None
        ]

        self.state.generated_packets += len(packets)

        return packets

    def generate_cycles(
        self,
        count: int,
    ) -> List[List[Dict[str, Any]]]:
        if count <= 0:
            raise ValueError("count must be greater than zero.")

        return [
            self.generate_cycle()
            for _ in range(count)
        ]

    def stream(
        self,
        *,
        cycles: Optional[int] = None,
        interval_seconds: float = 1.0,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Yield continuous simulation cycles.

        Parameters
        ----------
        cycles:
            Number of cycles to produce. None means continuous.

        interval_seconds:
            Delay between cycles.
        """

        if interval_seconds < 0:
            raise ValueError(
                "interval_seconds cannot be negative."
            )

        self.start()

        generated = 0

        try:
            while cycles is None or generated < cycles:
                yield self.generate_cycle()
                generated += 1

                if interval_seconds > 0:
                    time.sleep(interval_seconds)

        finally:
            self.stop()

    # ========================================================
    # RECEIVER INTEGRATION
    # ========================================================

    def send_cycle_to_receiver(
        self,
        receiver: Any,
        *,
        include_interaction: bool = True,
        raise_on_error: bool = True,
    ) -> List[Any]:
        """
        Generate one cycle and send every packet to a receiver.

        The receiver must provide:

            receive_batch(packets, raise_on_error=...)
        """

        if not hasattr(receiver, "receive_batch"):
            raise TypeError(
                "receiver must provide a receive_batch() method."
            )

        packets = self.generate_cycle(
            include_interaction=include_interaction
        )

        return receiver.receive_batch(
            packets,
            raise_on_error=raise_on_error,
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        return {
            "status": self.state.status.value,
            "scenario": self.config.scenario.value,
            "device_id": self.config.device_id,
            "cycle_number": self.state.cycle_number,
            "generated_packets": self.state.generated_packets,
            "dropped_packets": self.state.dropped_packets,
            "battery_level": round(
                self.state.battery_level,
                4,
            ),
            "network_strength": round(
                self.state.network_strength,
                4,
            ),
            "network_latency_ms": round(
                self.state.network_latency_ms,
                3,
            ),
            "latitude": round(self.state.latitude, 7),
            "longitude": round(self.state.longitude, 7),
            "heading_degrees": round(
                self.state.heading_degrees,
                3,
            ),
            "emergency_active": self.state.emergency_active,
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_simulator_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | PHONE SENSOR SIMULATOR TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating simulator configuration...")

        config = PhoneSimulatorConfig(
            scenario=SimulationScenario.NAVIGATION,
            random_seed=42,
            include_environment=True,
        )

        config.validate()

        print("[SUCCESS] Configuration is valid.")

        print("[2/6] Creating simulator...")

        simulator = PhoneSensorSimulator(config)
        simulator.start()

        if simulator.state.status != SimulatorStatus.RUNNING:
            raise AssertionError(
                "Simulator did not start correctly."
            )

        print("[SUCCESS] Simulator started.")

        print("[3/6] Generating one multimodal cycle...")

        packets = simulator.generate_cycle()

        if len(packets) < 8:
            raise AssertionError(
                "Expected at least eight generated packets."
            )

        print(
            f"[SUCCESS] Generated {len(packets)} packets."
        )

        print("[4/6] Checking modality coverage...")

        modalities = {
            packet["modality"]
            for packet in packets
        }

        expected = {
            "camera",
            "microphone",
            "gps",
            "accelerometer",
            "gyroscope",
            "magnetometer",
            "button",
            "earphone",
            "wearable",
            "environment",
        }

        missing = expected - modalities

        if missing:
            raise AssertionError(
                f"Missing expected modalities: {sorted(missing)}"
            )

        print("[SUCCESS] All expected modalities were generated.")

        print("[5/6] Checking packet schema fields...")

        required_fields = {
            "packet_id",
            "schema_version",
            "device_id",
            "modality",
            "sensor_type",
            "timestamp",
            "sequence_number",
            "payload",
            "metadata",
        }

        for packet in packets:
            absent = required_fields - set(packet)

            if absent:
                raise AssertionError(
                    f"Packet missing fields: {sorted(absent)}"
                )

        print("[SUCCESS] Packet envelopes are valid.")

        print("[6/6] Testing receiver integration...")

        try:
            from layer1.acquisition.multimodal_receiver import (
                MultimodalReceiver,
            )
            from layer1.config.settings import create_test_settings

            receiver = MultimodalReceiver(
                create_test_settings()
            )
            receiver.start()

            receipts = receiver.receive_batch(
                packets,
                raise_on_error=True,
            )

            if not all(receipt.accepted for receipt in receipts):
                raise AssertionError(
                    "One or more simulator packets were rejected."
                )

            print(
                "[SUCCESS] Simulator packets were accepted "
                "by MultimodalReceiver."
            )

        except ModuleNotFoundError:
            print(
                "[SKIPPED] Receiver integration imports were not "
                "available in this execution context."
            )

        simulator.stop()

        print("\nSimulator summary:")
        print(
            json.dumps(
                simulator.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nExample packet:")
        print(
            json.dumps(
                packets[0],
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print("[PASSED] PHONE SENSOR SIMULATOR IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] PHONE SENSOR SIMULATOR TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated NOONGIL Layer 1 smartphone "
            "sensor packets."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=[
            scenario.value
            for scenario in SimulationScenario
        ],
        default=SimulationScenario.NAVIGATION.value,
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--environment",
        action="store_true",
    )

    parser.add_argument(
        "--missing-packets",
        action="store_true",
    )

    parser.add_argument(
        "--degraded-quality",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.self_test:
        passed = run_simulator_self_test()

        if not passed:
            raise SystemExit(1)

        return

    if args.cycles <= 0:
        parser.error("--cycles must be greater than zero.")

    if args.interval < 0:
        parser.error("--interval cannot be negative.")

    config = PhoneSimulatorConfig(
        scenario=SimulationScenario(args.scenario),
        include_environment=args.environment,
        introduce_missing_packets=args.missing_packets,
        introduce_degraded_quality=args.degraded_quality,
    )

    simulator = PhoneSensorSimulator(config)

    for cycle_packets in simulator.stream(
        cycles=args.cycles,
        interval_seconds=args.interval,
    ):
        print(
            json.dumps(
                cycle_packets,
                indent=2,
                ensure_ascii=False,
            )
        )

    print("\nSimulation summary:")
    print(
        json.dumps(
            simulator.summary(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()