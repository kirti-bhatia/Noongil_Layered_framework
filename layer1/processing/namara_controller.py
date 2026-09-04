"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : NAMARA Controller
File    : layer1/processing/namara_controller.py
============================================================

NAMARA
------
NOONGIL Adaptive Multimodal Acquisition and Reliability Algorithm

Purpose
-------
NAMARA dynamically controls Layer 1 sensor acquisition using:

1. Current acquisition mode
2. Sensor reliability
3. Context relevance
4. Urgency
5. Energy cost
6. Phone battery
7. Network strength and latency
8. Sensor availability
9. Sensor quality
10. Emergency overrides

NAMARA does not perform semantic reasoning. It only generates an
adaptive acquisition plan for the next Layer 1 cycle.

Activation score
----------------
For modality i:

    A_i =
        w_r * reliability_i
      + w_c * context_relevance_i
      + w_u * urgency
      - w_e * energy_cost_i

A modality is activated when its score is above the configured
threshold, unless an emergency override forces activation.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json
import math
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from layer1.config.settings import (
    Layer1Settings,
    NAMARASettings,
    create_default_settings,
    create_test_settings,
)
from layer1.schemas.sensor_packet import AcquisitionMode
from layer1.utils.logger import (
    PipelineTimer,
    get_logger,
    log_exception,
    log_sensor_event,
)


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
    "interaction",
    "wearable",
    "environment",
}

DEFAULT_ENERGY_COSTS: Dict[str, float] = {
    "vision": 0.90,
    "audio": 0.60,
    "spatial": 0.55,
    "motion": 0.35,
    "interaction": 0.10,
    "wearable": 0.20,
    "environment": 0.70,
}

DEFAULT_BASE_SAMPLING_RATES: Dict[str, float] = {
    "vision": 10.0,
    "audio": 1.0,
    "spatial": 1.0,
    "motion": 25.0,
    "interaction": 1.0,
    "wearable": 0.5,
    "environment": 0.1,
}

DEFAULT_MINIMUM_SAMPLING_RATES: Dict[str, float] = {
    "vision": 1.0,
    "audio": 0.2,
    "spatial": 0.1,
    "motion": 5.0,
    "interaction": 0.1,
    "wearable": 0.1,
    "environment": 0.01,
}

DEFAULT_MAXIMUM_SAMPLING_RATES: Dict[str, float] = {
    "vision": 30.0,
    "audio": 4.0,
    "spatial": 5.0,
    "motion": 100.0,
    "interaction": 10.0,
    "wearable": 2.0,
    "environment": 1.0,
}


# ============================================================
# EXCEPTIONS
# ============================================================

class NAMARAError(Exception):
    """Base exception for NAMARA controller errors."""


class NAMARAValidationError(NAMARAError):
    """Raised when NAMARA input is invalid."""


class NAMARAProcessingError(NAMARAError):
    """Raised when an acquisition plan cannot be generated."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ModalityObservation:
    """
    Current signal-level state for one modality.
    """

    modality: str

    available: bool = True
    reliability: float = 0.75
    context_relevance: Optional[float] = None
    energy_cost: Optional[float] = None

    quality_score: Optional[float] = None
    freshness_score: Optional[float] = None
    sensor_health_score: Optional[float] = None

    current_sampling_rate_hz: Optional[float] = None
    limitations: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise NAMARAValidationError(
                f"Unsupported modality: {self.modality!r}"
            )

        for field_name, value in (
            ("reliability", self.reliability),
            ("context_relevance", self.context_relevance),
            ("energy_cost", self.energy_cost),
            ("quality_score", self.quality_score),
            ("freshness_score", self.freshness_score),
            ("sensor_health_score", self.sensor_health_score),
        ):
            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise NAMARAValidationError(
                    f"{field_name} for {self.modality!r} "
                    "must be between 0.0 and 1.0."
                )

        if self.current_sampling_rate_hz is not None:
            if (
                isinstance(self.current_sampling_rate_hz, bool)
                or not isinstance(
                    self.current_sampling_rate_hz,
                    (int, float),
                )
                or not math.isfinite(
                    float(self.current_sampling_rate_hz)
                )
                or self.current_sampling_rate_hz <= 0
            ):
                raise NAMARAValidationError(
                    "current_sampling_rate_hz must be positive."
                )


@dataclass
class NAMARAContext:
    """
    Runtime context used to generate an acquisition plan.
    """

    mode: AcquisitionMode = AcquisitionMode.AWARENESS

    urgency: float = 0.30

    battery_level: float = 1.0
    is_charging: bool = False

    network_strength: float = 1.0
    network_latency_ms: float = 0.0
    network_available: bool = True

    emergency_active: bool = False
    user_interaction_active: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for field_name, value in (
            ("urgency", self.urgency),
            ("battery_level", self.battery_level),
            ("network_strength", self.network_strength),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise NAMARAValidationError(
                    f"{field_name} must be between 0.0 and 1.0."
                )

        if (
            isinstance(self.network_latency_ms, bool)
            or not isinstance(
                self.network_latency_ms,
                (int, float),
            )
            or not math.isfinite(
                float(self.network_latency_ms)
            )
            or self.network_latency_ms < 0
        ):
            raise NAMARAValidationError(
                "network_latency_ms must be non-negative."
            )


@dataclass
class ModalityDecision:
    """
    NAMARA decision for one modality.
    """

    modality: str

    active: bool
    forced: bool

    activation_score: float
    activation_threshold: float

    reliability_score: float
    context_relevance_score: float
    urgency_score: float
    energy_cost_score: float

    target_sampling_rate_hz: float
    sampling_multiplier: float

    priority: str
    reason_codes: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NAMARAPlan:
    """
    Complete adaptive acquisition plan.
    """

    plan_id: str
    generated_at: str

    requested_mode: AcquisitionMode
    effective_mode: AcquisitionMode

    decisions: Dict[str, ModalityDecision]

    active_modalities: List[str]
    inactive_modalities: List[str]
    forced_modalities: List[str]

    battery_state: str
    network_state: str
    urgency: float

    total_estimated_energy_cost: float
    average_activation_score: float

    reason_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["requested_mode"] = self.requested_mode.value
        payload["effective_mode"] = self.effective_mode.value

        payload["decisions"] = {
            modality: decision.to_dict()
            for modality, decision in self.decisions.items()
        }

        return payload


@dataclass
class NAMARAStatistics:
    """
    Runtime NAMARA statistics.
    """

    total_plans: int = 0
    emergency_plans: int = 0
    low_power_plans: int = 0
    degraded_network_plans: int = 0

    forced_activations: int = 0
    inactive_decisions: int = 0

    cumulative_processing_seconds: float = 0.0
    last_plan_id: Optional[str] = None
    last_effective_mode: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_plans == 0:
            return 0.0

        return (
            self.cumulative_processing_seconds
            / self.total_plans
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

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


def priority_from_score(score: float) -> str:
    if score >= 0.85:
        return "critical"

    if score >= 0.65:
        return "high"

    if score >= 0.40:
        return "medium"

    return "low"


# ============================================================
# NAMARA CONTROLLER
# ============================================================

class NAMARAController:
    """
    Adaptive Layer 1 sensor acquisition controller.

    NAMARA consumes modality observations and runtime context,
    then creates a plan for the next acquisition cycle.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.namara_settings: NAMARASettings = (
            self.settings.namara
        )

        self.logger = get_logger(
            "processing.namara_controller"
        )

        self.statistics = NAMARAStatistics()

        self._last_plan: Optional[NAMARAPlan] = None
        self._plan_counter = 0

    # ========================================================
    # PUBLIC API
    # ========================================================

    def create_plan(
        self,
        observations: Iterable[ModalityObservation],
        context: Optional[NAMARAContext] = None,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> NAMARAPlan:
        """
        Generate one adaptive acquisition plan.
        """

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        started = time.perf_counter()

        observation_list = list(observations)
        context = context or NAMARAContext()

        try:
            with PipelineTimer(
                "namara.create_plan",
                logger=self.logger,
                metadata={
                    "requested_mode": context.mode.value,
                    "observation_count": len(
                        observation_list
                    ),
                },
            ):
                context.validate()

                observations_by_modality = (
                    self._normalize_observations(
                        observation_list
                    )
                )

                effective_mode, global_reasons = (
                    self._resolve_effective_mode(
                        context
                    )
                )

                decisions: Dict[
                    str,
                    ModalityDecision,
                ] = {}

                for modality in sorted(
                    SUPPORTED_MODALITIES
                ):
                    observation = (
                        observations_by_modality[
                            modality
                        ]
                    )

                    decision = self._make_decision(
                        modality=modality,
                        observation=observation,
                        context=context,
                        effective_mode=effective_mode,
                    )

                    decisions[modality] = decision

                decisions = self._apply_dependency_rules(
                    decisions=decisions,
                    context=context,
                    effective_mode=effective_mode,
                )

                plan = self._build_plan(
                    requested_mode=context.mode,
                    effective_mode=effective_mode,
                    decisions=decisions,
                    context=context,
                    reason_codes=global_reasons,
                )

                elapsed = time.perf_counter() - started

                self.statistics.total_plans += 1
                self.statistics.cumulative_processing_seconds += (
                    elapsed
                )
                self.statistics.last_plan_id = plan.plan_id
                self.statistics.last_effective_mode = (
                    plan.effective_mode.value
                )
                self.statistics.last_error = None

                if (
                    plan.effective_mode
                    == AcquisitionMode.EMERGENCY
                ):
                    self.statistics.emergency_plans += 1

                if (
                    plan.effective_mode
                    == AcquisitionMode.LOW_POWER
                ):
                    self.statistics.low_power_plans += 1

                if (
                    plan.effective_mode
                    == AcquisitionMode.DEGRADED_NETWORK
                ):
                    self.statistics.degraded_network_plans += 1

                self.statistics.forced_activations += len(
                    plan.forced_modalities
                )

                self.statistics.inactive_decisions += len(
                    plan.inactive_modalities
                )

                self._last_plan = plan

                log_sensor_event(
                    modality="interaction",
                    event="NAMARA acquisition plan generated",
                    sensor_type="namara_controller",
                    packet_id=plan.plan_id,
                    details={
                        "requested_mode": (
                            plan.requested_mode.value
                        ),
                        "effective_mode": (
                            plan.effective_mode.value
                        ),
                        "active_modalities": (
                            plan.active_modalities
                        ),
                        "inactive_modalities": (
                            plan.inactive_modalities
                        ),
                        "forced_modalities": (
                            plan.forced_modalities
                        ),
                        "battery_state": (
                            plan.battery_state
                        ),
                        "network_state": (
                            plan.network_state
                        ),
                        "average_activation_score": (
                            plan.average_activation_score
                        ),
                    },
                )

                return plan

        except Exception as error:
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "NAMARA plan generation failed",
                error=error,
                details={
                    "requested_mode": (
                        context.mode.value
                        if isinstance(
                            context.mode,
                            AcquisitionMode,
                        )
                        else str(context.mode)
                    ),
                    "observation_count": len(
                        observation_list
                    ),
                },
            )

            if should_raise:
                raise

            raise NAMARAProcessingError(
                f"NAMARA plan generation failed: {error}"
            ) from error

    # ========================================================
    # OBSERVATION NORMALIZATION
    # ========================================================

    def _normalize_observations(
        self,
        observations: List[ModalityObservation],
    ) -> Dict[str, ModalityObservation]:
        normalized: Dict[
            str,
            ModalityObservation,
        ] = {}

        for observation in observations:
            observation.validate()

            if observation.modality in normalized:
                raise NAMARAValidationError(
                    "Duplicate observation for modality "
                    f"{observation.modality!r}."
                )

            normalized[
                observation.modality
            ] = observation

        for modality in SUPPORTED_MODALITIES:
            if modality not in normalized:
                normalized[modality] = (
                    ModalityObservation(
                        modality=modality,
                        available=False,
                        reliability=0.0,
                        quality_score=0.0,
                        freshness_score=0.0,
                        sensor_health_score=0.0,
                        metadata={
                            "auto_created_missing_observation": (
                                True
                            )
                        },
                    )
                )

        return normalized

    # ========================================================
    # MODE RESOLUTION
    # ========================================================

    def _resolve_effective_mode(
        self,
        context: NAMARAContext,
    ) -> tuple[AcquisitionMode, List[str]]:
        reasons: List[str] = []

        if context.emergency_active:
            reasons.append(
                "emergency_override_active"
            )
            return AcquisitionMode.EMERGENCY, reasons

        if (
            context.battery_level
            <= self.namara_settings
            .battery_critical_threshold
            and not context.is_charging
        ):
            reasons.append(
                "critical_battery_override"
            )
            return AcquisitionMode.LOW_POWER, reasons

        network_degraded = (
            not context.network_available
            or context.network_strength
            < self.namara_settings
            .network_degraded_strength_threshold
            or context.network_latency_ms
            > self.namara_settings
            .network_high_latency_ms
        )

        if network_degraded:
            reasons.append(
                "degraded_network_override"
            )
            return (
                AcquisitionMode.DEGRADED_NETWORK,
                reasons,
            )

        if (
            context.battery_level
            <= self.namara_settings
            .battery_low_threshold
            and not context.is_charging
        ):
            reasons.append("low_battery_override")
            return AcquisitionMode.LOW_POWER, reasons

        return context.mode, reasons

    # ========================================================
    # DECISION GENERATION
    # ========================================================

    def _make_decision(
        self,
        *,
        modality: str,
        observation: ModalityObservation,
        context: NAMARAContext,
        effective_mode: AcquisitionMode,
    ) -> ModalityDecision:
        mode_priorities = (
            self.namara_settings
            .mode_sensor_priorities[
                effective_mode.value
            ]
        )

        context_relevance = (
            observation.context_relevance
            if observation.context_relevance is not None
            else mode_priorities.get(
                modality,
                0.0,
            )
        )

        energy_cost = (
            observation.energy_cost
            if observation.energy_cost is not None
            else DEFAULT_ENERGY_COSTS[
                modality
            ]
        )

        reliability = self._aggregate_reliability(
            observation
        )

        urgency = context.urgency

        weights = self.namara_settings.weights

        activation_score = (
            weights.reliability_weight * reliability
            + weights.context_weight
            * context_relevance
            + weights.urgency_weight * urgency
            - weights.energy_weight * energy_cost
        )

        activation_score = clamp(
            activation_score,
            0.0,
            1.0,
        )

        threshold = (
            self.namara_settings
            .emergency_activation_threshold
            if effective_mode
            == AcquisitionMode.EMERGENCY
            else self.namara_settings
            .activation_threshold
        )

        forced = False
        reasons: List[str] = []

        if not observation.available:
            active = False
            reasons.append("sensor_unavailable")
        else:
            active = activation_score >= threshold

        if (
            effective_mode
            == AcquisitionMode.EMERGENCY
            and observation.available
            and modality in {
                "vision",
                "audio",
                "spatial",
                "motion",
                "interaction",
                "wearable",
            }
        ):
            active = True
            forced = True
            reasons.append(
                "emergency_forced_activation"
            )

        if (
            modality == "interaction"
            and observation.available
        ):
            active = True
            forced = True
            reasons.append(
                "interaction_always_available"
            )

        if (
            effective_mode
            == AcquisitionMode.READING
            and modality == "vision"
            and observation.available
        ):
            active = True
            forced = True
            reasons.append(
                "reading_requires_vision"
            )

        if (
            effective_mode
            == AcquisitionMode.NAVIGATION
            and modality in {
                "vision",
                "spatial",
                "motion",
                "interaction",
            }
            and observation.available
        ):
            active = True
            forced = True
            reasons.append(
                "navigation_core_modality"
            )

        if active:
            reasons.append(
                "activation_score_passed"
                if activation_score >= threshold
                else "forced_despite_low_score"
            )
        else:
            reasons.append(
                "activation_score_below_threshold"
            )

        sampling_multiplier = (
            self._calculate_sampling_multiplier(
                modality=modality,
                observation=observation,
                context=context,
                effective_mode=effective_mode,
                active=active,
            )
        )

        target_rate = self._calculate_target_rate(
            modality=modality,
            observation=observation,
            multiplier=sampling_multiplier,
            active=active,
        )

        return ModalityDecision(
            modality=modality,
            active=active,
            forced=forced,
            activation_score=round(
                activation_score,
                6,
            ),
            activation_threshold=threshold,
            reliability_score=round(
                reliability,
                6,
            ),
            context_relevance_score=round(
                context_relevance,
                6,
            ),
            urgency_score=round(
                urgency,
                6,
            ),
            energy_cost_score=round(
                energy_cost,
                6,
            ),
            target_sampling_rate_hz=round(
                target_rate,
                6,
            ),
            sampling_multiplier=round(
                sampling_multiplier,
                6,
            ),
            priority=priority_from_score(
                activation_score
            ),
            reason_codes=reasons,
            metadata={
                "limitations": list(
                    observation.limitations
                ),
                "available": observation.available,
            },
        )

    def _aggregate_reliability(
        self,
        observation: ModalityObservation,
    ) -> float:
        components = [
            observation.reliability,
        ]

        for optional_score in (
            observation.quality_score,
            observation.freshness_score,
            observation.sensor_health_score,
        ):
            if optional_score is not None:
                components.append(optional_score)

        reliability = sum(components) / len(
            components
        )

        limitation_penalty = min(
            0.10 * len(
                observation.limitations
            ),
            0.40,
        )

        return clamp(
            reliability - limitation_penalty,
            0.0,
            1.0,
        )

    def _calculate_sampling_multiplier(
        self,
        *,
        modality: str,
        observation: ModalityObservation,
        context: NAMARAContext,
        effective_mode: AcquisitionMode,
        active: bool,
    ) -> float:
        if not active:
            return 0.0

        multiplier = 1.0

        if (
            effective_mode
            == AcquisitionMode.EMERGENCY
        ):
            multiplier *= (
                self.namara_settings
                .emergency_sampling_multiplier
            )

        elif (
            effective_mode
            == AcquisitionMode.LOW_POWER
        ):
            multiplier *= (
                self.namara_settings
                .low_power_sampling_multiplier
            )

        elif (
            effective_mode
            == AcquisitionMode.DEGRADED_NETWORK
        ):
            multiplier *= (
                self.namara_settings
                .degraded_network_sampling_multiplier
            )

        if (
            modality == "motion"
            and context.urgency >= 0.60
        ):
            multiplier *= (
                self.namara_settings
                .motion_sampling_multiplier
            )

        if observation.quality_score is not None:
            if observation.quality_score < 0.40:
                multiplier *= 0.75
            elif observation.quality_score >= 0.85:
                multiplier *= 1.10

        if (
            modality in {"vision", "audio", "environment"}
            and (
                not context.network_available
                or context.network_strength < 0.30
            )
        ):
            multiplier *= 0.70

        return clamp(
            multiplier,
            0.10,
            4.0,
        )

    def _calculate_target_rate(
        self,
        *,
        modality: str,
        observation: ModalityObservation,
        multiplier: float,
        active: bool,
    ) -> float:
        if not active:
            return 0.0

        base_rate = (
            observation.current_sampling_rate_hz
            if observation.current_sampling_rate_hz
            is not None
            else DEFAULT_BASE_SAMPLING_RATES[
                modality
            ]
        )

        target = base_rate * multiplier

        return clamp(
            target,
            DEFAULT_MINIMUM_SAMPLING_RATES[
                modality
            ],
            DEFAULT_MAXIMUM_SAMPLING_RATES[
                modality
            ],
        )

    # ========================================================
    # DEPENDENCY RULES
    # ========================================================

    def _apply_dependency_rules(
        self,
        *,
        decisions: Dict[str, ModalityDecision],
        context: NAMARAContext,
        effective_mode: AcquisitionMode,
    ) -> Dict[str, ModalityDecision]:
        """
        Apply safety and operational dependency rules.
        """

        if (
            decisions["vision"].active
            and not decisions["motion"].active
            and effective_mode
            in {
                AcquisitionMode.NAVIGATION,
                AcquisitionMode.EMERGENCY,
            }
            and decisions["motion"].metadata[
                "available"
            ]
        ):
            decisions["motion"].active = True
            decisions["motion"].forced = True
            decisions["motion"].reason_codes.append(
                "vision_motion_dependency"
            )
            decisions[
                "motion"
            ].target_sampling_rate_hz = max(
                decisions[
                    "motion"
                ].target_sampling_rate_hz,
                DEFAULT_MINIMUM_SAMPLING_RATES[
                    "motion"
                ],
            )

        if (
            decisions["spatial"].active
            and not decisions["motion"].active
            and effective_mode
            == AcquisitionMode.NAVIGATION
            and decisions["motion"].metadata[
                "available"
            ]
        ):
            decisions["motion"].active = True
            decisions["motion"].forced = True
            decisions["motion"].reason_codes.append(
                "spatial_motion_dependency"
            )
            decisions[
                "motion"
            ].target_sampling_rate_hz = max(
                decisions[
                    "motion"
                ].target_sampling_rate_hz,
                DEFAULT_BASE_SAMPLING_RATES[
                    "motion"
                ],
            )

        if (
            context.user_interaction_active
            and decisions["interaction"].metadata[
                "available"
            ]
        ):
            decisions["interaction"].active = True
            decisions["interaction"].forced = True
            decisions[
                "interaction"
            ].reason_codes.append(
                "active_user_interaction"
            )

        return decisions

    # ========================================================
    # PLAN BUILDING
    # ========================================================

    def _build_plan(
        self,
        *,
        requested_mode: AcquisitionMode,
        effective_mode: AcquisitionMode,
        decisions: Dict[str, ModalityDecision],
        context: NAMARAContext,
        reason_codes: List[str],
    ) -> NAMARAPlan:
        self._plan_counter += 1

        plan_id = (
            f"NAMARA_{self._plan_counter:06d}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        )

        active_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if decision.active
        )

        inactive_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if not decision.active
        )

        forced_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if decision.forced
        )

        active_energy_costs = [
            decision.energy_cost_score
            * decision.sampling_multiplier
            for decision in decisions.values()
            if decision.active
        ]

        estimated_energy = clamp(
            sum(active_energy_costs)
            / max(len(SUPPORTED_MODALITIES), 1),
            0.0,
            1.0,
        )

        average_score = (
            sum(
                decision.activation_score
                for decision in decisions.values()
            )
            / len(decisions)
        )

        return NAMARAPlan(
            plan_id=plan_id,
            generated_at=utc_now_iso(),
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            decisions=decisions,
            active_modalities=active_modalities,
            inactive_modalities=inactive_modalities,
            forced_modalities=forced_modalities,
            battery_state=self._battery_state(
                context
            ),
            network_state=self._network_state(
                context
            ),
            urgency=context.urgency,
            total_estimated_energy_cost=round(
                estimated_energy,
                6,
            ),
            average_activation_score=round(
                average_score,
                6,
            ),
            reason_codes=list(reason_codes),
            metadata={
                "battery_level": context.battery_level,
                "is_charging": context.is_charging,
                "network_strength": (
                    context.network_strength
                ),
                "network_latency_ms": (
                    context.network_latency_ms
                ),
                "network_available": (
                    context.network_available
                ),
                "emergency_active": (
                    context.emergency_active
                ),
                "user_interaction_active": (
                    context.user_interaction_active
                ),
            },
        )

    def _battery_state(
        self,
        context: NAMARAContext,
    ) -> str:
        if context.is_charging:
            return "charging"

        if (
            context.battery_level
            <= self.namara_settings
            .battery_critical_threshold
        ):
            return "critical"

        if (
            context.battery_level
            <= self.namara_settings
            .battery_low_threshold
        ):
            return "low"

        return "normal"

    def _network_state(
        self,
        context: NAMARAContext,
    ) -> str:
        if not context.network_available:
            return "offline"

        if (
            context.network_strength
            < self.namara_settings
            .network_degraded_strength_threshold
            or context.network_latency_ms
            > self.namara_settings
            .network_high_latency_ms
        ):
            return "degraded"

        return "normal"

    # ========================================================
    # STATE AND DIAGNOSTICS
    # ========================================================

    def get_last_plan(
        self,
    ) -> Optional[NAMARAPlan]:
        return self._last_plan

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "enabled": self.namara_settings.enabled,
            "last_plan_available": (
                self._last_plan is not None
            ),
            "last_plan_id": (
                self._last_plan.plan_id
                if self._last_plan
                else None
            ),
            "last_effective_mode": (
                self._last_plan
                .effective_mode.value
                if self._last_plan
                else None
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
        }


# ============================================================
# OBSERVATION FACTORY
# ============================================================

def create_default_observations(
    *,
    available_modalities: Optional[
        Iterable[str]
    ] = None,
) -> List[ModalityObservation]:
    """
    Create default observations for testing or early integration.
    """

    available_set = (
        set(available_modalities)
        if available_modalities is not None
        else set(SUPPORTED_MODALITIES)
    )

    observations: List[
        ModalityObservation
    ] = []

    for modality in sorted(
        SUPPORTED_MODALITIES
    ):
        observations.append(
            ModalityObservation(
                modality=modality,
                available=modality in available_set,
                reliability=(
                    0.85
                    if modality in available_set
                    else 0.0
                ),
                quality_score=(
                    0.80
                    if modality in available_set
                    else 0.0
                ),
                freshness_score=(
                    0.90
                    if modality in available_set
                    else 0.0
                ),
                sensor_health_score=(
                    0.90
                    if modality in available_set
                    else 0.0
                ),
                current_sampling_rate_hz=(
                    DEFAULT_BASE_SAMPLING_RATES[
                        modality
                    ]
                    if modality in available_set
                    else None
                ),
            )
        )

    return observations


# ============================================================
# SELF-TEST
# ============================================================

def run_namara_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | NAMARA CONTROLLER TEST")
    print("=" * 72)

    try:
        print("[1/7] Creating test settings...")

        settings = create_test_settings()
        controller = NAMARAController(settings)

        print("[SUCCESS] NAMARA initialized.")

        print("[2/7] Creating modality observations...")

        observations = create_default_observations(
            available_modalities={
                "vision",
                "audio",
                "spatial",
                "motion",
                "interaction",
                "wearable",
            }
        )

        print("[SUCCESS] Observations created.")

        print("[3/7] Testing navigation plan...")

        navigation_context = NAMARAContext(
            mode=AcquisitionMode.NAVIGATION,
            urgency=0.55,
            battery_level=0.85,
            network_strength=0.90,
            network_latency_ms=25.0,
            emergency_active=False,
            user_interaction_active=True,
        )

        navigation_plan = controller.create_plan(
            observations,
            navigation_context,
            raise_on_error=True,
        )

        required_navigation = {
            "vision",
            "spatial",
            "motion",
            "interaction",
        }

        if not required_navigation.issubset(
            set(
                navigation_plan
                .active_modalities
            )
        ):
            raise AssertionError(
                "Navigation core modalities were not active."
            )

        print("[SUCCESS] Navigation plan is valid.")

        print("[4/7] Testing low-power override...")

        low_power_context = NAMARAContext(
            mode=AcquisitionMode.AWARENESS,
            urgency=0.20,
            battery_level=0.08,
            is_charging=False,
            network_strength=0.90,
            network_latency_ms=20.0,
        )

        low_power_plan = controller.create_plan(
            observations,
            low_power_context,
            raise_on_error=True,
        )

        if (
            low_power_plan.effective_mode
            != AcquisitionMode.LOW_POWER
        ):
            raise AssertionError(
                "Critical battery did not trigger low-power mode."
            )

        print("[SUCCESS] Low-power override works.")

        print("[5/7] Testing degraded-network override...")

        degraded_context = NAMARAContext(
            mode=AcquisitionMode.AWARENESS,
            urgency=0.30,
            battery_level=0.80,
            network_strength=0.20,
            network_latency_ms=400.0,
        )

        degraded_plan = controller.create_plan(
            observations,
            degraded_context,
            raise_on_error=True,
        )

        if (
            degraded_plan.effective_mode
            != AcquisitionMode.DEGRADED_NETWORK
        ):
            raise AssertionError(
                "Degraded network override failed."
            )

        print("[SUCCESS] Network override works.")

        print("[6/7] Testing emergency override...")

        emergency_context = NAMARAContext(
            mode=AcquisitionMode.AWARENESS,
            urgency=1.0,
            battery_level=0.50,
            network_strength=0.50,
            network_latency_ms=100.0,
            emergency_active=True,
            user_interaction_active=True,
        )

        emergency_plan = controller.create_plan(
            observations,
            emergency_context,
            raise_on_error=True,
        )

        if (
            emergency_plan.effective_mode
            != AcquisitionMode.EMERGENCY
        ):
            raise AssertionError(
                "Emergency override failed."
            )

        required_emergency = {
            "vision",
            "audio",
            "spatial",
            "motion",
            "interaction",
            "wearable",
        }

        if not required_emergency.issubset(
            set(
                emergency_plan
                .active_modalities
            )
        ):
            raise AssertionError(
                "Emergency modalities were not activated."
            )

        print("[SUCCESS] Emergency override works.")

        print("[7/7] Checking diagnostics...")

        health = controller.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "NAMARA health check failed."
            )

        if (
            health["statistics"]["total_plans"]
            != 4
        ):
            raise AssertionError(
                "NAMARA plan count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nNavigation plan:")
        print(
            json.dumps(
                navigation_plan.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nNAMARA health:")
        print(
            json.dumps(
                health,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print("[PASSED] NAMARA CONTROLLER IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] NAMARA CONTROLLER TEST")
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        return False


if __name__ == "__main__":
    if not run_namara_self_test():
        raise SystemExit(1)