"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Confidence Estimator
File    : layer2/confidence/confidence_estimator.py
============================================================

Purpose
-------
Calculates modality-level confidence values and normalized fusion
weights for the Multimodal Fusion Engine.

Implemented equation
--------------------
                         C_i
    w_i = --------------------------------
          sum(C_j for j=1...M) + epsilon

Where:
- C_i is the effective confidence of modality i
- w_i is its normalized fusion weight
- epsilon prevents division by zero

The estimator does not generate predictions. It evaluates the
reliability of outputs produced by other Layer 2 modules.
============================================================
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from layer2.schemas.module_result import (
    ModuleResult,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    get_logger,
    log_event,
)


# ============================================================
# CONSTANTS
# ============================================================

CONFIDENCE_ESTIMATOR_VERSION = "1.0"

DEFAULT_EPSILON = 1e-8

DEFAULT_MODALITY_PRIORS = {
    "vision": 0.35,
    "audio": 0.20,
    "spatial": 0.20,
    "text": 0.15,
    "motion": 0.10,
}

STATUS_RELIABILITY_FACTORS = {
    "success": 1.00,
    "partial": 0.75,
    "skipped": 0.00,
    "failure": 0.00,
    "failed": 0.00,
    "error": 0.00,
}


# ============================================================
# EXCEPTION
# ============================================================

class ConfidenceEstimationError(Exception):
    """Raised when confidence estimation fails."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class ModalityConfidence:
    """Confidence information for one modality."""

    modality: str

    raw_confidence: float
    prior_reliability: float
    status_factor: float

    effective_confidence: float
    normalized_weight: float

    usable: bool
    status: str

    source_module: Optional[str]
    source_packet_id: Optional[str]

    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "modality": self.modality,
            "raw_confidence": round(
                self.raw_confidence,
                6,
            ),
            "prior_reliability": round(
                self.prior_reliability,
                6,
            ),
            "status_factor": round(
                self.status_factor,
                6,
            ),
            "effective_confidence": round(
                self.effective_confidence,
                6,
            ),
            "normalized_weight": round(
                self.normalized_weight,
                6,
            ),
            "usable": self.usable,
            "status": self.status,
            "source_module": (
                self.source_module
            ),
            "source_packet_id": (
                self.source_packet_id
            ),
            "warnings": list(
                self.warnings
            ),
        }


@dataclass
class ConfidenceEstimationOutput:
    """Complete confidence-estimation output."""

    modalities: Dict[
        str,
        ModalityConfidence
    ]

    fused_confidence: float

    total_effective_confidence: float
    weight_sum: float

    usable_modalities: List[str]
    excluded_modalities: List[str]

    epsilon: float

    warnings: List[str]

    @property
    def succeeded(self) -> bool:

        return bool(
            self.usable_modalities
        )

    def weight_for(
        self,
        modality: str,
    ) -> float:
        """Return a modality's normalized weight."""

        item = self.modalities.get(
            modality
        )

        if item is None:
            return 0.0

        return item.normalized_weight

    def confidence_for(
        self,
        modality: str,
    ) -> float:
        """Return effective modality confidence."""

        item = self.modalities.get(
            modality
        )

        if item is None:
            return 0.0

        return item.effective_confidence

    def to_dict(self) -> Dict[str, Any]:

        return {
            "modalities": {
                modality: item.to_dict()
                for modality, item
                in self.modalities.items()
            },
            "fused_confidence": round(
                self.fused_confidence,
                6,
            ),
            "total_effective_confidence": (
                round(
                    self.total_effective_confidence,
                    6,
                )
            ),
            "weight_sum": round(
                self.weight_sum,
                6,
            ),
            "usable_modalities": list(
                self.usable_modalities
            ),
            "excluded_modalities": list(
                self.excluded_modalities
            ),
            "epsilon": self.epsilon,
            "warnings": list(
                self.warnings
            ),
            "estimator_version": (
                CONFIDENCE_ESTIMATOR_VERSION
            ),
        }


# ============================================================
# CONFIDENCE ESTIMATOR
# ============================================================

class ConfidenceEstimator:
    """Estimate normalized multimodal confidence weights."""

    def __init__(
        self,
        *,
        modality_priors: Optional[
            Mapping[str, float]
        ] = None,
        epsilon: float = DEFAULT_EPSILON,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.epsilon = self._validate_epsilon(
            epsilon
        )

        supplied_priors = (
            modality_priors
            or DEFAULT_MODALITY_PRIORS
        )

        self.modality_priors = (
            self._validate_priors(
                supplied_priors
            )
        )

        self.logger = (
            logger
            or get_logger(
                "confidence_estimator"
            )
        )

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def estimate(
        self,
        modality_results: Mapping[
            str,
            ModuleResult,
        ],
    ) -> ConfidenceEstimationOutput:
        """
        Calculate effective confidence and normalized weights.

        modality_results example:
        {
            "vision": scene_result,
            "audio": speech_result,
            "text": ocr_result,
            "spatial": depth_result,
            "motion": activity_result,
        }
        """

        if not isinstance(
            modality_results,
            Mapping,
        ):
            raise ConfidenceEstimationError(
                "modality_results must be "
                "a mapping."
            )

        if not modality_results:
            raise ConfidenceEstimationError(
                "At least one modality result "
                "is required."
            )

        log_event(
            self.logger,
            event=(
                "confidence_estimation_started"
            ),
            message=(
                "Multimodal confidence estimation "
                "started."
            ),
            details={
                "modalities": list(
                    modality_results.keys()
                )
            },
        )

        intermediate = {}

        usable_modalities = []
        excluded_modalities = []

        global_warnings = []

        for raw_modality, result in (
            modality_results.items()
        ):
            modality = self._normalize_modality(
                raw_modality
            )

            if not modality:
                raise ConfidenceEstimationError(
                    "Modality names must be "
                    "non-empty strings."
                )

            if not isinstance(
                result,
                ModuleResult,
            ):
                raise ConfidenceEstimationError(
                    f"Result for {modality!r} "
                    "must be a ModuleResult."
                )

            raw_confidence = (
                self._extract_confidence(
                    result
                )
            )

            status = self._status_value(
                result
            )

            status_factor = (
                STATUS_RELIABILITY_FACTORS.get(
                    status,
                    0.50 if result.usable else 0.0,
                )
            )

            prior = self.modality_priors.get(
                modality,
                1.0,
            )

            usable = bool(
                result.usable
                and status_factor > 0.0
                and raw_confidence > 0.0
            )

            warnings = []

            if result.confidence is None:
                warnings.append(
                    "Source confidence is missing; "
                    "the modality was assigned zero "
                    "confidence."
                )

            if not result.usable:
                warnings.append(
                    "Source result is not usable."
                )

            if status_factor <= 0.0:
                warnings.append(
                    "Source status excludes this "
                    "modality from fusion."
                )

            effective_confidence = 0.0

            if usable:
                effective_confidence = (
                    raw_confidence
                    * prior
                    * status_factor
                )

                usable_modalities.append(
                    modality
                )

            else:
                excluded_modalities.append(
                    modality
                )

            intermediate[modality] = {
                "raw_confidence": (
                    raw_confidence
                ),
                "prior": prior,
                "status_factor": (
                    status_factor
                ),
                "effective_confidence": (
                    effective_confidence
                ),
                "usable": usable,
                "status": status,
                "source_module": getattr(
                    result,
                    "module_name",
                    None,
                ),
                "source_packet_id": getattr(
                    result,
                    "source_packet_id",
                    None,
                ),
                "warnings": warnings,
            }

        total_effective_confidence = sum(
            item[
                "effective_confidence"
            ]
            for item
            in intermediate.values()
        )

        if (
            total_effective_confidence
            <= self.epsilon
        ):
            global_warnings.append(
                "No modality supplied positive "
                "usable confidence."
            )

        modality_confidences = {}

        for modality, item in (
            intermediate.items()
        ):
            effective_confidence = item[
                "effective_confidence"
            ]

            if (
                total_effective_confidence
                > self.epsilon
            ):
                normalized_weight = (
                    effective_confidence
                    / (
                        total_effective_confidence
                        + self.epsilon
                    )
                )
            else:
                normalized_weight = 0.0

            modality_confidences[
                modality
            ] = ModalityConfidence(
                modality=modality,
                raw_confidence=(
                    item[
                        "raw_confidence"
                    ]
                ),
                prior_reliability=(
                    item["prior"]
                ),
                status_factor=(
                    item[
                        "status_factor"
                    ]
                ),
                effective_confidence=(
                    effective_confidence
                ),
                normalized_weight=(
                    normalized_weight
                ),
                usable=item["usable"],
                status=item["status"],
                source_module=(
                    item[
                        "source_module"
                    ]
                ),
                source_packet_id=(
                    item[
                        "source_packet_id"
                    ]
                ),
                warnings=item["warnings"],
            )

        weight_sum = sum(
            item.normalized_weight
            for item
            in modality_confidences.values()
        )

        fused_confidence = sum(
            item.normalized_weight
            * item.raw_confidence
            * item.status_factor
            for item
            in modality_confidences.values()
        )

        fused_confidence = self._clamp(
            fused_confidence
        )

        output = ConfidenceEstimationOutput(
            modalities=(
                modality_confidences
            ),
            fused_confidence=(
                fused_confidence
            ),
            total_effective_confidence=(
                total_effective_confidence
            ),
            weight_sum=weight_sum,
            usable_modalities=(
                usable_modalities
            ),
            excluded_modalities=(
                excluded_modalities
            ),
            epsilon=self.epsilon,
            warnings=global_warnings,
        )

        log_event(
            self.logger,
            event=(
                "confidence_estimation_completed"
            ),
            message=(
                "Multimodal confidence estimation "
                "completed."
            ),
            details={
                "fused_confidence": (
                    fused_confidence
                ),
                "weight_sum": weight_sum,
                "usable_modalities": (
                    usable_modalities
                ),
                "excluded_modalities": (
                    excluded_modalities
                ),
            },
        )

        return output

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate_epsilon(
        epsilon: Any,
    ) -> float:

        try:
            value = float(
                epsilon
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ConfidenceEstimationError(
                "epsilon must be numeric."
            ) from error

        if (
            not math.isfinite(value)
            or value <= 0.0
        ):
            raise ConfidenceEstimationError(
                "epsilon must be a positive "
                "finite number."
            )

        return value

    @classmethod
    def _validate_priors(
        cls,
        priors: Mapping[
            str,
            float,
        ],
    ) -> Dict[str, float]:

        if not isinstance(
            priors,
            Mapping,
        ):
            raise ConfidenceEstimationError(
                "modality_priors must be "
                "a mapping."
            )

        validated = {}

        for raw_modality, raw_prior in (
            priors.items()
        ):
            modality = cls._normalize_modality(
                raw_modality
            )

            if not modality:
                raise ConfidenceEstimationError(
                    "Prior modality names must "
                    "be non-empty."
                )

            try:
                prior = float(
                    raw_prior
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                raise ConfidenceEstimationError(
                    f"Prior for {modality!r} "
                    "must be numeric."
                ) from error

            if (
                not math.isfinite(prior)
                or prior < 0.0
            ):
                raise ConfidenceEstimationError(
                    f"Prior for {modality!r} "
                    "must be finite and "
                    "non-negative."
                )

            validated[
                modality
            ] = prior

        return validated

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _normalize_modality(
        modality: Any,
    ) -> str:

        if not isinstance(
            modality,
            str,
        ):
            return ""

        return (
            modality.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _status_value(
        result: ModuleResult,
    ) -> str:

        status = getattr(
            result,
            "status",
            "",
        )

        if hasattr(status, "value"):
            status = status.value

        return str(
            status
        ).strip().lower()

    @classmethod
    def _extract_confidence(
        cls,
        result: ModuleResult,
    ) -> float:

        confidence = result.confidence

        if confidence is None:
            return 0.0

        try:
            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            confidence
        ):
            return 0.0

        return cls._clamp(
            confidence
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def estimate_confidence(
    modality_results: Mapping[
        str,
        ModuleResult,
    ],
    *,
    modality_priors: Optional[
        Mapping[str, float]
    ] = None,
    epsilon: float = DEFAULT_EPSILON,
) -> ConfidenceEstimationOutput:

    estimator = ConfidenceEstimator(
        modality_priors=modality_priors,
        epsilon=epsilon,
    )

    return estimator.estimate(
        modality_results
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)

    print(
        "NOONGIL-X | CONFIDENCE ESTIMATOR "
        "SELF-TEST"
    )

    print("=" * 72)

    try:
        results = {
            "vision": ModuleResult.success(
                module_name=(
                    "scene_classifier"
                ),
                modality="vision",
                data={},
                confidence=0.90,
                processing_time_ms=10.0,
                source_packet_id=(
                    "MSP_TEST_001"
                ),
                warnings=[],
            ),
            "audio": ModuleResult.success(
                module_name=(
                    "sound_event_detector"
                ),
                modality="audio",
                data={},
                confidence=0.70,
                processing_time_ms=10.0,
                source_packet_id=(
                    "MSP_TEST_001"
                ),
                warnings=[],
            ),
            "spatial": ModuleResult.success(
                module_name=(
                    "depth_estimator"
                ),
                modality="spatial",
                data={},
                confidence=0.80,
                processing_time_ms=10.0,
                source_packet_id=(
                    "MSP_TEST_001"
                ),
                warnings=[],
            ),
            "text": ModuleResult.partial(
                module_name=(
                    "text_interpreter"
                ),
                modality="text",
                data={},
                confidence=0.60,
                processing_time_ms=10.0,
                source_packet_id=(
                    "MSP_TEST_001"
                ),
                warnings=[
                    "Low-contrast text."
                ],
            ),
            "motion": ModuleResult.failure(
                module_name=(
                    "activity_recognizer"
                ),
                modality="motion",
                error=(
                    "Motion sequence unavailable."
                ),
                processing_time_ms=10.0,
                source_packet_id=(
                    "MSP_TEST_001"
                ),
            ),
        }

        estimator = ConfidenceEstimator()

        output = estimator.estimate(
            results
        )

        if not output.succeeded:
            raise AssertionError(
                "Confidence estimation did "
                "not succeed."
            )

        if "motion" not in (
            output.excluded_modalities
        ):
            raise AssertionError(
                "Failed motion modality was "
                "not excluded."
            )

        if not math.isclose(
            output.weight_sum,
            1.0,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                "Normalized weights do not "
                "sum to one."
            )

        if not (
            0.0
            <= output.fused_confidence
            <= 1.0
        ):
            raise AssertionError(
                "Fused confidence is invalid."
            )

        for modality, item in (
            output.modalities.items()
        ):

            if not (
                0.0
                <= item.normalized_weight
                <= 1.0
            ):
                raise AssertionError(
                    f"Invalid weight for "
                    f"{modality}."
                )

        print(
            f"[PASS] Usable modalities: "
            f"{output.usable_modalities}"
        )

        print(
            f"[PASS] Excluded modalities: "
            f"{output.excluded_modalities}"
        )

        print(
            f"[PASS] Weight sum: "
            f"{output.weight_sum:.6f}"
        )

        print(
            f"[PASS] Fused confidence: "
            f"{output.fused_confidence:.6f}"
        )

        for modality, item in (
            output.modalities.items()
        ):

            print(
                f"[PASS] {modality}: "
                f"effective_confidence="
                f"{item.effective_confidence:.6f}, "
                f"weight="
                f"{item.normalized_weight:.6f}"
            )

        print(
            "[PASS] Invalid modalities excluded"
        )

        print(
            "[PASS] Status reliability applied"
        )

        print(
            "[PASS] Modality priors applied"
        )

        print(
            "[PASS] Epsilon safeguard applied"
        )

        print(
            "[PASS] Normalized fusion weights generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] CONFIDENCE ESTIMATOR "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        ConfidenceEstimationError,
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
            "confidence-estimator self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return (
        0
        if run_self_test()
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())