"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Decision Engine
============================================================

Purpose
-------
Generate the final Layer 4 decision using:

1. situation_understanding.json
2. cognitive_state.json
3. intent_reasoning.json
4. hazards.json
5. predictions.json
6. reasoning_fusion.json
7. context_graph.json

Output
------
decision_output.json
============================================================
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)
DECISION_DIR = os.path.dirname(CURRENT_FILE)
LAYER4_DIR = os.path.dirname(DECISION_DIR)
PROJECT_ROOT = os.path.dirname(LAYER4_DIR)

LAYER3_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output",
    "layer3",
)

LAYER4_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output",
    "layer4",
)

os.makedirs(
    LAYER4_OUTPUT_DIR,
    exist_ok=True,
)

if LAYER4_DIR not in sys.path:
    sys.path.insert(0, LAYER4_DIR)

from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import (
    clamp_confidence,
    confidence_label,
)


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

SITUATION_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "situation_understanding.json",
)

COGNITIVE_STATE_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "cognitive_state.json",
)

INTENT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "intent_reasoning.json",
)

HAZARDS_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "hazards.json",
)

PREDICTIONS_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "predictions.json",
)

REASONING_FUSION_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "reasoning_fusion.json",
)

CONTEXT_GRAPH_PATH = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json",
)

DECISION_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "decision_output.json",
)


# ============================================================
# CONSTANTS
# ============================================================

RISK_PRIORITY = {
    "safe": 0,
    "none": 0,
    "very_low": 1,
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

URGENCY_PRIORITY = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

COGNITIVE_PRIORITY = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

PRIORITY_LABELS = {
    0: "routine",
    1: "low",
    2: "medium",
    3: "high",
    4: "critical",
}

NAVIGATION_INTENTS = {
    "navigate_to_destination",
    "find_person_or_object",
}

MOVEMENT_INTENTS = {
    "continue_movement",
}

SAFETY_INTENTS = {
    "avoid_hazard",
    "stay_safe",
}

EMERGENCY_INTENTS = {
    "seek_emergency_help",
}

CRITICAL_HAZARD_TERMS = {
    "fire",
    "smoke",
    "medical_emergency",
    "collision",
    "accident",
    "fall",
    "help_call",
    "distress",
    "emergency",
}

EXTERNAL_SERVICE_ACTIONS = {
    "start_step_by_step_navigation",
    "activate_emergency_assistance",
    "contact_emergency_service",
    "request_external_help",
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("-", "_")
    return " ".join(text.split())


def normalize_token(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


def unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    result = []
    seen = set()

    for value in values:
        marker = str(value)

        if marker in seen:
            continue

        seen.add(marker)
        result.append(value)

    return result


def get_nested(
    data: Dict[str, Any],
    path: Sequence[str],
    default: Any = None,
) -> Any:
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def get_first_available(
    data: Dict[str, Any],
    nested_paths: Sequence[Sequence[str]],
    flat_keys: Sequence[str] = (),
    default: Any = None,
) -> Any:
    for path in nested_paths:
        value = get_nested(data, path, None)

        if value not in (None, "", [], {}):
            return value

    for key in flat_keys:
        value = data.get(key)

        if value not in (None, "", [], {}):
            return value

    return default


def contains_any(text: str, terms: Iterable[str]) -> bool:
    normalized = normalize_text(text).replace("_", " ")

    for term in terms:
        normalized_term = normalize_text(term).replace("_", " ")

        if normalized_term and normalized_term in normalized:
            return True

    return False


# ============================================================
# CONTEXT ACCESSORS
# ============================================================

def get_environment_type(
    situation: Dict[str, Any],
) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("environment_context", "environment_type"),
            ],
            flat_keys=("environment_type",),
            default="unknown",
        )
    )


def get_scene_type(
    situation: Dict[str, Any],
) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("environment_context", "scene_type"),
            ],
            flat_keys=("scene_type",),
            default="unknown",
        )
    )


def get_user_activities(
    situation: Dict[str, Any],
) -> List[str]:
    value = get_first_available(
        situation,
        nested_paths=[
            ("user_state", "activities"),
        ],
        flat_keys=("user_activities", "activities"),
        default=[],
    )

    return [
        normalize_token(item)
        for item in as_list(value)
        if normalize_token(item)
    ]


def get_user_location(
    situation: Dict[str, Any],
) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("user_state", "location"),
            ],
            flat_keys=("user_location", "location"),
            default="unknown",
        )
    )


def get_safety_status(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    hazards: Dict[str, Any],
) -> str:
    value = hazards.get("safety_status")

    if value in (None, ""):
        value = get_first_available(
            situation,
            nested_paths=[
                ("safety_context", "safety_status"),
            ],
            flat_keys=("safety_status",),
            default=None,
        )

    if value in (None, ""):
        value = cognitive_state.get(
            "safety_status",
            "unknown",
        )

    return normalize_token(value)


def get_primary_intent(
    intent_result: Dict[str, Any],
) -> str:
    return normalize_token(
        intent_result.get("primary_intent", "")
    )


def get_destination(
    intent_result: Dict[str, Any],
) -> Optional[str]:
    value = (
        intent_result.get("destination")
        or intent_result.get("target")
    )

    token = normalize_token(value)
    return token or None


def get_urgency(
    intent_result: Dict[str, Any],
) -> str:
    return normalize_token(
        intent_result.get("urgency", "low")
    )


def get_attention_focus(
    cognitive_state: Dict[str, Any],
) -> str:
    return normalize_token(
        cognitive_state.get(
            "attention_focus",
            "general_awareness",
        )
    )


def get_cognitive_priority(
    cognitive_state: Dict[str, Any],
) -> str:
    return normalize_token(
        cognitive_state.get(
            "cognitive_priority",
            "low",
        )
    )


def get_reasoning_mode(
    cognitive_state: Dict[str, Any],
) -> str:
    return normalize_token(
        cognitive_state.get(
            "reasoning_mode",
            "general",
        )
    )


def get_fusion_readiness(
    fusion_result: Dict[str, Any],
) -> str:
    return normalize_token(
        fusion_result.get(
            "decision_readiness",
            "decision_with_monitoring",
        )
    )


# ============================================================
# HAZARD AND PREDICTION EXTRACTION
# ============================================================

def extract_hazards(
    hazards: Dict[str, Any],
) -> List[Dict[str, Any]]:
    result = []

    for hazard in hazards.get("hazards", []):
        if not isinstance(hazard, dict):
            continue

        recommended_actions = (
            hazard.get("recommended_actions")
            or hazard.get("recommended_action")
            or hazard.get("actions")
            or hazard.get("action")
            or []
        )

        result.append(
            {
                "hazard_type": normalize_token(
                    hazard.get("hazard_type")
                    or hazard.get("type")
                    or hazard.get("name")
                    or "unknown_hazard"
                ),
                "risk_level": normalize_token(
                    hazard.get("risk_level")
                    or hazard.get("severity")
                    or hazard.get("priority")
                    or "medium"
                ),
                "risk_score": clamp_confidence(
                    safe_float(
                        hazard.get(
                            "risk_score",
                            hazard.get(
                                "confidence",
                                0.50,
                            ),
                        ),
                        0.50,
                    )
                ),
                "recommended_actions": [
                    str(item)
                    for item in as_list(
                        recommended_actions
                    )
                    if str(item).strip()
                ],
                "reason": hazard.get("reason", ""),
                "evidence": hazard.get("evidence", []),
                "source": hazard.get(
                    "source",
                    "hazard_reasoner",
                ),
            }
        )

    return result


def get_highest_priority_hazard(
    hazard_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not hazard_list:
        return None

    return max(
        hazard_list,
        key=lambda hazard: (
            RISK_PRIORITY.get(
                normalize_token(
                    hazard.get("risk_level")
                ),
                0,
            ),
            safe_float(
                hazard.get("risk_score"),
                0.0,
            ),
        ),
    )


def get_most_likely_prediction(
    predictions: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    item = predictions.get(
        "most_likely_prediction"
    )

    if isinstance(item, dict):
        return item

    prediction_list = [
        prediction
        for prediction in predictions.get(
            "predictions",
            [],
        )
        if isinstance(prediction, dict)
    ]

    if not prediction_list:
        return None

    return max(
        prediction_list,
        key=lambda prediction: safe_float(
            prediction.get("probability"),
            0.0,
        ),
    )


def extract_prediction_actions(
    predictions: Dict[str, Any],
) -> List[str]:
    actions = []

    top_level_actions = predictions.get(
        "recommended_preparations",
        [],
    )

    actions.extend(
        str(action)
        for action in as_list(top_level_actions)
        if str(action).strip()
    )

    for prediction in predictions.get(
        "predictions",
        [],
    ):
        if not isinstance(prediction, dict):
            continue

        probability = safe_float(
            prediction.get("probability"),
            0.0,
        )

        time_horizon = normalize_token(
            prediction.get("time_horizon")
        )

        if (
            probability < 0.65
            and time_horizon != "immediate"
        ):
            continue

        actions.extend(
            str(action)
            for action in as_list(
                prediction.get(
                    "recommended_preparation",
                    [],
                )
            )
            if str(action).strip()
        )

    return unique_preserve_order(actions)


def extract_required_assistance(
    intent_result: Dict[str, Any],
) -> List[str]:
    return unique_preserve_order(
        [
            str(item)
            for item in as_list(
                intent_result.get(
                    "required_assistance",
                    [],
                )
            )
            if str(item).strip()
        ]
    )


# ============================================================
# EMERGENCY AND MODE SELECTION
# ============================================================

def contains_critical_hazard(
    hazards: Dict[str, Any],
) -> bool:
    risk_level = normalize_token(
        hazards.get(
            "overall_risk_level",
            "low",
        )
    )

    if risk_level == "critical":
        return True

    for hazard in extract_hazards(hazards):
        hazard_type = hazard.get(
            "hazard_type",
            "",
        )

        if (
            hazard.get("risk_level") == "critical"
            or (
                hazard.get("risk_score", 0.0) >= 0.85
                and contains_any(
                    hazard_type,
                    CRITICAL_HAZARD_TERMS,
                )
            )
        ):
            return True

    return False


def contains_emergency_signal(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> bool:
    primary_intent = get_primary_intent(
        intent_result
    )
    attention_focus = get_attention_focus(
        cognitive_state
    )
    reasoning_mode = get_reasoning_mode(
        cognitive_state
    )
    future_state = normalize_token(
        fusion_result.get("future_state")
    )
    safety_status = get_safety_status(
        situation,
        cognitive_state,
        hazards,
    )

    if contains_critical_hazard(hazards):
        return True

    if primary_intent in EMERGENCY_INTENTS:
        return True

    if attention_focus == "emergency":
        return True

    if reasoning_mode == "emergency_response":
        return True

    if future_state == "critical_future_risk":
        return True

    if (
        safety_status == "unsafe"
        and normalize_token(
            hazards.get("overall_risk_level")
        ) in {"high", "critical"}
    ):
        return True

    return False


def determine_decision_mode(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> str:
    if contains_emergency_signal(
        situation,
        cognitive_state,
        intent_result,
        hazards,
        fusion_result,
    ):
        return "emergency_response"

    risk_level = normalize_token(
        hazards.get(
            "overall_risk_level",
            "low",
        )
    )

    primary_intent = get_primary_intent(
        intent_result
    )

    reasoning_focus = normalize_token(
        fusion_result.get(
            "reasoning_focus",
            "",
        )
    )

    future_state = normalize_token(
        fusion_result.get(
            "future_state",
            "",
        )
    )

    if (
        risk_level == "high"
        or reasoning_focus in {
            "safety_first_reasoning",
            "safety_assistance_reasoning",
        }
        or future_state == "high_attention_required"
    ):
        return "safety_intervention"

    if primary_intent in NAVIGATION_INTENTS:
        if get_destination(intent_result):
            return "navigation_assistance"

        return "navigation_clarification"

    if primary_intent in MOVEMENT_INTENTS:
        return "movement_assistance"

    if primary_intent in SAFETY_INTENTS:
        return "preventive_safety"

    return "general_assistance"


# ============================================================
# ACTION BUILDERS
# ============================================================

def build_emergency_action(
    highest_hazard: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    target = (
        highest_hazard.get("hazard_type")
        if highest_hazard
        else "detected_emergency"
    )

    return {
        "action_id": "ACT_EMERGENCY_RESPONSE",
        "action_type": "emergency_intervention",
        "action": "activate_emergency_assistance",
        "target": target,
        "instruction": (
            "Immediately warn the user, stop non-essential assistance, "
            "guide the user toward safety, and prepare emergency support."
        ),
        "expected_outcome": (
            "The user receives immediate safety guidance and emergency "
            "support is initiated when required."
        ),
        "requires_confirmation": False,
        "requires_external_service": True,
    }


def build_safety_action(
    highest_hazard: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    target = (
        highest_hazard.get("hazard_type")
        if highest_hazard
        else "environmental_hazard"
    )

    return {
        "action_id": "ACT_SAFETY_INTERVENTION",
        "action_type": "safety_assistance",
        "action": "provide_immediate_hazard_warning",
        "target": target,
        "instruction": (
            "Warn the user about the hazard, reduce movement risk, "
            "and identify a safer continuation path."
        ),
        "expected_outcome": (
            "The user becomes aware of the hazard and avoids unsafe movement."
        ),
        "requires_confirmation": False,
        "requires_external_service": False,
    }


def build_navigation_action(
    intent_result: Dict[str, Any],
) -> Dict[str, Any]:
    destination = (
        get_destination(intent_result)
        or "requested_destination"
    )

    return {
        "action_id": "ACT_NAVIGATION_ASSISTANCE",
        "action_type": "navigation_assistance",
        "action": "start_step_by_step_navigation",
        "target": destination,
        "instruction": (
            f"Guide the user toward {destination} using safe, "
            "step-by-step instructions with continuous obstacle monitoring."
        ),
        "expected_outcome": (
            f"The user progresses safely toward {destination}."
        ),
        "requires_confirmation": False,
        "requires_external_service": True,
    }


def build_navigation_clarification_action() -> Dict[str, Any]:
    return {
        "action_id": "ACT_NAVIGATION_CLARIFICATION",
        "action_type": "information_request",
        "action": "request_navigation_destination",
        "target": "user",
        "instruction": (
            "Ask the user to specify the intended destination before "
            "starting navigation."
        ),
        "expected_outcome": (
            "A valid destination is obtained for safe route planning."
        ),
        "requires_confirmation": True,
        "requires_external_service": False,
    }


def build_movement_action() -> Dict[str, Any]:
    return {
        "action_id": "ACT_MOVEMENT_SUPPORT",
        "action_type": "movement_assistance",
        "action": "continue_movement_monitoring",
        "target": "user_movement",
        "instruction": (
            "Continue monitoring movement direction, speed, obstacles, "
            "and environmental changes."
        ),
        "expected_outcome": (
            "The user continues moving with adaptive safety assistance."
        ),
        "requires_confirmation": False,
        "requires_external_service": False,
    }


def build_preventive_safety_action() -> Dict[str, Any]:
    return {
        "action_id": "ACT_PREVENTIVE_SAFETY",
        "action_type": "preventive_assistance",
        "action": "increase_environmental_monitoring",
        "target": "user_safety",
        "instruction": (
            "Increase environmental monitoring and provide warnings "
            "only when meaningful risk is detected."
        ),
        "expected_outcome": (
            "User safety is maintained without unnecessary interruption."
        ),
        "requires_confirmation": False,
        "requires_external_service": False,
    }


def build_general_action() -> Dict[str, Any]:
    return {
        "action_id": "ACT_GENERAL_AWARENESS",
        "action_type": "general_assistance",
        "action": "maintain_contextual_awareness",
        "target": "current_environment",
        "instruction": (
            "Continue observing the environment and remain ready to "
            "assist when the user's context changes."
        ),
        "expected_outcome": (
            "The system maintains awareness without unnecessary alerts."
        ),
        "requires_confirmation": False,
        "requires_external_service": False,
    }


def select_primary_action(
    decision_mode: str,
    intent_result: Dict[str, Any],
    highest_hazard: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    builders = {
        "emergency_response": lambda: build_emergency_action(
            highest_hazard
        ),
        "safety_intervention": lambda: build_safety_action(
            highest_hazard
        ),
        "navigation_assistance": lambda: build_navigation_action(
            intent_result
        ),
        "navigation_clarification": (
            build_navigation_clarification_action
        ),
        "movement_assistance": build_movement_action,
        "preventive_safety": build_preventive_safety_action,
        "general_assistance": build_general_action,
    }

    return builders.get(
        decision_mode,
        build_general_action,
    )()


# ============================================================
# SUPPORTING ACTIONS
# ============================================================

def generate_supporting_actions(
    decision_mode: str,
    intent_result: Dict[str, Any],
    predictions: Dict[str, Any],
    fusion_result: Dict[str, Any],
    highest_hazard: Optional[Dict[str, Any]],
) -> List[str]:
    actions = []

    actions.extend(
        extract_required_assistance(
            intent_result
        )
    )

    actions.extend(
        extract_prediction_actions(
            predictions
        )
    )

    actions.extend(
        str(action)
        for action in as_list(
            fusion_result.get(
                "recommended_reasoning_actions",
                [],
            )
        )
        if str(action).strip()
    )

    if highest_hazard:
        actions.extend(
            highest_hazard.get(
                "recommended_actions",
                [],
            )
        )

    mode_actions = {
        "emergency_response": [
            "issue_urgent_voice_alert",
            "pause_non_essential_tasks",
            "maintain_continuous_hazard_tracking",
            "prepare_emergency_contact_or_service",
        ],
        "safety_intervention": [
            "issue_clear_hazard_warning",
            "reduce_user_movement_risk",
            "search_for_safer_path",
            "continue_hazard_monitoring",
        ],
        "navigation_assistance": [
            "track_destination",
            "monitor_obstacles",
            "adapt_guidance_to_walking_speed",
            "recalculate_route_if_required",
        ],
        "navigation_clarification": [
            "request_destination",
            "avoid_route_generation_until_destination_is_known",
        ],
        "movement_assistance": [
            "monitor_walking_direction",
            "detect_path_changes",
            "warn_about_nearby_obstacles",
        ],
        "preventive_safety": [
            "maintain_environmental_monitoring",
            "avoid_unnecessary_alerts",
        ],
        "general_assistance": [
            "maintain_context_monitoring",
        ],
    }

    actions.extend(
        mode_actions.get(
            decision_mode,
            ["maintain_context_monitoring"],
        )
    )

    return unique_preserve_order(actions)


# ============================================================
# PRIORITY AND CONFIDENCE
# ============================================================

def determine_decision_priority(
    decision_mode: str,
    hazards: Dict[str, Any],
    intent_result: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> str:
    if decision_mode == "emergency_response":
        return "critical"

    scores = [
        RISK_PRIORITY.get(
            normalize_token(
                hazards.get(
                    "overall_risk_level",
                    "low",
                )
            ),
            1,
        ),
        URGENCY_PRIORITY.get(
            get_urgency(intent_result),
            1,
        ),
        COGNITIVE_PRIORITY.get(
            get_cognitive_priority(
                cognitive_state
            ),
            1,
        ),
    ]

    future_state = normalize_token(
        fusion_result.get("future_state")
    )

    if future_state == "high_attention_required":
        scores.append(3)

    elif future_state == "critical_future_risk":
        scores.append(4)

    return PRIORITY_LABELS.get(
        max(scores),
        "routine",
    )


def calculate_decision_confidence(
    intent_result: Dict[str, Any],
    predictions: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> float:
    intent_confidence = clamp_confidence(
        safe_float(
            intent_result.get(
                "intent_confidence",
                0.0,
            ),
            0.0,
        )
    )

    prediction_confidence = clamp_confidence(
        safe_float(
            predictions.get(
                "overall_prediction_confidence",
                0.0,
            ),
            0.0,
        )
    )

    reasoning_confidence = clamp_confidence(
        safe_float(
            fusion_result.get(
                "reasoning_confidence",
                0.0,
            ),
            0.0,
        )
    )

    readiness = get_fusion_readiness(
        fusion_result
    )

    readiness_score = {
        "ready_for_decision": 1.0,
        "decision_with_monitoring": 0.70,
        "requires_conflict_resolution": 0.35,
        "insufficient_reasoning_confidence": 0.25,
    }.get(readiness, 0.50)

    confidence = (
        reasoning_confidence * 0.45
        + intent_confidence * 0.25
        + prediction_confidence * 0.20
        + readiness_score * 0.10
    )

    return round(
        clamp_confidence(confidence),
        2,
    )


# ============================================================
# EXECUTION AND SAFETY
# ============================================================

def determine_execution_requirements(
    primary_action: Dict[str, Any],
    decision_mode: str,
    fusion_result: Dict[str, Any],
) -> Dict[str, Any]:
    if decision_mode == "emergency_response":
        execution_type = "immediate"
        timeout_seconds = 5

    elif decision_mode == "safety_intervention":
        execution_type = "immediate"
        timeout_seconds = 10

    elif decision_mode == "navigation_assistance":
        execution_type = "continuous"
        timeout_seconds = 30

    elif decision_mode == "navigation_clarification":
        execution_type = "interactive"
        timeout_seconds = 60

    else:
        execution_type = "adaptive"
        timeout_seconds = 60

    action_name = normalize_token(
        primary_action.get("action")
    )

    requires_external_service = bool(
        primary_action.get(
            "requires_external_service",
            False,
        )
        or action_name in EXTERNAL_SERVICE_ACTIONS
    )

    return {
        "execution_type": execution_type,
        "interruption_allowed": True,
        "requires_external_service": (
            requires_external_service
        ),
        "requires_user_confirmation": bool(
            primary_action.get(
                "requires_confirmation",
                False,
            )
        ),
        "timeout_seconds": timeout_seconds,
        "failure_policy": "fallback_to_safe_action",
        "monitoring_required": True,
        "decision_readiness": get_fusion_readiness(
            fusion_result
        ),
    }


def generate_safety_constraints(
    decision_mode: str,
    hazards: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> List[str]:
    constraints = [
        "do_not_provide_unverified_direction",
        "do_not_ignore_new_hazard_signals",
        "preserve_user_safety_over_task_completion",
        "adapt_action_when_environment_changes",
    ]

    risk_level = normalize_token(
        hazards.get(
            "overall_risk_level",
            "low",
        )
    )

    if risk_level in {"high", "critical"}:
        constraints.extend(
            [
                "stop_or_slow_user_movement_when_required",
                "prioritize_immediate_warning",
                "avoid_route_containing_detected_hazard",
            ]
        )

    if decision_mode == "emergency_response":
        constraints.extend(
            [
                "pause_non_emergency_actions",
                "maintain_continuous_user_feedback",
                "escalate_when_local_assistance_is_insufficient",
            ]
        )

    if get_fusion_readiness(
        fusion_result
    ) == "requires_conflict_resolution":
        constraints.append(
            "do_not_execute_irreversible_action_until_conflict_is_resolved"
        )

    return unique_preserve_order(
        constraints
    )


# ============================================================
# REASONING TRACE
# ============================================================

def build_reasoning_trace(
    decision_mode: str,
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
    fusion_result: Dict[str, Any],
    highest_hazard: Optional[Dict[str, Any]],
) -> List[str]:
    most_likely = get_most_likely_prediction(
        predictions
    )

    trace = [
        (
            "Environment identified as "
            f"{get_environment_type(situation)}."
        ),
        (
            "User location identified as "
            f"{get_user_location(situation)}."
        ),
        (
            "User intent identified as "
            f"{get_primary_intent(intent_result)}."
        ),
        (
            "Cognitive attention is focused on "
            f"{get_attention_focus(cognitive_state)}."
        ),
        (
            "Overall hazard level is "
            f"{hazards.get('overall_risk_level', 'low')}."
        ),
        (
            "Reasoning fusion state is "
            f"{fusion_result.get('future_state', 'unknown')}."
        ),
        (
            "Decision readiness is "
            f"{get_fusion_readiness(fusion_result)}."
        ),
    ]

    if highest_hazard:
        trace.append(
            "Highest-priority hazard is "
            f"{highest_hazard.get('hazard_type')}."
        )

    if most_likely:
        trace.append(
            "Most likely predicted outcome is "
            f"{most_likely.get('predicted_outcome')}."
        )

    conflicts = fusion_result.get(
        "reasoning_conflicts",
        [],
    )

    if conflicts:
        trace.append(
            f"{len(conflicts)} reasoning conflict(s) require monitoring."
        )

    trace.append(
        f"Decision mode selected as {decision_mode}."
    )

    return trace


# ============================================================
# DECISION STATUS
# ============================================================

def determine_decision_status(
    fusion_result: Dict[str, Any],
    decision_confidence: float,
) -> str:
    readiness = get_fusion_readiness(
        fusion_result
    )

    if readiness == "requires_conflict_resolution":
        return "requires_review"

    if readiness == "insufficient_reasoning_confidence":
        return "requires_more_context"

    if decision_confidence < 0.45:
        return "requires_more_context"

    if readiness == "decision_with_monitoring":
        return "ready_with_monitoring"

    return "ready_for_execution"


def determine_next_layer(
    decision_status: str,
) -> str:
    if decision_status in {
        "requires_review",
        "requires_more_context",
    }:
        return "layer4_context_reassessment"

    return "layer5_language_and_communication"


# ============================================================
# SUMMARY
# ============================================================

def generate_decision_summary(
    decision_mode: str,
    priority: str,
    primary_action: Dict[str, Any],
    confidence: float,
    decision_status: str,
) -> str:
    return (
        f"The Decision Engine selected {decision_mode} mode with "
        f"{priority} priority. The primary action is "
        f"{primary_action.get('action', 'unknown_action')} for "
        f"{primary_action.get('target', 'current_context')}. "
        f"Decision confidence is {confidence}, and the decision "
        f"status is {decision_status}."
    )


# ============================================================
# FINAL DECISION ENGINE
# ============================================================

def make_decision(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
    fusion_result: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Dict[str, Any]:
    log_section("Final Decision Generation")

    hazard_list = extract_hazards(
        hazards
    )

    highest_hazard = get_highest_priority_hazard(
        hazard_list
    )

    decision_mode = determine_decision_mode(
        situation,
        cognitive_state,
        intent_result,
        hazards,
        fusion_result,
    )

    primary_action = select_primary_action(
        decision_mode,
        intent_result,
        highest_hazard,
    )

    supporting_actions = generate_supporting_actions(
        decision_mode,
        intent_result,
        predictions,
        fusion_result,
        highest_hazard,
    )

    decision_priority = determine_decision_priority(
        decision_mode,
        hazards,
        intent_result,
        cognitive_state,
        fusion_result,
    )

    decision_confidence = calculate_decision_confidence(
        intent_result,
        predictions,
        fusion_result,
    )

    decision_status = determine_decision_status(
        fusion_result,
        decision_confidence,
    )

    execution_requirements = (
        determine_execution_requirements(
            primary_action,
            decision_mode,
            fusion_result,
        )
    )

    safety_constraints = generate_safety_constraints(
        decision_mode,
        hazards,
        fusion_result,
    )

    reasoning_trace = build_reasoning_trace(
        decision_mode,
        situation,
        cognitive_state,
        intent_result,
        hazards,
        predictions,
        fusion_result,
        highest_hazard,
    )

    result = {
        "timestamp": str(datetime.now()),
        "decision_id": (
            "DEC_"
            + datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )
        ),
        "decision_mode": decision_mode,
        "decision_priority": decision_priority,
        "decision_confidence": decision_confidence,
        "confidence_label": confidence_label(
            decision_confidence
        ),
        "decision_status": decision_status,

        "primary_action": primary_action,
        "supporting_action_count": len(
            supporting_actions
        ),
        "supporting_actions": supporting_actions,

        "highest_priority_hazard": highest_hazard,

        "current_context": {
            "environment_type": get_environment_type(
                situation
            ),
            "scene_type": get_scene_type(
                situation
            ),
            "user_location": get_user_location(
                situation
            ),
            "user_activities": get_user_activities(
                situation
            ),
            "situation_type": situation.get(
                "situation_type",
                "unknown",
            ),
            "primary_intent": get_primary_intent(
                intent_result
            ),
            "destination": get_destination(
                intent_result
            ),
            "risk_level": hazards.get(
                "overall_risk_level",
                "low",
            ),
            "future_state": fusion_result.get(
                "future_state",
                "unknown",
            ),
            "reasoning_focus": fusion_result.get(
                "reasoning_focus",
                "unknown",
            ),
            "attention_focus": get_attention_focus(
                cognitive_state
            ),
            "graph_node_count": context_graph.get(
                "node_count",
                len(context_graph.get("nodes", [])),
            ),
            "graph_edge_count": context_graph.get(
                "edge_count",
                len(context_graph.get("edges", [])),
            ),
        },

        "fusion_alignment": {
            "decision_readiness": get_fusion_readiness(
                fusion_result
            ),
            "reasoning_confidence": fusion_result.get(
                "reasoning_confidence",
                0.0,
            ),
            "future_risk_score": fusion_result.get(
                "future_risk_score",
                0.0,
            ),
            "conflict_count": fusion_result.get(
                "conflict_count",
                0,
            ),
        },

        "execution_requirements": (
            execution_requirements
        ),
        "safety_constraints": safety_constraints,
        "reasoning_trace": reasoning_trace,

        "next_layer": determine_next_layer(
            decision_status
        ),

        "summary": generate_decision_summary(
            decision_mode,
            decision_priority,
            primary_action,
            decision_confidence,
            decision_status,
        ),
    }

    log_info(
        f"Decision Mode: {decision_mode}"
    )
    log_info(
        f"Decision Priority: {decision_priority}"
    )
    log_info(
        "Primary Action: "
        f"{primary_action.get('action')}"
    )
    log_info(
        f"Decision Confidence: {decision_confidence}"
    )
    log_info(
        f"Decision Status: {decision_status}"
    )

    log_success(
        "Final Decision Generated"
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_required_inputs(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    predictions: Dict[str, Any],
    fusion_result: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    missing = []

    required = {
        "situation_understanding.json": situation,
        "cognitive_state.json": cognitive_state,
        "intent_reasoning.json": intent_result,
        "predictions.json": predictions,
        "reasoning_fusion.json": fusion_result,
        "context_graph.json": context_graph,
    }

    for filename, data in required.items():
        if not data:
            missing.append(filename)

    return len(missing) == 0, missing


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    module_start("DECISION ENGINE")

    situation = load_json(
        SITUATION_PATH
    )

    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
    )

    intent_result = load_json(
        INTENT_PATH
    )

    predictions = load_json(
        PREDICTIONS_PATH
    )

    fusion_result = load_json(
        REASONING_FUSION_PATH
    )

    context_graph = load_json(
        CONTEXT_GRAPH_PATH
    )

    hazards = load_json(
        HAZARDS_PATH,
        default={
            "hazard_count": 0,
            "hazards": [],
            "overall_risk_score": 0.0,
            "overall_risk_level": "low",
            "safety_status": "safe",
            "recommended_actions": [],
        },
    )

    inputs_valid, missing_inputs = (
        validate_required_inputs(
            situation,
            cognitive_state,
            intent_result,
            predictions,
            fusion_result,
            context_graph,
        )
    )

    if not inputs_valid:
        log_error(
            "Required Decision Engine inputs are missing: "
            + ", ".join(missing_inputs)
        )

        module_end("DECISION ENGINE")
        return

    if not hazards:
        log_warning(
            "hazards.json missing or empty. "
            "Using safe hazard defaults."
        )

        hazards = {
            "hazard_count": 0,
            "hazards": [],
            "overall_risk_score": 0.0,
            "overall_risk_level": "low",
            "safety_status": "safe",
            "recommended_actions": [],
        }

    result = make_decision(
        situation,
        cognitive_state,
        intent_result,
        hazards,
        predictions,
        fusion_result,
        context_graph,
    )

    save_json(
        result,
        DECISION_OUTPUT_PATH,
    )

    log_info(
        "Decision output saved to: "
        f"{DECISION_OUTPUT_PATH}"
    )

    module_end("DECISION ENGINE")


if __name__ == "__main__":
    main()