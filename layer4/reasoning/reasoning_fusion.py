"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Reasoning Fusion Engine
============================================================

Purpose
-------
Fuse situation understanding, cognitive state, intent,
hazards, predictions, and context-graph evidence into one
consistent reasoning state for the decision engine.

Inputs
------
1. situation_understanding.json
2. cognitive_state.json
3. intent_reasoning.json
4. hazards.json
5. predictions.json
6. context_graph.json

Output
------
reasoning_fusion.json
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
REASONING_DIR = os.path.dirname(CURRENT_FILE)
LAYER4_DIR = os.path.dirname(REASONING_DIR)
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

CONTEXT_GRAPH_PATH = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json",
)

REASONING_FUSION_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "reasoning_fusion.json",
)


# ============================================================
# CONSTANTS
# ============================================================

RISK_LEVEL_SCORES = {
    "safe": 0.00,
    "none": 0.00,
    "very_low": 0.10,
    "low": 0.20,
    "medium": 0.50,
    "moderate": 0.50,
    "high": 0.78,
    "critical": 1.00,
}

URGENCY_SCORES = {
    "none": 0.00,
    "low": 0.20,
    "medium": 0.50,
    "moderate": 0.50,
    "high": 0.78,
    "critical": 1.00,
}

PRIORITY_SCORES = {
    "none": 0.00,
    "low": 0.20,
    "medium": 0.50,
    "moderate": 0.50,
    "high": 0.78,
    "critical": 1.00,
}

IMMEDIATE_HORIZONS = {
    "immediate",
    "now",
    "near_immediate",
}

RISK_TERMS = {
    "risk",
    "hazard",
    "unsafe",
    "danger",
    "emergency",
    "fire",
    "smoke",
    "collision",
    "vehicle",
    "fall",
    "distress",
    "blocked",
    "obstacle",
    "traffic",
    "help",
}

EMERGENCY_TERMS = {
    "emergency",
    "fire",
    "smoke",
    "medical",
    "distress",
    "help_call",
    "collision",
    "critical",
}

NAVIGATION_INTENTS = {
    "navigate_to_destination",
    "continue_movement",
    "find_person_or_object",
}

SAFETY_INTENTS = {
    "avoid_hazard",
    "seek_emergency_help",
    "stay_safe",
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
# SIGNAL EXTRACTION
# ============================================================

def get_primary_intent(intent_result: Dict[str, Any]) -> str:
    return normalize_token(
        intent_result.get("primary_intent", "")
    )


def get_intent_confidence(intent_result: Dict[str, Any]) -> float:
    return clamp_confidence(
        safe_float(
            intent_result.get("intent_confidence"),
            0.50,
        )
    )


def get_urgency(intent_result: Dict[str, Any]) -> str:
    return normalize_token(
        intent_result.get("urgency", "low")
    )


def get_destination(intent_result: Dict[str, Any]) -> Optional[str]:
    value = (
        intent_result.get("destination")
        or intent_result.get("target")
    )

    token = normalize_token(value)
    return token or None


def get_cognitive_priority(
    cognitive_state: Dict[str, Any],
    situation: Dict[str, Any],
) -> str:
    value = get_first_available(
        situation,
        nested_paths=[
            ("cognitive_context", "cognitive_priority"),
        ],
        default=None,
    )

    if value in (None, ""):
        value = cognitive_state.get(
            "cognitive_priority",
            "low",
        )

    return normalize_token(value)


def get_reasoning_mode(
    cognitive_state: Dict[str, Any],
    situation: Dict[str, Any],
) -> str:
    value = get_first_available(
        situation,
        nested_paths=[
            ("cognitive_context", "reasoning_mode"),
        ],
        default=None,
    )

    if value in (None, ""):
        value = cognitive_state.get(
            "reasoning_mode",
            "general",
        )

    return normalize_token(value)


def get_environment_type(situation: Dict[str, Any]) -> str:
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


def get_prediction_list(
    predictions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        item
        for item in predictions.get("predictions", [])
        if isinstance(item, dict)
    ]


def get_hazard_list(
    hazards: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        item
        for item in hazards.get("hazards", [])
        if isinstance(item, dict)
    ]


def get_graph_event_types(
    context_graph: Dict[str, Any],
) -> List[str]:
    event_types = []

    for node in context_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue

        if normalize_token(node.get("category")) != "event":
            continue

        event_type = normalize_token(
            node.get("event_type")
            or node.get("type")
        )

        if event_type:
            event_types.append(event_type)

    return unique_preserve_order(event_types)


# ============================================================
# PREDICTION EXTRACTION
# ============================================================

def get_most_likely_prediction(
    predictions: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    most_likely = predictions.get(
        "most_likely_prediction"
    )

    if isinstance(most_likely, dict):
        return most_likely

    prediction_list = get_prediction_list(
        predictions
    )

    if not prediction_list:
        return None

    return max(
        prediction_list,
        key=lambda item: safe_float(
            item.get("probability"),
            0.0,
        ),
    )


def extract_high_priority_predictions(
    predictions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    high_priority = []

    for prediction in get_prediction_list(
        predictions
    ):
        probability = safe_float(
            prediction.get("probability"),
            0.0,
        )

        time_horizon = normalize_token(
            prediction.get("time_horizon")
        )

        priority = normalize_token(
            prediction.get("priority")
        )

        if (
            probability >= 0.75
            or time_horizon in IMMEDIATE_HORIZONS
            or priority in {"high", "critical"}
        ):
            high_priority.append(prediction)

    return sorted(
        high_priority,
        key=lambda item: safe_float(
            item.get("probability"),
            0.0,
        ),
        reverse=True,
    )


def extract_future_risk_predictions(
    predictions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    future_risks = []

    for prediction in get_prediction_list(
        predictions
    ):
        combined_text = " ".join(
            [
                str(
                    prediction.get(
                        "prediction_type",
                        "",
                    )
                ),
                str(
                    prediction.get(
                        "predicted_outcome",
                        "",
                    )
                ),
                str(
                    prediction.get(
                        "reason",
                        "",
                    )
                ),
            ]
        )

        if contains_any(combined_text, RISK_TERMS):
            future_risks.append(prediction)

    return sorted(
        future_risks,
        key=lambda item: safe_float(
            item.get("probability"),
            0.0,
        ),
        reverse=True,
    )


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def detect_reasoning_conflicts(
    predictions: Dict[str, Any],
    hazards: Dict[str, Any],
    intent_result: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    conflicts = []

    hazard_list = get_hazard_list(hazards)
    risk_level = normalize_token(
        hazards.get(
            "overall_risk_level",
            "low",
        )
    )
    safety_status = get_safety_status(
        situation,
        cognitive_state,
        hazards,
    )
    future_risks = extract_future_risk_predictions(
        predictions
    )
    primary_intent = get_primary_intent(
        intent_result
    )
    destination = get_destination(intent_result)

    if (
        safety_status == "safe"
        and risk_level in {"high", "critical"}
    ):
        conflicts.append(
            {
                "conflict_type": "safety_risk_mismatch",
                "severity": "high",
                "details": (
                    "Safety status is safe, but hazard risk level "
                    f"is {risk_level}."
                ),
            }
        )

    if (
        not hazard_list
        and risk_level in {"high", "critical"}
    ):
        conflicts.append(
            {
                "conflict_type": "unsupported_high_risk",
                "severity": "high",
                "details": (
                    "High risk level is reported without detected hazards."
                ),
            }
        )

    if (
        not hazard_list
        and future_risks
        and risk_level in {"safe", "none", "very_low", "low"}
    ):
        conflicts.append(
            {
                "conflict_type": "prediction_hazard_disagreement",
                "severity": "medium",
                "details": (
                    "Risk-related predictions exist, but the hazard "
                    "reasoner found no supporting hazards."
                ),
            }
        )

    if (
        primary_intent == "navigate_to_destination"
        and not destination
    ):
        conflicts.append(
            {
                "conflict_type": "missing_navigation_destination",
                "severity": "medium",
                "details": (
                    "Navigation intent is present without a destination."
                ),
            }
        )

    return conflicts


def conflict_penalty(
    conflicts: List[Dict[str, Any]],
) -> float:
    penalty = 0.0

    for conflict in conflicts:
        severity = normalize_token(
            conflict.get("severity")
        )

        if severity == "high":
            penalty += 0.12
        elif severity == "medium":
            penalty += 0.06
        else:
            penalty += 0.03

    return min(penalty, 0.30)


# ============================================================
# FUSION SCORING
# ============================================================

def calculate_signal_scores(
    predictions: Dict[str, Any],
    hazards: Dict[str, Any],
    intent_result: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    situation: Dict[str, Any],
) -> Dict[str, float]:
    prediction_confidence = clamp_confidence(
        safe_float(
            predictions.get(
                "overall_prediction_confidence"
            ),
            0.0,
        )
    )

    hazard_risk_score = clamp_confidence(
        safe_float(
            hazards.get("overall_risk_score"),
            RISK_LEVEL_SCORES.get(
                normalize_token(
                    hazards.get(
                        "overall_risk_level",
                        "low",
                    )
                ),
                0.20,
            ),
        )
    )

    intent_confidence = get_intent_confidence(
        intent_result
    )

    urgency_score = URGENCY_SCORES.get(
        get_urgency(intent_result),
        0.20,
    )

    cognitive_score = PRIORITY_SCORES.get(
        get_cognitive_priority(
            cognitive_state,
            situation,
        ),
        0.20,
    )

    return {
        "prediction_confidence": round(
            prediction_confidence,
            2,
        ),
        "hazard_risk_score": round(
            hazard_risk_score,
            2,
        ),
        "intent_confidence": round(
            intent_confidence,
            2,
        ),
        "urgency_score": round(
            urgency_score,
            2,
        ),
        "cognitive_priority_score": round(
            cognitive_score,
            2,
        ),
    }


def calculate_reasoning_confidence(
    signal_scores: Dict[str, float],
    conflicts: List[Dict[str, Any]],
) -> float:
    score = (
        signal_scores["prediction_confidence"] * 0.35
        + signal_scores["intent_confidence"] * 0.30
        + signal_scores["cognitive_priority_score"] * 0.20
        + (1.0 - signal_scores["hazard_risk_score"]) * 0.15
        
    )

    score -= conflict_penalty(conflicts)

    return round(
        clamp_confidence(score),
        2,
    )


def calculate_future_risk_score(
    predictions: Dict[str, Any],
    hazards: Dict[str, Any],
    intent_result: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    situation: Dict[str, Any],
) -> float:
    signal_scores = calculate_signal_scores(
        predictions,
        hazards,
        intent_result,
        cognitive_state,
        situation,
    )

    future_risk_predictions = (
        extract_future_risk_predictions(
            predictions
        )
    )

    if future_risk_predictions:
        future_risk_probability = max(
            safe_float(
                item.get("probability"),
                0.0,
            )
            for item in future_risk_predictions
        )
    else:
        future_risk_probability = 0.0

    future_risk_score = (
        signal_scores["hazard_risk_score"] * 0.55
        + future_risk_probability * 0.25
        + signal_scores["urgency_score"] * 0.12
        + signal_scores["cognitive_priority_score"] * 0.08
    )

    return round(
        clamp_confidence(future_risk_score),
        2,
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_future_state(
    future_risk_score: float,
    hazards: Dict[str, Any],
) -> str:
    risk_level = normalize_token(
        hazards.get(
            "overall_risk_level",
            "low",
        )
    )

    if (
        risk_level == "critical"
        or future_risk_score >= 0.85
    ):
        return "critical_future_risk"

    if (
        risk_level == "high"
        or future_risk_score >= 0.65
    ):
        return "high_attention_required"

    if (
        risk_level in {"medium", "moderate"}
        or future_risk_score >= 0.40
    ):
        return "moderate_monitoring_required"

    if future_risk_score >= 0.18:
        return "low_risk_continuation"

    return "safe_continuation"


def decide_reasoning_focus(
    most_likely_prediction: Optional[Dict[str, Any]],
    hazards: Dict[str, Any],
    intent_result: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
) -> str:
    risk_level = normalize_token(
        hazards.get(
            "overall_risk_level",
            "low",
        )
    )

    primary_intent = get_primary_intent(
        intent_result
    )

    environment_type = get_environment_type(
        situation
    )

    reasoning_mode = get_reasoning_mode(
        cognitive_state,
        situation,
    )

    if risk_level in {"critical", "high"}:
        return "safety_first_reasoning"

    if primary_intent in SAFETY_INTENTS:
        return "safety_assistance_reasoning"

    if primary_intent in NAVIGATION_INTENTS:
        return "navigation_progress_reasoning"

    if environment_type in {
        "traffic_area",
        "urban_environment",
        "transport_environment",
    }:
        return "traffic_awareness_reasoning"

    if most_likely_prediction:
        combined = " ".join(
            [
                str(
                    most_likely_prediction.get(
                        "prediction_type",
                        "",
                    )
                ),
                str(
                    most_likely_prediction.get(
                        "predicted_outcome",
                        "",
                    )
                ),
            ]
        )

        if contains_any(combined, EMERGENCY_TERMS):
            return "emergency_reasoning"

        if contains_any(
            combined,
            {"move", "walking", "navigation", "destination"},
        ):
            return "movement_reasoning"

    if reasoning_mode and reasoning_mode != "general":
        return f"{reasoning_mode}_reasoning"

    return "general_context_reasoning"


def determine_decision_readiness(
    reasoning_confidence: float,
    conflicts: List[Dict[str, Any]],
    most_likely_prediction: Optional[Dict[str, Any]],
) -> str:
    high_conflicts = [
        conflict
        for conflict in conflicts
        if normalize_token(
            conflict.get("severity")
        ) == "high"
    ]

    if high_conflicts:
        return "requires_conflict_resolution"

    if (
        reasoning_confidence >= 0.70
        and most_likely_prediction
    ):
        return "ready_for_decision"

    if reasoning_confidence >= 0.45:
        return "decision_with_monitoring"

    return "insufficient_reasoning_confidence"


# ============================================================
# ACTION FOCUS
# ============================================================

def generate_reasoning_actions(
    future_state: str,
    reasoning_focus: str,
    conflicts: List[Dict[str, Any]],
    predictions: Dict[str, Any],
    hazards: Dict[str, Any],
) -> List[str]:
    actions = []

    if future_state == "critical_future_risk":
        actions.extend(
            [
                "prioritize_immediate_safety",
                "activate_emergency_reasoning",
                "prepare_external_assistance",
            ]
        )

    elif future_state == "high_attention_required":
        actions.extend(
            [
                "increase_monitoring",
                "prepare_fast_safety_guidance",
                "reduce_reasoning_uncertainty",
            ]
        )

    elif future_state == "moderate_monitoring_required":
        actions.extend(
            [
                "continue_active_monitoring",
                "reassess_context_if_conditions_change",
            ]
        )

    else:
        actions.append(
            "continue_contextual_monitoring"
        )

    if reasoning_focus == "navigation_progress_reasoning":
        actions.extend(
            [
                "continue_destination_tracking",
                "maintain_route_guidance",
                "monitor_navigation_progress",
            ]
        )

    elif reasoning_focus in {
        "safety_first_reasoning",
        "safety_assistance_reasoning",
        "emergency_reasoning",
    }:
        actions.extend(
            [
                "prioritize_hazard_evidence",
                "prepare_safe_action",
            ]
        )

    if conflicts:
        actions.append(
            "resolve_reasoning_conflicts"
        )

    for action in predictions.get(
        "recommended_preparations",
        [],
    ):
        actions.append(str(action))

    for action in hazards.get(
        "recommended_actions",
        [],
    ):
        actions.append(str(action))

    return unique_preserve_order(actions)


# ============================================================
# SUMMARY
# ============================================================

def generate_fusion_summary(
    future_state: str,
    reasoning_focus: str,
    most_likely_prediction: Optional[Dict[str, Any]],
    future_risk_score: float,
    reasoning_confidence: float,
    decision_readiness: str,
    conflict_count: int,
) -> str:
    if most_likely_prediction:
        outcome = most_likely_prediction.get(
            "predicted_outcome",
            "unknown",
        )
    else:
        outcome = "no clear predicted outcome"

    return (
        f"The fused reasoning state is {future_state}. "
        f"The current reasoning focus is {reasoning_focus}. "
        f"The most likely outcome is {outcome}. "
        f"Future risk score is {future_risk_score}, while fused "
        f"reasoning confidence is {reasoning_confidence}. "
        f"Decision readiness is {decision_readiness}. "
        f"Detected reasoning conflicts: {conflict_count}."
    )


# ============================================================
# MAIN FUSION FUNCTION
# ============================================================

def fuse_reasoning(
    predictions: Dict[str, Any],
    hazards: Dict[str, Any],
    intent_result: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Dict[str, Any]:
    log_section("Reasoning Fusion")

    most_likely_prediction = (
        get_most_likely_prediction(
            predictions
        )
    )

    high_priority_predictions = (
        extract_high_priority_predictions(
            predictions
        )
    )

    future_risk_predictions = (
        extract_future_risk_predictions(
            predictions
        )
    )

    conflicts = detect_reasoning_conflicts(
        predictions,
        hazards,
        intent_result,
        situation,
        cognitive_state,
    )

    signal_scores = calculate_signal_scores(
        predictions,
        hazards,
        intent_result,
        cognitive_state,
        situation,
    )

    reasoning_confidence = (
        calculate_reasoning_confidence(
            signal_scores,
            conflicts,
        )
    )

    future_risk_score = calculate_future_risk_score(
        predictions,
        hazards,
        intent_result,
        cognitive_state,
        situation,
    )

    future_state = classify_future_state(
        future_risk_score,
        hazards,
    )

    reasoning_focus = decide_reasoning_focus(
        most_likely_prediction,
        hazards,
        intent_result,
        situation,
        cognitive_state,
    )

    decision_readiness = (
        determine_decision_readiness(
            reasoning_confidence,
            conflicts,
            most_likely_prediction,
        )
    )

    recommended_actions = (
        generate_reasoning_actions(
            future_state,
            reasoning_focus,
            conflicts,
            predictions,
            hazards,
        )
    )

    result = {
        "timestamp": str(datetime.now()),
        "fusion_type": "multi_source_reasoning_fusion",

        "current_context": {
            "environment_type": get_environment_type(
                situation
            ),
            "primary_intent": get_primary_intent(
                intent_result
            ),
            "destination": get_destination(
                intent_result
            ),
            "urgency": get_urgency(
                intent_result
            ),
            "cognitive_priority": get_cognitive_priority(
                cognitive_state,
                situation,
            ),
            "reasoning_mode": get_reasoning_mode(
                cognitive_state,
                situation,
            ),
            "safety_status": get_safety_status(
                situation,
                cognitive_state,
                hazards,
            ),
            "graph_event_types": get_graph_event_types(
                context_graph
            ),
        },

        "signal_scores": signal_scores,

        "future_state": future_state,
        "reasoning_focus": reasoning_focus,
        "decision_readiness": decision_readiness,

        "most_likely_prediction": (
            most_likely_prediction
        ),

        "high_priority_prediction_count": len(
            high_priority_predictions
        ),
        "high_priority_predictions": (
            high_priority_predictions
        ),

        "future_risk_prediction_count": len(
            future_risk_predictions
        ),
        "future_risk_predictions": (
            future_risk_predictions
        ),

        "hazard_count": len(
            get_hazard_list(hazards)
        ),
        "current_risk_level": hazards.get(
            "overall_risk_level",
            "low",
        ),

        "future_risk_score": future_risk_score,
        "future_risk_label": confidence_label(
            future_risk_score
        ),

        "reasoning_confidence": reasoning_confidence,
        "reasoning_confidence_label": (
            confidence_label(
                reasoning_confidence
            )
        ),

        "conflict_count": len(conflicts),
        "reasoning_conflicts": conflicts,

        "recommended_reasoning_actions": (
            recommended_actions
        ),

        "summary": generate_fusion_summary(
            future_state,
            reasoning_focus,
            most_likely_prediction,
            future_risk_score,
            reasoning_confidence,
            decision_readiness,
            len(conflicts),
        ),
    }

    log_info(
        f"Future State: {future_state}"
    )
    log_info(
        f"Reasoning Focus: {reasoning_focus}"
    )
    log_info(
        f"Future Risk Score: {future_risk_score}"
    )
    log_info(
        f"Reasoning Confidence: {reasoning_confidence}"
    )
    log_info(
        f"Decision Readiness: {decision_readiness}"
    )
    log_info(
        f"Reasoning Conflicts: {len(conflicts)}"
    )

    log_success("Reasoning Fusion Complete")

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_required_inputs(
    predictions: Dict[str, Any],
    intent_result: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> bool:
    valid = True

    if not predictions:
        log_error(
            "predictions.json Missing Or Empty"
        )
        valid = False

    if not intent_result:
        log_error(
            "intent_reasoning.json Missing Or Empty"
        )
        valid = False

    if not situation:
        log_error(
            "situation_understanding.json Missing Or Empty"
        )
        valid = False

    if not cognitive_state:
        log_error(
            "cognitive_state.json Missing Or Empty"
        )
        valid = False

    if not context_graph:
        log_error(
            "context_graph.json Missing Or Empty"
        )
        valid = False

    return valid


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    module_start("REASONING FUSION")

    os.makedirs(
        LAYER4_OUTPUT_DIR,
        exist_ok=True,
    )

    predictions = load_json(
        PREDICTIONS_PATH
    )

    intent_result = load_json(
        INTENT_PATH
    )

    situation = load_json(
        SITUATION_PATH
    )

    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
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

    if not validate_required_inputs(
        predictions,
        intent_result,
        situation,
        cognitive_state,
        context_graph,
    ):
        module_end("REASONING FUSION")
        return

    if not hazards:
        log_warning(
            "hazards.json missing or empty. "
            "Using safe default."
        )

        hazards = {
            "hazard_count": 0,
            "hazards": [],
            "overall_risk_score": 0.0,
            "overall_risk_level": "low",
            "safety_status": "safe",
            "recommended_actions": [],
        }

    result = fuse_reasoning(
        predictions,
        hazards,
        intent_result,
        situation,
        cognitive_state,
        context_graph,
    )

    save_json(
        result,
        REASONING_FUSION_OUTPUT_PATH,
    )

    log_info(
        "Reasoning fusion output saved to: "
        f"{REASONING_FUSION_OUTPUT_PATH}"
    )

    module_end("REASONING FUSION")


if __name__ == "__main__":
    main()