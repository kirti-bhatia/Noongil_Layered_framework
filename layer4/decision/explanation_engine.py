"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Explanation Engine
============================================================

Purpose
-------
Explain why the Decision Engine selected its final action.

Inputs
------
1. decision_output.json
2. reasoning_fusion.json
3. situation_understanding.json
4. cognitive_state.json
5. intent_reasoning.json
6. hazards.json
7. predictions.json
8. context_graph.json

Output
------
explanation_output.json
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

if LAYER4_DIR not in sys.path:
    sys.path.insert(0, LAYER4_DIR)

from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import confidence_label


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

DECISION_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "decision_output.json",
)

FUSION_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "reasoning_fusion.json",
)

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

EXPLANATION_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "explanation_output.json",
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    return " ".join(text.split())


def normalize_token(value: Any) -> str:
    return normalize_text(value).lower().replace("-", "_").replace(" ", "_")


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


def get_primary_intent(
    intent_result: Dict[str, Any],
) -> str:
    return normalize_token(
        intent_result.get("primary_intent", "unknown")
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


def get_most_likely_prediction(
    predictions: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    item = predictions.get("most_likely_prediction")

    if isinstance(item, dict):
        return item

    prediction_list = [
        prediction
        for prediction in predictions.get("predictions", [])
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


# ============================================================
# EXPLANATION EVIDENCE
# ============================================================

def extract_decision_evidence(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Dict[str, Any]:
    primary_action = decision.get("primary_action", {})
    most_likely = get_most_likely_prediction(predictions)

    evidence = {
        "environment": {
            "environment_type": get_environment_type(
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
        },

        "intent": {
            "primary_intent": get_primary_intent(
                intent_result
            ),
            "destination": get_destination(
                intent_result
            ),
            "intent_confidence": safe_float(
                intent_result.get(
                    "intent_confidence",
                    0.0,
                ),
                0.0,
            ),
            "urgency": intent_result.get(
                "urgency",
                "low",
            ),
            "required_assistance": intent_result.get(
                "required_assistance",
                [],
            ),
        },

        "cognitive_state": {
            "attention_focus": cognitive_state.get(
                "attention_focus",
                "unknown",
            ),
            "primary_goal": cognitive_state.get(
                "primary_goal",
                "unknown",
            ),
            "cognitive_priority": cognitive_state.get(
                "cognitive_priority",
                "low",
            ),
            "reasoning_mode": cognitive_state.get(
                "reasoning_mode",
                "general",
            ),
            "safety_status": cognitive_state.get(
                "safety_status",
                "unknown",
            ),
        },

        "hazard_reasoning": {
            "hazard_count": hazards.get(
                "hazard_count",
                len(hazards.get("hazards", [])),
            ),
            "overall_risk_score": hazards.get(
                "overall_risk_score",
                0.0,
            ),
            "overall_risk_level": hazards.get(
                "overall_risk_level",
                "low",
            ),
            "safety_status": hazards.get(
                "safety_status",
                "safe",
            ),
            "hazards": hazards.get(
                "hazards",
                [],
            ),
        },

        "prediction_reasoning": {
            "prediction_count": predictions.get(
                "prediction_count",
                len(predictions.get("predictions", [])),
            ),
            "overall_prediction_confidence": (
                predictions.get(
                    "overall_prediction_confidence",
                    0.0,
                )
            ),
            "most_likely_prediction": most_likely,
        },

        "fusion_reasoning": {
            "future_state": fusion.get(
                "future_state",
                "unknown",
            ),
            "reasoning_focus": fusion.get(
                "reasoning_focus",
                "unknown",
            ),
            "decision_readiness": fusion.get(
                "decision_readiness",
                "unknown",
            ),
            "future_risk_score": fusion.get(
                "future_risk_score",
                0.0,
            ),
            "reasoning_confidence": fusion.get(
                "reasoning_confidence",
                0.0,
            ),
            "conflict_count": fusion.get(
                "conflict_count",
                0,
            ),
            "reasoning_conflicts": fusion.get(
                "reasoning_conflicts",
                [],
            ),
        },

        "decision": {
            "decision_mode": decision.get(
                "decision_mode",
                "unknown",
            ),
            "decision_priority": decision.get(
                "decision_priority",
                "unknown",
            ),
            "decision_confidence": decision.get(
                "decision_confidence",
                0.0,
            ),
            "decision_status": decision.get(
                "decision_status",
                "unknown",
            ),
            "primary_action": primary_action,
            "supporting_actions": decision.get(
                "supporting_actions",
                [],
            ),
        },

        "graph_support": {
            "node_count": context_graph.get(
                "node_count",
                len(context_graph.get("nodes", [])),
            ),
            "edge_count": context_graph.get(
                "edge_count",
                len(context_graph.get("edges", [])),
            ),
        },
    }

    return evidence


# ============================================================
# REASON GENERATION
# ============================================================

def explain_intent(
    intent_result: Dict[str, Any],
) -> str:
    primary_intent = get_primary_intent(
        intent_result
    )

    destination = get_destination(
        intent_result
    )

    confidence = safe_float(
        intent_result.get(
            "intent_confidence",
            0.0,
        )
    )

    if primary_intent == "navigate_to_destination":
        if destination:
            return (
                f"The user intends to navigate to {destination}, "
                f"with intent confidence {confidence}."
            )

        return (
            "The user has a navigation intent, but the destination "
            "is not yet available."
        )

    return (
        f"The detected primary intent is {primary_intent}, "
        f"with confidence {confidence}."
    )


def explain_hazards(
    hazards: Dict[str, Any],
) -> str:
    hazard_count = hazards.get(
        "hazard_count",
        len(hazards.get("hazards", [])),
    )

    risk_level = hazards.get(
        "overall_risk_level",
        "low",
    )

    risk_score = hazards.get(
        "overall_risk_score",
        0.0,
    )

    if hazard_count == 0:
        return (
            "No active hazard was detected, so no safety override "
            "was required."
        )

    return (
        f"{hazard_count} hazard(s) were detected. The overall risk "
        f"level is {risk_level} with score {risk_score}."
    )


def explain_prediction(
    predictions: Dict[str, Any],
) -> str:
    most_likely = get_most_likely_prediction(
        predictions
    )

    if not most_likely:
        return (
            "No sufficiently supported future outcome was available."
        )

    outcome = most_likely.get(
        "predicted_outcome",
        "unknown",
    )

    probability = most_likely.get(
        "probability",
        0.0,
    )

    reason = most_likely.get(
        "reason",
        "",
    )

    explanation = (
        f"The most likely future outcome is {outcome}, "
        f"with probability {probability}."
    )

    if reason:
        explanation += f" This was predicted because {reason}"

    return explanation


def explain_fusion(
    fusion: Dict[str, Any],
) -> str:
    future_state = fusion.get(
        "future_state",
        "unknown",
    )

    reasoning_focus = fusion.get(
        "reasoning_focus",
        "unknown",
    )

    readiness = fusion.get(
        "decision_readiness",
        "unknown",
    )

    confidence = fusion.get(
        "reasoning_confidence",
        0.0,
    )

    conflict_count = fusion.get(
        "conflict_count",
        0,
    )

    return (
        f"The fused reasoning state is {future_state}, with focus on "
        f"{reasoning_focus}. Reasoning confidence is {confidence}, "
        f"decision readiness is {readiness}, and {conflict_count} "
        f"reasoning conflict(s) were detected."
    )


def explain_primary_action(
    decision: Dict[str, Any],
) -> str:
    primary_action = decision.get(
        "primary_action",
        {},
    )

    action = primary_action.get(
        "action",
        "unknown_action",
    )

    target = primary_action.get(
        "target",
        "current_context",
    )

    expected_outcome = primary_action.get(
        "expected_outcome",
        "",
    )

    explanation = (
        f"The selected primary action is {action} for {target}."
    )

    if expected_outcome:
        explanation += (
            f" The expected result is: {expected_outcome}"
        )

    return explanation


# ============================================================
# USER-FACING EXPLANATION
# ============================================================

def generate_user_explanation(
    decision: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
) -> str:
    mode = normalize_token(
        decision.get(
            "decision_mode",
            "general_assistance",
        )
    )

    primary_action = decision.get(
        "primary_action",
        {},
    )

    destination = get_destination(
        intent_result
    )

    hazard_count = hazards.get(
        "hazard_count",
        len(hazards.get("hazards", [])),
    )

    if mode == "navigation_assistance":
        if destination:
            if hazard_count == 0:
                return (
                    f"I will guide you toward {destination}. "
                    "No immediate hazard is currently detected, so I can "
                    "continue with step-by-step navigation while monitoring "
                    "the surroundings."
                )

            return (
                f"I will guide you toward {destination}, but I will first "
                "account for the detected hazards and select the safer route."
            )

    if mode == "navigation_clarification":
        return (
            "I understand that you need navigation assistance, but I need "
            "your destination before I can begin."
        )

    if mode == "emergency_response":
        target = primary_action.get(
            "target",
            "the detected emergency",
        )

        return (
            f"I detected an urgent safety concern related to {target}. "
            "I am prioritizing immediate safety guidance and emergency support."
        )

    if mode == "safety_intervention":
        target = primary_action.get(
            "target",
            "the detected hazard",
        )

        return (
            f"I detected a safety risk related to {target}. "
            "I will warn you and help identify a safer way to continue."
        )

    if mode == "movement_assistance":
        return (
            "I will continue monitoring your movement and nearby obstacles "
            "to support safe progress."
        )

    if mode == "preventive_safety":
        return (
            "I will increase environmental monitoring and provide warnings "
            "only when a meaningful risk is detected."
        )

    return (
        "I will continue monitoring the current context and remain ready "
        "to assist when needed."
    )


# ============================================================
# TRACE AND FACTORS
# ============================================================

def build_explanation_trace(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    situation: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
) -> List[str]:
    trace = [
        (
            f"Environment interpreted as "
            f"{get_environment_type(situation)}."
        ),
        explain_intent(intent_result),
        explain_hazards(hazards),
        explain_prediction(predictions),
        explain_fusion(fusion),
        explain_primary_action(decision),
    ]

    for item in decision.get(
        "reasoning_trace",
        [],
    ):
        trace.append(str(item))

    return unique_preserve_order(trace)


def identify_decision_factors(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    factors = []

    factors.append(
        {
            "factor": "user_intent",
            "value": get_primary_intent(
                intent_result
            ),
            "importance": "high",
            "support": (
                intent_result.get(
                    "intent_confidence",
                    0.0,
                )
            ),
        }
    )

    if get_destination(intent_result):
        factors.append(
            {
                "factor": "destination",
                "value": get_destination(
                    intent_result
                ),
                "importance": "high",
                "support": "explicit_or_inferred_target",
            }
        )

    factors.append(
        {
            "factor": "current_risk",
            "value": hazards.get(
                "overall_risk_level",
                "low",
            ),
            "importance": (
                "critical"
                if hazards.get(
                    "overall_risk_level"
                ) in {"high", "critical"}
                else "medium"
            ),
            "support": hazards.get(
                "overall_risk_score",
                0.0,
            ),
        }
    )

    most_likely = get_most_likely_prediction(
        predictions
    )

    if most_likely:
        factors.append(
            {
                "factor": "predicted_outcome",
                "value": most_likely.get(
                    "predicted_outcome",
                    "unknown",
                ),
                "importance": "high",
                "support": most_likely.get(
                    "probability",
                    0.0,
                ),
            }
        )

    factors.append(
        {
            "factor": "reasoning_readiness",
            "value": fusion.get(
                "decision_readiness",
                "unknown",
            ),
            "importance": "high",
            "support": fusion.get(
                "reasoning_confidence",
                0.0,
            ),
        }
    )

    factors.append(
        {
            "factor": "selected_decision_mode",
            "value": decision.get(
                "decision_mode",
                "unknown",
            ),
            "importance": "high",
            "support": decision.get(
                "decision_confidence",
                0.0,
            ),
        }
    )

    return factors


# ============================================================
# EXPLANATION QUALITY
# ============================================================

def calculate_explanation_confidence(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    intent_result: Dict[str, Any],
    predictions: Dict[str, Any],
) -> float:
    decision_confidence = safe_float(
        decision.get(
            "decision_confidence",
            0.0,
        )
    )

    reasoning_confidence = safe_float(
        fusion.get(
            "reasoning_confidence",
            0.0,
        )
    )

    intent_confidence = safe_float(
        intent_result.get(
            "intent_confidence",
            0.0,
        )
    )

    prediction_confidence = safe_float(
        predictions.get(
            "overall_prediction_confidence",
            0.0,
        )
    )

    score = (
        decision_confidence * 0.40
        + reasoning_confidence * 0.30
        + intent_confidence * 0.20
        + prediction_confidence * 0.10
    )

    return round(
        max(0.0, min(1.0, score)),
        2,
    )


def identify_limitations(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
) -> List[str]:
    limitations = []

    if not predictions.get("predictions"):
        limitations.append(
            "No supported prediction was available."
        )

    if fusion.get("conflict_count", 0) > 0:
        limitations.append(
            "Reasoning conflicts were detected and may reduce reliability."
        )

    if decision.get(
        "decision_status"
    ) in {
        "requires_review",
        "requires_more_context",
    }:
        limitations.append(
            "The decision is not fully ready for execution."
        )

    if hazards.get(
        "overall_risk_level"
    ) in {
        "high",
        "critical",
    }:
        limitations.append(
            "The environment may change rapidly because of active hazards."
        )

    return limitations


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================

def generate_explanation(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    predictions: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Dict[str, Any]:
    log_section("Decision Explanation Generation")

    evidence = extract_decision_evidence(
        decision,
        fusion,
        situation,
        cognitive_state,
        intent_result,
        hazards,
        predictions,
        context_graph,
    )

    explanation_trace = build_explanation_trace(
        decision,
        fusion,
        situation,
        intent_result,
        hazards,
        predictions,
    )

    decision_factors = identify_decision_factors(
        decision,
        fusion,
        intent_result,
        hazards,
        predictions,
    )

    explanation_confidence = (
        calculate_explanation_confidence(
            decision,
            fusion,
            intent_result,
            predictions,
        )
    )

    user_explanation = generate_user_explanation(
        decision,
        intent_result,
        hazards,
    )

    limitations = identify_limitations(
        decision,
        fusion,
        hazards,
        predictions,
    )

    result = {
        "timestamp": str(datetime.now()),
        "explanation_id": (
            "EXP_"
            + datetime.now().strftime(
                "%Y%m%d%H%M%S%f"
            )
        ),
        "decision_id": decision.get(
            "decision_id",
            "unknown",
        ),
        "explanation_type": "evidence_based_decision_explanation",

        "decision_mode": decision.get(
            "decision_mode",
            "unknown",
        ),
        "decision_status": decision.get(
            "decision_status",
            "unknown",
        ),

        "user_explanation": user_explanation,

        "technical_explanation": {
            "intent_explanation": explain_intent(
                intent_result
            ),
            "hazard_explanation": explain_hazards(
                hazards
            ),
            "prediction_explanation": explain_prediction(
                predictions
            ),
            "fusion_explanation": explain_fusion(
                fusion
            ),
            "action_explanation": explain_primary_action(
                decision
            ),
        },

        "decision_factor_count": len(
            decision_factors
        ),
        "decision_factors": decision_factors,

        "evidence": evidence,

        "explanation_trace": explanation_trace,

        "explanation_confidence": (
            explanation_confidence
        ),
        "confidence_label": confidence_label(
            explanation_confidence
        ),

        "limitations": limitations,

        "next_layer_payload": {
            "message_type": "decision_explanation",
            "communication_priority": decision.get(
                "decision_priority",
                "medium",
            ),
            "user_message": user_explanation,
            "action_instruction": get_nested(
                decision,
                ("primary_action", "instruction"),
                "",
            ),
            "requires_confirmation": bool(
                get_nested(
                    decision,
                    (
                        "execution_requirements",
                        "requires_user_confirmation",
                    ),
                    False,
                )
            ),
        },

        "summary": (
            f"The selected decision is "
            f"{decision.get('decision_mode', 'unknown')} with "
            f"{decision.get('decision_confidence', 0.0)} confidence. "
            f"The explanation confidence is {explanation_confidence}. "
            f"The user-facing explanation is ready for Layer 5."
        ),
    }

    log_info(
        "Decision Mode: "
        f"{result.get('decision_mode')}"
    )
    log_info(
        "Explanation Confidence: "
        f"{explanation_confidence}"
    )
    log_info(
        "Decision Factors: "
        f"{len(decision_factors)}"
    )
    log_info(
        "Limitations: "
        f"{len(limitations)}"
    )

    log_success(
        "Decision Explanation Generated"
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_required_inputs(
    decision: Dict[str, Any],
    fusion: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    predictions: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    required = {
        "decision_output.json": decision,
        "reasoning_fusion.json": fusion,
        "situation_understanding.json": situation,
        "cognitive_state.json": cognitive_state,
        "intent_reasoning.json": intent_result,
        "predictions.json": predictions,
        "context_graph.json": context_graph,
    }

    missing = [
        filename
        for filename, data in required.items()
        if not data
    ]

    return len(missing) == 0, missing


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    module_start("EXPLANATION ENGINE")

    decision = load_json(
        DECISION_PATH
    )

    fusion = load_json(
        FUSION_PATH
    )

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

    valid, missing = validate_required_inputs(
        decision,
        fusion,
        situation,
        cognitive_state,
        intent_result,
        predictions,
        context_graph,
    )

    if not valid:
        log_error(
            "Required Explanation Engine inputs are missing: "
            + ", ".join(missing)
        )

        module_end("EXPLANATION ENGINE")
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

    result = generate_explanation(
        decision,
        fusion,
        situation,
        cognitive_state,
        intent_result,
        hazards,
        predictions,
        context_graph,
    )

    save_json(
        result,
        EXPLANATION_OUTPUT_PATH,
    )

    log_info(
        "Explanation output saved to: "
        f"{EXPLANATION_OUTPUT_PATH}"
    )

    module_end("EXPLANATION ENGINE")


if __name__ == "__main__":
    main()