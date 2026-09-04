"""
============================================================
NOONGIL-X


NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Prediction Engine
============================================================

Purpose
-------
Predict near-future outcomes using:

1. situation_understanding.json
2. cognitive_state.json
3. intent_reasoning.json
4. hazards.json
5. context_graph.json
6. commonsense_rules.json
7. navigation_rules.json
8. risk_rules.json
9. emergency_rules.json

The engine combines graph evidence, current state, intent,
hazards, and contextual rules. It avoids broad keyword-only
rule matching and removes unsupported predictions.
============================================================
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)
REASONING_DIR = os.path.dirname(CURRENT_FILE)
LAYER4_DIR = os.path.dirname(REASONING_DIR)
PROJECT_ROOT = os.path.dirname(LAYER4_DIR)

LAYER3_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "layer3")
LAYER4_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "layer4")
KNOWLEDGE_DIR = os.path.join(LAYER4_DIR, "knowledge")

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

CONTEXT_GRAPH_PATH = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json",
)

COMMON_SENSE_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "commonsense_rules.json",
)

RISK_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "risk_rules.json",
)

NAVIGATION_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "navigation_rules.json",
)

EMERGENCY_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "emergency_rules.json",
)

PREDICTION_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "predictions.json",
)


# ============================================================
# IMPORTS
# ============================================================

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
# CONSTANTS
# ============================================================

MOTION_ACTIVITIES = {
    "walking",
    "running",
    "moving",
    "crossing",
    "climbing",
    "descending",
    "travelling",
    "traveling",
}

STATIC_ACTIVITIES = {
    "resting",
    "sitting",
    "standing",
    "sleeping",
    "reading",
    "watching",
    "talking",
}

NAVIGATION_INTENTS = {
    "navigate_to_destination",
    "continue_movement",
    "find_person_or_object",
}

EMERGENCY_INTENTS = {
    "seek_emergency_help",
    "avoid_hazard",
    "stay_safe",
}

INDOOR_ENVIRONMENTS = {
    "home_environment",
    "classroom_environment",
    "office_environment",
    "indoor_environment",
    "shopping_mall",
}

OUTDOOR_ENVIRONMENTS = {
    "park_environment",
    "outdoor_environment",
    "urban_environment",
    "traffic_area",
    "transport_environment",
}

CRITICAL_HAZARD_TERMS = {
    "fire",
    "smoke",
    "medical_emergency",
    "accident",
    "collision",
    "fall",
    "help_call",
    "distress",
    "emergency",
}

NAVIGATION_HAZARD_TERMS = {
    "obstacle",
    "blocked",
    "blocking",
    "stairs",
    "vehicle",
    "traffic",
    "crowd",
    "construction",
    "spill",
    "drop",
}

ARRIVAL_RELATIONS = {
    "at",
    "reached",
    "arrived_at",
    "near_destination",
}

BLOCKING_RELATIONS = {
    "blocking",
    "blocks",
    "obstructing",
    "obstructs",
    "in_path_of",
}

APPROACH_RELATIONS = {
    "approaching",
    "moving_towards",
    "moving_toward",
    "crossing_path",
}

RISK_LEVEL_SCORES = {
    "none": 0.0,
    "very_low": 0.1,
    "low": 0.2,
    "medium": 0.5,
    "moderate": 0.5,
    "high": 0.75,
    "critical": 0.95,
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
    seen = set()
    result = []

    for value in values:
        marker = str(value)

        if marker not in seen:
            seen.add(marker)
            result.append(value)

    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        term_text = normalize_text(term).replace("_", " ")

        if term_text and term_text in normalized:
            return True

    return False


def risk_score_from_level(level: str) -> float:
    return RISK_LEVEL_SCORES.get(
        normalize_token(level),
        0.0,
    )


# ============================================================
# INPUT ACCESSORS
# ============================================================

def get_environment_type(situation: Dict[str, Any]) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("environment_context", "environment_type"),
            ],
            flat_keys=("environment_type",),
            default="",
        )
    )


def get_scene_type(situation: Dict[str, Any]) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("environment_context", "scene_type"),
            ],
            flat_keys=("scene_type",),
            default="",
        )
    )


def get_user_location(situation: Dict[str, Any]) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("user_state", "location"),
            ],
            flat_keys=("user_location", "location"),
            default="",
        )
    )


def get_user_activities(situation: Dict[str, Any]) -> List[str]:
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


def get_nearby_entities(situation: Dict[str, Any]) -> List[str]:
    value = get_first_available(
        situation,
        nested_paths=[
            ("environment_context", "nearby_entities"),
        ],
        flat_keys=("nearby_entities",),
        default=[],
    )

    return [
        normalize_token(item)
        for item in as_list(value)
        if normalize_token(item)
    ]


def get_event_types(situation: Dict[str, Any]) -> List[str]:
    value = get_first_available(
        situation,
        nested_paths=[
            ("event_context", "event_types"),
            ("event_context", "events"),
        ],
        flat_keys=("event_types", "events"),
        default=[],
    )

    result = []

    for item in as_list(value):
        if isinstance(item, dict):
            event_type = (
                item.get("event_type")
                or item.get("type")
                or item.get("name")
            )
        else:
            event_type = item

        token = normalize_token(event_type)

        if token:
            result.append(token)

    return unique_preserve_order(result)


def get_primary_intent(intent_result: Dict[str, Any]) -> str:
    return normalize_token(
        intent_result.get("primary_intent", "")
    )


def get_destination(intent_result: Dict[str, Any]) -> Optional[str]:
    value = (
        intent_result.get("destination")
        or intent_result.get("target")
    )

    token = normalize_token(value)

    return token or None


def get_intent_confidence(intent_result: Dict[str, Any]) -> float:
    return safe_float(
        intent_result.get("intent_confidence"),
        0.5,
    )


def get_urgency(intent_result: Dict[str, Any]) -> str:
    return normalize_token(
        intent_result.get("urgency", "low")
    )


def get_cognitive_priority(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
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


# ============================================================
# GRAPH HELPERS
# ============================================================

def extract_graph_nodes(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        node
        for node in context_graph.get("nodes", [])
        if isinstance(node, dict)
    ]


def extract_graph_edges(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        edge
        for edge in context_graph.get("edges", [])
        if isinstance(edge, dict)
    ]


def extract_graph_event_types(
    context_graph: Dict[str, Any],
) -> List[str]:
    result = []

    for node in extract_graph_nodes(context_graph):
        if normalize_token(node.get("category")) != "event":
            continue

        event_type = normalize_token(
            node.get("event_type")
            or node.get("type")
        )

        if event_type:
            result.append(event_type)

    return unique_preserve_order(result)


def graph_has_relation(
    context_graph: Dict[str, Any],
    relations: Iterable[str],
    source: Optional[str] = None,
    target: Optional[str] = None,
) -> bool:
    relation_set = {
        normalize_token(item)
        for item in relations
    }

    normalized_source = normalize_token(source)
    normalized_target = normalize_token(target)

    for edge in extract_graph_edges(context_graph):
        relation = normalize_token(
            edge.get("relation")
            or edge.get("type")
            or edge.get("label")
        )

        if relation not in relation_set:
            continue

        edge_source = normalize_token(edge.get("source"))
        edge_target = normalize_token(edge.get("target"))

        if normalized_source and edge_source != normalized_source:
            continue

        if normalized_target and edge_target != normalized_target:
            continue

        return True

    return False


def get_graph_terms(
    context_graph: Dict[str, Any],
) -> List[str]:
    terms = []

    for node in extract_graph_nodes(context_graph):
        for key in (
            "id",
            "label",
            "name",
            "entity_type",
            "category",
            "event_type",
        ):
            token = normalize_token(node.get(key))

            if token:
                terms.append(token)

    for edge in extract_graph_edges(context_graph):
        for key in (
            "source",
            "target",
            "relation",
            "type",
            "label",
        ):
            token = normalize_token(edge.get(key))

            if token:
                terms.append(token)

    return unique_preserve_order(terms)


# ============================================================
# RULE HELPERS
# ============================================================

def collect_rule_terms(rule: Dict[str, Any]) -> List[str]:
    terms = []

    for key in (
        "keywords",
        "triggers",
        "conditions",
        "entities",
        "event_types",
        "contexts",
        "situations",
        "hazards",
    ):
        for item in as_list(rule.get(key)):
            if isinstance(item, dict):
                for nested_value in item.values():
                    token = normalize_text(nested_value)

                    if token:
                        terms.append(token)
            else:
                token = normalize_text(item)

                if token:
                    terms.append(token)

    for key in (
        "name",
        "rule_name",
        "description",
        "condition",
    ):
        token = normalize_text(rule.get(key))

        if token:
            terms.append(token)

    return unique_preserve_order(terms)


def extract_rule_actions(rule: Dict[str, Any]) -> List[str]:
    actions = (
        rule.get("actions")
        or rule.get("recommended_actions")
        or rule.get("recommended_action")
        or []
    )

    return [
        str(action)
        for action in as_list(actions)
        if str(action).strip()
    ]


def get_rule_probability(
    rule: Dict[str, Any],
    default: float,
) -> float:
    for key in (
        "probability",
        "confidence",
        "prediction_score",
    ):
        if key in rule:
            return clamp_confidence(
                safe_float(rule.get(key), default)
            )

    priority = normalize_token(
        rule.get("priority")
        or rule.get("risk_level")
        or rule.get("severity")
    )

    priority_scores = {
        "low": 0.55,
        "medium": 0.70,
        "high": 0.85,
        "critical": 0.95,
        "maximum": 0.98,
    }

    return priority_scores.get(priority, default)


def match_rules(
    rules_data: Dict[str, Any],
    evidence_terms: List[str],
    evidence_text: str,
) -> List[Dict[str, Any]]:
    evidence_set = {
        normalize_token(term)
        for term in evidence_terms
        if normalize_token(term)
    }

    matches = []

    for rule in rules_data.get("rules", []):
        if not isinstance(rule, dict):
            continue

        rule_terms = collect_rule_terms(rule)

        if not rule_terms:
            continue

        matched_terms = []

        for term in rule_terms:
            token = normalize_token(term)

            if token in evidence_set or contains_any(
                evidence_text,
                [term],
            ):
                matched_terms.append(term)

        minimum_matches = int(
            max(
                1,
                safe_float(
                    rule.get("minimum_matches"),
                    1.0,
                ),
            )
        )

        if len(matched_terms) < minimum_matches:
            continue

        matches.append(
            {
                "rule": rule,
                "matched_terms": unique_preserve_order(
                    matched_terms
                ),
            }
        )

    return matches


# ============================================================
# PREDICTION BUILDER
# ============================================================

def create_prediction(
    prediction_type: str,
    predicted_outcome: str,
    probability: float,
    time_horizon: str,
    source: str,
    reason: str,
    recommended_preparation: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
    matched_rule_id: Optional[str] = None,
    matched_rule_name: Optional[str] = None,
    priority: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "prediction_id": None,
        "prediction_type": normalize_token(prediction_type),
        "predicted_outcome": normalize_token(predicted_outcome),
        "probability": round(
            clamp_confidence(probability),
            2,
        ),
        "time_horizon": normalize_token(time_horizon),
        "source": source,
        "reason": reason,
        "recommended_preparation": unique_preserve_order(
            recommended_preparation or []
        ),
        "evidence": unique_preserve_order(
            evidence or []
        ),
        "matched_rule_id": matched_rule_id,
        "matched_rule_name": matched_rule_name,
        "priority": priority,
    }


# ============================================================
# STATE-BASED PREDICTIONS
# ============================================================

def predict_from_current_state(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    predictions = []

    activities = get_user_activities(situation)
    primary_intent = get_primary_intent(intent_result)
    destination = get_destination(intent_result)
    intent_confidence = get_intent_confidence(intent_result)
    environment_type = get_environment_type(situation)

    log_subsection("Predicting From Current State")

    if any(
        activity in MOTION_ACTIVITIES
        for activity in activities
    ):
        predictions.append(
            create_prediction(
                prediction_type="movement_continuation",
                predicted_outcome="user_will_likely_continue_moving",
                probability=0.72,
                time_horizon="immediate",
                source="current_state",
                reason=(
                    "The user is currently performing a movement "
                    "activity, making short-term movement continuation likely."
                ),
                recommended_preparation=[
                    "continue_path_monitoring",
                    "maintain_obstacle_awareness",
                ],
                evidence=[
                    f"activities={activities}",
                ],
            )
        )

    if (
        primary_intent == "navigate_to_destination"
        and destination
    ):
        probability = min(
            0.90,
            0.65 + 0.25 * intent_confidence,
        )

        predictions.append(
            create_prediction(
                prediction_type="goal_directed_movement",
                predicted_outcome=(
                    f"user_will_move_toward_{destination}"
                ),
                probability=probability,
                time_horizon="short_term",
                source="intent_state",
                reason=(
                    "The detected intent identifies a specific "
                    "navigation destination."
                ),
                recommended_preparation=[
                    "track_destination",
                    "prepare_step_by_step_guidance",
                    "monitor_route_progress",
                ],
                evidence=[
                    f"primary_intent={primary_intent}",
                    f"destination={destination}",
                    f"intent_confidence={intent_confidence}",
                ],
            )
        )

    if (
        primary_intent == "navigate_to_destination"
        and not destination
    ):
        predictions.append(
            create_prediction(
                prediction_type="destination_clarification",
                predicted_outcome="system_will_need_destination_information",
                probability=0.80,
                time_horizon="immediate",
                source="intent_state",
                reason=(
                    "Navigation intent is present, but no destination "
                    "is available."
                ),
                recommended_preparation=[
                    "request_destination",
                    "wait_for_user_input",
                ],
                evidence=[
                    f"primary_intent={primary_intent}",
                    "destination_missing",
                ],
            )
        )

    if (
        primary_intent in NAVIGATION_INTENTS
        and environment_type in INDOOR_ENVIRONMENTS
    ):
        predictions.append(
            create_prediction(
                prediction_type="indoor_navigation_support",
                predicted_outcome=(
                    "user_may_need_landmark_based_guidance"
                ),
                probability=0.67,
                time_horizon="short_term",
                source="environment_state",
                reason=(
                    "The user has a navigation-related intent in an "
                    "indoor environment."
                ),
                recommended_preparation=[
                    "use_indoor_landmarks",
                    "track_nearby_objects",
                ],
                evidence=[
                    f"environment_type={environment_type}",
                    f"primary_intent={primary_intent}",
                ],
            )
        )

    if (
        primary_intent in NAVIGATION_INTENTS
        and environment_type in OUTDOOR_ENVIRONMENTS
    ):
        predictions.append(
            create_prediction(
                prediction_type="outdoor_navigation_support",
                predicted_outcome=(
                    "user_may_need_location_and_traffic_awareness"
                ),
                probability=0.70,
                time_horizon="short_term",
                source="environment_state",
                reason=(
                    "The user has a navigation-related intent in an "
                    "outdoor or traffic-related environment."
                ),
                recommended_preparation=[
                    "monitor_location",
                    "monitor_traffic_hazards",
                    "prepare_route_updates",
                ],
                evidence=[
                    f"environment_type={environment_type}",
                    f"primary_intent={primary_intent}",
                ],
            )
        )

    if (
        activities
        and all(
            activity in STATIC_ACTIVITIES
            for activity in activities
        )
        and primary_intent not in NAVIGATION_INTENTS
    ):
        predictions.append(
            create_prediction(
                prediction_type="activity_continuation",
                predicted_outcome=(
                    "user_will_likely_continue_current_activity"
                ),
                probability=0.62,
                time_horizon="short_term",
                source="current_state",
                reason=(
                    "The user is currently engaged in a stable, "
                    "non-movement activity."
                ),
                recommended_preparation=[
                    "maintain_context_awareness",
                    "avoid_unnecessary_interruption",
                ],
                evidence=[
                    f"activities={activities}",
                ],
            )
        )

    return predictions


# ============================================================
# GRAPH-BASED PREDICTIONS
# ============================================================

def predict_from_context_graph(
    context_graph: Dict[str, Any],
    intent_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    predictions = []

    destination = get_destination(intent_result)
    event_types = extract_graph_event_types(
        context_graph
    )

    log_subsection("Predicting From Context Graph")

    if graph_has_relation(
        context_graph,
        BLOCKING_RELATIONS,
    ):
        predictions.append(
            create_prediction(
                prediction_type="route_change_possible",
                predicted_outcome="user_may_need_alternate_route",
                probability=0.86,
                time_horizon="immediate",
                source="context_graph_relation",
                reason=(
                    "The context graph contains a blocking or "
                    "obstruction relation."
                ),
                recommended_preparation=[
                    "identify_blockage",
                    "calculate_alternate_route",
                    "warn_user",
                ],
                evidence=[
                    "blocking_relation_detected",
                ],
            )
        )

    if graph_has_relation(
        context_graph,
        APPROACH_RELATIONS,
    ):
        predictions.append(
            create_prediction(
                prediction_type="approaching_entity_risk",
                predicted_outcome=(
                    "moving_entity_may_enter_user_path"
                ),
                probability=0.78,
                time_horizon="immediate",
                source="context_graph_relation",
                reason=(
                    "The context graph indicates that an entity is "
                    "approaching or crossing the user's path."
                ),
                recommended_preparation=[
                    "increase_spatial_monitoring",
                    "prepare_collision_warning",
                ],
                evidence=[
                    "approach_relation_detected",
                ],
            )
        )

    if destination and graph_has_relation(
        context_graph,
        ARRIVAL_RELATIONS,
        source="user",
        target=destination,
    ):
        predictions.append(
            create_prediction(
                prediction_type="destination_arrival",
                predicted_outcome=(
                    f"user_will_reach_{destination}"
                ),
                probability=0.90,
                time_horizon="immediate",
                source="context_graph_relation",
                reason=(
                    "The context graph indicates that the user is "
                    "at or near the destination."
                ),
                recommended_preparation=[
                    "announce_arrival",
                    "confirm_destination",
                    "stop_navigation_when_confirmed",
                ],
                evidence=[
                    f"user_near_destination={destination}",
                ],
            )
        )

    if any(
        event in {
            "navigation_request",
            "movement",
            "home_activity",
            "conversation_event",
        }
        for event in event_types
    ):
        predictions.append(
            create_prediction(
                prediction_type="event_continuation",
                predicted_outcome=(
                    "current_context_event_may_continue"
                ),
                probability=0.60,
                time_horizon="short_term",
                source="context_graph_event",
                reason=(
                    "The graph contains active event nodes that may "
                    "persist into the near future."
                ),
                recommended_preparation=[
                    "monitor_event_progress",
                    "update_context_graph",
                ],
                evidence=[
                    f"event_types={event_types}",
                ],
            )
        )

    return predictions


# ============================================================
# HAZARD-BASED PREDICTIONS
# ============================================================

def predict_from_hazards(
    hazards: Dict[str, Any],
) -> List[Dict[str, Any]]:
    predictions = []

    hazard_list = hazards.get("hazards", [])
    risk_level = normalize_token(
        hazards.get("overall_risk_level", "low")
    )
    overall_risk_score = safe_float(
        hazards.get("overall_risk_score"),
        risk_score_from_level(risk_level),
    )

    log_subsection("Predicting From Hazards")

    if not hazard_list:
        return predictions

    if risk_level in {
        "medium",
        "moderate",
        "high",
        "critical",
    }:
        predictions.append(
            create_prediction(
                prediction_type="safety_intervention",
                predicted_outcome=(
                    "system_will_prioritize_safety_response"
                ),
                probability=max(
                    0.75,
                    overall_risk_score,
                ),
                time_horizon="immediate",
                source="hazard_reasoning",
                reason=(
                    f"The overall hazard risk level is {risk_level}."
                ),
                recommended_preparation=[
                    "prioritize_safety",
                    "issue_context_appropriate_warning",
                    "prepare_safe_action",
                ],
                evidence=[
                    f"overall_risk_level={risk_level}",
                    f"overall_risk_score={overall_risk_score}",
                ],
            )
        )

    for hazard in hazard_list:
        hazard_type = normalize_token(
            hazard.get("hazard_type")
        )
        risk_score = safe_float(
            hazard.get("risk_score"),
            0.5,
        )
        confidence = safe_float(
            hazard.get("confidence"),
            0.7,
        )
        evidence = [
            str(item)
            for item in hazard.get("evidence", [])
        ]

        probability = clamp_confidence(
            0.60 * risk_score
            + 0.40 * confidence
        )

        if contains_any(
            hazard_type,
            CRITICAL_HAZARD_TERMS,
        ):
            predictions.append(
                create_prediction(
                    prediction_type="emergency_escalation",
                    predicted_outcome=(
                        "emergency_response_may_be_required"
                    ),
                    probability=max(
                        0.85,
                        probability,
                    ),
                    time_horizon="immediate",
                    source="hazard_reasoning",
                    reason=(
                        f"Critical hazard evidence was detected: "
                        f"{hazard_type}."
                    ),
                    recommended_preparation=[
                        "activate_emergency_protocol",
                        "guide_user_to_safety",
                        "request_external_help_if_needed",
                    ],
                    evidence=[
                        f"hazard_type={hazard_type}",
                        *evidence,
                    ],
                )
            )

        elif contains_any(
            hazard_type,
            NAVIGATION_HAZARD_TERMS,
        ):
            predictions.append(
                create_prediction(
                    prediction_type="navigation_disruption",
                    predicted_outcome=(
                        "current_route_may_become_unsafe_or_unavailable"
                    ),
                    probability=probability,
                    time_horizon="immediate",
                    source="hazard_reasoning",
                    reason=(
                        f"Navigation-related hazard detected: "
                        f"{hazard_type}."
                    ),
                    recommended_preparation=[
                        "warn_user",
                        "reassess_route",
                        "prepare_alternate_path",
                    ],
                    evidence=[
                        f"hazard_type={hazard_type}",
                        *evidence,
                    ],
                )
            )

    return predictions


# ============================================================
# KNOWLEDGE-BASED PREDICTIONS
# ============================================================

def build_prediction_evidence(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Tuple[List[str], str]:
    hazard_terms = []

    for hazard in hazards.get("hazards", []):
        if not isinstance(hazard, dict):
            continue

        for key in (
            "hazard_type",
            "entity",
            "relation",
            "target",
        ):
            token = normalize_token(hazard.get(key))

            if token:
                hazard_terms.append(token)

    terms = unique_preserve_order(
        get_graph_terms(context_graph)
        + extract_graph_event_types(context_graph)
        + get_user_activities(situation)
        + get_nearby_entities(situation)
        + get_event_types(situation)
        + hazard_terms
        + [
            get_environment_type(situation),
            get_scene_type(situation),
            get_user_location(situation),
            get_primary_intent(intent_result),
            get_destination(intent_result),
            get_urgency(intent_result),
            get_cognitive_priority(
                situation,
                cognitive_state,
            ),
            normalize_token(
                hazards.get(
                    "overall_risk_level",
                    "",
                )
            ),
        ]
    )

    terms = [
        term
        for term in terms
        if term
    ]

    evidence_text = " ".join(
        term.replace("_", " ")
        for term in terms
    )

    return terms, evidence_text


def predict_from_rule_matches(
    source_name: str,
    rules_data: Dict[str, Any],
    evidence_terms: List[str],
    evidence_text: str,
) -> List[Dict[str, Any]]:
    predictions = []

    matches = match_rules(
        rules_data,
        evidence_terms,
        evidence_text,
    )

    for match in matches:
        rule = match["rule"]
        matched_terms = match["matched_terms"]

        outcome = (
            rule.get("predicted_outcome")
            or rule.get("outcome")
            or rule.get("prediction")
        )

        if not outcome:
            continue

        probability = get_rule_probability(
            rule,
            default=0.70,
        )

        predictions.append(
            create_prediction(
                prediction_type=(
                    rule.get("prediction_type")
                    or f"{source_name}_prediction"
                ),
                predicted_outcome=str(outcome),
                probability=probability,
                time_horizon=(
                    rule.get("time_horizon")
                    or "short_term"
                ),
                source=source_name,
                reason=(
                    f"Rule matched using contextual evidence: "
                    f"{', '.join(matched_terms)}."
                ),
                recommended_preparation=(
                    extract_rule_actions(rule)
                ),
                evidence=[
                    f"matched_term={term}"
                    for term in matched_terms
                ],
                matched_rule_id=(
                    rule.get("rule_id")
                    or rule.get("id")
                ),
                matched_rule_name=(
                    rule.get("name")
                    or rule.get("rule_name")
                ),
                priority=rule.get("priority"),
            )
        )

    return predictions


def predict_from_common_sense_categories(
    common_sense: Dict[str, Any],
    situation: Dict[str, Any],
    intent_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    predictions = []

    categories = common_sense.get(
        "categories",
        {},
    )

    activities = get_user_activities(situation)
    nearby_entities = get_nearby_entities(situation)
    primary_intent = get_primary_intent(intent_result)

    log_subsection("Predicting From Common Sense")

    if any(
        activity in MOTION_ACTIVITIES
        for activity in activities
    ):
        predictions.append(
            create_prediction(
                prediction_type="common_sense_movement",
                predicted_outcome=(
                    "user_will_likely_continue_current_motion"
                ),
                probability=0.70,
                time_horizon="immediate",
                source="common_sense",
                reason=(
                    "Movement commonly continues briefly unless "
                    "interrupted by a goal change or hazard."
                ),
                recommended_preparation=[
                    "continue_motion_monitoring",
                    "maintain_obstacle_awareness",
                ],
                evidence=[
                    f"activities={activities}",
                ],
            )
        )

    if (
        primary_intent == "navigate_to_destination"
        and get_destination(intent_result)
    ):
        predictions.append(
            create_prediction(
                prediction_type="common_sense_goal_progress",
                predicted_outcome=(
                    "user_will_attempt_to_progress_toward_destination"
                ),
                probability=0.78,
                time_horizon="short_term",
                source="common_sense",
                reason=(
                    "Users with a known navigation goal generally "
                    "attempt to move toward that goal."
                ),
                recommended_preparation=[
                    "maintain_route_guidance",
                    "track_progress",
                ],
                evidence=[
                    f"primary_intent={primary_intent}",
                    f"destination={get_destination(intent_result)}",
                ],
            )
        )

    if any(
        contains_any(entity, NAVIGATION_HAZARD_TERMS)
        for entity in nearby_entities
    ):
        predictions.append(
            create_prediction(
                prediction_type="common_sense_navigation_difficulty",
                predicted_outcome=(
                    "user_may_require_additional_navigation_support"
                ),
                probability=0.72,
                time_horizon="immediate",
                source="common_sense",
                reason=(
                    "Nearby obstacle-like or traffic-related entities "
                    "can increase navigation difficulty."
                ),
                recommended_preparation=[
                    "prepare_obstacle_warning",
                    "prepare_safer_path",
                ],
                evidence=[
                    f"nearby_entities={nearby_entities}",
                ],
            )
        )

    return predictions


# ============================================================
# FILTERING AND MERGING
# ============================================================

def prediction_supported(
    prediction: Dict[str, Any],
) -> bool:
    probability = safe_float(
        prediction.get("probability"),
        0.0,
    )

    evidence = prediction.get("evidence", [])
    source = prediction.get("source", "")

    if probability < 0.45:
        return False

    if source in {
        "current_state",
        "intent_state",
        "environment_state",
        "context_graph_relation",
        "context_graph_event",
        "hazard_reasoning",
        "common_sense",
    }:
        return True

    return bool(evidence)


def merge_duplicate_predictions(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for prediction in predictions:
        key = (
            normalize_token(
                prediction.get("prediction_type")
            ),
            normalize_token(
                prediction.get("predicted_outcome")
            ),
        )

        if key not in merged:
            merged[key] = prediction.copy()
            continue

        existing = merged[key]

        existing["probability"] = max(
            safe_float(
                existing.get("probability"),
                0.0,
            ),
            safe_float(
                prediction.get("probability"),
                0.0,
            ),
        )

        existing["recommended_preparation"] = (
            unique_preserve_order(
                as_list(
                    existing.get(
                        "recommended_preparation"
                    )
                )
                + as_list(
                    prediction.get(
                        "recommended_preparation"
                    )
                )
            )
        )

        existing["evidence"] = unique_preserve_order(
            as_list(existing.get("evidence"))
            + as_list(prediction.get("evidence"))
        )

    result = sorted(
        merged.values(),
        key=lambda item: safe_float(
            item.get("probability"),
            0.0,
        ),
        reverse=True,
    )

    for index, prediction in enumerate(
        result,
        start=1,
    ):
        prediction["prediction_id"] = (
            f"PRD_{index:03d}"
        )

    return result


# ============================================================
# CONFIDENCE AND SUMMARY
# ============================================================

def calculate_overall_prediction_confidence(
    predictions: List[Dict[str, Any]],
) -> float:
    if not predictions:
        return 0.0

    probabilities = sorted(
        [
            safe_float(
                prediction.get("probability"),
                0.0,
            )
            for prediction in predictions
        ],
        reverse=True,
    )

    highest = probabilities[0]
    average = (
        sum(probabilities)
        / len(probabilities)
    )

    overall = (
        0.60 * highest
        + 0.40 * average
    )

    return round(
        clamp_confidence(overall),
        2,
    )


def find_most_likely_prediction(
    predictions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not predictions:
        return None

    return max(
        predictions,
        key=lambda prediction: safe_float(
            prediction.get("probability"),
            0.0,
        ),
    )


def collect_preparations(
    predictions: List[Dict[str, Any]],
) -> List[str]:
    actions = []

    for prediction in predictions:
        actions.extend(
            prediction.get(
                "recommended_preparation",
                [],
            )
        )

    return unique_preserve_order(actions)


def generate_prediction_summary(
    predictions: List[Dict[str, Any]],
    overall_confidence: float,
) -> str:
    if not predictions:
        return (
            "No sufficiently supported near-future outcome "
            "was identified from the current context."
        )

    most_likely = find_most_likely_prediction(
        predictions
    )

    return (
        f"{len(predictions)} supported near-future outcome(s) "
        f"were predicted. The most likely outcome is "
        f"{most_likely.get('predicted_outcome')} with probability "
        f"{most_likely.get('probability')}. Overall prediction "
        f"confidence is {overall_confidence}."
    )


# ============================================================
# MAIN PREDICTION ENGINE
# ============================================================

def predict_future_outcomes(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    hazards: Dict[str, Any],
    context_graph: Dict[str, Any],
    common_sense: Dict[str, Any],
    navigation_rules: Dict[str, Any],
    risk_rules: Dict[str, Any],
    emergency_rules: Dict[str, Any],
) -> Dict[str, Any]:
    log_section("Prediction Engine")

    evidence_terms, evidence_text = (
        build_prediction_evidence(
            situation,
            cognitive_state,
            intent_result,
            hazards,
            context_graph,
        )
    )

    prediction_groups = {
        "state_predictions": (
            predict_from_current_state(
                situation,
                cognitive_state,
                intent_result,
                context_graph,
            )
        ),
        "graph_predictions": (
            predict_from_context_graph(
                context_graph,
                intent_result,
            )
        ),
        "hazard_predictions": (
            predict_from_hazards(hazards)
        ),
        "common_sense_predictions": (
            predict_from_common_sense_categories(
                common_sense,
                situation,
                intent_result,
            )
        ),
        "navigation_rule_predictions": (
            predict_from_rule_matches(
                "navigation_rules",
                navigation_rules,
                evidence_terms,
                evidence_text,
            )
        ),
        "risk_rule_predictions": (
            predict_from_rule_matches(
                "risk_rules",
                risk_rules,
                evidence_terms,
                evidence_text,
            )
        ),
        "emergency_rule_predictions": (
            predict_from_rule_matches(
                "emergency_rules",
                emergency_rules,
                evidence_terms,
                evidence_text,
            )
        ),
    }

    all_predictions = []

    for group_name, group_predictions in (
        prediction_groups.items()
    ):
        log_debug(
            f"{group_name}: "
            f"{len(group_predictions)} prediction(s)"
        )
        all_predictions.extend(
            group_predictions
        )

    supported_predictions = [
        prediction
        for prediction in all_predictions
        if prediction_supported(prediction)
    ]

    predictions = merge_duplicate_predictions(
        supported_predictions
    )

    overall_confidence = (
        calculate_overall_prediction_confidence(
            predictions
        )
    )

    most_likely = find_most_likely_prediction(
        predictions
    )

    result = {
        "timestamp": str(datetime.now()),
        "prediction_count": len(predictions),
        "prediction_groups": prediction_groups,
        "predictions": predictions,
        "most_likely_prediction": most_likely,
        "overall_prediction_confidence": (
            overall_confidence
        ),
        "confidence_label": confidence_label(
            overall_confidence
        ),
        "recommended_preparations": (
            collect_preparations(predictions)
        ),
        "evidence_summary": {
            "environment_type": (
                get_environment_type(situation)
            ),
            "activities": (
                get_user_activities(situation)
            ),
            "primary_intent": (
                get_primary_intent(intent_result)
            ),
            "destination": (
                get_destination(intent_result)
            ),
            "overall_risk_level": (
                hazards.get(
                    "overall_risk_level",
                    "low",
                )
            ),
            "graph_event_types": (
                extract_graph_event_types(
                    context_graph
                )
            ),
        },
        "summary": generate_prediction_summary(
            predictions,
            overall_confidence,
        ),
    }

    log_info(
        f"Prediction Count: {len(predictions)}"
    )
    log_info(
        "Overall Prediction Confidence: "
        f"{overall_confidence}"
    )

    if most_likely:
        log_info(
            "Most Likely Outcome: "
            f"{most_likely.get('predicted_outcome')}"
        )

    log_success("Prediction Engine Complete")

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_required_inputs(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_result: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> bool:
    valid = True

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

    if not intent_result:
        log_error(
            "intent_reasoning.json Missing Or Empty"
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
    module_start("PREDICTION ENGINE")

    os.makedirs(
        LAYER4_OUTPUT_DIR,
        exist_ok=True,
    )

    situation = load_json(SITUATION_PATH)
    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
    )
    intent_result = load_json(INTENT_PATH)
    context_graph = load_json(
        CONTEXT_GRAPH_PATH
    )

    if not validate_required_inputs(
        situation,
        cognitive_state,
        intent_result,
        context_graph,
    ):
        module_end("PREDICTION ENGINE")
        return

    hazards = load_json(
        HAZARDS_PATH,
        default={
            "hazards": [],
            "overall_risk_score": 0.0,
            "overall_risk_level": "low",
            "safety_status": "safe",
        },
    )

    common_sense = load_json(
        COMMON_SENSE_PATH
    )
    navigation_rules = load_json(
        NAVIGATION_RULES_PATH
    )
    risk_rules = load_json(
        RISK_RULES_PATH
    )
    emergency_rules = load_json(
        EMERGENCY_RULES_PATH
    )

    if not hazards:
        log_warning(
            "hazards.json missing or empty. "
            "Using a safe default."
        )
        hazards = {
            "hazards": [],
            "overall_risk_score": 0.0,
            "overall_risk_level": "low",
            "safety_status": "safe",
        }

    if not common_sense:
        log_warning(
            "commonsense_rules.json missing or empty. "
            "Continuing without common-sense enrichment."
        )
        common_sense = {}

    if not navigation_rules:
        log_warning(
            "navigation_rules.json missing or empty. "
            "Continuing without navigation-rule enrichment."
        )
        navigation_rules = {}

    if not risk_rules:
        log_warning(
            "risk_rules.json missing or empty. "
            "Continuing without risk-rule enrichment."
        )
        risk_rules = {}

    if not emergency_rules:
        log_warning(
            "emergency_rules.json missing or empty. "
            "Continuing without emergency-rule enrichment."
        )
        emergency_rules = {}

    result = predict_future_outcomes(
        situation,
        cognitive_state,
        intent_result,
        hazards,
        context_graph,
        common_sense,
        navigation_rules,
        risk_rules,
        emergency_rules,
    )

    save_json(
        result,
        PREDICTION_OUTPUT_PATH,
    )

    log_info(
        "Prediction output saved to: "
        f"{PREDICTION_OUTPUT_PATH}"
    )

    module_end("PREDICTION ENGINE")


if __name__ == "__main__":
    main()