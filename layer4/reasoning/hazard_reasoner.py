"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Hazard Reasoner
============================================================

Purpose
-------
Detect and evaluate hazards using:

1. situation_understanding.json
2. cognitive_state.json
3. intent_reasoning.json
4. context_graph.json
5. risk_rules.json
6. emergency_rules.json
7. navigation_rules.json

The module avoids treating normal objects as hazards unless
there is supporting contextual, relational, event, or rule evidence.
============================================================
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
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

INTENT_REASONING_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "intent_reasoning.json",
)

CONTEXT_GRAPH_PATH = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json",
)

RISK_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "risk_rules.json",
)

EMERGENCY_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "emergency_rules.json",
)

NAVIGATION_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "navigation_rules.json",
)

HAZARD_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "hazards.json",
)


# ============================================================
# IMPORTS
# ============================================================

if LAYER4_DIR not in sys.path:
    sys.path.insert(0, LAYER4_DIR)

from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import clamp_confidence, confidence_label


# ============================================================
# CONSTANTS
# ============================================================

RISK_LEVEL_SCORES = {
    "none": 0.0,
    "very_low": 0.1,
    "low": 0.2,
    "medium": 0.5,
    "moderate": 0.5,
    "high": 0.75,
    "severe": 0.9,
    "critical": 1.0,
    "extreme": 1.0,
}

RISK_LEVEL_ORDER = {
    "none": 0,
    "very_low": 1,
    "low": 2,
    "medium": 3,
    "moderate": 3,
    "high": 4,
    "severe": 5,
    "critical": 6,
    "extreme": 6,
}

HAZARD_EVENT_TYPES = {
    "hazard_event",
    "danger_event",
    "collision_risk",
    "collision_event",
    "obstacle_event",
    "traffic_risk",
    "unsafe_condition",
    "fall_risk",
    "fire_event",
    "smoke_event",
    "medical_emergency",
    "accident",
    "distress_event",
    "emergency_event",
}

CRITICAL_EVENT_TYPES = {
    "fire_event",
    "medical_emergency",
    "accident",
    "distress_event",
    "emergency_event",
    "fall_event",
}

HAZARD_ENTITY_TYPES = {
    "hazard",
    "obstacle",
    "danger",
    "vehicle",
    "fire",
    "smoke",
    "weapon",
    "spill",
    "stairs",
    "drop",
    "construction",
    "traffic",
}

NORMAL_OBJECT_TERMS = {
    "television",
    "tv",
    "sofa",
    "chair",
    "table",
    "bed",
    "desk",
    "door",
    "wall",
    "cup",
    "book",
    "phone",
    "person",
    "child",
    "teacher",
    "student",
    "shop",
    "bench",
    "projector",
}

HAZARD_TERMS = {
    "hazard",
    "danger",
    "unsafe",
    "collision",
    "obstacle",
    "blocked",
    "blocking",
    "fire",
    "smoke",
    "vehicle",
    "traffic",
    "stairs",
    "drop",
    "edge",
    "spill",
    "wet floor",
    "broken glass",
    "scream",
    "help call",
    "help_call",
    "accident",
    "fall",
    "falling",
    "sharp",
    "hot surface",
    "electric shock",
    "construction",
    "crowd surge",
}

SAFE_RELATIONS = {
    "near",
    "nearby",
    "located_in",
    "contains",
    "beside",
    "in_front_of",
    "behind",
    "on",
    "inside",
    "associated_with",
    "performing",
}

HAZARD_RELATIONS = {
    "blocking",
    "blocks",
    "obstructing",
    "obstructs",
    "approaching",
    "moving_towards",
    "moving_toward",
    "colliding_with",
    "collision_with",
    "too_close_to",
    "in_path_of",
    "hazard_to",
    "danger_to",
    "risk_to",
    "falling_towards",
    "on_fire",
    "emitting_smoke",
    "causing_risk",
    "crossing_path",
}

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

ENVIRONMENT_BASE_RISK = {
    "home_environment": 0.05,
    "classroom_environment": 0.05,
    "office_environment": 0.05,
    "indoor_environment": 0.10,
    "shopping_mall": 0.15,
    "park_environment": 0.15,
    "outdoor_environment": 0.20,
    "urban_environment": 0.25,
    "traffic_area": 0.40,
    "transport_environment": 0.30,
    "construction_environment": 0.50,
    "emergency_environment": 0.80,
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


def risk_level_from_score(score: float) -> str:
    score = clamp_confidence(score)

    if score >= 0.85:
        return "critical"

    if score >= 0.65:
        return "high"

    if score >= 0.35:
        return "medium"

    if score > 0.0:
        return "low"

    return "none"


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


def get_user_activities(situation: Dict[str, Any]) -> List[str]:
    activities = get_first_available(
        situation,
        nested_paths=[
            ("user_state", "activities"),
        ],
        flat_keys=("user_activities", "activities"),
        default=[],
    )

    return [
        normalize_token(activity)
        for activity in as_list(activities)
        if normalize_token(activity)
    ]


def get_nearby_entities(situation: Dict[str, Any]) -> List[str]:
    entities = get_first_available(
        situation,
        nested_paths=[
            ("environment_context", "nearby_entities"),
        ],
        flat_keys=("nearby_entities",),
        default=[],
    )

    return [
        normalize_token(entity)
        for entity in as_list(entities)
        if normalize_token(entity)
    ]


def get_audio_cues(situation: Dict[str, Any]) -> List[str]:
    cues = get_first_available(
        situation,
        nested_paths=[
            ("environment_context", "audio_cues"),
        ],
        flat_keys=("audio_cues", "sounds"),
        default=[],
    )

    result = []

    for cue in as_list(cues):
        if isinstance(cue, dict):
            label = (
                cue.get("label")
                or cue.get("name")
                or cue.get("id")
            )
        else:
            label = cue

        token = normalize_token(label)

        if token:
            result.append(token)

    return result


def get_safety_status(situation: Dict[str, Any]) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("safety_context", "safety_status"),
            ],
            flat_keys=("safety_status",),
            default="safe",
        )
    )


def get_existing_risk_level(situation: Dict[str, Any]) -> str:
    return normalize_token(
        get_first_available(
            situation,
            nested_paths=[
                ("safety_context", "risk_level"),
            ],
            flat_keys=("risk_level",),
            default="low",
        )
    )


def get_context_confidence(situation: Dict[str, Any]) -> float:
    return safe_float(
        situation.get("context_confidence", 0.5),
        default=0.5,
    )


def get_attention_focus(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
) -> str:
    value = get_first_available(
        situation,
        nested_paths=[
            ("cognitive_context", "attention_focus"),
        ],
        default=None,
    )

    if value in (None, ""):
        value = cognitive_state.get("attention_focus", "")

    return normalize_token(value)


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

def build_node_map(
    context_graph: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    node_map = {}

    for node in context_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue

        node_id = normalize_token(node.get("id"))

        if node_id:
            node_map[node_id] = node

    return node_map


def extract_event_nodes(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    events = []

    for node in context_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue

        category = normalize_token(node.get("category"))

        if category == "event":
            events.append(node)

    return events


def extract_event_types(
    context_graph: Dict[str, Any],
) -> List[str]:
    return unique_preserve_order(
        normalize_token(
            node.get("event_type")
            or node.get("type")
            or "unknown"
        )
        for node in extract_event_nodes(context_graph)
    )


def extract_graph_terms(
    context_graph: Dict[str, Any],
) -> List[str]:
    terms = []

    for node in context_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue

        for key in (
            "id",
            "label",
            "name",
            "entity_type",
            "category",
            "event_type",
        ):
            value = normalize_token(node.get(key))

            if value:
                terms.append(value)

    for edge in context_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue

        for key in (
            "source",
            "target",
            "relation",
            "type",
            "label",
        ):
            value = normalize_token(edge.get(key))

            if value:
                terms.append(value)

    return unique_preserve_order(terms)


def find_hazard_relations(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    relations = []

    for edge in context_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue

        relation = normalize_token(
            edge.get("relation")
            or edge.get("type")
            or edge.get("label")
        )

        if relation not in HAZARD_RELATIONS:
            continue

        relations.append(
            {
                "source": normalize_token(edge.get("source")),
                "relation": relation,
                "target": normalize_token(edge.get("target")),
                "confidence": safe_float(
                    edge.get("confidence"),
                    1.0,
                ),
            }
        )

    return relations


def find_explicit_hazard_nodes(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hazards = []

    for node in context_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue

        node_id = normalize_token(node.get("id"))
        entity_type = normalize_token(
            node.get("entity_type")
            or node.get("type")
        )
        category = normalize_token(node.get("category"))
        event_type = normalize_token(node.get("event_type"))

        explicit_hazard = (
            entity_type in HAZARD_ENTITY_TYPES
            or category in {"hazard", "danger", "risk"}
            or event_type in HAZARD_EVENT_TYPES
        )

        if not explicit_hazard:
            continue

        hazards.append(node)

    return hazards


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
        for value in as_list(rule.get(key)):
            if isinstance(value, dict):
                for nested in value.values():
                    token = normalize_text(nested)

                    if token:
                        terms.append(token)
            else:
                token = normalize_text(value)

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


def get_rule_risk_score(rule: Dict[str, Any]) -> float:
    for key in (
        "risk_score",
        "severity_score",
        "score",
        "weight",
    ):
        if key in rule:
            score = safe_float(rule.get(key), -1.0)

            if score >= 0.0:
                return clamp_confidence(score)

    level = normalize_token(
        rule.get("risk_level")
        or rule.get("severity")
        or rule.get("priority")
        or ""
    )

    return risk_score_from_level(level)


def match_contextual_rules(
    rules_data: Dict[str, Any],
    evidence_terms: List[str],
    evidence_text: str,
) -> List[Dict[str, Any]]:
    matched_rules = []
    evidence_set = {
        normalize_token(term)
        for term in evidence_terms
        if normalize_token(term)
    }

    for rule in rules_data.get("rules", []):
        if not isinstance(rule, dict):
            continue

        rule_terms = collect_rule_terms(rule)

        if not rule_terms:
            continue

        matched_terms = []

        for term in rule_terms:
            token = normalize_token(term)

            exact_match = token in evidence_set
            phrase_match = contains_any(evidence_text, [term])

            if exact_match or phrase_match:
                matched_terms.append(term)

        if not matched_terms:
            continue

        minimum_matches = safe_float(
            rule.get("minimum_matches"),
            1.0,
        )

        if len(matched_terms) < int(max(1, minimum_matches)):
            continue

        matched_rules.append(
            {
                "rule_id": (
                    rule.get("rule_id")
                    or rule.get("id")
                ),
                "name": (
                    rule.get("name")
                    or rule.get("rule_name")
                ),
                "risk_level": (
                    rule.get("risk_level")
                    or rule.get("severity")
                    or rule.get("priority")
                ),
                "risk_score": get_rule_risk_score(rule),
                "matched_terms": unique_preserve_order(
                    matched_terms
                ),
                "actions": unique_preserve_order(
                    as_list(
                        rule.get("actions")
                        or rule.get("recommended_actions")
                    )
                ),
            }
        )

    return matched_rules


# ============================================================
# HAZARD BUILDERS
# ============================================================

def create_hazard(
    hazard_type: str,
    source: str,
    description: str,
    risk_score: float,
    confidence: float,
    entity: Optional[str] = None,
    relation: Optional[str] = None,
    target: Optional[str] = None,
    recommended_actions: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    risk_score = round(
        clamp_confidence(risk_score),
        2,
    )
    confidence = round(
        clamp_confidence(confidence),
        2,
    )

    return {
        "hazard_id": None,
        "hazard_type": normalize_token(hazard_type),
        "source": source,
        "entity": entity,
        "relation": relation,
        "target": target,
        "description": description,
        "risk_score": risk_score,
        "risk_level": risk_level_from_score(risk_score),
        "confidence": confidence,
        "recommended_actions": unique_preserve_order(
            recommended_actions or []
        ),
        "evidence": unique_preserve_order(
            evidence or []
        ),
    }


def detect_event_hazards(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hazards = []

    for node in extract_event_nodes(context_graph):
        event_type = normalize_token(
            node.get("event_type")
            or node.get("type")
            or "unknown"
        )

        if event_type not in HAZARD_EVENT_TYPES:
            continue

        confidence = safe_float(
            node.get("confidence"),
            0.85,
        )

        if event_type in CRITICAL_EVENT_TYPES:
            score = 0.90
            actions = [
                "issue_immediate_alert",
                "activate_emergency_protocol",
                "guide_user_to_safety",
            ]
        else:
            score = 0.65
            actions = [
                "warn_user",
                "monitor_hazard",
                "provide_safe_direction",
            ]

        hazards.append(
            create_hazard(
                hazard_type=event_type,
                source="context_graph_event",
                description=(
                    f"Hazard-related event detected: {event_type}"
                ),
                risk_score=score,
                confidence=confidence,
                entity=normalize_token(node.get("id")),
                recommended_actions=actions,
                evidence=[f"event_type={event_type}"],
            )
        )

    return hazards


def detect_relation_hazards(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hazards = []

    for edge in find_hazard_relations(context_graph):
        source = edge["source"]
        relation = edge["relation"]
        target = edge["target"]
        confidence = edge["confidence"]

        risk_score = 0.65

        if relation in {
            "colliding_with",
            "collision_with",
            "falling_towards",
            "on_fire",
            "emitting_smoke",
        }:
            risk_score = 0.85

        hazards.append(
            create_hazard(
                hazard_type="relational_hazard",
                source="context_graph_relation",
                description=(
                    f"{source} is {relation} {target}"
                ),
                risk_score=risk_score,
                confidence=confidence,
                entity=source,
                relation=relation,
                target=target,
                recommended_actions=[
                    "warn_user",
                    "avoid_hazard",
                    "reassess_path",
                ],
                evidence=[
                    f"{source} {relation} {target}"
                ],
            )
        )

    return hazards


def detect_explicit_node_hazards(
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hazards = []

    for node in find_explicit_hazard_nodes(context_graph):
        node_id = normalize_token(node.get("id"))
        entity_type = normalize_token(
            node.get("entity_type")
            or node.get("type")
        )
        event_type = normalize_token(node.get("event_type"))
        category = normalize_token(node.get("category"))

        if (
            node_id in NORMAL_OBJECT_TERMS
            and entity_type not in HAZARD_ENTITY_TYPES
            and category not in {"hazard", "danger", "risk"}
            and event_type not in HAZARD_EVENT_TYPES
        ):
            continue

        confidence = safe_float(
            node.get("confidence"),
            0.80,
        )

        severity = normalize_token(
            node.get("severity")
            or node.get("risk_level")
            or ""
        )

        risk_score = (
            risk_score_from_level(severity)
            if severity
            else 0.60
        )

        hazards.append(
            create_hazard(
                hazard_type=(
                    event_type
                    or entity_type
                    or category
                    or "environmental_hazard"
                ),
                source="context_graph_node",
                description=(
                    f"Explicit hazard node detected: {node_id}"
                ),
                risk_score=risk_score,
                confidence=confidence,
                entity=node_id,
                recommended_actions=[
                    "warn_user",
                    "monitor_hazard",
                    "provide_safe_direction",
                ],
                evidence=[
                    f"entity_type={entity_type}",
                    f"category={category}",
                    f"event_type={event_type}",
                ],
            )
        )

    return hazards


def detect_audio_hazards(
    situation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hazards = []

    for cue in get_audio_cues(situation):
        cue_text = cue.replace("_", " ")

        if not contains_any(cue_text, HAZARD_TERMS):
            continue

        critical = contains_any(
            cue_text,
            {
                "scream",
                "help call",
                "fire alarm",
                "explosion",
            },
        )

        score = 0.85 if critical else 0.60

        hazards.append(
            create_hazard(
                hazard_type="audio_hazard",
                source="audio_context",
                description=(
                    f"Hazard-related audio cue detected: {cue}"
                ),
                risk_score=score,
                confidence=0.80,
                entity=cue,
                recommended_actions=[
                    "alert_user",
                    "identify_audio_source",
                    "monitor_environment",
                ],
                evidence=[f"audio_cue={cue}"],
            )
        )

    return hazards


def detect_safety_context_hazard(
    situation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    safety_status = get_safety_status(situation)
    existing_risk = get_existing_risk_level(situation)

    if (
        safety_status == "safe"
        and existing_risk in {"none", "very_low", "low"}
    ):
        return []

    score = max(
        risk_score_from_level(existing_risk),
        0.50 if safety_status == "caution" else 0.0,
        0.75 if safety_status == "unsafe" else 0.0,
        0.95 if safety_status in {"critical", "danger"} else 0.0,
    )

    return [
        create_hazard(
            hazard_type="safety_context_risk",
            source="situation_understanding",
            description=(
                "Situation understanding reports "
                f"safety_status={safety_status} and "
                f"risk_level={existing_risk}"
            ),
            risk_score=score,
            confidence=get_context_confidence(situation),
            recommended_actions=[
                "increase_safety_monitoring",
                "warn_user_if_risk_persists",
            ],
            evidence=[
                f"safety_status={safety_status}",
                f"risk_level={existing_risk}",
            ],
        )
    ]


def detect_rule_based_hazards(
    risk_rules: Dict[str, Any],
    emergency_rules: Dict[str, Any],
    navigation_rules: Dict[str, Any],
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_reasoning: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    graph_terms = extract_graph_terms(context_graph)
    event_types = extract_event_types(context_graph)

    evidence_terms = unique_preserve_order(
        graph_terms
        + event_types
        + get_nearby_entities(situation)
        + get_audio_cues(situation)
        + get_user_activities(situation)
        + [
            get_environment_type(situation),
            get_scene_type(situation),
            get_safety_status(situation),
            get_existing_risk_level(situation),
            get_attention_focus(
                situation,
                cognitive_state,
            ),
            normalize_token(
                intent_reasoning.get(
                    "primary_intent",
                    "",
                )
            ),
            normalize_token(
                intent_reasoning.get(
                    "destination",
                    "",
                )
            ),
        ]
    )

    evidence_text = " ".join(
        term.replace("_", " ")
        for term in evidence_terms
        if term
    )

    all_matches = []

    for source_name, rules_data in (
        ("risk_rules", risk_rules),
        ("emergency_rules", emergency_rules),
        ("navigation_rules", navigation_rules),
    ):
        matched = match_contextual_rules(
            rules_data,
            evidence_terms,
            evidence_text,
        )

        for rule in matched:
            rule["source_name"] = source_name
            all_matches.append(rule)

    hazards = []

    for rule in all_matches:
        score = rule.get("risk_score", 0.0)

        if score <= 0.0:
            continue

        matched_terms = [
            normalize_token(term)
            for term in rule.get("matched_terms", [])
        ]

        meaningful_hazard_match = any(
            contains_any(term, HAZARD_TERMS)
            or term in HAZARD_EVENT_TYPES
            or term in HAZARD_ENTITY_TYPES
            for term in matched_terms
        )

        if not meaningful_hazard_match:
            continue

        hazards.append(
            create_hazard(
                hazard_type="rule_based_hazard",
                source=rule["source_name"],
                description=(
                    f"Matched hazard rule: {rule.get('name')}"
                ),
                risk_score=score,
                confidence=0.75,
                recommended_actions=[
                    str(action)
                    for action in rule.get("actions", [])
                    if str(action).strip()
                ],
                evidence=[
                    f"matched_term={term}"
                    for term in rule.get(
                        "matched_terms",
                        [],
                    )
                ],
            )
        )

    return hazards


# ============================================================
# FALSE-POSITIVE FILTERING
# ============================================================

def is_supported_hazard(
    hazard: Dict[str, Any],
    situation: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> bool:
    source = hazard.get("source")
    entity = normalize_token(hazard.get("entity"))
    relation = normalize_token(hazard.get("relation"))
    evidence_text = " ".join(
        str(item)
        for item in hazard.get("evidence", [])
    )

    if source in {
        "context_graph_event",
        "context_graph_relation",
        "audio_context",
        "situation_understanding",
    }:
        return True

    if relation in HAZARD_RELATIONS:
        return True

    if contains_any(evidence_text, HAZARD_TERMS):
        return True

    if entity in NORMAL_OBJECT_TERMS:
        return False

    node_map = build_node_map(context_graph)
    node = node_map.get(entity, {})

    entity_type = normalize_token(
        node.get("entity_type")
        or node.get("type")
    )
    category = normalize_token(node.get("category"))

    if (
        entity_type in HAZARD_ENTITY_TYPES
        or category in {"hazard", "danger", "risk"}
    ):
        return True

    return False


def merge_duplicate_hazards(
    hazards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for hazard in hazards:
        key = (
            normalize_token(hazard.get("hazard_type")),
            normalize_token(hazard.get("entity")),
            normalize_token(hazard.get("relation")),
            normalize_token(hazard.get("target")),
        )

        if key not in grouped:
            grouped[key] = hazard.copy()
            continue

        existing = grouped[key]

        existing["risk_score"] = max(
            safe_float(existing.get("risk_score")),
            safe_float(hazard.get("risk_score")),
        )
        existing["confidence"] = max(
            safe_float(existing.get("confidence")),
            safe_float(hazard.get("confidence")),
        )
        existing["risk_level"] = risk_level_from_score(
            existing["risk_score"]
        )
        existing["recommended_actions"] = (
            unique_preserve_order(
                as_list(
                    existing.get("recommended_actions")
                )
                + as_list(
                    hazard.get("recommended_actions")
                )
            )
        )
        existing["evidence"] = unique_preserve_order(
            as_list(existing.get("evidence"))
            + as_list(hazard.get("evidence"))
        )

    merged = list(grouped.values())

    for index, hazard in enumerate(merged, start=1):
        hazard["hazard_id"] = f"HZD_{index:03d}"

    return merged


# ============================================================
# RISK AGGREGATION
# ============================================================

def calculate_overall_risk(
    hazards: List[Dict[str, Any]],
    situation: Dict[str, Any],
) -> Tuple[float, str]:
    if not hazards:
        return 0.0, "low"

    weighted_scores = []

    for hazard in hazards:
        risk_score = safe_float(
            hazard.get("risk_score"),
            0.0,
        )
        confidence = safe_float(
            hazard.get("confidence"),
            0.5,
        )

        weighted_scores.append(
            risk_score * confidence
        )

    highest = max(weighted_scores)
    average = sum(weighted_scores) / len(weighted_scores)

    overall = (
        0.70 * highest
        + 0.30 * average
    )

    environment_type = get_environment_type(situation)
    environment_base = ENVIRONMENT_BASE_RISK.get(
        environment_type,
        0.05,
    )

    overall = max(
        overall,
        environment_base
        if overall > 0.0
        else 0.0,
    )

    overall = round(
        clamp_confidence(overall),
        2,
    )

    return overall, risk_level_from_score(overall)


def infer_output_safety_status(
    risk_level: str,
) -> str:
    if risk_level in {"critical", "high"}:
        return "unsafe"

    if risk_level == "medium":
        return "caution"

    return "safe"


def collect_recommended_actions(
    hazards: List[Dict[str, Any]],
) -> List[str]:
    actions = []

    for hazard in hazards:
        actions.extend(
            hazard.get(
                "recommended_actions",
                [],
            )
        )

    return unique_preserve_order(actions)


def generate_hazard_summary(
    hazards: List[Dict[str, Any]],
    overall_risk_score: float,
    overall_risk_level: str,
    safety_status: str,
) -> str:
    if not hazards:
        return (
            "No supported hazards were detected. "
            "The overall risk level is low and the "
            "current safety status is safe."
        )

    hazard_types = unique_preserve_order(
        hazard.get("hazard_type", "unknown")
        for hazard in hazards
    )

    return (
        f"{len(hazards)} supported hazard(s) were detected: "
        f"{', '.join(hazard_types)}. "
        f"The overall risk score is {overall_risk_score}, "
        f"corresponding to a {overall_risk_level} risk level. "
        f"The current safety status is {safety_status}."
    )


# ============================================================
# MAIN REASONING
# ============================================================

def reason_about_hazards(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_reasoning: Dict[str, Any],
    context_graph: Dict[str, Any],
    risk_rules: Dict[str, Any],
    emergency_rules: Dict[str, Any],
    navigation_rules: Dict[str, Any],
) -> Dict[str, Any]:
    log_section("Hazard Reasoning")

    detected_hazards = []

    detected_hazards.extend(
        detect_event_hazards(context_graph)
    )
    detected_hazards.extend(
        detect_relation_hazards(context_graph)
    )
    detected_hazards.extend(
        detect_explicit_node_hazards(context_graph)
    )
    detected_hazards.extend(
        detect_audio_hazards(situation)
    )
    detected_hazards.extend(
        detect_safety_context_hazard(situation)
    )
    detected_hazards.extend(
        detect_rule_based_hazards(
            risk_rules,
            emergency_rules,
            navigation_rules,
            situation,
            cognitive_state,
            intent_reasoning,
            context_graph,
        )
    )

    supported_hazards = [
        hazard
        for hazard in detected_hazards
        if is_supported_hazard(
            hazard,
            situation,
            context_graph,
        )
    ]

    hazards = merge_duplicate_hazards(
        supported_hazards
    )

    overall_risk_score, overall_risk_level = (
        calculate_overall_risk(
            hazards,
            situation,
        )
    )

    safety_status = infer_output_safety_status(
        overall_risk_level
    )

    recommended_actions = collect_recommended_actions(
        hazards
    )

    result = {
        "timestamp": str(datetime.now()),
        "hazard_count": len(hazards),
        "hazards": hazards,
        "overall_risk_score": overall_risk_score,
        "overall_risk_level": overall_risk_level,
        "risk_confidence_label": confidence_label(
            overall_risk_score
        ),
        "safety_status": safety_status,
        "recommended_actions": recommended_actions,
        "summary": generate_hazard_summary(
            hazards,
            overall_risk_score,
            overall_risk_level,
            safety_status,
        ),
    }

    log_info(f"Hazard Count: {len(hazards)}")
    log_info(
        f"Overall Risk Score: {overall_risk_score}"
    )
    log_info(
        f"Overall Risk Level: {overall_risk_level}"
    )
    log_info(f"Safety Status: {safety_status}")

    log_success("Hazard Reasoning Complete")

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_required_inputs(
    situation: Dict[str, Any],
    cognitive_state: Dict[str, Any],
    intent_reasoning: Dict[str, Any],
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

    if not intent_reasoning:
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
    module_start("HAZARD REASONER")

    os.makedirs(
        LAYER4_OUTPUT_DIR,
        exist_ok=True,
    )

    situation = load_json(SITUATION_PATH)
    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
    )
    intent_reasoning = load_json(
        INTENT_REASONING_PATH
    )
    context_graph = load_json(
        CONTEXT_GRAPH_PATH
    )

    if not validate_required_inputs(
        situation,
        cognitive_state,
        intent_reasoning,
        context_graph,
    ):
        module_end("HAZARD REASONER")
        return

    risk_rules = load_json(RISK_RULES_PATH)
    emergency_rules = load_json(
        EMERGENCY_RULES_PATH
    )
    navigation_rules = load_json(
        NAVIGATION_RULES_PATH
    )

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

    if not navigation_rules:
        log_warning(
            "navigation_rules.json missing or empty. "
            "Continuing without navigation-rule enrichment."
        )
        navigation_rules = {}

    hazard_result = reason_about_hazards(
        situation,
        cognitive_state,
        intent_reasoning,
        context_graph,
        risk_rules,
        emergency_rules,
        navigation_rules,
    )

    save_json(
        hazard_result,
        HAZARD_OUTPUT_PATH,
    )

    log_info(
        f"Hazard output saved to: {HAZARD_OUTPUT_PATH}"
    )

    module_end("HAZARD REASONER")


if __name__ == "__main__":
    main()