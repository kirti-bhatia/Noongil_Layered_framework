"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Intent Reasoner
============================================================
Purpose:
Infer user intent from nested Layer 4 outputs, the context graph,
and optional knowledge rules. The module supports navigation,
safety, emergency, reading, identification, search, information,
communication, memory, activity support, and general awareness.
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

SITUATION_PATH = os.path.join(LAYER4_OUTPUT_DIR, "situation_understanding.json")
COGNITIVE_STATE_PATH = os.path.join(LAYER4_OUTPUT_DIR, "cognitive_state.json")
CONTEXT_GRAPH_PATH = os.path.join(LAYER3_OUTPUT_DIR, "context_graph.json")
INTENT_OUTPUT_PATH = os.path.join(LAYER4_OUTPUT_DIR, "intent_reasoning.json")

COMMON_SENSE_CANDIDATES = [
    os.path.join(KNOWLEDGE_DIR, "common_sense.json"),
    os.path.join(KNOWLEDGE_DIR, "commonsense_rules.json"),
    os.path.join(KNOWLEDGE_DIR, "common_sense_rules.json"),
]
NAVIGATION_RULES_CANDIDATES = [
    os.path.join(KNOWLEDGE_DIR, "navigation_rules.json")
]
EMERGENCY_RULES_CANDIDATES = [
    os.path.join(KNOWLEDGE_DIR, "emergency_rules.json")
]

if LAYER4_DIR not in sys.path:
    sys.path.insert(0, LAYER4_DIR)

from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import clamp_confidence, confidence_label


# ============================================================
# INTENT DEFINITIONS
# ============================================================

INTENT_CATEGORIES = {
    "navigate_to_destination": "navigation",
    "continue_movement": "navigation",
    "avoid_hazard": "safety",
    "stay_safe": "safety",
    "seek_emergency_help": "emergency",
    "read_text": "perception_assistance",
    "identify_object": "perception_assistance",
    "find_person_or_object": "search",
    "request_information": "information",
    "communicate_with_person": "communication",
    "remember_event": "memory",
    "support_current_activity": "activity_assistance",
    "maintain_general_awareness": "general",
}

TEMPORAL_TYPES = {
    "time", "date", "duration", "number", "quantity", "cardinal", "ordinal"
}
TEMPORAL_CATEGORIES = {"time", "temporal"}
TEMPORAL_TERMS = {
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night"
}
GENERIC_TARGETS = {
    "", "none", "null", "unknown", "destination", "location", "place",
    "object", "person", "thing", "something", "somewhere", "user"
}

REQUEST_RELATIONS = {
    "requesting", "requests", "requested", "wants", "want", "needs", "need",
    "goal", "has_goal", "looking_for", "searching_for", "seeking", "targeting",
    "destination", "moving_to", "going_to", "navigate_to", "reach", "approach", "find"
}

NAVIGATION_EVENTS = {
    "navigation_request", "route_request", "destination_request",
    "movement", "travel_event", "wayfinding_request"
}
EMERGENCY_EVENTS = {
    "emergency", "emergency_event", "fire_event", "accident", "fall_event",
    "medical_emergency", "distress_event", "help_request"
}
HAZARD_EVENTS = {
    "hazard_event", "collision_risk", "obstacle_event", "traffic_risk",
    "unsafe_condition", "danger_event"
}
READING_EVENTS = {
    "read_request", "ocr_request", "text_reading", "document_reading", "sign_reading"
}
IDENTIFICATION_EVENTS = {
    "object_identification", "identify_request", "recognition_request", "scene_identification"
}
SEARCH_EVENTS = {
    "search_request", "find_request", "person_search", "object_search",
    "lost_person", "lost_object"
}
INFORMATION_EVENTS = {
    "information_request", "question_event", "query_event", "explanation_request"
}
COMMUNICATION_EVENTS = {
    "conversation_event", "communication_request", "call_request",
    "message_request", "social_interaction"
}
MEMORY_EVENTS = {
    "memory_request", "reminder_request", "remember_event", "recall_request"
}

EMERGENCY_TERMS = {
    "emergency", "fire", "smoke", "scream", "help_call", "help call",
    "accident", "bleeding", "collapsed", "unconscious", "medical emergency"
}
HAZARD_TERMS = {
    "hazard", "obstacle", "collision", "vehicle", "traffic", "stairs",
    "drop", "edge", "wet floor", "blocked path", "unsafe", "risk"
}
READ_TERMS = {
    "read", "text", "ocr", "sign", "label", "document", "menu",
    "medicine", "board", "screen"
}
IDENTIFY_TERMS = {
    "identify", "recognize", "what is", "who is", "describe object", "describe scene"
}
SEARCH_TERMS = {"find", "locate", "search", "looking for", "lost", "where is"}
INFORMATION_TERMS = {"what", "when", "where", "why", "how", "which", "information", "explain"}
COMMUNICATION_TERMS = {"call", "message", "contact", "speak", "talk", "communicate"}
MEMORY_TERMS = {"remember", "remind", "recall", "save this", "store this", "memorize"}
MOVEMENT_TERMS = {"walking", "moving", "running", "traveling", "travelling", "crossing"}


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().replace("-", "_").split())


def normalize_token(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    output = []
    for value in values:
        marker = str(value)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_existing_path(paths: Sequence[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_optional_json(paths: Sequence[str]) -> Dict[str, Any]:
    path = first_existing_path(paths)
    if not path:
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def get_nested(data: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
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
    normalized = " " + normalize_text(text).replace("_", " ") + " "
    for term in terms:
        term_text = normalize_text(term).replace("_", " ")
        if term_text and term_text in normalized:
            return True
    return False


def add_evidence(evidence, intent, source, description, weight):
    evidence[intent].append({
        "source": source,
        "description": description,
        "weight": round(weight, 3),
    })


# ============================================================
# SITUATION ACCESSORS
# ============================================================

def get_user_goal(situation):
    return normalize_token(get_first_available(
        situation,
        [("user_state", "goal"), ("cognitive_context", "primary_goal")],
        ("user_goal", "goal", "primary_goal"),
        "",
    ))


def get_user_targets(situation):
    values = get_first_available(
        situation,
        [("user_state", "targets")],
        ("user_targets", "targets"),
        [],
    )
    return [normalize_token(v) for v in as_list(values) if normalize_token(v)]


def get_user_activities(situation):
    values = get_first_available(
        situation,
        [("user_state", "activities")],
        ("user_activities", "activities"),
        [],
    )
    return [normalize_token(v) for v in as_list(values) if normalize_token(v)]


def get_environment_type(situation):
    return normalize_token(get_first_available(
        situation,
        [("environment_context", "environment_type")],
        ("environment_type",),
        "",
    ))


def get_scene_type(situation):
    return normalize_token(get_first_available(
        situation,
        [("environment_context", "scene_type")],
        ("scene_type",),
        "",
    ))


def get_nearby_entities(situation):
    values = get_first_available(
        situation,
        [("environment_context", "nearby_entities")],
        ("nearby_entities",),
        [],
    )
    return [normalize_token(v) for v in as_list(values) if normalize_token(v)]


def get_audio_cues(situation):
    values = get_first_available(
        situation,
        [("environment_context", "audio_cues")],
        ("audio_cues", "sounds"),
        [],
    )
    output = []
    for item in as_list(values):
        if isinstance(item, dict):
            item = item.get("label") or item.get("name") or item.get("id")
        token = normalize_token(item)
        if token:
            output.append(token)
    return output


def get_safety_status(situation):
    return normalize_token(get_first_available(
        situation,
        [("safety_context", "safety_status")],
        ("safety_status",),
        "safe",
    ))


def get_risk_level(situation):
    return normalize_token(get_first_available(
        situation,
        [("safety_context", "risk_level")],
        ("risk_level",),
        "low",
    ))


def get_situation_type(situation):
    return normalize_token(situation.get("situation_type", ""))


def get_context_confidence(situation):
    return safe_float(situation.get("context_confidence", 0.5), 0.5)


def get_attention_focus(situation, cognitive_state):
    value = get_nested(situation, ("cognitive_context", "attention_focus"), None)
    if value in (None, ""):
        value = cognitive_state.get("attention_focus", "")
    return normalize_token(value)


def get_primary_goal(situation, cognitive_state):
    return get_user_goal(situation) or normalize_token(cognitive_state.get("primary_goal", ""))


def get_cognitive_priority(situation, cognitive_state):
    value = get_nested(situation, ("cognitive_context", "cognitive_priority"), None)
    if value in (None, ""):
        value = cognitive_state.get("cognitive_priority", "low")
    return normalize_token(value)


def get_reasoning_mode(situation, cognitive_state):
    value = get_nested(situation, ("cognitive_context", "reasoning_mode"), None)
    if value in (None, ""):
        value = cognitive_state.get("reasoning_mode", "general")
    return normalize_token(value)


# ============================================================
# GRAPH HELPERS
# ============================================================

def build_node_map(context_graph):
    node_map = {}
    for node in context_graph.get("nodes", []):
        if isinstance(node, dict):
            node_id = normalize_token(node.get("id"))
            if node_id:
                node_map[node_id] = node
    return node_map


def get_node_type(node):
    return normalize_token(node.get("entity_type") or node.get("type") or node.get("node_type") or "")


def get_node_category(node):
    return normalize_token(node.get("category", ""))


def is_valid_target(target, node_map):
    target_key = normalize_token(target)
    if target_key in GENERIC_TARGETS or target_key.startswith("evt_"):
        return False
    if target_key in TEMPORAL_TERMS:
        return False
    node = node_map.get(target_key, {})
    if get_node_type(node) in TEMPORAL_TYPES:
        return False
    if get_node_category(node) in TEMPORAL_CATEGORIES | {"event"}:
        return False
    return True


def extract_user_requested_targets(context_graph):
    node_map = build_node_map(context_graph)
    scores = defaultdict(float)
    weights = {
        "navigate_to": 1.00, "reach": 1.00, "destination": 0.95,
        "moving_to": 0.90, "going_to": 0.90, "looking_for": 0.90,
        "searching_for": 0.90, "find": 0.90, "requesting": 0.80,
        "requests": 0.80, "wants": 0.75, "needs": 0.75,
        "goal": 0.75, "has_goal": 0.75, "seeking": 0.70,
        "targeting": 0.70, "approach": 0.65,
    }
    for edge in context_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = normalize_token(edge.get("source"))
        relation = normalize_token(edge.get("relation") or edge.get("type") or edge.get("label"))
        target = normalize_token(edge.get("target"))
        if source != "user" or relation not in REQUEST_RELATIONS:
            continue
        if not is_valid_target(target, node_map):
            continue
        confidence = safe_float(edge.get("confidence"), 1.0)
        scores[target] += weights.get(relation, 0.60) * max(0.1, confidence)
    return [item[0] for item in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def extract_event_types(context_graph):
    output = []
    for node in context_graph.get("nodes", []):
        if isinstance(node, dict) and get_node_category(node) == "event":
            value = normalize_token(node.get("event_type") or node.get("type") or "unknown")
            if value:
                output.append(value)
    return unique_preserve_order(output)


def extract_activity_nodes(context_graph):
    output = []
    for node in context_graph.get("nodes", []):
        if isinstance(node, dict) and get_node_type(node) == "activity":
            value = normalize_token(node.get("id"))
            if value:
                output.append(value)
    return unique_preserve_order(output)


def extract_graph_terms(context_graph):
    terms = []
    for node in context_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for key in ("id", "label", "name", "entity_type", "category", "event_type"):
            value = normalize_token(node.get(key))
            if value:
                terms.append(value)
    for edge in context_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        for key in ("source", "target", "relation", "type", "label"):
            value = normalize_token(edge.get(key))
            if value:
                terms.append(value)
    return unique_preserve_order(terms)


# ============================================================
# RULE HELPERS
# ============================================================

def collect_rule_keywords(rule):
    keywords = []
    for key in ("keywords", "triggers", "conditions", "event_types", "situations", "contexts", "intents"):
        for item in as_list(rule.get(key)):
            if isinstance(item, dict):
                for value in item.values():
                    if normalize_text(value):
                        keywords.append(normalize_text(value))
            elif normalize_text(item):
                keywords.append(normalize_text(item))
    for key in ("name", "description", "rule_name"):
        if normalize_text(rule.get(key)):
            keywords.append(normalize_text(rule.get(key)))
    return unique_preserve_order(keywords)


def match_rules(rules_data, primary_intent, evidence_text, allowed_priorities=None):
    matched = []
    for rule in rules_data.get("rules", []):
        if not isinstance(rule, dict):
            continue
        priority = normalize_token(rule.get("priority", ""))
        if allowed_priorities and priority not in allowed_priorities:
            continue
        rule_intents = {
            normalize_token(v)
            for v in as_list(rule.get("intent") or rule.get("intents") or rule.get("applies_to"))
            if normalize_token(v)
        }
        intent_match = not rule_intents or primary_intent in rule_intents
        keywords = collect_rule_keywords(rule)
        keyword_match = not keywords or any(contains_any(evidence_text, [k]) for k in keywords)
        name = normalize_text(rule.get("name") or rule.get("rule_name") or "")
        semantic_match = (
            primary_intent.replace("_", " ") in name
            or (primary_intent == "navigate_to_destination" and contains_any(name, {"navigation", "route", "destination", "navigate"}))
            or (primary_intent == "seek_emergency_help" and contains_any(name, {"emergency", "critical", "distress", "help"}))
            or (primary_intent in {"avoid_hazard", "stay_safe"} and contains_any(name, {"hazard", "risk", "safety", "obstacle"}))
        )
        if intent_match and (keyword_match or semantic_match):
            matched.append({
                "rule_id": rule.get("rule_id") or rule.get("id"),
                "name": rule.get("name") or rule.get("rule_name"),
                "priority": rule.get("priority"),
                "actions": unique_preserve_order(as_list(rule.get("actions") or rule.get("recommended_actions"))),
            })
    return matched


# ============================================================
# EVIDENCE BUILDING
# ============================================================

def build_evidence_text(situation, cognitive_state, context_graph):
    parts = []
    parts += extract_event_types(context_graph)
    parts += extract_activity_nodes(context_graph)
    parts += extract_graph_terms(context_graph)
    parts += get_user_targets(situation)
    parts += get_user_activities(situation)
    parts += get_nearby_entities(situation)
    parts += get_audio_cues(situation)
    parts += [
        get_primary_goal(situation, cognitive_state),
        get_attention_focus(situation, cognitive_state),
        get_situation_type(situation),
        get_safety_status(situation),
        get_risk_level(situation),
        get_environment_type(situation),
        get_scene_type(situation),
        get_reasoning_mode(situation, cognitive_state),
        normalize_text(situation.get("human_readable_summary", "")),
        normalize_text(cognitive_state.get("summary", "")),
    ]
    return " ".join(normalize_text(v).replace("_", " ") for v in parts if normalize_text(v))


def infer_intent_scores(situation, cognitive_state, context_graph, common_sense):
    scores = defaultdict(float)
    evidence = defaultdict(list)

    event_types = set(extract_event_types(context_graph))
    graph_targets = extract_user_requested_targets(context_graph)
    situation_targets = get_user_targets(situation)
    activities = set(get_user_activities(situation) + extract_activity_nodes(context_graph))
    attention = get_attention_focus(situation, cognitive_state)
    goal = get_primary_goal(situation, cognitive_state)
    situation_type = get_situation_type(situation)
    safety = get_safety_status(situation)
    risk = get_risk_level(situation)
    reasoning_mode = get_reasoning_mode(situation, cognitive_state)
    text = build_evidence_text(situation, cognitive_state, context_graph)

    emergency_events = event_types & EMERGENCY_EVENTS
    if emergency_events:
        scores["seek_emergency_help"] += 4.5
        add_evidence(evidence, "seek_emergency_help", "context_graph", f"Emergency events: {sorted(emergency_events)}", 4.5)
    if attention == "emergency":
        scores["seek_emergency_help"] += 3.0
        add_evidence(evidence, "seek_emergency_help", "cognitive_state", "Attention focus is emergency", 3.0)
    if contains_any(text, EMERGENCY_TERMS):
        scores["seek_emergency_help"] += 2.0
        add_evidence(evidence, "seek_emergency_help", "multimodal_context", "Emergency-related signals detected", 2.0)
    if safety in {"critical", "danger"} or risk in {"critical", "extreme"}:
        scores["seek_emergency_help"] += 3.0
        add_evidence(evidence, "seek_emergency_help", "safety_context", f"Safety={safety}, risk={risk}", 3.0)

    hazard_events = event_types & HAZARD_EVENTS
    if hazard_events:
        scores["avoid_hazard"] += 4.0
        add_evidence(evidence, "avoid_hazard", "context_graph", f"Hazard events: {sorted(hazard_events)}", 4.0)
    if safety == "unsafe":
        scores["avoid_hazard"] += 3.0
        add_evidence(evidence, "avoid_hazard", "safety_context", "Environment is unsafe", 3.0)
    if risk in {"high", "severe"}:
        scores["avoid_hazard"] += 2.5
        add_evidence(evidence, "avoid_hazard", "safety_context", f"Risk level is {risk}", 2.5)
    if contains_any(text, HAZARD_TERMS):
        scores["avoid_hazard"] += 1.5
        add_evidence(evidence, "avoid_hazard", "multimodal_context", "Hazard-related signals detected", 1.5)
    if attention == "safety":
        scores["stay_safe"] += 2.0
        add_evidence(evidence, "stay_safe", "cognitive_state", "Attention focus is safety", 2.0)
    if safety == "caution" or risk == "medium":
        scores["stay_safe"] += 2.5
        add_evidence(evidence, "stay_safe", "safety_context", f"Safety={safety}, risk={risk}", 2.5)

    navigation_events = event_types & NAVIGATION_EVENTS
    if navigation_events:
        scores["navigate_to_destination"] += 3.5
        add_evidence(evidence, "navigate_to_destination", "context_graph", f"Navigation events: {sorted(navigation_events)}", 3.5)
    if graph_targets:
        scores["navigate_to_destination"] += 3.0
        add_evidence(evidence, "navigate_to_destination", "context_graph", f"Requested targets: {graph_targets}", 3.0)
    if situation_targets:
        scores["navigate_to_destination"] += 2.0
        add_evidence(evidence, "navigate_to_destination", "situation_understanding", f"Situation targets: {situation_targets}", 2.0)
    if attention == "navigation":
        scores["navigate_to_destination"] += 2.5
        add_evidence(evidence, "navigate_to_destination", "cognitive_state", "Attention focus is navigation", 2.5)
    if situation_type == "navigation_assistance":
        scores["navigate_to_destination"] += 2.5
        add_evidence(evidence, "navigate_to_destination", "situation_understanding", "Navigation-assistance situation", 2.5)
    if goal.startswith(("reach_", "navigate_", "go_to_", "find_route_")):
        scores["navigate_to_destination"] += 3.0
        add_evidence(evidence, "navigate_to_destination", "cognitive_state", f"Navigation goal: {goal}", 3.0)

    if activities & {"walking", "running", "moving", "travelling", "traveling"}:
        scores["continue_movement"] += 2.5
        add_evidence(evidence, "continue_movement", "activity_context", f"Movement activities: {sorted(activities)}", 2.5)
    if "movement" in event_types:
        scores["continue_movement"] += 2.0
        add_evidence(evidence, "continue_movement", "context_graph", "Movement event detected", 2.0)
    if contains_any(text, MOVEMENT_TERMS):
        scores["continue_movement"] += 1.0

    reading_events = event_types & READING_EVENTS
    if reading_events:
        scores["read_text"] += 4.0
        add_evidence(evidence, "read_text", "context_graph", f"Reading events: {sorted(reading_events)}", 4.0)
    if goal.startswith(("read_", "scan_", "recognize_text_")):
        scores["read_text"] += 3.5
        add_evidence(evidence, "read_text", "cognitive_state", f"Reading goal: {goal}", 3.5)
    if attention in {"text", "reading", "ocr"}:
        scores["read_text"] += 2.5
    if contains_any(text, READ_TERMS):
        scores["read_text"] += 1.2

    identification_events = event_types & IDENTIFICATION_EVENTS
    if identification_events:
        scores["identify_object"] += 4.0
        add_evidence(evidence, "identify_object", "context_graph", f"Identification events: {sorted(identification_events)}", 4.0)
    if goal.startswith(("identify_", "recognize_", "describe_")):
        scores["identify_object"] += 3.5
        add_evidence(evidence, "identify_object", "cognitive_state", f"Identification goal: {goal}", 3.5)
    if attention in {"object", "object_identification", "scene", "recognition"}:
        scores["identify_object"] += 2.5
    if contains_any(text, IDENTIFY_TERMS):
        scores["identify_object"] += 1.2

    search_events = event_types & SEARCH_EVENTS
    if search_events:
        scores["find_person_or_object"] += 4.0
        add_evidence(evidence, "find_person_or_object", "context_graph", f"Search events: {sorted(search_events)}", 4.0)
    if goal.startswith(("find_", "locate_", "search_for_")):
        scores["find_person_or_object"] += 3.5
        add_evidence(evidence, "find_person_or_object", "cognitive_state", f"Search goal: {goal}", 3.5)
    if attention in {"search", "person_search", "object_search"}:
        scores["find_person_or_object"] += 2.5
    if contains_any(text, SEARCH_TERMS):
        scores["find_person_or_object"] += 1.0

    information_events = event_types & INFORMATION_EVENTS
    if information_events:
        scores["request_information"] += 4.0
        add_evidence(evidence, "request_information", "context_graph", f"Information events: {sorted(information_events)}", 4.0)
    if goal.startswith(("ask_", "know_", "explain_", "get_information_")):
        scores["request_information"] += 3.0
    if attention in {"information", "question", "query", "explanation"}:
        scores["request_information"] += 2.5
    if contains_any(text, INFORMATION_TERMS):
        scores["request_information"] += 0.8

    communication_events = event_types & COMMUNICATION_EVENTS
    if communication_events:
        scores["communicate_with_person"] += 2.5
        add_evidence(evidence, "communicate_with_person", "context_graph", f"Communication events: {sorted(communication_events)}", 2.5)
    if goal.startswith(("call_", "message_", "contact_", "talk_to_")):
        scores["communicate_with_person"] += 3.5
    if attention in {"communication", "conversation", "social"}:
        scores["communicate_with_person"] += 2.0
    if contains_any(text, COMMUNICATION_TERMS):
        scores["communicate_with_person"] += 1.0

    memory_events = event_types & MEMORY_EVENTS
    if memory_events:
        scores["remember_event"] += 4.0
        add_evidence(evidence, "remember_event", "context_graph", f"Memory events: {sorted(memory_events)}", 4.0)
    if goal.startswith(("remember_", "remind_", "recall_", "store_")):
        scores["remember_event"] += 3.5
    if attention in {"memory", "reminder", "recall"}:
        scores["remember_event"] += 2.5
    if contains_any(text, MEMORY_TERMS):
        scores["remember_event"] += 1.0

    if activities and not activities & {"walking", "running", "moving", "traveling", "travelling"}:
        scores["support_current_activity"] += 1.2
        add_evidence(evidence, "support_current_activity", "activity_context", f"Current activities: {sorted(activities)}", 1.2)
    if reasoning_mode in {"assistance", "activity_support", "guidance"}:
        scores["support_current_activity"] += 0.8

    scores["maintain_general_awareness"] += 0.6
    add_evidence(evidence, "maintain_general_awareness", "fallback", "General awareness fallback", 0.6)

    return dict(scores), dict(evidence)


def select_primary_and_secondary(scores):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return "maintain_general_awareness", []
    primary, primary_score = ranked[0]
    secondary = []
    for intent, score in ranked[1:]:
        if score < 1.0:
            continue
        if primary_score > 0 and score / primary_score < 0.45:
            continue
        secondary.append({
            "intent": intent,
            "category": INTENT_CATEGORIES.get(intent, "general"),
            "score": round(score, 2),
        })
        if len(secondary) == 3:
            break
    return primary, secondary


# ============================================================
# TARGET AND DESTINATION
# ============================================================

def goal_to_target(goal):
    prefixes = (
        "reach_", "navigate_to_", "go_to_", "find_route_to_", "find_",
        "locate_", "search_for_", "read_", "identify_", "recognize_",
        "call_", "message_", "contact_"
    )
    for prefix in prefixes:
        if goal.startswith(prefix):
            value = goal[len(prefix):].strip("_")
            return value or None
    return None


def infer_destination_and_target(primary_intent, situation, cognitive_state, context_graph):
    graph_targets = extract_user_requested_targets(context_graph)
    situation_targets = get_user_targets(situation)
    goal_target = goal_to_target(get_primary_goal(situation, cognitive_state))
    candidates = unique_preserve_order(graph_targets + situation_targets + ([goal_target] if goal_target else []))
    target = candidates[0] if candidates else None
    if primary_intent in {"navigate_to_destination", "continue_movement"}:
        return target, target
    if primary_intent in {
        "read_text", "identify_object", "find_person_or_object",
        "communicate_with_person", "remember_event", "request_information"
    }:
        return None, target
    return None, target


# ============================================================
# ASSISTANCE, CONFIDENCE, URGENCY
# ============================================================

def infer_required_assistance(
    primary_intent, situation, cognitive_state, destination, target,
    matched_navigation_rules, matched_emergency_rules
):
    assistance_map = {
        "navigate_to_destination": [
            "route_guidance", "destination_tracking", "step_by_step_navigation",
            "obstacle_awareness", "route_replanning"
        ],
        "continue_movement": [
            "movement_tracking", "obstacle_monitoring", "path_continuity_support", "pace_adaptation"
        ],
        "avoid_hazard": [
            "immediate_hazard_alert", "safe_direction_guidance", "obstacle_avoidance", "risk_reassessment"
        ],
        "stay_safe": ["hazard_monitoring", "risk_alert_generation", "safety_guidance"],
        "seek_emergency_help": [
            "emergency_alert", "safe_exit_guidance", "caregiver_notification",
            "location_sharing", "emergency_service_escalation"
        ],
        "read_text": ["text_region_detection", "ocr_processing", "text_summarization", "speech_output"],
        "identify_object": ["object_recognition", "scene_description", "spatial_positioning", "speech_output"],
        "find_person_or_object": ["target_search", "target_tracking", "spatial_guidance", "search_progress_updates"],
        "request_information": ["context_retrieval", "knowledge_lookup", "answer_generation", "speech_output"],
        "communicate_with_person": [
            "contact_resolution", "communication_channel_selection", "message_or_call_assistance", "confirmation_feedback"
        ],
        "remember_event": ["memory_recording", "event_structuring", "temporal_association", "future_recall_support"],
        "support_current_activity": ["activity_monitoring", "contextual_assistance", "step_guidance"],
        "maintain_general_awareness": ["context_monitoring", "environment_change_detection", "important_event_alerting"],
    }
    assistance = list(assistance_map.get(primary_intent, ["context_monitoring"]))
    activities = set(get_user_activities(situation))
    environment = get_environment_type(situation)
    safety = get_safety_status(situation)
    priority = get_cognitive_priority(situation, cognitive_state)

    if primary_intent == "navigate_to_destination" and "walking" in activities:
        assistance.append("walking_speed_adaptation")
    if primary_intent in {"navigate_to_destination", "continue_movement"} and environment in {
        "traffic_area", "urban_environment", "shopping_mall", "indoor_environment",
        "outdoor_environment", "transport_environment"
    }:
        assistance.append("dynamic_obstacle_awareness")
    if destination:
        assistance.append("destination_confirmation")
    if target and primary_intent in {"read_text", "identify_object", "find_person_or_object", "communicate_with_person"}:
        assistance.append("target_confirmation")
    if safety in {"caution", "unsafe", "critical", "danger"}:
        assistance.append("continuous_safety_monitoring")
    if priority in {"high", "critical"}:
        assistance.append("priority_response")
    for rule in matched_navigation_rules + matched_emergency_rules:
        assistance.extend(str(v) for v in rule.get("actions", []) if str(v).strip())
    return unique_preserve_order(assistance)


def calculate_intent_confidence(primary_intent, scores, evidence, situation):
    ranked = sorted(scores.values(), reverse=True)
    primary_score = scores.get(primary_intent, 0.0)
    second_score = ranked[1] if len(ranked) > 1 else 0.0
    strength = min(primary_score / 8.0, 1.0)
    separation = 1.0 if primary_score <= 0 else max(0.0, min((primary_score - second_score) / max(primary_score, 1.0), 1.0))
    evidence_strength = min(len(evidence.get(primary_intent, [])) / 5.0, 1.0)
    context_confidence = get_context_confidence(situation)
    confidence = 0.20 + 0.35 * strength + 0.20 * separation + 0.15 * evidence_strength + 0.10 * context_confidence
    if primary_intent == "maintain_general_awareness":
        confidence = min(confidence, 0.75)
    return round(clamp_confidence(confidence), 2)


def infer_urgency(primary_intent, situation, cognitive_state):
    safety = get_safety_status(situation)
    risk = get_risk_level(situation)
    priority = get_cognitive_priority(situation, cognitive_state)
    if primary_intent == "seek_emergency_help" or safety in {"critical", "danger"} or risk in {"critical", "extreme"}:
        return "critical"
    if primary_intent == "avoid_hazard" or safety == "unsafe" or risk in {"high", "severe"} or priority == "high":
        return "high"
    if priority == "critical":
        return "critical"
    if safety == "caution" or risk == "medium" or priority == "medium":
        return "medium"
    return "low"


def generate_summary(primary_intent, category, destination, target, confidence, urgency, assistance):
    if destination:
        target_text = f"The inferred destination is {destination}."
    elif target:
        target_text = f"The inferred target is {target}."
    else:
        target_text = "No specific destination or target was identified."
    return (
        f"The user's primary intent is {primary_intent}, belonging to the {category} category. "
        f"{target_text} Intent confidence is {confidence}. Urgency is {urgency}. "
        f"Required assistance includes {', '.join(assistance)}."
    )


# ============================================================
# MAIN REASONING
# ============================================================

def reason_about_intent(situation, cognitive_state, context_graph, common_sense, navigation_rules, emergency_rules):
    log_section("Intent Reasoning")

    scores, evidence = infer_intent_scores(
        situation, cognitive_state, context_graph, common_sense
    )
    primary_intent, secondary_intents = select_primary_and_secondary(scores)
    category = INTENT_CATEGORIES.get(primary_intent, "general")
    destination, target = infer_destination_and_target(
        primary_intent, situation, cognitive_state, context_graph
    )
    evidence_text = build_evidence_text(situation, cognitive_state, context_graph)

    matched_navigation_rules = []
    if primary_intent in {"navigate_to_destination", "continue_movement"}:
        matched_navigation_rules = match_rules(
            navigation_rules, primary_intent, evidence_text
        )

    matched_emergency_rules = []
    if primary_intent in {"seek_emergency_help", "avoid_hazard", "stay_safe"}:
        allowed = None
        if primary_intent == "seek_emergency_help":
            allowed = {"critical", "maximum", "high", "emergency"}
        matched_emergency_rules = match_rules(
            emergency_rules, primary_intent, evidence_text, allowed
        )

    assistance = infer_required_assistance(
        primary_intent, situation, cognitive_state, destination, target,
        matched_navigation_rules, matched_emergency_rules
    )
    confidence = calculate_intent_confidence(
        primary_intent, scores, evidence, situation
    )
    urgency = infer_urgency(primary_intent, situation, cognitive_state)

    ranked_scores = [
        {
            "intent": intent,
            "category": INTENT_CATEGORIES.get(intent, "general"),
            "score": round(score, 2),
        }
        for intent, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    result = {
        "timestamp": str(datetime.now()),
        "primary_intent": primary_intent,
        "intent_category": category,
        "destination": destination,
        "target": target,
        "intent_confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "urgency": urgency,
        "required_assistance": assistance,
        "secondary_intents": secondary_intents,
        "intent_scores": ranked_scores,
        "evidence": evidence.get(primary_intent, []),
        "matched_navigation_rules": matched_navigation_rules,
        "matched_emergency_rules": matched_emergency_rules,
        "summary": generate_summary(
            primary_intent, category, destination, target,
            confidence, urgency, assistance
        ),
    }

    log_info(f"Primary Intent: {primary_intent}")
    log_info(f"Intent Category: {category}")
    log_info(f"Destination: {destination}")
    log_info(f"Target: {target}")
    log_info(f"Confidence: {confidence}")
    log_info(f"Urgency: {urgency}")
    log_info(f"Required Assistance: {assistance}")
    log_success("Intent Reasoning Complete")
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    module_start("INTENT REASONER")
    os.makedirs(LAYER4_OUTPUT_DIR, exist_ok=True)

    situation = load_json(SITUATION_PATH)
    cognitive_state = load_json(COGNITIVE_STATE_PATH)
    context_graph = load_json(CONTEXT_GRAPH_PATH)

    if not situation:
        log_error("situation_understanding.json Missing Or Empty")
        module_end("INTENT REASONER")
        return
    if not cognitive_state:
        log_error("cognitive_state.json Missing Or Empty")
        module_end("INTENT REASONER")
        return
    if not context_graph:
        log_error("context_graph.json Missing Or Empty")
        module_end("INTENT REASONER")
        return

    common_sense = load_optional_json(COMMON_SENSE_CANDIDATES)
    navigation_rules = load_optional_json(NAVIGATION_RULES_CANDIDATES)
    emergency_rules = load_optional_json(EMERGENCY_RULES_CANDIDATES)

    if not common_sense:
        log_warning("Common-sense knowledge file not found or empty; continuing with contextual reasoning.")
    if not navigation_rules:
        log_warning("navigation_rules.json not found or empty; continuing without navigation-rule enrichment.")
    if not emergency_rules:
        log_warning("emergency_rules.json not found or empty; continuing without emergency-rule enrichment.")

    result = reason_about_intent(
        situation,
        cognitive_state,
        context_graph,
        common_sense,
        navigation_rules,
        emergency_rules,
    )

    save_json(result, INTENT_OUTPUT_PATH)
    log_info(f"Intent output saved to: {INTENT_OUTPUT_PATH}")
    module_end("INTENT REASONER")


if __name__ == "__main__":
    main()