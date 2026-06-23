
"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Intent Reasoner
============================================================
Purpose:
Infer user intent using:
1. situation_understanding.json
2. cognitive_state.json
3. context_graph.json
4. knowledge/common_sense.json
5. knowledge/navigation_rules.json
6. knowledge/emergency_rules.json
============================================================
"""

import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

PROJECT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "layer4"
)

KNOWLEDGE_DIR = os.path.join(
    BASE_DIR,
    "knowledge"
)

sys.path.append(BASE_DIR)

from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import clamp_confidence, confidence_label


# ============================================================
# INPUT PATHS
# ============================================================

SITUATION_PATH = os.path.join(
    PROJECT_OUTPUT_DIR,
    "situation_understanding.json"
)

COGNITIVE_STATE_PATH = os.path.join(
    PROJECT_OUTPUT_DIR,
    "cognitive_state.json"
)

CONTEXT_GRAPH_PATH = os.path.join(
    PROJECT_OUTPUT_DIR,
    "context_graph.json"
)


# ============================================================
# KNOWLEDGE PATHS
# ============================================================

COMMON_SENSE_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "common_sense.json"
)

NAVIGATION_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "navigation_rules.json"
)

EMERGENCY_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "emergency_rules.json"
)


# ============================================================
# OUTPUT PATH
# ============================================================

INTENT_OUTPUT_PATH = os.path.join(
    PROJECT_OUTPUT_DIR,
    "intent_reasoning.json"
)


# ============================================================
# GRAPH HELPERS
# ============================================================

def extract_user_requested_target(context_graph):
    for edge in context_graph.get("edges", []):
        source = str(edge.get("source", "")).lower()
        relation = str(edge.get("relation", "")).lower()
        target = edge.get("target", None)

        if source == "user" and relation in [
            "requesting",
            "wants",
            "needs",
            "goal",
            "looking_for",
            "searching_for"
        ]:
            return target

    return None


def extract_event_types(context_graph):
    event_types = []

    for node in context_graph.get("nodes", []):
        if node.get("category") == "event":
            event_types.append(
                str(node.get("event_type", "unknown")).lower()
            )

    return event_types


def extract_activity_nodes(context_graph):
    activities = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "activity":
            activities.append(
                str(node.get("id", "")).lower()
            )

    return activities


def extract_graph_entities(context_graph):
    entities = []

    for node in context_graph.get("nodes", []):
        node_id = str(node.get("id", "")).lower()
        entity_type = str(node.get("entity_type", "")).lower()
        category = str(node.get("category", "")).lower()

        entities.append(node_id)
        entities.append(entity_type)
        entities.append(category)

    return list(set(entities))


# ============================================================
# KNOWLEDGE HELPERS
# ============================================================

def get_common_sense_facts(common_sense):
    facts = []

    categories = common_sense.get("categories", {})

    for category, statements in categories.items():
        for statement in statements:
            facts.append({
                "category": category,
                "statement": statement
            })

    return facts


def match_navigation_rules(primary_intent, navigation_rules):
    matched_rules = []

    if primary_intent != "navigate_to_destination":
        return matched_rules

    for rule in navigation_rules.get("rules", []):
        rule_name = str(rule.get("name", "")).lower()

        if (
            "navigate" in rule_name
            or "destination" in rule_name
            or "route" in rule_name
            or "navigation" in rule_name
        ):
            matched_rules.append({
                "rule_id": rule.get("rule_id"),
                "name": rule.get("name"),
                "priority": rule.get("priority"),
                "actions": rule.get("actions", [])
            })

    return matched_rules


def match_emergency_rules(primary_intent, emergency_rules):
    matched_rules = []

    if primary_intent != "seek_emergency_help":
        return matched_rules

    for rule in emergency_rules.get("rules", []):
        priority = str(rule.get("priority", "")).lower()

        if priority in ["critical", "maximum", "high"]:
            matched_rules.append({
                "rule_id": rule.get("rule_id"),
                "name": rule.get("name"),
                "priority": rule.get("priority"),
                "actions": rule.get("actions", [])
            })

    return matched_rules


# ============================================================
# INTENT REASONING
# ============================================================

def infer_primary_intent(
        situation,
        cognitive_state,
        context_graph,
        common_sense
):
    attention_focus = str(
        cognitive_state.get("attention_focus", "")
    ).lower()

    situation_type = str(
        situation.get("situation_type", "")
    ).lower()

    user_goal = str(
        situation.get("user_goal", "")
    ).lower()

    safety_status = str(
        situation.get("safety_status", "safe")
    ).lower()

    event_types = extract_event_types(context_graph)
    requested_target = extract_user_requested_target(context_graph)
    graph_entities = extract_graph_entities(context_graph)

    all_text = " ".join(
        event_types
        + graph_entities
        + [
            attention_focus,
            situation_type,
            user_goal,
            safety_status
        ]
    )

    if (
        "emergency" in all_text
        or "fire" in all_text
        or "smoke" in all_text
        or "help_call" in all_text
        or "scream" in all_text
        or safety_status == "unsafe"
        or attention_focus == "emergency"
    ):
        return "seek_emergency_help"

    if (
        attention_focus == "safety"
        or situation_type == "safety_monitoring"
        or safety_status in ["caution", "unsafe"]
    ):
        return "stay_safe"

    if (
        attention_focus == "navigation"
        or situation_type == "navigation_assistance"
        or "navigation_request" in event_types
        or requested_target is not None
        or user_goal.startswith("reach_")
        or "exit" in all_text
        or "gate" in all_text
        or "destination" in all_text
    ):
        return "navigate_to_destination"

    if (
        "walking" in all_text
        or "movement" in event_types
        or "moving" in all_text
    ):
        return "continue_movement"

    return "maintain_general_awareness"


def infer_intent_category(primary_intent):
    if primary_intent in [
        "navigate_to_destination",
        "continue_movement"
    ]:
        return "navigation"

    if primary_intent in [
        "stay_safe",
        "seek_emergency_help"
    ]:
        return "safety"

    return "general"


def infer_destination(situation, context_graph):
    requested_target = extract_user_requested_target(context_graph)

    if requested_target:
        return requested_target

    user_goal = str(
        situation.get("user_goal", "")
    ).lower()

    if user_goal.startswith("reach_"):
        return user_goal.replace("reach_", "")

    nearby_entities = situation.get("nearby_entities", [])

    for entity in nearby_entities:
        entity = str(entity).lower()

        if entity in [
            "exit",
            "gate",
            "door",
            "shop",
            "information_desk",
            "counter",
            "platform"
        ]:
            return entity

    return None


def infer_required_assistance(
        primary_intent,
        situation,
        cognitive_state,
        matched_navigation_rules,
        matched_emergency_rules
):
    assistance = []

    safety_status = str(
        situation.get("safety_status", "safe")
    ).lower()

    environment_type = str(
        situation.get("environment_type", "")
    ).lower()

    user_activities = [
        str(activity).lower()
        for activity in situation.get("user_activities", [])
    ]

    if primary_intent == "navigate_to_destination":
        assistance.extend([
            "route_guidance",
            "destination_tracking",
            "step_by_step_navigation"
        ])

        if "walking" in user_activities:
            assistance.append("walking_speed_adaptation")

        if environment_type in [
            "traffic_area",
            "urban_environment",
            "shopping_mall",
            "indoor_environment"
        ]:
            assistance.append("obstacle_awareness")

        for rule in matched_navigation_rules:
            assistance.extend(rule.get("actions", []))

    elif primary_intent == "stay_safe":
        assistance.extend([
            "hazard_monitoring",
            "risk_alert_generation",
            "safety_guidance"
        ])

    elif primary_intent == "seek_emergency_help":
        assistance.extend([
            "emergency_alert",
            "safe_exit_guidance",
            "caregiver_notification",
            "location_sharing"
        ])

        for rule in matched_emergency_rules:
            assistance.extend(rule.get("actions", []))

    elif primary_intent == "continue_movement":
        assistance.extend([
            "movement_tracking",
            "obstacle_monitoring",
            "path_continuity_support"
        ])

    else:
        assistance.append("context_monitoring")

    if safety_status in ["caution", "unsafe"]:
        assistance.append("safety_guidance")

    return list(set(assistance))


def calculate_intent_confidence(
        primary_intent,
        situation,
        cognitive_state,
        context_graph,
        matched_navigation_rules,
        matched_emergency_rules
):
    confidence = 0.50

    event_types = extract_event_types(context_graph)
    requested_target = extract_user_requested_target(context_graph)

    attention_focus = str(
        cognitive_state.get("attention_focus", "")
    ).lower()

    situation_type = str(
        situation.get("situation_type", "")
    ).lower()

    user_goal = str(
        situation.get("user_goal", "")
    ).lower()

    safety_status = str(
        situation.get("safety_status", "safe")
    ).lower()

    if primary_intent == "navigate_to_destination":
        if requested_target:
            confidence += 0.20
        if "navigation_request" in event_types:
            confidence += 0.15
        if attention_focus == "navigation":
            confidence += 0.10
        if situation_type == "navigation_assistance":
            confidence += 0.10
        if user_goal.startswith("reach_"):
            confidence += 0.10
        if matched_navigation_rules:
            confidence += 0.10

    elif primary_intent == "seek_emergency_help":
        if "emergency" in event_types:
            confidence += 0.25
        if attention_focus == "emergency":
            confidence += 0.20
        if safety_status == "unsafe":
            confidence += 0.15
        if matched_emergency_rules:
            confidence += 0.10

    elif primary_intent == "stay_safe":
        if safety_status in ["caution", "unsafe"]:
            confidence += 0.20
        if attention_focus == "safety":
            confidence += 0.15

    elif primary_intent == "continue_movement":
        confidence += 0.15

    else:
        confidence += 0.10

    return round(clamp_confidence(confidence), 2)


def infer_urgency(situation, cognitive_state, primary_intent):
    safety_status = str(
        situation.get("safety_status", "safe")
    ).lower()

    priority = str(
        cognitive_state.get("cognitive_priority", "low")
    ).lower()

    if primary_intent == "seek_emergency_help":
        return "critical"

    if safety_status == "unsafe" or priority == "critical":
        return "critical"

    if safety_status == "caution" or priority == "high":
        return "high"

    if priority == "medium":
        return "medium"

    return "low"


def generate_intent_summary(
        primary_intent,
        destination,
        confidence,
        assistance,
        urgency
):
    destination_text = destination if destination else "no specific destination"

    return (
        f"The user's primary intent is {primary_intent}. "
        f"Destination: {destination_text}. "
        f"Confidence: {confidence}. "
        f"Urgency: {urgency}. "
        f"Required assistance: {', '.join(assistance)}."
    )


# ============================================================
# MAIN REASONING
# ============================================================

def reason_about_intent(
        situation,
        cognitive_state,
        context_graph,
        common_sense,
        navigation_rules,
        emergency_rules
):
    log_section("Intent Reasoning")

    primary_intent = infer_primary_intent(
        situation,
        cognitive_state,
        context_graph,
        common_sense
    )

    intent_category = infer_intent_category(primary_intent)

    matched_navigation_rules = match_navigation_rules(
        primary_intent,
        navigation_rules
    )

    matched_emergency_rules = match_emergency_rules(
        primary_intent,
        emergency_rules
    )

    destination = infer_destination(
        situation,
        context_graph
    )

    required_assistance = infer_required_assistance(
        primary_intent,
        situation,
        cognitive_state,
        matched_navigation_rules,
        matched_emergency_rules
    )

    confidence = calculate_intent_confidence(
        primary_intent,
        situation,
        cognitive_state,
        context_graph,
        matched_navigation_rules,
        matched_emergency_rules
    )

    urgency = infer_urgency(
        situation,
        cognitive_state,
        primary_intent
    )

    intent_result = {
        "timestamp": str(datetime.now()),
        "primary_intent": primary_intent,
        "intent_category": intent_category,
        "destination": destination,
        "intent_confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "urgency": urgency,
        "required_assistance": required_assistance,
        "matched_navigation_rules": matched_navigation_rules,
        "matched_emergency_rules": matched_emergency_rules,
        "summary": generate_intent_summary(
            primary_intent,
            destination,
            confidence,
            required_assistance,
            urgency
        )
    }

    log_info(f"Primary Intent: {primary_intent}")
    log_info(f"Intent Category: {intent_category}")
    log_info(f"Destination: {destination}")
    log_info(f"Confidence: {confidence}")
    log_info(f"Urgency: {urgency}")
    log_info(f"Required Assistance: {required_assistance}")

    log_success("Intent Reasoning Complete")

    return intent_result


# ============================================================
# MAIN
# ============================================================

def main():
    module_start("INTENT REASONER")

    situation = load_json(SITUATION_PATH)
    cognitive_state = load_json(COGNITIVE_STATE_PATH)
    context_graph = load_json(CONTEXT_GRAPH_PATH)

    common_sense = load_json(COMMON_SENSE_PATH)
    navigation_rules = load_json(NAVIGATION_RULES_PATH)
    emergency_rules = load_json(EMERGENCY_RULES_PATH)

    if not situation:
        log_error("situation_understanding.json Missing Or Empty")
        return

    if not cognitive_state:
        log_error("cognitive_state.json Missing Or Empty")
        return

    if not context_graph:
        log_error("context_graph.json Missing Or Empty")
        return

    if not common_sense:
        log_error("common_sense.json Missing Or Empty")
        return

    if not navigation_rules:
        log_error("navigation_rules.json Missing Or Empty")
        return

    if not emergency_rules:
        log_error("emergency_rules.json Missing Or Empty")
        return

    intent_result = reason_about_intent(
        situation,
        cognitive_state,
        context_graph,
        common_sense,
        navigation_rules,
        emergency_rules
    )

    save_json(
        intent_result,
        INTENT_OUTPUT_PATH
    )

    module_end("INTENT REASONER")


if __name__ == "__main__":
    main()