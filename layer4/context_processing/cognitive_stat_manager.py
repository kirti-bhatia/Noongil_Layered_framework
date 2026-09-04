"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Cognitive State Manager
============================================================
Purpose:
Create the current cognitive state of NOONGIL-X using:
1. analyzed_context.json
2. context_graph.json

This module decides:
- What should the system focus on?
- What is the user's current goal?
- What is the safety condition?
- What reasoning mode should be activated?
============================================================
"""

import json
from datetime import datetime

from layer4.config.paths import (
    ANALYZED_CONTEXT_PATH,
    CONTEXT_GRAPH_PATH,
    COGNITIVE_STATE_PATH,
    ensure_output_directories,
)
from layer4.utils.file_loader import load_json
from layer4.utils.json_writer import save_json


# ============================================================
# GRAPH EXTRACTION
# ============================================================

def extract_graph_signals(context_graph):
    print("\n[INFO] Extracting Graph Signals...")

    nodes = context_graph.get("nodes", [])
    edges = context_graph.get("edges", [])

    node_ids = []
    event_types = []
    activity_nodes = []
    audio_nodes = []
    user_targets = []

    # --------------------------------------------------------
    # NODE EXTRACTION
    # --------------------------------------------------------

    for node in nodes:
        node_id = str(node.get("id", "")).lower()
        entity_type = str(node.get("entity_type", "")).lower()
        event_type = str(node.get("event_type", "")).lower()

        if node_id:
            node_ids.append(node_id)

        if event_type:
            event_types.append(event_type)

        if entity_type == "activity" and node_id:
            activity_nodes.append(node_id)

        if entity_type in {"audio", "sound"} and node_id:
            audio_nodes.append(node_id)

    # --------------------------------------------------------
    # EDGE EXTRACTION
    # --------------------------------------------------------

    goal_relations = {
        "requesting",
        "requests",
        "wants",
        "needs",
        "goal",
        "targeting",
        "navigating_to",
        "moving_towards"
    }

    for edge in edges:
        source = str(edge.get("source", "")).lower()
        target = str(edge.get("target", "")).lower()

        relation = str(
            edge.get("relation")
            or edge.get("relation_type")
            or edge.get("type")
            or ""
        ).lower()

        if (
            source == "user"
            and relation in goal_relations
            and target
        ):
            user_targets.append(target)

    # --------------------------------------------------------
    # REMOVE DUPLICATES WHILE PRESERVING ORDER
    # --------------------------------------------------------

    node_ids = list(dict.fromkeys(node_ids))
    event_types = list(dict.fromkeys(event_types))
    activity_nodes = list(dict.fromkeys(activity_nodes))
    audio_nodes = list(dict.fromkeys(audio_nodes))
    user_targets = list(dict.fromkeys(user_targets))

    signals = {
        "node_ids": node_ids,
        "event_types": event_types,
        "activity_nodes": activity_nodes,
        "audio_nodes": audio_nodes,
        "user_targets": user_targets
    }

    print(f"[INFO] Nodes Found: {len(node_ids)}")
    print(f"[INFO] Events Found: {event_types}")
    print(f"[INFO] Activities Found: {activity_nodes}")
    print(f"[INFO] Audio Signals Found: {audio_nodes}")
    print(f"[INFO] User Targets Found: {user_targets}")

    return signals


# ============================================================
# ATTENTION FOCUS
# ============================================================

def determine_attention_focus(analyzed_context, graph_signals):
    risk_level = str(
        analyzed_context.get("risk_level", "low")
    ).lower()

    event_types = graph_signals.get("event_types", [])
    user_targets = graph_signals.get("user_targets", [])
    node_ids = graph_signals.get("node_ids", [])

    if risk_level == "high":
        return "safety"

    if (
        "emergency" in event_types
        or "fall_detected" in event_types
    ):
        return "emergency"

    if (
        "navigation_request" in event_types
        or user_targets
    ):
        return "navigation"

    hazard_nodes = {
        "obstacle",
        "vehicle",
        "traffic",
        "stairs",
        "fire",
        "smoke"
    }

    if any(node in hazard_nodes for node in node_ids):
        return "hazard_monitoring"

    return "general_awareness"


# ============================================================
# PRIMARY GOAL
# ============================================================

def determine_primary_goal(graph_signals):
    user_targets = graph_signals.get("user_targets", [])
    event_types = graph_signals.get("event_types", [])

    if user_targets:
        return f"reach_{user_targets[0]}"

    if "navigation_request" in event_types:
        return "assist_navigation"

    if "movement" in event_types:
        return "monitor_user_movement"

    if "home_activity" in event_types:
        return "support_home_activity"

    if "conversation_event" in event_types:
        return "understand_user_request"

    return "maintain_awareness"


# ============================================================
# SAFETY STATUS
# ============================================================

def determine_safety_status(risk_level):
    risk_level = str(risk_level).lower()

    if risk_level == "high":
        return "unsafe"

    if risk_level == "medium":
        return "caution"

    return "safe"


# ============================================================
# COGNITIVE PRIORITY
# ============================================================

def determine_priority(risk_level, attention_focus):
    risk_level = str(risk_level).lower()

    if risk_level == "high":
        return "critical"

    if attention_focus == "emergency":
        return "critical"

    if risk_level == "medium":
        return "high"

    if attention_focus in {
        "navigation",
        "hazard_monitoring"
    }:
        return "medium"

    return "low"


# ============================================================
# REASONING MODE
# ============================================================

def determine_reasoning_mode(attention_focus):
    if attention_focus == "emergency":
        return "emergency_response"

    if attention_focus == "safety":
        return "risk_avoidance"

    if attention_focus == "navigation":
        return "assistance"

    if attention_focus == "hazard_monitoring":
        return "safety_monitoring"

    return "observation"


# ============================================================
# ACTIVE CONTEXTS
# ============================================================

def build_active_contexts(analyzed_context, graph_signals):
    active_contexts = []

    scene_type = str(
        analyzed_context.get("scene_type", "unknown")
    ).lower()

    risk_level = str(
        analyzed_context.get("risk_level", "low")
    ).lower()

    if scene_type:
        active_contexts.append(scene_type)

    for activity in graph_signals.get("activity_nodes", []):
        if activity:
            active_contexts.append(activity)

    for audio in graph_signals.get("audio_nodes", []):
        if audio:
            active_contexts.append(audio)

    for event in graph_signals.get("event_types", []):
        if event:
            active_contexts.append(event)

    if risk_level:
        active_contexts.append(f"risk_{risk_level}")

    return list(dict.fromkeys(active_contexts))


# ============================================================
# COGNITIVE STATE GENERATION
# ============================================================

def generate_cognitive_state(analyzed_context, context_graph):
    print("\n[INFO] Generating Cognitive State...")

    graph_signals = extract_graph_signals(context_graph)

    risk_level = analyzed_context.get(
        "risk_level",
        "low"
    )

    attention_focus = determine_attention_focus(
        analyzed_context,
        graph_signals
    )

    primary_goal = determine_primary_goal(
        graph_signals
    )

    safety_status = determine_safety_status(
        risk_level
    )

    cognitive_priority = determine_priority(
        risk_level,
        attention_focus
    )

    reasoning_mode = determine_reasoning_mode(
        attention_focus
    )

    active_contexts = build_active_contexts(
        analyzed_context,
        graph_signals
    )

    cognitive_state = {
        "timestamp": str(datetime.now()),
        "attention_focus": attention_focus,
        "primary_goal": primary_goal,
        "safety_status": safety_status,
        "cognitive_priority": cognitive_priority,
        "reasoning_mode": reasoning_mode,
        "active_contexts": active_contexts,
        "graph_signals": graph_signals,
        "summary": (
            f"Focus: {attention_focus}. "
            f"Goal: {primary_goal}. "
            f"Safety: {safety_status}. "
            f"Priority: {cognitive_priority}. "
            f"Reasoning mode: {reasoning_mode}."
        )
    }

    print(f"[INFO] Attention Focus: {attention_focus}")
    print(f"[INFO] Primary Goal: {primary_goal}")
    print(f"[INFO] Safety Status: {safety_status}")
    print(f"[INFO] Cognitive Priority: {cognitive_priority}")
    print(f"[INFO] Reasoning Mode: {reasoning_mode}")
    print(f"[INFO] Active Contexts: {active_contexts}")

    print("[SUCCESS] Cognitive State Generated")

    return cognitive_state


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_output_directories()

    print("\n" + "=" * 60)
    print("NOONGIL-X COGNITIVE STATE MANAGER")
    print("=" * 60)

    analyzed_context = load_json(
        ANALYZED_CONTEXT_PATH
    )

    context_graph = load_json(
        CONTEXT_GRAPH_PATH
    )

    if not analyzed_context:
        print(
            "[ERROR] analyzed_context.json "
            "Missing Or Empty"
        )
        return

    if not context_graph:
        print(
            "[ERROR] context_graph.json "
            "Missing Or Empty"
        )
        return

    cognitive_state = generate_cognitive_state(
        analyzed_context,
        context_graph
    )

    save_json(
        cognitive_state,
        COGNITIVE_STATE_PATH
    )

    print("\n" + "=" * 60)
    print("COGNITIVE STATE SUMMARY")
    print("=" * 60)
    print(json.dumps(cognitive_state, indent=4))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()