"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Situation Understanding
============================================================
Purpose:
Understand the current situation by combining:
1. context_graph.json
2. analyzed_context.json
3. cognitive_state.json

This module explains:
- Where the user is
- What the user is doing
- What the user wants
- What is happening around the user
- Whether the situation is safe or risky
============================================================
"""

import json
import os
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

CONTEXT_GRAPH_PATH = os.path.join(OUTPUT_DIR, "context_graph.json")
ANALYZED_CONTEXT_PATH = os.path.join(OUTPUT_DIR, "analyzed_context.json")
COGNITIVE_STATE_PATH = os.path.join(OUTPUT_DIR, "cognitive_state.json")

SITUATION_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "situation_understanding.json"
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File Not Found: {filepath}")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[SUCCESS] Loaded: {filepath}")
        return data

    except Exception as e:
        print(f"[ERROR] Failed To Load JSON: {e}")
        return {}


def save_json(data, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"[SUCCESS] Saved: {filepath}")

    except Exception as e:
        print(f"[ERROR] Failed To Save JSON: {e}")


# ============================================================
# GRAPH HELPERS
# ============================================================

def get_node_map(context_graph):
    node_map = {}

    for node in context_graph.get("nodes", []):
        node_id = node.get("id", "")
        if node_id:
            node_map[node_id.lower()] = node

    return node_map


def extract_locations(context_graph):
    locations = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "location":
            locations.append(node.get("id"))

    return locations


def extract_activities(context_graph):
    activities = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "activity":
            activities.append(node.get("id"))

    return activities


def extract_audio_cues(context_graph):
    audio_cues = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "audio":
            audio_cues.append(node.get("id"))

    return audio_cues


def extract_objects(context_graph):
    objects = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "object":
            objects.append(node.get("id"))

    return objects


def extract_events(context_graph):
    events = []

    for node in context_graph.get("nodes", []):
        if node.get("category") == "event":
            events.append({
                "id": node.get("id"),
                "event_type": node.get("event_type", "unknown")
            })

    return events


def extract_user_location(context_graph):
    for edge in context_graph.get("edges", []):
        source = edge.get("source", "").lower()
        relation = edge.get("relation", "").lower()
        target = edge.get("target", "")

        if source == "user" and relation in ["located_in", "inside", "at"]:
            return target

    locations = extract_locations(context_graph)
    if locations:
        return locations[0]

    return "unknown"


def extract_user_activity(context_graph):
    activities = []

    for edge in context_graph.get("edges", []):
        source = edge.get("source", "").lower()
        relation = edge.get("relation", "").lower()
        target = edge.get("target", "")

        if source == "user" and relation in ["performing", "doing"]:
            activities.append(target)

    if not activities:
        activities = extract_activities(context_graph)

    return activities


def extract_user_goal(context_graph, cognitive_state):
    for edge in context_graph.get("edges", []):
        source = edge.get("source", "").lower()
        relation = edge.get("relation", "").lower()
        target = edge.get("target", "")

        if source == "user" and relation in ["requesting", "wants", "needs", "goal"]:
            return f"reach_{target}"

    return cognitive_state.get("primary_goal", "unknown")


def extract_nearby_entities(context_graph, user_location):
    nearby = []

    for edge in context_graph.get("edges", []):
        target = edge.get("target", "").lower()
        relation = edge.get("relation", "").lower()
        source = edge.get("source", "")

        if target == user_location.lower() and relation in ["inside", "located_in", "near"]:
            if source.lower() != "user":
                nearby.append(source)

    return nearby


# ============================================================
# SITUATION CLASSIFICATION
# ============================================================

def classify_environment(user_location, analyzed_context):
    scene_type = analyzed_context.get("scene_type", "unknown")

    location = user_location.lower()

    if "temple" in location:
        return "religious_place"

    if "park" in location:
        return "outdoor_recreation_area"

    if "road" in location or "street" in location:
        return "traffic_area"

    if "home" in location:
        return "home_environment"

    if "college" in location or "school" in location:
        return "education_environment"

    return scene_type


def classify_situation_type(cognitive_state):
    attention_focus = cognitive_state.get("attention_focus", "general_awareness")

    if attention_focus == "navigation":
        return "navigation_assistance"

    if attention_focus == "safety":
        return "safety_monitoring"

    if attention_focus == "emergency":
        return "emergency_response"

    if attention_focus == "hazard_monitoring":
        return "hazard_awareness"

    return "general_observation"


def generate_human_summary(
    user_location,
    user_activities,
    user_goal,
    nearby_entities,
    audio_cues,
    safety_status
):
    activity_text = ", ".join(user_activities) if user_activities else "unknown activity"
    nearby_text = ", ".join(nearby_entities) if nearby_entities else "no important nearby entities"
    audio_text = ", ".join(audio_cues) if audio_cues else "no strong audio cues"

    return (
        f"The user is currently in {user_location}. "
        f"The user appears to be doing: {activity_text}. "
        f"The current goal is {user_goal}. "
        f"Nearby entities include: {nearby_text}. "
        f"Audio context includes: {audio_text}. "
        f"The current safety status is {safety_status}."
    )


# ============================================================
# MAIN SITUATION UNDERSTANDING
# ============================================================

def understand_situation(context_graph, analyzed_context, cognitive_state):
    print("\n[INFO] Understanding Current Situation...")

    user_location = extract_user_location(context_graph)
    user_activities = extract_user_activity(context_graph)
    user_goal = extract_user_goal(context_graph, cognitive_state)

    nearby_entities = extract_nearby_entities(
        context_graph,
        user_location
    )

    audio_cues = extract_audio_cues(context_graph)
    events = extract_events(context_graph)

    environment_type = classify_environment(
        user_location,
        analyzed_context
    )

    situation_type = classify_situation_type(cognitive_state)

    safety_status = cognitive_state.get("safety_status", "unknown")

    human_summary = generate_human_summary(
        user_location,
        user_activities,
        user_goal,
        nearby_entities,
        audio_cues,
        safety_status
    )

    situation = {
        "timestamp": str(datetime.now()),
        "user_location": user_location,
        "user_activities": user_activities,
        "user_goal": user_goal,
        "nearby_entities": nearby_entities,
        "audio_cues": audio_cues,
        "events": events,
        "environment_type": environment_type,
        "situation_type": situation_type,
        "safety_status": safety_status,
        "attention_focus": cognitive_state.get("attention_focus", "unknown"),
        "cognitive_priority": cognitive_state.get("cognitive_priority", "low"),
        "human_readable_summary": human_summary
    }

    print(f"[INFO] User Location: {user_location}")
    print(f"[INFO] User Activities: {user_activities}")
    print(f"[INFO] User Goal: {user_goal}")
    print(f"[INFO] Nearby Entities: {nearby_entities}")
    print(f"[INFO] Environment Type: {environment_type}")
    print(f"[INFO] Situation Type: {situation_type}")
    print(f"[INFO] Safety Status: {safety_status}")

    print("[SUCCESS] Situation Understanding Complete")

    return situation


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("NOONGIL-X SITUATION UNDERSTANDING")
    print("=" * 60)

    context_graph = load_json(CONTEXT_GRAPH_PATH)
    analyzed_context = load_json(ANALYZED_CONTEXT_PATH)
    cognitive_state = load_json(COGNITIVE_STATE_PATH)

    if not context_graph:
        print("[ERROR] context_graph.json Missing Or Empty")
        return

    if not analyzed_context:
        print("[ERROR] analyzed_context.json Missing Or Empty")
        return

    if not cognitive_state:
        print("[ERROR] cognitive_state.json Missing Or Empty")
        return

    situation = understand_situation(
        context_graph,
        analyzed_context,
        cognitive_state
    )

    save_json(situation, SITUATION_OUTPUT_PATH)

    print("\n============================================================")
    print("SITUATION UNDERSTANDING SUMMARY")
    print("============================================================")
    print(json.dumps(situation, indent=4))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()