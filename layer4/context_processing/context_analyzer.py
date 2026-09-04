"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Context Analyzer
============================================================
Purpose:
Analyze memory/context data and create a cognitive
understanding of the current situation.
============================================================
"""

import json
from datetime import datetime

from layer4.config.paths import (
    CONTEXT_GRAPH_PATH,
    ANALYZED_CONTEXT_PATH,
    ensure_output_directories,
)
from layer4.utils.file_loader import load_json
from layer4.utils.json_writer import save_json


# ============================================================
# CONTEXT ANALYSIS
# ============================================================

def analyze_context(graph_data):

    print("\n[INFO] Starting Context Analysis...")

    analysis = {
        "timestamp": str(datetime.now()),
        "scene_type": "general_environment",
        "important_entities": [],
        "events": [],
        "activities": [],
        "user_context": {
            "user_present": False,
            "people_detected": 0
        },
        "risk_level": "low",
        "summary": ""
    }

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        print("[WARNING] No Nodes Found")
        return analysis

    # --------------------------------------------------------
    # SEPARATE NODE TYPES
    # --------------------------------------------------------

    entity_nodes = [
        node
        for node in nodes
        if node.get("category") == "entity"
    ]

    event_nodes = [
        node
        for node in nodes
        if node.get("category") == "event"
    ]

    agent_nodes = [
        node
        for node in nodes
        if node.get("category") == "agent"
    ]

    # --------------------------------------------------------
    # ENTITY COLLECTION
    # --------------------------------------------------------

    important_entities = []

    for node in entity_nodes:
        important_entities.append({
            "name": node.get("id", "unknown_entity"),
            "type": node.get("entity_type", "unknown")
        })

    analysis["important_entities"] = important_entities

    print(f"[INFO] Found {len(important_entities)} Entities")

    # --------------------------------------------------------
    # EVENT COLLECTION
    # --------------------------------------------------------

    events = []

    for node in event_nodes:
        events.append({
            "event_id": node.get("id", "unknown_event"),
            "event_type": node.get("event_type", "unknown")
        })

    analysis["events"] = events

    print(f"[INFO] Found {len(events)} Events")

    # --------------------------------------------------------
    # ACTIVITY COLLECTION
    # --------------------------------------------------------

    activities = []

    for node in entity_nodes:
        if str(node.get("entity_type", "")).lower() == "activity":
            activity_name = node.get("id")

            if activity_name:
                activities.append(activity_name)

    analysis["activities"] = activities

    print(f"[INFO] Found {len(activities)} Activities")

    # --------------------------------------------------------
    # SCENE DETECTION
    # --------------------------------------------------------

    detected_scene = "general_environment"

    # First preference: location entity
    for node in entity_nodes:
        if str(node.get("entity_type", "")).lower() == "location":
            detected_scene = str(
                node.get("id", "general_environment")
            ).lower()
            break

    # Second preference: user located_in relation
    if detected_scene == "general_environment":
        for edge in edges:

            relation_type = (
                edge.get("relation")
                or edge.get("relation_type")
                or edge.get("type")
            )

            if (
                str(edge.get("source", "")).lower() == "user"
                and str(relation_type).lower() == "located_in"
            ):
                detected_scene = str(
                    edge.get("target", "general_environment")
                ).lower()
                break

    analysis["scene_type"] = detected_scene

    print(f"[INFO] Scene Detected: {detected_scene}")

    # --------------------------------------------------------
    # USER CONTEXT
    # --------------------------------------------------------

    user_present = any(
        str(node.get("id", "")).lower() == "user"
        for node in agent_nodes
    )

    person_names = {
        "person",
        "human",
        "child",
        "adult",
        "elderly",
        "teacher",
        "student",
        "pedestrian"
    }

    people_count = 0

    for node in entity_nodes:

        entity_name = str(node.get("id", "")).lower()
        entity_type = str(node.get("entity_type", "")).lower()

        if (
            entity_name in person_names
            or entity_type in person_names
        ):
            people_count += 1

    analysis["user_context"] = {
        "user_present": user_present,
        "people_detected": people_count
    }

    print(f"[INFO] User Present: {user_present}")
    print(f"[INFO] Other People Detected: {people_count}")

    # --------------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------------

    risk_level = "low"

    medium_risk_terms = {
        "vehicle",
        "car",
        "truck",
        "stairs",
        "traffic",
        "crosswalk",
        "obstacle"
    }

    high_risk_terms = {
        "fire",
        "collision",
        "accident",
        "emergency",
        "help_call",
        "smoke",
        "weapon"
    }

    observed_terms = set()

    for entity in important_entities:
        observed_terms.add(str(entity["name"]).lower())
        observed_terms.add(str(entity["type"]).lower())

    for event in events:
        observed_terms.add(str(event["event_type"]).lower())

    if observed_terms.intersection(high_risk_terms):
        risk_level = "high"

    elif observed_terms.intersection(medium_risk_terms):
        risk_level = "medium"

    elif people_count > 5:
        risk_level = "medium"

    analysis["risk_level"] = risk_level

    print(f"[INFO] Risk Level: {risk_level}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = (
        f"Scene: {detected_scene}. "
        f"Entities detected: {len(important_entities)}. "
        f"Events detected: {len(events)}. "
        f"Activities detected: {len(activities)}. "
        f"Other people detected: {people_count}. "
        f"Risk level: {risk_level}."
    )

    analysis["summary"] = summary

    print("[SUCCESS] Context Analysis Completed")

    return analysis


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_output_directories()

    print("\n" + "=" * 60)
    print("NOONGIL-X CONTEXT ANALYZER")
    print("=" * 60)

    graph_data = load_json(CONTEXT_GRAPH_PATH)

    if not graph_data:
        print("[ERROR] No Context Graph Available")
        return

    analysis = analyze_context(graph_data)

    save_json(analysis, ANALYZED_CONTEXT_PATH)

    print("\n" + "=" * 60)
    print("CONTEXT ANALYSIS SUMMARY")
    print("=" * 60)
    print(json.dumps(analysis, indent=4))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()