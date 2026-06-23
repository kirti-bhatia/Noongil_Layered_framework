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
import os
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

GRAPH_PATH = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "context_graph.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "analyzed_context.json"
)


# ============================================================
# LOAD JSON
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
        print(f"[ERROR] Loading Failed: {e}")
        return {}


# ============================================================
# CONTEXT ANALYSIS
# ============================================================

def analyze_context(graph_data):

    print("\n[INFO] Starting Context Analysis...")

    analysis = {
        "timestamp": str(datetime.now()),
        "scene_type": "unknown",
        "important_entities": [],
        "user_context": {},
        "risk_level": "low",
        "summary": ""
    }

    nodes = graph_data.get("nodes", [])

    if not nodes:
        print("[WARNING] No Nodes Found")
        return analysis

    # --------------------------------------------------------
    # ENTITY COLLECTION
    # --------------------------------------------------------

    entities = []

    for node in nodes:

        entity_name = node.get("id", "unknown_entity")
        entity_type =(
            node.get("entity_type")
            or node.get("event_type")
            or node.get("category")
            or "unknown"
        )

        entities.append({
            "name": entity_name,
            "type": entity_type
        })

    analysis["important_entities"] = entities

    print(f"[INFO] Found {len(entities)} Entities")

    # --------------------------------------------------------
    # SCENE DETECTION
    # --------------------------------------------------------

    scene_keywords = {
        "road": "outdoor_navigation",
        "car": "traffic_environment",
        "vehicle": "traffic_environment",
        "building": "urban_area",
        "person": "human_activity",
        "stairs": "mobility_environment",
        "crosswalk": "road_crossing"
    }

    detected_scene = "general_environment"

    for entity in entities:

        name = entity["name"].lower()

        for keyword, scene in scene_keywords.items():

            if keyword in name:
                detected_scene = scene
                break

    analysis["scene_type"] = detected_scene

    print(f"[INFO] Scene Detected: {detected_scene}")

    # --------------------------------------------------------
    # USER CONTEXT
    # --------------------------------------------------------

    user_context = {
        "user_present": False,
        "people_detected": 0
    }

    people_count = 0

    for entity in entities:
        entity_name = entity["name"].lower()
        entity_type = entity["type"].lower()
        if entity_name in ["person","user","human"]:
            people_count +=1
        elif entity_type in ["person","user","human"]:
            people_count +=1

    user_context["people_detected"] = people_count

    if people_count > 0:
        user_context["user_present"] = True

    analysis["user_context"] = user_context

    print(f"[INFO] People Detected: {people_count}")

    # --------------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------------

    risk_level = "low"

    danger_keywords = [
        "vehicle",
        "car",
        "truck",
        "stairs",
        "fire",
        "traffic"
    ]

    for entity in entities:

        name = entity["name"].lower()

        if any(word in name for word in danger_keywords):
            risk_level = "medium"

    if people_count > 5:
        risk_level = "high"

    analysis["risk_level"] = risk_level

    print(f"[INFO] Risk Level: {risk_level}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = (
        f"Scene: {detected_scene}. "
        f"Entities detected: {len(entities)}. "
        f"People: {people_count}. "
        f"Risk Level: {risk_level}."
    )

    analysis["summary"] = summary

    print("[SUCCESS] Context Analysis Completed")

    return analysis


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, filepath):

    try:

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"[SUCCESS] Saved: {filepath}")

    except Exception as e:
        print(f"[ERROR] Save Failed: {e}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X CONTEXT ANALYZER")
    print("=" * 60)

    graph_data = load_json(GRAPH_PATH)

    if not graph_data:
        print("[ERROR] No Context Graph Available")
        return

    analysis = analyze_context(graph_data)

    save_json(analysis, OUTPUT_PATH)

    print("\n============================================================")
    print("CONTEXT ANALYSIS SUMMARY")
    print("============================================================")
    print(json.dumps(analysis, indent=4))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()