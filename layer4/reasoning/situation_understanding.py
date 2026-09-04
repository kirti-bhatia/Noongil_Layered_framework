"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Situation Understanding
============================================================

Purpose:
Generate a structured understanding of the current situation
by combining:

1. context_graph.json
2. analyzed_context.json
3. cognitive_state.json

The module determines:

- User location
- User activities
- User goal
- Nearby entities
- Relevant objects and people
- Audio cues
- Current events
- Environment type
- Situation type
- Safety condition
- Cognitive priority
- Human-readable situation summary

The implementation uses graph structure, entity metadata,
relations, events and cognitive outputs so that it can operate
across different real-world scenarios.
============================================================
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from layer4.config.paths import (
    CONTEXT_GRAPH_PATH,
    ANALYZED_CONTEXT_PATH,
    COGNITIVE_STATE_PATH,
    SITUATION_UNDERSTANDING_PATH,
    ensure_output_directories,
)

from layer4.utils.file_loader import load_json
from layer4.utils.json_writer import save_json


# ============================================================
# GENERAL UTILITIES
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Convert a value to normalized lowercase text.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


def unique_values(values: List[Any]) -> List[Any]:
    """
    Remove duplicate values while preserving their order.
    """

    return list(dict.fromkeys(values))


def get_relation_type(edge: Dict[str, Any]) -> str:
    """
    Read a relation name from different possible Layer 3
    relation-field formats.
    """

    return normalize_text(
        edge.get("relation")
        or edge.get("relation_type")
        or edge.get("type")
        or edge.get("label")
    )


def get_node_category(node: Dict[str, Any]) -> str:
    """
    Return the normalized category of a graph node.
    """

    return normalize_text(node.get("category"))


def get_entity_type(node: Dict[str, Any]) -> str:
    """
    Return the normalized entity type of a graph node.
    """

    return normalize_text(
        node.get("entity_type")
        or node.get("type")
    )


def get_event_type(node: Dict[str, Any]) -> str:
    """
    Return the normalized event type of a graph node.
    """

    return normalize_text(
        node.get("event_type")
        or node.get("type")
    )


# ============================================================
# GRAPH HELPERS
# ============================================================

def get_node_map(
    context_graph: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Create a lookup dictionary using normalized node IDs.
    """

    node_map = {}

    for node in context_graph.get("nodes", []):
        node_id = normalize_text(node.get("id"))

        if node_id:
            node_map[node_id] = node

    return node_map


def get_nodes_by_entity_type(
    context_graph: Dict[str, Any],
    allowed_types: Set[str]
) -> List[str]:
    """
    Extract node IDs matching one or more entity types.
    """

    results = []

    normalized_types = {
        normalize_text(entity_type)
        for entity_type in allowed_types
    }

    for node in context_graph.get("nodes", []):
        node_id = node.get("id")
        entity_type = get_entity_type(node)

        if node_id and entity_type in normalized_types:
            results.append(node_id)

    return unique_values(results)


def extract_locations(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract all location entities.
    """

    return get_nodes_by_entity_type(
        context_graph,
        {"location", "place", "scene"}
    )


def extract_activities(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract all activity entities.
    """

    return get_nodes_by_entity_type(
        context_graph,
        {"activity", "action", "motion"}
    )


def extract_audio_cues(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract audio and sound entities.
    """

    return get_nodes_by_entity_type(
        context_graph,
        {
            "audio",
            "sound",
            "audio_event",
            "sound_event"
        }
    )


def extract_objects(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract physical object entities.
    """

    return get_nodes_by_entity_type(
        context_graph,
        {
            "object",
            "device",
            "vehicle",
            "obstacle",
            "furniture"
        }
    )


def extract_people(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract people other than the user.
    """

    people_types = {
        "person",
        "human",
        "child",
        "adult",
        "elderly",
        "teacher",
        "student",
        "pedestrian",
        "friend",
        "worker",
        "staff"
    }

    people = []

    for node in context_graph.get("nodes", []):
        node_id = node.get("id")
        entity_type = get_entity_type(node)
        node_name = normalize_text(node_id)

        if not node_id or node_name == "user":
            continue

        if (
            entity_type in people_types
            or node_name in people_types
        ):
            people.append(node_id)

    return unique_values(people)


def extract_events(
    context_graph: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract event nodes with useful event metadata.
    """

    events = []

    for node in context_graph.get("nodes", []):
        category = get_node_category(node)
        event_type = get_event_type(node)

        if category != "event" and not event_type:
            continue

        event_id = node.get("id")

        if not event_id:
            continue

        event_data = {
            "id": event_id,
            "event_type": event_type or "unknown"
        }

        optional_fields = [
            "actor",
            "target",
            "location",
            "activity",
            "confidence",
            "timestamp"
        ]

        for field in optional_fields:
            value = node.get(field)

            if value is not None:
                event_data[field] = value

        events.append(event_data)

    return events


# ============================================================
# USER STATE EXTRACTION
# ============================================================

def extract_user_location(
    context_graph: Dict[str, Any],
    analyzed_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Determine the user's current location.

    Priority:
    1. Explicit graph relation
    2. Location contained in an event
    3. Location entity
    4. Scene from analyzed context
    """

    location_relations = {
        "located_in",
        "inside",
        "at",
        "present_in",
        "positioned_in",
        "standing_in",
        "sitting_in"
    }

    for edge in context_graph.get("edges", []):
        source = normalize_text(edge.get("source"))
        target = edge.get("target")
        relation = get_relation_type(edge)

        if (
            source == "user"
            and relation in location_relations
            and target
        ):
            return str(target)

    for node in context_graph.get("nodes", []):
        if get_node_category(node) != "event":
            continue

        event_location = node.get("location")

        if event_location:
            return str(event_location)

    locations = extract_locations(context_graph)

    if locations:
        return locations[0]

    if analyzed_context:
        scene_type = analyzed_context.get("scene_type")

        if scene_type:
            return str(scene_type)

    return "unknown"


def extract_user_activity(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract activities associated with the user.

    Uses explicit user relations first and falls back to
    activity entities and event attributes.
    """

    activity_relations = {
        "performing",
        "doing",
        "engaged_in",
        "currently_doing",
        "activity",
        "walking",
        "running",
        "sitting",
        "standing",
        "resting"
    }

    activities = []

    for edge in context_graph.get("edges", []):
        source = normalize_text(edge.get("source"))
        target = edge.get("target")
        relation = get_relation_type(edge)

        if source != "user" or not target:
            continue

        if relation in activity_relations:
            activities.append(str(target))

    for node in context_graph.get("nodes", []):
        if get_node_category(node) != "event":
            continue

        actor = normalize_text(node.get("actor"))
        activity = node.get("activity")

        if actor == "user" and activity:
            activities.append(str(activity))

    if not activities:
        activities.extend(extract_activities(context_graph))

    return unique_values(activities)


def extract_user_targets(
    context_graph: Dict[str, Any]
) -> List[str]:
    """
    Extract meaningful user goal targets.

    Time references and event IDs are excluded from goal
    targets because they do not normally represent destinations
    or actionable objects.
    """

    node_map = get_node_map(context_graph)

    goal_relations = {
        "requesting",
        "requests",
        "wants",
        "needs",
        "goal",
        "targeting",
        "destination",
        "navigating_to",
        "moving_towards",
        "going_to",
        "looking_for",
        "searching_for",
        "reach",
        "approaching"
    }

    excluded_target_types = {
        "time",
        "date",
        "duration",
        "number",
        "quantity",
        "cardinal",
        "ordinal"
}
    
    excluded_target_categories = {
    "event",
    "time",
    "temporal"
}

    excluded_temporal_terms = {
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "today",
    "tomorrow",
    "yesterday"
}

    targets = []

    for edge in context_graph.get("edges", []):
        source = normalize_text(edge.get("source"))
        target = edge.get("target")
        relation = get_relation_type(edge)

        if (
            source != "user"
            or relation not in goal_relations
            or not target
        ):
            continue

        target_key = normalize_text(target)
        target_node = node_map.get(target_key, {})
        target_type = get_entity_type(target_node)
        target_category = get_node_category(target_node)

        if target_type in excluded_target_types:
            continue

        if target_category in excluded_target_categories:
            continue
        if target_key in excluded_temporal_terms:
            continue

        targets.append(str(target))

    return unique_values(targets)


def extract_user_goal(
    context_graph: Dict[str, Any],
    cognitive_state: Dict[str, Any]
) -> str:
    """
    Determine the user's goal.

    The cognitive-state goal is preferred because it has already
    combined context and graph signals. Graph targets are used as
    a fallback.
    """

    cognitive_goal = normalize_text(
        cognitive_state.get("primary_goal")
    )

    invalid_goals = {
        "",
        "unknown",
        "maintain_awareness",
        "understand_user_request"
    }

    if cognitive_goal not in invalid_goals:
        return cognitive_state.get("primary_goal")

    targets = extract_user_targets(context_graph)

    if targets:
        return f"reach_{normalize_text(targets[0])}"

    event_goal_mapping = {
        "navigation_request": "assist_navigation",
        "movement": "monitor_user_movement",
        "emergency": "provide_emergency_assistance",
        "fall_detected": "respond_to_fall",
        "conversation_event": "understand_user_request",
        "home_activity": "support_home_activity"
    }

    for event in extract_events(context_graph):
        event_type = normalize_text(event.get("event_type"))

        if event_type in event_goal_mapping:
            return event_goal_mapping[event_type]

    return cognitive_state.get(
        "primary_goal",
        "maintain_awareness"
    )


# ============================================================
# SPATIAL AND NEARBY CONTEXT
# ============================================================

def extract_nearby_entities(
    context_graph: Dict[str, Any],
    user_location: str
) -> List[str]:
    """
    Extract entities near the user or in the user's environment.

    Supports:
    - entity -> location relations
    - user -> entity proximity relations
    - entity -> user proximity relations
    """

    nearby = []
    normalized_location = normalize_text(user_location)

    location_relations = {
        "inside",
        "located_in",
        "at",
        "present_in",
        "contained_in"
    }

    proximity_relations = {
        "near",
        "nearby",
        "next_to",
        "beside",
        "in_front_of",
        "behind",
        "adjacent_to",
        "close_to",
        "approaching"
    }

    for edge in context_graph.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        source_key = normalize_text(source)
        target_key = normalize_text(target)
        relation = get_relation_type(edge)

        if (
            target_key == normalized_location
            and relation in location_relations
            and source_key != "user"
        ):
            nearby.append(source)

        if (
            source_key == normalized_location
            and relation in location_relations
            and target_key != "user"
        ):
            nearby.append(target)

        if (
            source_key == "user"
            and relation in proximity_relations
            and target_key
        ):
            nearby.append(target)

        if (
            target_key == "user"
            and relation in proximity_relations
            and source_key
        ):
            nearby.append(source)

    # Fallback for graphs that do not yet contain full spatial edges.
    if not nearby:
        nearby.extend(extract_objects(context_graph))
        nearby.extend(extract_people(context_graph))

    excluded_values = {
        "",
        "user",
        normalized_location
    }

    nearby = [
        entity
        for entity in nearby
        if normalize_text(entity) not in excluded_values
    ]

    return unique_values(nearby)


# ============================================================
# ENVIRONMENT CLASSIFICATION
# ============================================================

def classify_environment(
    user_location: str,
    analyzed_context: Dict[str, Any],
    context_graph: Optional[Dict[str, Any]] = None
) -> str:
    """
    Classify the environment using location, scene, entities,
    events and contextual evidence.
    """

    location = normalize_text(user_location)
    scene_type = normalize_text(
        analyzed_context.get("scene_type")
    )

    environment_rules = {
        "home_environment": {
            "home",
            "house",
            "apartment",
            "bedroom",
            "living_room",
            "kitchen"
        },
        "education_environment": {
            "school",
            "college",
            "university",
            "classroom",
            "library",
            "laboratory"
        },
        "traffic_environment": {
            "road",
            "street",
            "highway",
            "crosswalk",
            "intersection",
            "traffic"
        },
        "transportation_environment": {
            "station",
            "airport",
            "bus_stop",
            "railway_station",
            "metro",
            "terminal"
        },
        "shopping_environment": {
            "mall",
            "shopping_mall",
            "market",
            "shop",
            "supermarket",
            "store"
        },
        "healthcare_environment": {
            "hospital",
            "clinic",
            "pharmacy",
            "medical_center"
        },
        "work_environment": {
            "office",
            "workplace",
            "factory",
            "warehouse"
        },
        "outdoor_recreation_environment": {
            "park",
            "garden",
            "playground",
            "trail"
        },
        "religious_environment": {
            "temple",
            "church",
            "mosque",
            "gurudwara",
            "shrine"
        },
        "food_service_environment": {
            "restaurant",
            "cafe",
            "canteen",
            "food_court"
        },
        "emergency_environment": {
            "fire",
            "accident",
            "collision",
            "emergency"
        }
    }

    evidence_terms = {
        location,
        scene_type
    }

    if context_graph:
        for node in context_graph.get("nodes", []):
            evidence_terms.add(
                normalize_text(node.get("id"))
            )
            evidence_terms.add(get_entity_type(node))
            evidence_terms.add(get_event_type(node))

    for environment, keywords in environment_rules.items():
        for evidence in evidence_terms:
            if not evidence:
                continue

            if any(
                keyword == evidence or keyword in evidence
                for keyword in keywords
            ):
                return environment

    if scene_type and scene_type not in {
        "unknown",
        "general_environment"
    }:
        if scene_type.endswith("_environment"):
            return scene_type

        return f"{scene_type}_environment"

    return "general_environment"


# ============================================================
# SITUATION CLASSIFICATION
# ============================================================

def classify_situation_type(
    cognitive_state: Dict[str, Any],
    analyzed_context: Dict[str, Any],
    events: List[Dict[str, Any]]
) -> str:
    """
    Classify the active situation using safety, cognitive focus,
    risk and event evidence.
    """

    attention_focus = normalize_text(
        cognitive_state.get("attention_focus")
    )

    reasoning_mode = normalize_text(
        cognitive_state.get("reasoning_mode")
    )

    risk_level = normalize_text(
        analyzed_context.get("risk_level")
    )

    event_types = {
        normalize_text(event.get("event_type"))
        for event in events
    }

    emergency_events = {
        "emergency",
        "fall_detected",
        "fire_detected",
        "collision",
        "accident",
        "help_call"
    }

    navigation_events = {
        "navigation_request",
        "route_request",
        "movement",
        "destination_request"
    }

    conversation_events = {
        "conversation_event",
        "speech_event",
        "question",
        "request"
    }

    if (
        attention_focus == "emergency"
        or reasoning_mode == "emergency_response"
        or event_types.intersection(emergency_events)
    ):
        return "emergency_response"

    if (
        risk_level == "high"
        or attention_focus == "safety"
        or reasoning_mode == "risk_avoidance"
    ):
        return "safety_critical_situation"

    if (
        attention_focus == "hazard_monitoring"
        or reasoning_mode == "safety_monitoring"
    ):
        return "hazard_awareness"

    if (
        attention_focus == "navigation"
        or event_types.intersection(navigation_events)
    ):
        return "navigation_assistance"

    if event_types.intersection(conversation_events):
        return "communication_assistance"

    if "home_activity" in event_types:
        return "daily_activity_assistance"

    return "general_observation"


# ============================================================
# RELEVANT CONTEXT GENERATION
# ============================================================

def identify_relevant_entities(
    context_graph: Dict[str, Any],
    nearby_entities: List[str]
) -> List[Dict[str, Any]]:
    """
    Return structured information about nearby entities.
    """

    node_map = get_node_map(context_graph)
    relevant_entities = []

    for entity_id in nearby_entities:
        node = node_map.get(
            normalize_text(entity_id),
            {}
        )

        entity_data = {
            "name": entity_id,
            "category": node.get(
                "category",
                "entity"
            ),
            "entity_type": (
                node.get("entity_type")
                or node.get("type")
                or "unknown"
            )
        }

        if node.get("confidence") is not None:
            entity_data["confidence"] = node.get("confidence")

        relevant_entities.append(entity_data)

    return relevant_entities


def determine_context_confidence(
    user_location: str,
    user_activities: List[str],
    user_goal: str,
    events: List[Dict[str, Any]],
    nearby_entities: List[str]
) -> float:
    """
    Estimate situation-understanding completeness.

    This is not a machine-learning probability. It represents
    how much contextual evidence was available.
    """

    evidence_scores = []

    evidence_scores.append(
        1.0 if normalize_text(user_location) != "unknown" else 0.0
    )

    evidence_scores.append(
        1.0 if user_activities else 0.0
    )

    evidence_scores.append(
        1.0
        if normalize_text(user_goal) not in {
            "",
            "unknown",
            "maintain_awareness"
        }
        else 0.0
    )

    evidence_scores.append(
        1.0 if events else 0.0
    )

    evidence_scores.append(
        1.0 if nearby_entities else 0.0
    )

    confidence = sum(evidence_scores) / len(evidence_scores)

    return round(confidence, 2)


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================

def generate_human_summary(
    user_location: str,
    user_activities: List[str],
    user_goal: str,
    nearby_entities: List[str],
    audio_cues: List[str],
    safety_status: str,
    environment_type: str,
    situation_type: str
) -> str:
    """
    Generate a concise human-readable description.
    """

    activity_text = (
        ", ".join(user_activities)
        if user_activities
        else "no confirmed activity"
    )

    nearby_text = (
        ", ".join(nearby_entities)
        if nearby_entities
        else "no important nearby entities"
    )

    audio_text = (
        ", ".join(audio_cues)
        if audio_cues
        else "no significant audio cues"
    )

    return (
        f"The user is currently located in {user_location}. "
        f"The environment is classified as {environment_type}. "
        f"The user appears to be engaged in {activity_text}. "
        f"The current goal is {user_goal}. "
        f"Relevant nearby entities include {nearby_text}. "
        f"The audio context contains {audio_text}. "
        f"The situation is classified as {situation_type}, "
        f"and the current safety status is {safety_status}."
    )


# ============================================================
# MAIN SITUATION UNDERSTANDING
# ============================================================

def understand_situation(
    context_graph: Dict[str, Any],
    analyzed_context: Dict[str, Any],
    cognitive_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine graph, analyzed context and cognitive state into
    one structured situation representation.
    """

    print("\n[INFO] Understanding Current Situation...")

    user_location = extract_user_location(
        context_graph,
        analyzed_context
    )

    user_activities = extract_user_activity(
        context_graph
    )

    user_targets = extract_user_targets(
        context_graph
    )

    user_goal = extract_user_goal(
        context_graph,
        cognitive_state
    )

    nearby_entities = extract_nearby_entities(
        context_graph,
        user_location
    )

    relevant_entities = identify_relevant_entities(
        context_graph,
        nearby_entities
    )

    audio_cues = extract_audio_cues(
        context_graph
    )

    objects = extract_objects(
        context_graph
    )

    people = extract_people(
        context_graph
    )

    events = extract_events(
        context_graph
    )

    environment_type = classify_environment(
        user_location,
        analyzed_context,
        context_graph
    )

    situation_type = classify_situation_type(
        cognitive_state,
        analyzed_context,
        events
    )

    safety_status = cognitive_state.get(
        "safety_status",
        "unknown"
    )

    context_confidence = determine_context_confidence(
        user_location,
        user_activities,
        user_goal,
        events,
        nearby_entities
    )

    human_summary = generate_human_summary(
        user_location,
        user_activities,
        user_goal,
        nearby_entities,
        audio_cues,
        safety_status,
        environment_type,
        situation_type
    )

    situation = {
        "timestamp": str(datetime.now()),

        "user_state": {
            "location": user_location,
            "activities": user_activities,
            "goal": user_goal,
            "targets": user_targets
        },

        "environment_context": {
            "scene_type": analyzed_context.get(
                "scene_type",
                "unknown"
            ),
            "environment_type": environment_type,
            "nearby_entities": nearby_entities,
            "relevant_entities": relevant_entities,
            "objects": objects,
            "people": people,
            "audio_cues": audio_cues
        },

        "event_context": {
            "events": events,
            "event_count": len(events)
        },

        "cognitive_context": {
            "attention_focus": cognitive_state.get(
                "attention_focus",
                "unknown"
            ),
            "primary_goal": cognitive_state.get(
                "primary_goal",
                user_goal
            ),
            "reasoning_mode": cognitive_state.get(
                "reasoning_mode",
                "observation"
            ),
            "cognitive_priority": cognitive_state.get(
                "cognitive_priority",
                "low"
            ),
            "active_contexts": cognitive_state.get(
                "active_contexts",
                []
            )
        },

        "safety_context": {
            "risk_level": analyzed_context.get(
                "risk_level",
                "low"
            ),
            "safety_status": safety_status
        },

        "situation_type": situation_type,
        "context_confidence": context_confidence,
        "human_readable_summary": human_summary
    }

    print(f"[INFO] User Location: {user_location}")
    print(f"[INFO] User Activities: {user_activities}")
    print(f"[INFO] User Goal: {user_goal}")
    print(f"[INFO] User Targets: {user_targets}")
    print(f"[INFO] Nearby Entities: {nearby_entities}")
    print(f"[INFO] Environment Type: {environment_type}")
    print(f"[INFO] Situation Type: {situation_type}")
    print(f"[INFO] Safety Status: {safety_status}")
    print(f"[INFO] Context Confidence: {context_confidence}")

    print("[SUCCESS] Situation Understanding Complete")

    return situation


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Execute the Situation Understanding module.
    """

    ensure_output_directories()

    print("\n" + "=" * 60)
    print("NOONGIL-X SITUATION UNDERSTANDING")
    print("=" * 60)

    context_graph = load_json(
        CONTEXT_GRAPH_PATH
    )

    analyzed_context = load_json(
        ANALYZED_CONTEXT_PATH
    )

    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
    )

    if not context_graph:
        print(
            "[ERROR] context_graph.json "
            "Missing Or Empty"
        )
        return

    if not analyzed_context:
        print(
            "[ERROR] analyzed_context.json "
            "Missing Or Empty"
        )
        return

    if not cognitive_state:
        print(
            "[ERROR] cognitive_state.json "
            "Missing Or Empty"
        )
        return

    situation = understand_situation(
        context_graph,
        analyzed_context,
        cognitive_state
    )

    save_json(
        situation,
        SITUATION_UNDERSTANDING_PATH
    )

    print("\n" + "=" * 60)
    print("SITUATION UNDERSTANDING SUMMARY")
    print("=" * 60)
    print(json.dumps(situation, indent=4))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()