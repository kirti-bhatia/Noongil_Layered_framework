"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Hazard Reasoner
============================================================
Purpose:
Detect possible hazards using:
1. situation_understanding.json
2. cognitive_state.json
3. intent_reasoning.json
4. context_graph.json
============================================================
"""

import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output","layer4")

sys.path.append(BASE_DIR)

from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import clamp_confidence, confidence_label


SITUATION_PATH = os.path.join(PROJECT_OUTPUT_DIR, "situation_understanding.json")
COGNITIVE_STATE_PATH = os.path.join(PROJECT_OUTPUT_DIR, "cognitive_state.json")
INTENT_PATH = os.path.join(PROJECT_OUTPUT_DIR, "intent_reasoning.json")
CONTEXT_GRAPH_PATH = os.path.join(PROJECT_OUTPUT_DIR, "context_graph.json")

HAZARD_OUTPUT_PATH = os.path.join(PROJECT_OUTPUT_DIR, "hazards.json")

KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

COMMON_SENSE_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "common_sense.json"
)

EMERGENCY_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "emergency_rules.json"
)

RISK_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "risk_rules.json"
)

NAVIGATION_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "navigation_rules.json"
)
# ============================================================
# GRAPH HELPERS
# ============================================================

def extract_graph_objects(context_graph):
    objects = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "object":
            objects.append(node.get("id", "").lower())

    return objects


def extract_graph_events(context_graph):
    events = []

    for node in context_graph.get("nodes", []):
        if node.get("category") == "event":
            events.append({
                "id": node.get("id", ""),
                "event_type": node.get("event_type", "unknown")
            })

    return events


def extract_audio_cues(context_graph):
    audio = []

    for node in context_graph.get("nodes", []):
        if node.get("entity_type") == "audio":
            audio.append(node.get("id", "").lower())

    return audio


# ============================================================
# HAZARD DETECTION RULES
# ============================================================

def detect_hazards_using_knowledge(
        situation,
        intent_result,
        context_graph,
        risk_rules,
        emergency_rules,
        navigation_rules
):
    """
    Knowledge Driven Hazard Detection
    """

    hazards = []

    nearby_entities = [
        str(x).lower()
        for x in situation.get(
            "nearby_entities",
            []
        )
    ]

    audio_cues = [
        str(x).lower()
        for x in situation.get(
            "audio_cues",
            []
        )
    ]

    intent = intent_result.get(
        "primary_intent",
        ""
    ).lower()

    # =====================================
    # RISK RULES
    # =====================================

    for rule in risk_rules.get(
            "rules",
            []
    ):

        rule_name = rule.get(
            "name",
            ""
        ).lower()

        matched = False

        if "fire" in rule_name and "fire" in str(nearby_entities):
            matched = True

        elif "smoke" in rule_name and "smoke" in str(nearby_entities):
            matched = True

        elif "lost child" in rule_name and (
                "child" in str(nearby_entities)
                and
                "help" in str(audio_cues)
        ):
            matched = True

        elif "person calling for help" in rule_name and (
                "help" in str(audio_cues)
        ):
            matched = True

        elif "heavy crowd" in rule_name and (
                "crowd" in str(nearby_entities)
        ):
            matched = True

        if matched:

            hazards.append({

                "rule_id":
                    rule.get("rule_id"),

                "hazard_type":
                    rule_name,

                "severity":
                    rule.get(
                        "priority",
                        "medium"
                    ),

                "risk_score":
                    rule.get(
                        "risk_score",
                        50
                    ),

                "recommended_action":
                    rule.get(
                        "recommended_action",
                        ""
                    ),

                "source":
                    "risk_rules",

                "confidence":
                    clamp_confidence(
                        rule.get(
                            "risk_score",
                            50
                        ) / 100
                    )
            })

    # =====================================
    # EMERGENCY RULES
    # =====================================

    for rule in emergency_rules.get(
            "rules",
            []
    ):

        name = rule.get(
            "name",
            ""
        ).lower()

        if "fire" in name and "fire" in str(
                nearby_entities
        ):

            hazards.append({

                "rule_id":
                    rule.get(
                        "rule_id"
                    ),

                "hazard_type":
                    "emergency_fire",

                "severity":
                    rule.get(
                        "priority",
                        "critical"
                    ),

                "recommended_actions":
                    rule.get(
                        "actions",
                        []
                    ),

                "source":
                    "emergency_rules",

                "confidence":
                    0.95
            })

    # =====================================
    # NAVIGATION RULES
    # =====================================

    if intent == "navigate_to_destination":

        for rule in navigation_rules.get(
                "rules",
                []
        ):

            if "route blocked" in rule.get(
                    "name",
                    ""
            ).lower():

                hazards.append({

                    "rule_id":
                        rule.get(
                            "rule_id"
                        ),

                    "hazard_type":
                        "navigation_risk",

                    "severity":
                        rule.get(
                            "priority",
                            "medium"
                        ),

                    "recommended_actions":
                        rule.get(
                            "actions",
                            []
                        ),

                    "source":
                        "navigation_rules",

                    "confidence":
                        0.75
                })

    return hazards

# ============================================================
# RISK SCORING
# ============================================================

def severity_to_score(severity):
    severity = severity.lower()

    if severity == "critical":
        return 1.0

    if severity == "high":
        return 0.8

    if severity == "medium":
        return 0.55

    if severity == "low":
        return 0.3

    return 0.1


def calculate_overall_risk(hazards):
    if not hazards:
        return 0.0, "safe"

    scores = []

    for hazard in hazards:
        severity_score = severity_to_score(
            hazard.get("severity", "low")
        )

        confidence = hazard.get("confidence", 0.5)

        # scores.append(severity_score * confidence)
        scores.append(
    hazard.get(
        "risk_score",
        severity_score * confidence * 100
    ) / 100
)

    risk_score = clamp_confidence(max(scores))  #clamp is from utility file
    if risk_score >= 0.85:
        risk_level = "critical"
    elif risk_score >= 0.70:
        risk_level = "high"
    elif risk_score >= 0.45:
        risk_level = "medium"
    elif risk_score >= 0.20:
        risk_level = "low"
    else:
        risk_level = "safe"

    return risk_score, risk_level


def remove_duplicate_hazards(hazards):
    unique = {}
    for hazard in hazards:
        key = hazard.get("hazard_type", "unknown")
        old = unique.get(key)

        if old is None:
            unique[key] = hazard
        else:
            if hazard.get("confidence", 0) > old.get("confidence", 0):
                unique[key] = hazard

    return list(unique.values())


def generate_hazard_summary(hazards, risk_score, risk_level):
    if not hazards:
        return "No significant hazards detected. Current situation appears safe."

    hazard_names = [
        hazard.get("hazard_type", "unknown")
        for hazard in hazards
    ]

    return (
        f"Detected {len(hazards)} possible hazard(s): "
        f"{', '.join(hazard_names)}. "
        f"Overall risk level is {risk_level} "
        f"with risk score {risk_score}."
    )


# ============================================================
# MAIN HAZARD REASONING
# ============================================================

def reason_about_hazards(
   
        situation,
        cognitive_state,
        intent_result,
        context_graph,
        risk_rules,
        emergency_rules,
        navigation_rules

):
    log_section("Hazard Reasoning")

    hazards = []

    hazards.extend(
    detect_hazards_using_knowledge(
        situation,
        intent_result,
        context_graph,
        risk_rules,
        emergency_rules,
        navigation_rules
    )
)


    hazards = remove_duplicate_hazards(hazards)

    risk_score, risk_level = calculate_overall_risk(hazards)

    result = {
        "timestamp": str(datetime.now()),
        "hazard_count": len(hazards),
        "hazards": hazards,
        "overall_risk_score": risk_score,
        "overall_risk_level": risk_level,
        "risk_confidence_label": confidence_label(risk_score),
        "safety_status": cognitive_state.get("safety_status", "unknown"),
        "summary": generate_hazard_summary(
            hazards,
            risk_score,
            risk_level
        )
    }

    log_info(f"Hazard Count: {len(hazards)}")
    log_info(f"Overall Risk Score: {risk_score}")
    log_info(f"Overall Risk Level: {risk_level}")
    log_success("Hazard Reasoning Complete")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    module_start("HAZARD REASONER")

    situation = load_json(SITUATION_PATH)
    cognitive_state = load_json(COGNITIVE_STATE_PATH)
    intent_result = load_json(INTENT_PATH)
    context_graph = load_json(CONTEXT_GRAPH_PATH)

    risk_rules = load_json(
    RISK_RULES_PATH)
    emergency_rules = load_json(
    EMERGENCY_RULES_PATH)

    navigation_rules = load_json(
    NAVIGATION_RULES_PATH
)

    if not situation:
        log_error("situation_understanding.json Missing Or Empty")
        return

    if not cognitive_state:
        log_error("cognitive_state.json Missing Or Empty")
        return

    if not intent_result:
        log_error("intent_reasoning.json Missing Or Empty")
        return

    if not context_graph:
        log_error("context_graph.json Missing Or Empty")
        return

    # result = reason_about_hazards(
    #     situation,
    #     cognitive_state,
    #     intent_result,
    #     context_graph
    # )
    result = reason_about_hazards(
    situation,
    cognitive_state,
    intent_result,
    context_graph,
    risk_rules,
    emergency_rules,
    navigation_rules
)

    save_json(result, HAZARD_OUTPUT_PATH)

    module_end("HAZARD REASONER")


if __name__ == "__main__":
    main()