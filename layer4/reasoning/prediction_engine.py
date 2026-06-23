"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Prediction Engine
============================================================

Purpose:
Predict possible near-future outcomes from the current
situation, user intent, cognitive state, and hazards.

Inputs:
1. situation_understanding.json
2. cognitive_state.json
3. intent_reasoning.json
4. hazards.json

Output:
predictions.json

============================================================
"""

import os
import sys
from datetime import datetime
from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import (
    clamp_confidence,
    confidence_label
)

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

LAYER4_OUTPUT_DIR = os.path.join(
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


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

SITUATION_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "situation_understanding.json"
)

COGNITIVE_STATE_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "cognitive_state.json"
)

INTENT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "intent_reasoning.json"
)

HAZARDS_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "hazards.json"
)

PREDICTION_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "predictions.json"
)


# ============================================================
# KNOWLEDGE PATHS
# ============================================================

COMMON_SENSE_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "common_sense.json"
)

RISK_RULES_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "risk_rules.json"
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
# KNOWLEDGE-DRIVEN PREDICTION FUNCTIONS
# ============================================================

def predict_using_common_sense(
        situation,
        intent_result,
        cognitive_state,
        common_sense
):

    predictions = []

    user_activities = [
        str(activity).lower()
        for activity in situation.get("user_activities", [])
    ]

    nearby_entities = [
        str(entity).lower()
        for entity in situation.get("nearby_entities", [])
    ]

    primary_intent = intent_result.get(
        "primary_intent",
        ""
    )

    log_subsection("Predicting Using Common Sense")

    categories = common_sense.get("categories", {})

    prediction_facts = categories.get("prediction", [])
    mobility_facts = categories.get("mobility", [])
    navigation_facts = categories.get("navigation", [])

    if "walking" in user_activities:

        predictions.append({
            "prediction_type": "movement_continuation",
            "predicted_outcome": "user_will_likely_continue_walking",
            "probability": 0.75,
            "time_horizon": "immediate",
            "source": "common_sense",
            "matched_knowledge": prediction_facts,
            "reason": (
                "Common sense indicates people generally continue "
                "their current movement direction."
            ),
            "recommended_preparation": (
                "Continue path monitoring and obstacle awareness."
            )
        })

    if primary_intent == "navigate_to_destination":

        predictions.append({
            "prediction_type": "goal_directed_movement",
            "predicted_outcome": "user_will_likely_move_toward_destination",
            "probability": 0.82,
            "time_horizon": "short_term",
            "source": "common_sense",
            "matched_knowledge": navigation_facts,
            "reason": (
                "Common sense indicates people usually move toward "
                "their intended destination."
            ),
            "recommended_preparation": (
                "Keep destination tracking and route guidance active."
            )
        })

    if any(
        entity in nearby_entities
        for entity in ["obstacle", "stairs", "crowd", "vehicle"]
    ):

        predictions.append({
            "prediction_type": "navigation_difficulty_possible",
            "predicted_outcome": "user_may_need_safer_navigation_support",
            "probability": 0.70,
            "time_horizon": "immediate",
            "source": "common_sense",
            "matched_knowledge": mobility_facts,
            "reason": (
                "Common sense indicates obstacles and crowded paths "
                "increase navigation difficulty."
            ),
            "recommended_preparation": (
                "Prepare obstacle warnings and safer path suggestions."
            )
        })

    return predictions


def predict_using_navigation_rules(
        situation,
        intent_result,
        navigation_rules
):

    predictions = []

    primary_intent = intent_result.get(
        "primary_intent",
        ""
    )

    destination = intent_result.get(
        "destination",
        None
    )

    environment_type = str(
        situation.get("environment_type", "")
    ).lower()

    nearby_entities = [
        str(entity).lower()
        for entity in situation.get("nearby_entities", [])
    ]

    log_subsection("Predicting Using Navigation Rules")

    for rule in navigation_rules.get("rules", []):

        rule_name = str(rule.get("name", "")).lower()
        rule_id = rule.get("rule_id")
        actions = rule.get("actions", [])
        priority = rule.get("priority", "medium")

        if (
            primary_intent == "navigate_to_destination"
            and
            (
                "navigate" in rule_name
                or
                "destination" in rule_name
            )
        ):

            outcome = "user_will_continue_navigation"

            if destination:
                outcome = f"user_will_move_toward_{destination}"

            predictions.append({
                "prediction_type": "navigation_progress",
                "predicted_outcome": outcome,
                "probability": 0.85,
                "time_horizon": "short_term",
                "source": "navigation_rules",
                "matched_rule_id": rule_id,
                "matched_rule_name": rule.get("name"),
                "priority": priority,
                "reason": (
                    "Navigation rule matched because the user's "
                    "primary intent is navigation."
                ),
                "recommended_preparation": actions
            })

        if (
            "route blocked" in rule_name
            and
            "blocked" in nearby_entities
        ):

            predictions.append({
                "prediction_type": "route_change_possible",
                "predicted_outcome": "user_may_need_alternate_route",
                "probability": 0.82,
                "time_horizon": "immediate",
                "source": "navigation_rules",
                "matched_rule_id": rule_id,
                "matched_rule_name": rule.get("name"),
                "priority": priority,
                "reason": "Blocked route condition may affect navigation.",
                "recommended_preparation": actions
            })

        if (
            "indoor" in rule_name
            and
            environment_type in [
                "indoor",
                "indoor_environment",
                "shopping_mall"
            ]
        ):

            predictions.append({
                "prediction_type": "indoor_navigation_support",
                "predicted_outcome": "user_may_need_landmark_based_guidance",
                "probability": 0.72,
                "time_horizon": "short_term",
                "source": "navigation_rules",
                "matched_rule_id": rule_id,
                "matched_rule_name": rule.get("name"),
                "priority": priority,
                "reason": "Indoor navigation rule matched.",
                "recommended_preparation": actions
            })

        if (
            "outdoor" in rule_name
            and
            environment_type in [
                "outdoor",
                "outdoor_environment",
                "traffic_area",
                "urban_environment"
            ]
        ):

            predictions.append({
                "prediction_type": "outdoor_navigation_support",
                "predicted_outcome": "user_may_need_gps_and_traffic_awareness",
                "probability": 0.74,
                "time_horizon": "short_term",
                "source": "navigation_rules",
                "matched_rule_id": rule_id,
                "matched_rule_name": rule.get("name"),
                "priority": priority,
                "reason": "Outdoor navigation rule matched.",
                "recommended_preparation": actions
            })

    return predictions


def predict_using_risk_rules(
        hazards,
        risk_rules
):

    predictions = []

    hazard_list = hazards.get("hazards", [])

    overall_risk_level = hazards.get(
        "overall_risk_level",
        "safe"
    )

    log_subsection("Predicting Using Risk Rules")

    if overall_risk_level in ["medium", "high", "critical"]:

        predictions.append({
            "prediction_type": "risk_response_needed",
            "predicted_outcome": "system_may_need_safety_first_decision",
            "probability": 0.82,
            "time_horizon": "immediate",
            "source": "risk_rules",
            "reason": f"Overall risk level is {overall_risk_level}.",
            "recommended_preparation": (
                "Prioritize hazard-aware decisions and safety guidance."
            )
        })

    for hazard in hazard_list:

        hazard_type = str(
            hazard.get("hazard_type", "")
        ).lower()

        for rule in risk_rules.get("rules", []):

            rule_name = str(
                rule.get("name", "")
            ).lower()

            rule_id = rule.get("rule_id")
            risk_score = rule.get("risk_score", 50)
            action = rule.get("recommended_action", "")

            matched = False

            if "fire" in hazard_type and "fire" in rule_name:
                matched = True

            elif "smoke" in hazard_type and "smoke" in rule_name:
                matched = True

            elif "vehicle" in hazard_type and "vehicle" in rule_name:
                matched = True

            elif "crowd" in hazard_type and "crowd" in rule_name:
                matched = True

            elif "obstacle" in hazard_type and "obstacle" in rule_name:
                matched = True

            elif "lost" in hazard_type and "lost" in rule_name:
                matched = True

            elif "help" in hazard_type and "help" in rule_name:
                matched = True

            if matched:

                predictions.append({
                    "prediction_type": "risk_based_future_outcome",
                    "predicted_outcome": (
                        f"{hazard_type}_may_require_{action}"
                    ),
                    "probability": clamp_confidence(risk_score / 100),
                    "time_horizon": "immediate",
                    "source": "risk_rules",
                    "matched_rule_id": rule_id,
                    "matched_rule_name": rule.get("name"),
                    "risk_score": risk_score,
                    "reason": (
                        f"Hazard {hazard_type} matched risk rule "
                        f"{rule_id}."
                    ),
                    "recommended_preparation": action
                })

    return predictions


def predict_using_emergency_rules(
        intent_result,
        hazards,
        emergency_rules
):

    predictions = []

    primary_intent = intent_result.get(
        "primary_intent",
        ""
    )

    urgency = intent_result.get(
        "urgency",
        "low"
    )

    hazard_list = hazards.get(
        "hazards",
        []
    )

    hazard_text = " ".join([
        str(hazard.get("hazard_type", "")).lower()
        for hazard in hazard_list
    ])

    log_subsection("Predicting Using Emergency Rules")

    for rule in emergency_rules.get("rules", []):

        rule_name = str(rule.get("name", "")).lower()
        rule_id = rule.get("rule_id")
        priority = str(rule.get("priority", "medium")).lower()
        actions = rule.get("actions", [])

        matched = False

        if primary_intent == "seek_emergency_help":
            matched = True

        if urgency == "critical":
            matched = True

        if "fire" in hazard_text and "fire" in rule_name:
            matched = True

        if "smoke" in hazard_text and "smoke" in rule_name:
            matched = True

        if "lost" in hazard_text and "lost" in rule_name:
            matched = True

        if "help" in hazard_text and "help" in rule_name:
            matched = True

        if "medical" in hazard_text and "medical" in rule_name:
            matched = True

        if matched:

            probability = 0.80

            if priority in ["critical", "maximum"]:
                probability = 0.95

            elif priority == "high":
                probability = 0.88

            predictions.append({
                "prediction_type": "emergency_rule_prediction",
                "predicted_outcome": (
                    "emergency_response_may_be_required"
                ),
                "probability": probability,
                "time_horizon": "immediate",
                "source": "emergency_rules",
                "matched_rule_id": rule_id,
                "matched_rule_name": rule.get("name"),
                "priority": priority,
                "reason": (
                    f"Emergency rule {rule_id} matched current "
                    "intent or hazard context."
                ),
                "recommended_preparation": actions
            })

    return predictions


















# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicate_predictions(predictions):

    unique = {}

    for prediction in predictions:

        key = prediction.get(
            "prediction_type",
            "unknown"
        )

        old_prediction = unique.get(key)

        if old_prediction is None:

            unique[key] = prediction

        else:

            if prediction.get(
                "probability",
                0.0
            ) > old_prediction.get(
                "probability",
                0.0
            ):

                unique[key] = prediction

    return list(
        unique.values()
    )


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_overall_prediction_confidence(predictions):

    if not predictions:

        return 0.0

    total_probability = 0.0

    for prediction in predictions:

        total_probability += prediction.get(
            "probability",
            0.0
        )

    average_probability = total_probability / len(
        predictions
    )

    return clamp_confidence(
        average_probability
    )


def find_most_likely_prediction(predictions):

    if not predictions:

        return None

    return max(
        predictions,
        key=lambda prediction: prediction.get(
            "probability",
            0.0
        )
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_prediction_summary(
        predictions,
        overall_confidence
):

    if not predictions:

        return (
            "No clear future outcome predicted from "
            "the current context."
        )

    most_likely = find_most_likely_prediction(
        predictions
    )

    return (
        f"{len(predictions)} possible future outcome(s) predicted. "
        f"Most likely outcome: "
        f"{most_likely.get('predicted_outcome')}. "
        f"Overall prediction confidence: {overall_confidence}."
    )


# ============================================================
# MAIN PREDICTION ENGINE
# ============================================================

def predict_future_outcomes(
           situation,
        cognitive_state,
        intent_result,
        hazards,
        common_sense,
        navigation_rules,
        risk_rules,
        emergency_rules
):

    log_section("Prediction Engine")

    prediction_groups = {
    "common_sense_predictions": predict_using_common_sense(
        situation,
        intent_result,
        cognitive_state,
        common_sense
    ),

    "navigation_rule_predictions": predict_using_navigation_rules(
        situation,
        intent_result,
        navigation_rules
    ),

    "risk_rule_predictions": predict_using_risk_rules(
        hazards,
        risk_rules
    ),

    "emergency_rule_predictions": predict_using_emergency_rules(
        intent_result,
        hazards,
        emergency_rules
    )
}











    all_predictions = []

    for group_name, group_predictions in prediction_groups.items():

        log_debug(
            f"{group_name}: {len(group_predictions)} prediction(s)"
        )

        all_predictions.extend(
            group_predictions
        )

    all_predictions = remove_duplicate_predictions(
        all_predictions
    )

    overall_confidence = calculate_overall_prediction_confidence(
        all_predictions
    )

    most_likely = find_most_likely_prediction(
        all_predictions
    )

    result = {
        "timestamp": str(datetime.now()),
        "prediction_count": len(all_predictions),
        "prediction_groups": prediction_groups,
        "predictions": all_predictions,
        "most_likely_prediction": most_likely,
        "overall_prediction_confidence": overall_confidence,
        "confidence_label": confidence_label(
            overall_confidence
        ),
        "summary": generate_prediction_summary(
            all_predictions,
            overall_confidence
        )
    }

    log_info(f"Prediction Count: {len(all_predictions)}")
    log_info(
        f"Overall Prediction Confidence: {overall_confidence}"
    )

    if most_likely:

        log_info(
            f"Most Likely Outcome: "
            f"{most_likely.get('predicted_outcome')}"
        )

    log_success("Prediction Engine Complete")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    module_start("PREDICTION ENGINE")

    situation = load_json(
        SITUATION_PATH
    )

    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
    )
    common_sense = load_json(COMMON_SENSE_PATH)
    navigation_rules = load_json(NAVIGATION_RULES_PATH)
    risk_rules = load_json(RISK_RULES_PATH)
    emergency_rules = load_json(EMERGENCY_RULES_PATH)
    intent_result = load_json(
        INTENT_PATH
    )

    hazards = load_json(
        HAZARDS_PATH,
        default={
            "hazards": [],
            "overall_risk_level": "safe"
        }
    )

    if not situation:

        log_error(
            "situation_understanding.json Missing Or Empty"
        )

        return

    if not cognitive_state:

        log_error(
            "cognitive_state.json Missing Or Empty"
        )

        return

    if not intent_result:

        log_error(
            "intent_reasoning.json Missing Or Empty"
        )

        return

    if not hazards:

        log_warning(
            "hazards.json Missing Or Empty, using safe default"
        )

        hazards = {
            "hazards": [],
            "overall_risk_level": "safe"
        }

    result = predict_future_outcomes(
    situation,
    cognitive_state,
    intent_result,
    hazards,
    common_sense,
    navigation_rules,
    risk_rules,
    emergency_rules
)

    save_json(
        result,
        PREDICTION_OUTPUT_PATH
    )

    module_end("PREDICTION ENGINE")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()