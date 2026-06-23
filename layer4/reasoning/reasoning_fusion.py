"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module : Fusion Prediction
============================================================

Purpose:
Fuse prediction results with hazards, intent, situation,
and cognitive state to create a final future-risk understanding.

Inputs:
1. predictions.json
2. hazards.json
3. intent_reasoning.json
4. situation_understanding.json
5. cognitive_state.json

Output:
fusion_prediction.json

============================================================
"""

import os
import sys
from datetime import datetime


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

LAYER4_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "layer4"
)

os.makedirs(
    LAYER4_OUTPUT_DIR,
    exist_ok=True
)

sys.path.append(BASE_DIR)


from utils.file_loader import load_json
from utils.json_writer import save_json
from utils.logger import *
from utils.confidence_calculator import (
    clamp_confidence,
    confidence_label
)


# ============================================================
# INPUT PATHS
# ============================================================

PREDICTIONS_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "predictions.json"
)

HAZARDS_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "hazards.json"
)

INTENT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "intent_reasoning.json"
)

SITUATION_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "situation_understanding.json"
)

COGNITIVE_STATE_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "cognitive_state.json"
)


# ============================================================
# OUTPUT PATH
# ============================================================

FUSION_PREDICTION_OUTPUT_PATH = os.path.join(
    LAYER4_OUTPUT_DIR,
    "fusion_prediction.json"
)


# ============================================================
# RISK SCORE MAPPING
# ============================================================

def risk_level_to_score(risk_level):

    risk_level = str(risk_level).lower()

    if risk_level == "critical":
        return 1.0

    if risk_level == "high":
        return 0.80

    if risk_level == "medium":
        return 0.55

    if risk_level == "low":
        return 0.30

    if risk_level == "safe":
        return 0.0

    return 0.10


def urgency_to_score(urgency):

    urgency = str(urgency).lower()

    if urgency == "critical":
        return 1.0

    if urgency == "high":
        return 0.75

    if urgency == "medium":
        return 0.45

    if urgency == "low":
        return 0.20

    return 0.10


# ============================================================
# EXTRACT IMPORTANT SIGNALS
# ============================================================

def get_most_likely_prediction(predictions):

    most_likely = predictions.get(
        "most_likely_prediction",
        None
    )

    if most_likely:
        return most_likely

    prediction_list = predictions.get(
        "predictions",
        []
    )

    if not prediction_list:
        return None

    return max(
        prediction_list,
        key=lambda item: item.get(
            "probability",
            0.0
        )
    )


def extract_high_priority_predictions(predictions):

    high_priority = []

    prediction_list = predictions.get(
        "predictions",
        []
    )

    for prediction in prediction_list:

        probability = prediction.get(
            "probability",
            0.0
        )

        time_horizon = prediction.get(
            "time_horizon",
            ""
        )

        if (
            probability >= 0.75
            or time_horizon == "immediate"
        ):

            high_priority.append(
                prediction
            )

    return high_priority


def extract_future_risk_predictions(predictions):

    future_risks = []

    prediction_list = predictions.get(
        "predictions",
        []
    )

    risk_keywords = [
        "risk",
        "emergency",
        "hazard",
        "vehicle",
        "crossing",
        "help",
        "unsafe",
        "fall",
        "distress"
    ]

    for prediction in prediction_list:

        combined_text = (
            prediction.get("prediction_type", "")
            + " "
            + prediction.get("predicted_outcome", "")
            + " "
            + prediction.get("reason", "")
        ).lower()

        for keyword in risk_keywords:

            if keyword in combined_text:

                future_risks.append(
                    prediction
                )

                break

    return future_risks


# ============================================================
# FUSION LOGIC
# ============================================================

def calculate_fused_prediction_score(
        predictions,
        hazards,
        intent_result,
        cognitive_state
):

    prediction_confidence = predictions.get(
        "overall_prediction_confidence",
        0.0
    )

    hazard_risk_score = hazards.get(
        "overall_risk_score",
        0.0
    )

    urgency_score = urgency_to_score(
        intent_result.get(
            "urgency",
            "low"
        )
    )

    cognitive_priority = cognitive_state.get(
        "cognitive_priority",
        "low"
    )

    cognitive_score = urgency_to_score(
        cognitive_priority
    )

    fused_score = (
        prediction_confidence * 0.35
        + hazard_risk_score * 0.35
        + urgency_score * 0.15
        + cognitive_score * 0.15
    )

    return clamp_confidence(
        fused_score
    )


def classify_future_state(fused_score, hazards):

    risk_level = hazards.get(
        "overall_risk_level",
        "safe"
    )

    if risk_level == "critical" or fused_score >= 0.85:
        return "critical_future_risk"

    if risk_level == "high" or fused_score >= 0.70:
        return "high_attention_required"

    if risk_level == "medium" or fused_score >= 0.45:
        return "moderate_monitoring_required"

    if risk_level == "low" or fused_score >= 0.20:
        return "low_risk_continuation"

    return "safe_continuation"


def decide_prediction_focus(
        most_likely_prediction,
        hazards,
        intent_result,
        situation
):

    risk_level = hazards.get(
        "overall_risk_level",
        "safe"
    )

    primary_intent = intent_result.get(
        "primary_intent",
        ""
    )

    environment_type = situation.get(
        "environment_type",
        ""
    )

    if risk_level in ["critical", "high"]:
        return "safety_first_prediction"

    if primary_intent == "navigate_to_destination":
        return "navigation_progress_prediction"

    if environment_type == "traffic_area":
        return "traffic_safety_prediction"

    if most_likely_prediction:

        predicted_outcome = most_likely_prediction.get(
            "predicted_outcome",
            ""
        )

        if "emergency" in predicted_outcome:
            return "emergency_prediction"

        if "walking" in predicted_outcome:
            return "movement_prediction"

    return "general_context_prediction"


def generate_fusion_action_hint(
        future_state,
        prediction_focus
):

    if future_state == "critical_future_risk":

        return (
            "Prepare immediate safety response and prioritize "
            "emergency-aware decision making."
        )

    if future_state == "high_attention_required":

        return (
            "Increase monitoring, reduce uncertainty, and prepare "
            "fast safety guidance."
        )

    if prediction_focus == "navigation_progress_prediction":

        return (
            "Continue route guidance, destination tracking, and "
            "walking-speed adaptation."
        )

    if prediction_focus == "traffic_safety_prediction":

        return (
            "Prepare traffic crossing support, vehicle alerts, and "
            "obstacle awareness."
        )

    if future_state == "safe_continuation":

        return (
            "Continue normal contextual monitoring without unnecessary alerts."
        )

    return (
        "Maintain adaptive monitoring and prepare assistance if context changes."
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_fusion_summary(
        future_state,
        prediction_focus,
        most_likely_prediction,
        fused_score
):

    if most_likely_prediction:

        outcome = most_likely_prediction.get(
            "predicted_outcome",
            "unknown"
        )

    else:

        outcome = "no clear predicted outcome"

    return (
        f"Fusion prediction state is {future_state}. "
        f"Prediction focus is {prediction_focus}. "
        f"Most likely outcome is {outcome}. "
        f"Fused prediction score is {fused_score}."
    )


# ============================================================
# MAIN FUSION FUNCTION
# ============================================================

def fuse_predictions(
        predictions,
        hazards,
        intent_result,
        situation,
        cognitive_state
):

    log_section("Fusion Prediction")

    most_likely_prediction = get_most_likely_prediction(
        predictions
    )

    high_priority_predictions = extract_high_priority_predictions(
        predictions
    )

    future_risk_predictions = extract_future_risk_predictions(
        predictions
    )

    fused_score = calculate_fused_prediction_score(
        predictions,
        hazards,
        intent_result,
        cognitive_state
    )

    future_state = classify_future_state(
        fused_score,
        hazards
    )

    prediction_focus = decide_prediction_focus(
        most_likely_prediction,
        hazards,
        intent_result,
        situation
    )

    action_hint = generate_fusion_action_hint(
        future_state,
        prediction_focus
    )

    result = {
        "timestamp": str(datetime.now()),

        "fusion_type": "prediction_fusion",

        "future_state": future_state,

        "prediction_focus": prediction_focus,

        "most_likely_prediction": most_likely_prediction,

        "high_priority_prediction_count": len(
            high_priority_predictions
        ),

        "high_priority_predictions": high_priority_predictions,

        "future_risk_prediction_count": len(
            future_risk_predictions
        ),

        "future_risk_predictions": future_risk_predictions,

        "fused_prediction_score": fused_score,

        "fused_prediction_confidence_label": confidence_label(
            fused_score
        ),

        "current_risk_level": hazards.get(
            "overall_risk_level",
            "safe"
        ),

        "current_intent": intent_result.get(
            "primary_intent",
            "unknown"
        ),

        "current_environment": situation.get(
            "environment_type",
            "unknown"
        ),

        "recommended_reasoning_focus": action_hint,

        "summary": generate_fusion_summary(
            future_state,
            prediction_focus,
            most_likely_prediction,
            fused_score
        )
    }

    log_info(f"Future State: {future_state}")
    log_info(f"Prediction Focus: {prediction_focus}")
    log_info(f"Fused Prediction Score: {fused_score}")
    log_info(
        f"High Priority Predictions: "
        f"{len(high_priority_predictions)}"
    )
    log_info(
        f"Future Risk Predictions: "
        f"{len(future_risk_predictions)}"
    )

    log_success("Fusion Prediction Complete")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    module_start("FUSION PREDICTION")

    predictions = load_json(
        PREDICTIONS_PATH
    )

    hazards = load_json(
        HAZARDS_PATH,
        default={
            "hazards": [],
            "overall_risk_level": "safe",
            "overall_risk_score": 0.0
        }
    )

    intent_result = load_json(
        INTENT_PATH
    )

    situation = load_json(
        SITUATION_PATH
    )

    cognitive_state = load_json(
        COGNITIVE_STATE_PATH
    )

    if not predictions:

        log_error(
            "predictions.json Missing Or Empty"
        )

        return

    if not hazards:

        log_warning(
            "hazards.json Missing Or Empty, using safe default"
        )

        hazards = {
            "hazards": [],
            "overall_risk_level": "safe",
            "overall_risk_score": 0.0
        }

    if not intent_result:

        log_error(
            "intent_reasoning.json Missing Or Empty"
        )

        return

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

    result = fuse_predictions(
        predictions,
        hazards,
        intent_result,
        situation,
        cognitive_state
    )

    save_json(
        result,
        FUSION_PREDICTION_OUTPUT_PATH
    )

    module_end("FUSION PREDICTION")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()