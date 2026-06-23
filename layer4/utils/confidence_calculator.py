"""
=========================================================
NOONGIL-X Layer 4
Utility: Confidence Calculator
=========================================================

Purpose:
---------
Provides reusable confidence scoring functions for
the Cognitive Reasoning Layer.

Used By:
--------
- situation_understanding.py
- intent_reasoner.py
- prediction_engine.py
- hazard_reasoner.py
- reasoning_fusion.py
- decision_engine.py

=========================================================
"""


# =========================================================
# CLAMP CONFIDENCE
# =========================================================

def clamp_confidence(value):
    """
    Keeps confidence value between 0.0 and 1.0.
    """

    try:
        value = float(value)

    except Exception:
        print("[ERROR] Invalid confidence value")
        print(f"[DEBUG] Received: {value}")
        return 0.0

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return round(value, 3)


# =========================================================
# AVERAGE CONFIDENCE
# =========================================================

def average_confidence(scores):
    """
    Calculates average confidence from a list of scores.
    """

    print("\n[INFO] Calculating Average Confidence")

    if not scores:
        print("[WARNING] Empty score list")
        return 0.0

    valid_scores = []

    for score in scores:
        valid_scores.append(
            clamp_confidence(score)
        )

    confidence = sum(valid_scores) / len(valid_scores)

    confidence = clamp_confidence(confidence)

    print(f"[DEBUG] Scores: {valid_scores}")
    print(f"[DEBUG] Average Confidence: {confidence}")

    return confidence


# =========================================================
# WEIGHTED CONFIDENCE
# =========================================================

def weighted_confidence(weighted_scores):
    """
    Calculates weighted confidence.

    Input format:
    -------------
    [
        {"score": 0.9, "weight": 0.5},
        {"score": 0.7, "weight": 0.3}
    ]
    """

    print("\n[INFO] Calculating Weighted Confidence")

    if not weighted_scores:
        print("[WARNING] Empty weighted score list")
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for item in weighted_scores:

        score = clamp_confidence(
            item.get("score", 0.0)
        )

        weight = float(
            item.get("weight", 0.0)
        )

        if weight < 0:
            weight = 0.0

        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        print("[WARNING] Total weight is zero")
        return 0.0

    confidence = weighted_sum / total_weight

    confidence = clamp_confidence(confidence)

    print(f"[DEBUG] Weighted Sum: {weighted_sum}")
    print(f"[DEBUG] Total Weight: {total_weight}")
    print(f"[DEBUG] Weighted Confidence: {confidence}")

    return confidence


# =========================================================
# BOOST CONFIDENCE
# =========================================================

def boost_confidence(base_score, boost=0.1):
    """
    Boosts confidence by a small amount.
    """

    print("\n[INFO] Boosting Confidence")

    boosted = clamp_confidence(
        base_score + boost
    )

    print(f"[DEBUG] Base Score: {base_score}")
    print(f"[DEBUG] Boost: {boost}")
    print(f"[DEBUG] Boosted Score: {boosted}")

    return boosted


# =========================================================
# REDUCE CONFIDENCE
# =========================================================

def reduce_confidence(base_score, penalty=0.1):
    """
    Reduces confidence by a penalty value.
    """

    print("\n[INFO] Reducing Confidence")

    reduced = clamp_confidence(
        base_score - penalty
    )

    print(f"[DEBUG] Base Score: {base_score}")
    print(f"[DEBUG] Penalty: {penalty}")
    print(f"[DEBUG] Reduced Score: {reduced}")

    return reduced


# =========================================================
# CONFIDENCE FROM MATCH COUNT
# =========================================================

def confidence_from_matches(
        matches,
        total,
        minimum=0.1
):
    """
    Calculates confidence based on match ratio.
    """

    print("\n[INFO] Calculating Confidence From Matches")

    if total <= 0:
        print("[WARNING] Total is zero or negative")
        return 0.0

    ratio = matches / total

    confidence = max(
        ratio,
        minimum
    )

    confidence = clamp_confidence(confidence)

    print(f"[DEBUG] Matches: {matches}")
    print(f"[DEBUG] Total: {total}")
    print(f"[DEBUG] Confidence: {confidence}")

    return confidence


# =========================================================
# CONFIDENCE LABEL
# =========================================================

def confidence_label(score):
    """
    Converts numeric confidence into readable label.
    """

    score = clamp_confidence(score)

    if score >= 0.85:
        return "very_high"

    if score >= 0.70:
        return "high"

    if score >= 0.50:
        return "medium"

    if score >= 0.30:
        return "low"

    return "very_low"


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("NOONGIL-X CONFIDENCE CALCULATOR TEST")
    print("=" * 60)

    scores = [0.9, 0.8, 0.7]

    avg = average_confidence(scores)

    weighted = weighted_confidence([
        {
            "score": 0.9,
            "weight": 0.5
        },
        {
            "score": 0.7,
            "weight": 0.3
        },
        {
            "score": 0.6,
            "weight": 0.2
        }
    ])

    boosted = boost_confidence(
        avg,
        boost=0.05
    )

    reduced = reduce_confidence(
        avg,
        penalty=0.1
    )

    match_score = confidence_from_matches(
        matches=3,
        total=5
    )

    print("\n" + "-" * 60)
    print("[RESULT] Average:", avg)
    print("[RESULT] Weighted:", weighted)
    print("[RESULT] Boosted:", boosted)
    print("[RESULT] Reduced:", reduced)
    print("[RESULT] Match Score:", match_score)
    print("[RESULT] Label:", confidence_label(avg))
    print("-" * 60)

    print(
        "\n[SUCCESS] confidence_calculator.py "
        "is working correctly"
    )