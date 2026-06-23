"""
=========================================================
NOONGIL-X Layer 4
Utility: Logger
=========================================================

Purpose:
---------
Provides standardized logging across the
Cognitive Reasoning Layer.

Used By:
--------
- context_analyzer.py
- cognitive_state_manager.py
- situation_understanding.py
- intent_reasoner.py
- prediction_engine.py
- hazard_reasoner.py
- reasoning_fusion.py
- decision_engine.py
- explanation_engine.py
- reasoning_pipeline.py

=========================================================
"""

from datetime import datetime


# =========================================================
# TIMESTAMP
# =========================================================

def get_timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# INFO
# =========================================================

def log_info(message):

    print(
        f"[{get_timestamp()}] "
        f"[INFO] "
        f"{message}"
    )


# =========================================================
# SUCCESS
# =========================================================

def log_success(message):

    print(
        f"[{get_timestamp()}] "
        f"[SUCCESS] "
        f"{message}"
    )


# =========================================================
# WARNING
# =========================================================

def log_warning(message):

    print(
        f"[{get_timestamp()}] "
        f"[WARNING] "
        f"{message}"
    )


# =========================================================
# ERROR
# =========================================================

def log_error(message):

    print(
        f"[{get_timestamp()}] "
        f"[ERROR] "
        f"{message}"
    )


# =========================================================
# DEBUG
# =========================================================

def log_debug(message):

    print(
        f"[{get_timestamp()}] "
        f"[DEBUG] "
        f"{message}"
    )


# =========================================================
# SECTION HEADER
# =========================================================

def log_section(title):

    print("\n")
    print("=" * 60)

    print(title.upper())

    print("=" * 60)


# =========================================================
# SUBSECTION HEADER
# =========================================================

def log_subsection(title):

    print("\n")
    print("-" * 60)

    print(title)

    print("-" * 60)


# =========================================================
# MODULE START
# =========================================================

def module_start(module_name):

    print("\n")
    print("=" * 60)

    print(
        f"NOONGIL-X | {module_name}"
    )

    print("=" * 60)


# =========================================================
# MODULE END
# =========================================================

def module_end(module_name):

    print("\n")

    log_success(
        f"{module_name} Completed"
    )

    print("=" * 60)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    module_start(
        "LOGGER TEST"
    )

    log_info(
        "Loading Context"
    )

    log_debug(
        "Node Count = 10"
    )

    log_warning(
        "No Semantic Facts Found"
    )

    log_error(
        "Example Error Message"
    )

    log_success(
        "Test Successful"
    )

    module_end(
        "LOGGER TEST"
    )