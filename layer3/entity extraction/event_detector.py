"""
=========================================================
NOONGIL-X Layer 3
Event Detector
=========================================================

Purpose:
---------
Detect cognitive events from Layer 2 output.

Input:
------
test/park_walking.json

Output:
-------
output/detected_events.json
=========================================================
"""

import json
import os
import uuid
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

LAYER3_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "layer3"
)
OUTPUT_FILE = Path(LAYER3_OUTPUT_DIR) / "events.json"
os.makedirs(
    LAYER3_OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "layer2_output.json"
)

# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "events.json"
)
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

# =========================================================
# EVENT ID GENERATOR
# =========================================================

def generate_event_id():

    return f"EVT_{str(uuid.uuid4())[:8]}"


# =========================================================
# LOAD JSON
# =========================================================

def load_layer2_output():

    print("\n[INFO] Loading Layer 2 Output")

    try:

        with open(
            INPUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print("[SUCCESS] JSON Loaded")

        return data

    except Exception as e:

        print("[ERROR] Failed Loading JSON")
        print(e)

        return None


# =========================================================
# EVENT CREATOR
# =========================================================

def create_event(
        event_type,
        actor="user",
        target=None,
        activity=None,
        location=None,
        timestamp=None
):

    return {

        "event_id":
        generate_event_id(),

        "event_type":
        event_type,

        "actor":
        actor,

        "target":
        target,

        "activity":
        activity,

        "location":
        location,

        "timestamp":
        timestamp
    }


# =========================================================
# NAVIGATION EVENTS
# =========================================================

def detect_navigation_events(data):

    print("\n[INFO] Checking Navigation Events")

    events = []

    transcripts = data.get(
        "speech_transcript",
        []
    )

    location = (
        data
        .get("scene", {})
        .get("type")
    )

    timestamp = data.get(
        "timestamp"
    )

    keywords = [

        "guide",
        "navigate",
        "direction",
        "where",
        "gate",
        "hospital",
        "take me"
    ]

    for sentence in transcripts:

        lower = sentence.lower()

        if any(
            keyword in lower
            for keyword in keywords
        ):

            print(
                f"[EVENT FOUND] "
                f"Navigation Request"
            )

            events.append(

                create_event(

                    event_type=
                    "navigation_request",

                    target=
                    "destination",

                    location=
                    location,

                    timestamp=
                    timestamp
                )
            )

    return events


# =========================================================
# MOVEMENT EVENTS
# =========================================================

def detect_movement_events(data):

    print("\n[INFO] Checking Movement Events")

    events = []

    activity = (
        data
        .get("user_activity", {})
        .get("state")
    )

    location = (
        data
        .get("scene", {})
        .get("type")
    )

    timestamp = data.get(
        "timestamp"
    )

    movement_states = [

        "walking",
        "running",
        "cycling",
        "driving"
    ]

    if activity in movement_states:

        print(
            f"[EVENT FOUND] "
            f"Movement: {activity}"
        )

        events.append(

            create_event(

                event_type=
                "movement",

                activity=
                activity,

                location=
                location,

                timestamp=
                timestamp
            )
        )

    return events


# =========================================================
# EMERGENCY EVENTS
# =========================================================

def detect_emergency_events(data):

    print("\n[INFO] Checking Emergency Events")

    events = []

    transcripts = data.get(
        "speech_transcript",
        []
    )

    sounds = data.get(
        "sounds",
        []
    )

    timestamp = data.get(
        "timestamp"
    )

    location = (
        data
        .get("scene", {})
        .get("type")
    )

    strong_emergency_words = [

        
        "emergency",
        "accident",
        "danger",
        "fell"
    ]

    weak_emergency_words = ["help"]

    # Speech detection

    for sentence in transcripts:

        lower = sentence.lower()
        has_strong = any( word in lower for word in strong_emergency_words )

        has_weak_with_strong = any( word in lower for word in weak_emergency_words ) 
        if has_strong or has_weak_with_strong:
            print(
                "[EVENT FOUND] "
                "Emergency From Speech"
            )

            events.append(

                create_event(

                    event_type=
                    "emergency",

                    location=
                    location,

                    timestamp=
                    timestamp
                )
            )

    # Sound detection

    for sound in sounds:

        label = sound.get(
            "label",
            ""
        )

        if (
            "alarm" in label
            or
            "siren" in label
        ):

            print(
                "[EVENT FOUND] "
                "Emergency From Sound"
            )

            events.append(

                create_event(

                    event_type=
                    "emergency",

                    location=
                    location,

                    timestamp=
                    timestamp
                )
            )

    return events
# =========================================================
# Conversation Events
# =========================================================

def detect_conversation_events(data):

    print("\n[INFO] Checking Conversation Events")

    events = []

    transcripts = data.get("speech_transcript", [])

    if transcripts:
        events.append({
            "event_id": generate_event_id(),
            "event_type": "conversation_event",
            "description": "User is involved in a conversation.",
            "source": "speech",
            "confidence": 0.80
        })

        print("[EVENT FOUND] Conversation Event")

    return events
# =========================================================
# SCENE BASED EVENTS
# =========================================================

def detect_scene_events(data):

    print("\n[INFO] Checking Scene Events")

    events = []

    scene = (
        data
        .get("scene", {})
        .get("type")
    )

    timestamp = data.get(
        "timestamp"
    )

    if scene == "classroom":

        print(
            "[EVENT FOUND] Learning"
        )

        events.append(

            create_event(

                event_type=
                "learning",

                location=
                scene,

                timestamp=
                timestamp
            )
        )

    elif scene == "shopping_mall":

        print(
            "[EVENT FOUND] Shopping"
        )

        events.append(

            create_event(

                event_type=
                "shopping",

                location=
                scene,

                timestamp=
                timestamp
            )
        )

    elif scene == "home":

        print(
            "[EVENT FOUND] Home Activity"
        )

        events.append(

            create_event(

                event_type=
                "home_activity",

                location=
                scene,

                timestamp=
                timestamp
            )
        )

    return events


# =========================================================
# SAVE EVENTS
# =========================================================

def save_events(events):

    print("\n[INFO] Saving Events")

    output = {

        "event_count":
        len(events),

        "events":
        events
    }

    os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Saved To "
        f"{OUTPUT_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X EVENT DETECTOR")
    print("=" * 60)

    data = load_layer2_output()

    if data is None:
        return

    events = []

    events.extend(
        detect_navigation_events(data)
    )

    events.extend(
        detect_movement_events(data)
    )

    events.extend(
        detect_emergency_events(data)
    )

    events.extend(
        detect_scene_events(data)
    )
    events.extend(
    detect_conversation_events(data)
)

    print(
        f"\n[INFO] Total Events: "
        f"{len(events)}"
    )

    save_events(events)

    print(
        "\n[SUCCESS] EVENT DETECTION COMPLETE"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()