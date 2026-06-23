"""
=========================================================
NOONGIL-X Layer 3
Entity Detection Module

File:
entity_detector.py

Purpose:
--------
Receives Layer 2 Unified Context Representation (UCR)
and extracts entities for Context Graph Memory.

Output:
--------
detected_entities.json

Author:
NOONGIL-X Research Architecture
=========================================================
"""
from dotenv import load_dotenv
import os
import json
import uuid
from pathlib import Path

import spacy

# from layer3_pipeline import BASE_DIR

load_dotenv()
# =========================================================
# CONFIGURATION
# =========================================================
# BASE_DIR = Path(__file__).resolve().parent.parent
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

os.makedirs(
    LAYER3_OUTPUT_DIR,
    exist_ok=True
)

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
    "entities.json"
)
# Load SpaCy Model
print("[INFO] Loading SpaCy Model...")

try:
    nlp = spacy.load("en_core_web_sm")
    print("[SUCCESS] SpaCy Loaded")
except Exception as e:
    print("[ERROR] Failed to load SpaCy")
    print(e)
    exit()


# =========================================================
# ENTITY TYPE MAPPING
# =========================================================

LOCATION_SCENES = {
    "park",
    "home",
    "classroom",
    "street",
    "road",
    "shopping_mall",
    "mall",
    "cafe",
    "hospital",
    "office"
}


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def generate_entity_id():
    """
    Generate unique entity ID.
    """
    return f"ENT_{str(uuid.uuid4())[:8]}"


def create_entity(name, entity_type, source):
    """
    Standardized entity structure.
    """

    return {
        "entity_id": generate_entity_id(),
        "name": name,
        "type": entity_type,
        "source": source
    }


# =========================================================
# LOAD JSON INPUT
# =========================================================

def load_layer2_output(file_path):
    """
    Load Layer 2 output JSON.
    """

    print("\n[INFO] Loading Layer 2 Output")

    try:

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print("[SUCCESS] JSON Loaded")

        return data

    except Exception as e:

        print("[ERROR] Failed Loading JSON")
        print(e)

        return None


# =========================================================
# SCENE ENTITY EXTRACTION
# =========================================================

def extract_scene_entities(data):

    print("\n[INFO] Extracting Scene Entities")

    entities = []

    scene = data.get("scene", {})

    scene_type = scene.get("type")

    if scene_type:

        entities.append(
            create_entity(
                scene_type,
                "location",
                "scene"
            )
        )

        print(f"[FOUND] Scene Entity: {scene_type}")

    return entities


# =========================================================
# OBJECT ENTITY EXTRACTION
# =========================================================

def extract_object_entities(data):

    print("\n[INFO] Extracting Object Entities")

    entities = []

    objects = data.get("objects", [])

    for obj in objects:

        label = obj.get("label")

        if not label:
            continue

        entity = create_entity(
            label,
            "object",
            "vision"
        )

        entities.append(entity)

        print(f"[FOUND] Object Entity: {label}")

    return entities


# =========================================================
# SOUND ENTITY EXTRACTION
# =========================================================

def extract_sound_entities(data):

    print("\n[INFO] Extracting Sound Entities")

    entities = []

    sounds = data.get("sounds", [])

    for sound in sounds:

        label = sound.get("label")

        if not label:
            continue

        entity = create_entity(
            label,
            "audio",
            "sound"
        )

        entities.append(entity)

        print(f"[FOUND] Sound Entity: {label}")

    return entities


# =========================================================
# ACTIVITY ENTITY EXTRACTION
# =========================================================

def extract_activity_entities(data):

    print("\n[INFO] Extracting Activity Entities")

    entities = []

    activity = data.get("user_activity", {})

    state = activity.get("state")

    if state:

        entity = create_entity(
            state,
            "activity",
            "activity"
        )

        entities.append(entity)

        print(f"[FOUND] Activity: {state}")

    return entities


# =========================================================
# SPEECH NLP ENTITY EXTRACTION
# =========================================================

def extract_speech_entities(data):

    print("\n[INFO] Extracting Speech Entities")

    entities = []

    transcripts = data.get("speech_transcript", [])

    for sentence in transcripts:

        print(f"\n[NLP] Processing: {sentence}")

        doc = nlp(sentence)

        # -------------------------------------------------
        # Named Entity Recognition
        # -------------------------------------------------

        for ent in doc.ents:

            entity = create_entity(
                ent.text,
                ent.label_,
                "speech"
            )

            entities.append(entity)

            print(
                f"[FOUND] NLP Entity -> "
                f"{ent.text} ({ent.label_})"
            )

        # -------------------------------------------------
        # Noun Extraction
        # Helps detect gate, store, hospital etc.
        # -------------------------------------------------

        for token in doc:

            if token.pos_ == "NOUN":

                entity = create_entity(
                    token.text.lower(),
                    "noun",
                    "speech"
                )

                entities.append(entity)

                print(
                    f"[FOUND] Noun Entity -> "
                    f"{token.text}"
                )

    return entities


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(entity_list):

    print("\n[INFO] Removing Duplicate Entities")

    unique = {}
    
    for entity in entity_list:

        key = (
            entity["name"].lower(),
            entity["type"]
        )

        if key not in unique:
            unique[key] = entity

    result = list(unique.values())

    print(
        f"[INFO] Final Unique Entities: "
        f"{len(result)}"
    )

    return result


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_entities(data, entities):

    print("\n[INFO] Saving Results")

    output = {

        "timestamp":
        data.get("timestamp"),

        "entity_count":
        len(entities),

        "entities":
        entities
    }

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
        f"[SUCCESS] Saved to {OUTPUT_FILE}"
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X LAYER 3")
    print("ENTITY DETECTOR")
    print("=" * 60)

    data = load_layer2_output(INPUT_FILE)

    if data is None:
        return

    entities = []

    entities.extend(
        extract_scene_entities(data)
    )

    entities.extend(
        extract_object_entities(data)
    )

    entities.extend(
        extract_sound_entities(data)
    )

    entities.extend(
        extract_activity_entities(data)
    )

    entities.extend(
        extract_speech_entities(data)
    )

    entities = remove_duplicates(entities)

    print("\n[INFO] Entity Summary")

    for entity in entities:

        print(
            f"{entity['entity_id']} | "
            f"{entity['name']} | "
            f"{entity['type']}"
        )

    save_entities(data, entities)

    print("\n[SUCCESS] ENTITY DETECTION COMPLETE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()