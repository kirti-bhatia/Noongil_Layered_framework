from pathlib import Path


LAYER4_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAYER4_DIR.parent

LAYER3_OUTPUT_DIR = PROJECT_ROOT / "output" / "layer3"
LAYER4_OUTPUT_DIR = PROJECT_ROOT / "output" / "layer4"
KNOWLEDGE_DIR = LAYER4_DIR / "knowledge"


# Layer 3 inputs
CONTEXT_GRAPH_PATH = LAYER3_OUTPUT_DIR / "context_graph.json"
ENTITIES_PATH = LAYER3_OUTPUT_DIR / "entities.json"
RELATIONS_PATH = LAYER3_OUTPUT_DIR / "relations.json"
EVENTS_PATH = LAYER3_OUTPUT_DIR / "events.json"
EPISODIC_MEMORY_PATH = LAYER3_OUTPUT_DIR / "episodic_memory.json"
EPISODIC_SUMMARY_PATH = LAYER3_OUTPUT_DIR / "episodic_summary.json"
SEMANTIC_MEMORY_PATH = LAYER3_OUTPUT_DIR / "semantic_memory.json"


# Layer 4 intermediate outputs
ANALYZED_CONTEXT_PATH = LAYER4_OUTPUT_DIR / "analyzed_context.json"
COGNITIVE_STATE_PATH = LAYER4_OUTPUT_DIR / "cognitive_state.json"
SITUATION_UNDERSTANDING_PATH = (
    LAYER4_OUTPUT_DIR / "situation_understanding.json"
)
INTENT_REASONING_PATH = LAYER4_OUTPUT_DIR / "intent_reasoning.json"
HAZARDS_PATH = LAYER4_OUTPUT_DIR / "hazards.json"
PREDICTIONS_PATH = LAYER4_OUTPUT_DIR / "predictions.json"
FUSED_REASONING_PATH = LAYER4_OUTPUT_DIR / "fused_reasoning.json"
DECISION_OUTPUT_PATH = LAYER4_OUTPUT_DIR / "decision_output.json"
EXPLANATION_OUTPUT_PATH = LAYER4_OUTPUT_DIR / "explanation_output.json"


# Knowledge files
COMMONSENSE_RULES_PATH = KNOWLEDGE_DIR / "commonsense_rules.json"
EMERGENCY_RULES_PATH = KNOWLEDGE_DIR / "emergency_rules.json"
NAVIGATION_RULES_PATH = KNOWLEDGE_DIR / "navigation_rules.json"
RISK_RULES_PATH = KNOWLEDGE_DIR / "risk_rules.json"


def ensure_output_directories() -> None:
    """Create required output directories."""
    LAYER3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LAYER4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)