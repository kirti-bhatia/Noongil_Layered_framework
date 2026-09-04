# NOONGIL-X Layer 3 — Dynamic Context Graph and Memory

Layer 3 converts the structured perception output from Layer 2 into a contextual representation. It connects entities, events and relationships in a dynamic graph and maintains episodic and semantic memory for reasoning.

## Pipeline

```text
Layer 2 output
→ Entity detection
→ Event detection
→ Relation detection
→ Context-graph construction and update
→ Episodic-memory storage and retrieval
→ Semantic extraction, storage and retrieval
```

## Main capabilities

- Entity extraction from perceived objects, scenes, text and activities
- Event identification
- Relationship extraction
- Context-graph construction, update and queries
- Episodic-memory storage, retrieval and summarization
- Semantic-fact extraction and persistent semantic memory

## Structure

```text
layer3/
├── entity extraction/
├── graph_memory/
├── episodic_memory/
├── semantic_memory/
└── pipeline/layer3_pipeline.py
```

## Input and output

Canonical input: `output/layer2/layer2_output.json`

Important outputs in `output/layer3/`:

- `entities.json`
- `events.json`
- `relations.json`
- `context_graph.json`
- `episodic_memory.json`
- `episodic_summary.json`
- `semantic_memory.json`

## Run

The unified project runner automatically moves the selected Layer 2 result to the canonical input location and executes the Layer 3 stages:

```bash
python run_noongil.py
```

## Prototype status

Layer 3 provides a functional file-based graph and memory pipeline. Persistent database storage and continuous real-time graph updates remain future deployment work.

