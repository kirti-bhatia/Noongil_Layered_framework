# NOONGIL-X Layer 4 — Reasoning and Intelligence

Layer 4 interprets the context graph produced by Layer 3 and generates a safety-aware decision and user-facing explanation. The submitted prototype uses implemented contextual and rule-based reasoning components.

## Pipeline

```text
Context analysis
→ Cognitive-state estimation
→ Situation understanding
→ Intent reasoning
→ Hazard reasoning
→ Prediction
→ Reasoning fusion
→ Decision generation
→ Explanation generation
```

## Main capabilities

- Context and cognitive-state analysis
- Situation understanding
- User-intent inference
- Hazard and risk evaluation
- Short-term prediction
- Confidence-aware reasoning fusion
- Primary-action selection
- Evidence-based user explanation

## Structure

```text
layer4/
├── context_processing/
├── reasoning/
├── decision/
├── knowledge/
├── pipeline/reasoning_pipeline.py
└── utils/
```

## Input and output

Primary input: `output/layer3/context_graph.json`

Important outputs in `output/layer4/`:

- `analyzed_context.json`
- `cognitive_state.json`
- `situation_understanding.json`
- `intent_reasoning.json`
- `hazards.json`
- `predictions.json`
- `reasoning_fusion.json`
- `decision_output.json`
- `explanation_output.json`
- `reasoning_pipeline_report.json`

## Run

```bash
python -m layer4.pipeline.reasoning_pipeline
```

## Prototype status and limitations

The rule-based reasoning, decision and explanation path is implemented. Empty LLM-client, LLM-reasoner and schema-validation placeholders are not part of the demonstrated execution path and must not be described as completed features. Layer 4 currently operates on scenario-generated context rather than a continuously updated live environment.

