# NOONGIL-X Layer 2 — Multimodal Perception

Layer 2 transforms a synchronized Layer 1 sensor packet into a structured description of the observed environment. It combines visual, textual, audio and spatial evidence for use by the context-graph layer.

## Main capabilities

- Scene classification
- Object detection and tracking
- Activity recognition
- OCR and text interpretation
- Speech recognition and environmental-sound detection
- Relative-depth estimation
- Object direction and proximity localization
- Multimodal feature alignment and confidence-aware fusion
- Structured Layer 2 JSON generation

## Models used by the prototype

- YOLOv8 for object detection
- CLIP for scene classification
- Whisper for speech recognition
- PaddleOCR for text recognition
- AST for sound-event classification
- MiDaS for monocular relative-depth estimation

## Structure

```text
layer2/
├── input_reception/
├── vision_perception/
├── audio/
├── text/
├── spatial/
├── multimodal_fusion/
├── confidence/
├── output/
├── schemas/
├── config/
└── run_layer2.py
```

## Input and output

Input: `layer1_output_test_scenarios/<scenario>/layer1_sensor_packet.json`

Output: `output/layer2/pipeline/<scenario>/<packet_id>_layer2_output.json`

The output contains the detected scene, objects, text, sounds, speech transcript, user activity, location, depth/proximity observations, confidence and diagnostic information.

## Run

```bash
python -m layer2.run_layer2 --scenario park_walking --mode navigation
```

## Limitations

Monocular depth primarily provides relative proximity. Metric distance must be presented as an estimate and depends on valid camera calibration or another metric-depth source. Model packages and weights must be available before the first run.
