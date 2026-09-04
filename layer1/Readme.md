# NOONGIL-X Layer 1 — Multimodal Input

Layer 1 acquires, validates and synchronizes multimodal sensor observations for the NOONGIL-X assistive-AI prototype. It converts camera, microphone, location, motion, interaction, device and wearable inputs into a consistent sensor packet for Layer 2.

## Main capabilities

- Multimodal input reception and validation
- Phone-sensor simulation for prototype testing
- Timestamp-based modality synchronization
- Modality-level and packet-level confidence estimation
- Missing-modality handling and recovery metadata
- Sensor-packet construction and output dispatch

## Structure

```text
layer1/
├── acquisition/     # Input receiver and phone-sensor simulator
├── modalities/      # Vision, audio, spatial, motion and device adapters
├── processing/      # Synchronization, confidence and recovery
├── output/          # Sensor-packet builder and dispatcher
├── schemas/         # Layer 1 packet definitions
├── config/          # Runtime settings and paths
└── run_layer1.py    # Layer 1 entry point
```

## Output

Layer 1 produces a JSON sensor packet containing synchronized modality data, media paths, timestamps, confidence scores and packet metadata. For scenario-based evaluation, each runnable scenario supplies `layer1_sensor_packet.json` as the saved Layer 1 output.

## Run

From the project root:

```bash
python -m layer1.run_layer1 --help
```

## Prototype status

Layer 1 is implemented as a modular prototype and has self-tests for reception, synchronization, confidence estimation and packet generation. The submitted demonstration uses saved scenario packets rather than continuous live mobile sensing.
