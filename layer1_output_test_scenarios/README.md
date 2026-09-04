# Layer 1 Output Test Scenarios

These fixtures let Layer 2 run before live Layer 1 integration. Each scenario contains a real frame, a mono 16-kHz WAV sample, sensor readings, a Layer 1-style packet, and the expected Layer 2 semantic output used by the frozen Layer 3.

## Per-scenario files

- `frame.jpg`: visual input for vision, object detection, OCR, and depth modules.
- `audio.wav`: speech input for ASR/audio processing.
- `sensor_data.json`: GPS, motion, interaction, device, and wearable test readings.
- `layer1_sensor_packet.json`: the complete simulated Layer 1 handoff.
- `expected_layer2_output.json`: comparison target compatible with the existing Layer 3 tests.

Use `manifest.json` to enumerate all scenarios. Media paths inside packets are relative to this folder.
