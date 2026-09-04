# NOONGIL-X Layer 2 User Test Dataset

This bundle contains 20 new image-and-audio test situations. It is separate
from the original eight Layer 2 scenarios.

Each scenario contains:

- `image.png`: visual input
- `audio.wav`: 16 kHz mono audio input
- `transcript.txt`: exact expected speech
- `ground_truth.json`: expected scene, object labels, sounds and OCR text

Case 13 is intentionally silent and is a negative-control audio test.

## Important evaluation note

The object annotations are class-label annotations, not bounding-box
annotations. They support label precision, recall and F1. They do not support
formal object-detection mAP. Inspect the pictures before publishing results and
adjust an object count if the generated picture differs visibly from its prompt.

## Suggested use

Open `layer2/final_test_app.py`, upload `image.png` and `audio.wav`, then copy
the corresponding values from `ground_truth.json` into the optional ground
truth fields.
