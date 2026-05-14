# Archived Work

This folder contains earlier iterations of the project that are preserved for historical reference but are **not part of the final deliverable**. The current production scripts and trained model are at the repository root.

## Contents

- `old_inference_scripts/` — Seven prior versions of the live inference script (`live_asl.py` through `live_asl7april21.py`). These reference older model files that have moved to `old_models/` and will not run without path updates. Preserved for documentation of the iteration process.

- `old_capture_scripts/` — Two earlier versions of the webcam capture script before the final 256×256 square-crop implementation.

- `old_models/` — Trained model weights from versions v3, v4, v5, and the 5-class specialist model. The corresponding `class_names_*.json` files needed to load each model are alongside them.

- `old_models/pre_v3_36class_era/` — The earliest models from February and early March, before the project migrated to the Kaggle ASL Alphabet dataset and reduced from 36 classes (digits + lowercase letters) to 27 classes (A-Z + space). Includes the original from-scratch CNN baseline and the first MobileNetV2 transfer-learning attempts.

- `old_notebooks/` — Earlier Colab training notebooks. `Untitled0.ipynb` was the v3 trainer; `MidtermReadyModel.ipynb` was the v3-v4 transition; `April7smallDataTest.ipynb` was the 5-class specialist experiment.

- `project_brief.pdf` — The original CPS 485 assignment description.

For the full timeline mapping these versions to the work that produced them, see [`../NOTES.md`](../NOTES.md).